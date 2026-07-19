"""Feature-based cost estimator (D6): predict token cost from more than size.

The base :class:`~agentpause.estimator.Estimator` predicts the next step from
one signal — the current context length — plus a moving-average correction.
That's a good scalar model, but consumption also depends on *which tool* runs,
*which model* answers, the temperature, the typical output length. This
estimator learns a small ridge-regularized linear model over those features, so
the prediction adapts to the shape of the workload instead of just its size.

Drop-in by design. It subclasses ``Estimator`` and keeps the exact same method
signatures the scheduler calls (``estimate(input_tokens)``, ``sigma(...)``,
``record(input_tokens, realized)``), so it works with no wiring changes — with
only the context-length feature it degrades to a sensible linear-in-size model.
To feed the richer features without touching the scheduler, set them as ambient
context for the next step::

    est = FeatureEstimator()
    sched = PredictiveScheduler(backend=..., telemetry=..., estimator=est)

    est.set_context(tool="web_search", model="gpt-4o", temperature=0.7)
    reply = session.call()      # estimate() and record() pick up the context

Until ``min_samples`` steps are seen (or if the fit is degenerate) it falls back
to the base estimator, so early decisions are never wild. It also learns
per-step **latency** the same way (``estimate_latency``), feeding the time
budget dimension of :func:`~agentpause.risk.decide`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .estimator import Estimator

__all__ = ["FeatureEstimator"]

CTX = "ctx"     # the always-present feature: current context length in tokens


def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting.

    Returns None if the system is singular (caller falls back).
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r != col and abs(m[r][col]) > 1e-15:
                f = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


class _Ridge:
    """A tiny standardized ridge regressor over a fixed set of feature keys."""

    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.keys: List[str] = []
        self.mu: Dict[str, float] = {}
        self.sd: Dict[str, float] = {}
        self.w: Optional[List[float]] = None   # [intercept, *per-key weights]

    def fit(self, samples: List[Tuple[Dict[str, float], float]]) -> bool:
        n = len(samples)
        keys = sorted({k for f, _ in samples for k in f})
        if not keys or n < len(keys) + 1:
            self.w = None
            return False
        # standardize each feature so ridge penalizes them on equal footing
        self.keys = keys
        for k in keys:
            vals = [f.get(k, 0.0) for f, _ in samples]
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / n
            self.mu[k] = mu
            self.sd[k] = var ** 0.5 or 1.0
        rows = [self._vec(f) for f, _ in samples]
        y = [t for _, t in samples]
        d = len(keys) + 1                      # + intercept
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d
        for vec, target in zip(rows, y):
            for i in range(d):
                xty[i] += vec[i] * target
                for j in range(d):
                    xtx[i][j] += vec[i] * vec[j]
        for i in range(1, d):                  # regularize weights, not intercept
            xtx[i][i] += self.ridge
        w = _solve(xtx, xty)
        self.w = w
        return w is not None

    def _vec(self, feat: Dict[str, float]) -> List[float]:
        out = [1.0]
        for k in self.keys:
            out.append((feat.get(k, 0.0) - self.mu[k]) / self.sd[k])
        return out

    def predict(self, feat: Dict[str, float]) -> Optional[float]:
        if self.w is None:
            return None
        vec = self._vec(feat)
        return sum(wi * xi for wi, xi in zip(self.w, vec))


class FeatureEstimator(Estimator):
    """Ridge-regression cost estimator over workload features (D6).

    Args:
        min_samples: steps required before the regression is trusted; below it
            (or on a degenerate fit) the base estimator is used.
        ridge: L2 strength on the standardized feature weights.
        sample_cap: most recent samples kept for the fit (bounds cost + drift).
        Other args forwarded to :class:`Estimator`.
    """

    def __init__(
        self,
        min_samples: int = 6,
        ridge: float = 1.0,
        sample_cap: int = 500,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.min_samples = min_samples
        self.sample_cap = sample_cap
        self._ambient: Dict[str, float] = {}
        self._samples: List[Tuple[Dict[str, float], float, Optional[float]]] = []
        self._tokens = _Ridge(ridge)
        self._latency = _Ridge(ridge)
        self._tok_resid: List[float] = []

    # -- ambient features ---------------------------------------------------

    def set_context(self, **features: float) -> "FeatureEstimator":
        """Set features for the NEXT estimate/record (cleared after record).

        Non-numeric values are one-hot encoded (``tool="search"`` becomes the
        feature ``tool=search`` = 1.0), so categorical inputs just work.
        """
        self._ambient = self._encode(features)
        return self

    @staticmethod
    def _encode(features: Dict[str, object]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in features.items():
            if isinstance(v, bool):
                out[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                out[k] = float(v)
            else:
                out[f"{k}={v}"] = 1.0            # one-hot for categoricals
        return out

    def _features(self, input_tokens: int) -> Dict[str, float]:
        f = dict(self._ambient)
        f[CTX] = float(input_tokens)
        return f

    # -- estimation (overrides) ---------------------------------------------

    def estimate(self, input_tokens: int) -> int:
        pred = self._tokens.predict(self._features(input_tokens))
        if pred is None or len(self._samples) < self.min_samples:
            return super().estimate(input_tokens)
        return max(self.min_estimate, int(round(pred)))

    def sigma(self, fallback_estimate: int) -> float:
        if self._tokens.w is not None and len(self._tok_resid) >= 4:
            n = len(self._tok_resid)
            mean = sum(self._tok_resid) / n
            return (sum((r - mean) ** 2 for r in self._tok_resid) / n) ** 0.5
        return super().sigma(fallback_estimate)

    def estimate_latency(self, input_tokens: int) -> Optional[float]:
        """Predicted seconds for the next step, or None until enough data."""
        if len(self._samples) < self.min_samples:
            return None
        return self._latency.predict(self._features(input_tokens))

    # -- learning (override) ------------------------------------------------

    def record(self, input_tokens: int, realized: int,
               latency: Optional[float] = None, **features: float) -> None:
        """Learn from a completed step; keeps base stats AND the regression.

        Extra keyword features merge over the ambient context; both are cleared
        afterward so they never leak into the next step.
        """
        feat = self._features(input_tokens)
        feat.update(self._encode(features))
        super().record(input_tokens, realized)          # base ε/σ/quantile intact
        self._samples.append((feat, float(realized), latency))
        if len(self._samples) > self.sample_cap:
            self._samples = self._samples[-self.sample_cap:]
        self._refit()
        self._ambient = {}

    def _refit(self) -> None:
        if len(self._samples) < self.min_samples:
            return
        tok = [(f, y) for f, y, _ in self._samples]
        if self._tokens.fit(tok):
            self._tok_resid = [y - (self._tokens.predict(f) or 0.0) for f, y in tok]
        lat = [(f, s) for f, _, s in self._samples if s is not None]
        if len(lat) >= self.min_samples:
            self._latency.fit(lat)

    # -- persistence (extends F9.3) -----------------------------------------

    def to_dict(self) -> dict:
        state = super().to_dict()
        state["feature_samples"] = [
            {"f": f, "y": y, "lat": s} for f, y, s in self._samples[-self.sample_cap:]
        ]
        return state

    def load(self, state: dict) -> None:
        super().load(state)
        self._samples = [
            (dict(s["f"]), float(s["y"]), s.get("lat"))
            for s in state.get("feature_samples", [])
        ]
        self._refit()
