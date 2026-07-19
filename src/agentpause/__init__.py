"""agentpause — predictive scheduling for autonomous LLM agents.

Suspend an agent gracefully *before* it hits a provider rate limit, and resume
it without redoing work. The core works on any provider (cloud or local); true
KV-cache warm start is an optional plugin for self-hosted runtimes.
"""

from .errors import (
    AgentPauseError,
    BackendError,
    CheckpointError,
    RateLimitHit,
    TelemetryError,
)
from .breaker import CircuitBreaker, CircuitOpenError
from .estimator import Estimator
from .regression import FeatureEstimator
from .fallback import FallbackBackend
from .refill import RegimeDetector
from .retry import RetryPolicy
from .risk import Budget, Decision, RiskModel, decide, should_checkpoint
from .router import BudgetRouter
from .coordinator import MultiAgentCoordinator
from .state import Checkpoint, StateStore
from .tools import ToolQuota
from .scheduler import PredictiveScheduler, Session

__version__ = "0.3.0"

__all__ = [
    "PredictiveScheduler",
    "Session",
    "Estimator",
    "FeatureEstimator",
    "RiskModel",
    "Budget",
    "Decision",
    "decide",
    "should_checkpoint",
    "Checkpoint",
    "StateStore",
    "RetryPolicy",
    "FallbackBackend",
    "BudgetRouter",
    "MultiAgentCoordinator",
    "CircuitBreaker",
    "CircuitOpenError",
    "RegimeDetector",
    "ToolQuota",
    "AgentPauseError",
    "RateLimitHit",
    "TelemetryError",
    "CheckpointError",
    "BackendError",
    "__version__",
]
