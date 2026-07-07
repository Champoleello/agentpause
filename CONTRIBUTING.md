# Contributing to agentpause

Thanks for considering a contribution!

## Setup

```bash
git clone https://github.com/Champoleello/agentpause
cd agentpause
pip install -e ".[dev]"
pytest        # the whole suite runs offline, no API keys needed
```

## Ground rules

- **Test-first**: every behavior change comes with a test. The suite must
  stay runnable offline — inject fakes (`completion_fn`, `interrupt_fn`,
  `sleep_fn`, ...) instead of calling real providers.
- **The core stays dependency-free**: provider/framework code belongs in
  `src/agentpause/adapters/`, guarded by optional extras in `pyproject.toml`.
- **Telemetry is never trusted from a checkpoint**: any code path that
  resumes work must re-read the budget fresh (see the telemetry-ping rule
  in the README).
- Keep errors typed: raise subclasses of `AgentPauseError`.

## Pull requests

1. Fork, branch from `main`.
2. `pytest -q` green locally.
3. Update `CHANGELOG.md` under an *Unreleased* heading.
4. Open the PR with a short description of the behavior change.

## Reporting issues

Include: Python version, provider/model string, whether the failure involves
a real API or the offline fakes, and the shortest script that reproduces it.
