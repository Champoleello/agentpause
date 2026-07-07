"""Provider adapters: turn real LLM providers into the two callables
(backend, telemetry) that :class:`agentpause.PredictiveScheduler` expects.

Adapters are optional — the core has zero dependencies. Import them
explicitly, e.g.::

    from agentpause.adapters.litellm import LiteLLMAdapter
"""
