"""
ModelRouter — dispatches LLM calls to different adapters per capsule.

Enables multi-model execution: different agents in the same hierarchy can
be routed to different LLM endpoints (e.g. fast/cheap model for fact-checking,
powerful model for synthesis) without changing the executor.

Usage::

    router = ModelRouter(
        default=AnthropicAdapter(),
        routes={
            "summarizer": OpenAIAdapter(model="gpt-4o"),
            "fact_checker": AnthropicAdapter(model="claude-haiku-4-5-20251001"),
        }
    )
    executor = CapsuleExecutor(adapter=router, ...)

The router satisfies the LLMAdapter Protocol itself — the executor sees it as
a single adapter. Internally it selects the registered adapter for the current
capsule name, falling back to the default for unregistered names.

The executor sets `router.current_capsule` before each LLM call so the router
can dispatch correctly.

Design plan ref: §5.2 Phase 6 (multi-model execution)
"""

from __future__ import annotations

from ..core.types import LLMAdapter, LLMMessage


class ModelRouter:
    """
    Routes LLM calls to different adapters per capsule name.

    Args:
        default: Adapter used for any capsule not in *routes*.
        routes: Mapping of capsule_name → adapter.

    The router is itself an LLMAdapter — pass it directly to CapsuleExecutor.
    Before each leaf's LLM call the executor sets `router.current_capsule`
    to the leaf name so the router can select the correct backend.
    """

    def __init__(
        self,
        default: LLMAdapter,
        routes: dict[str, LLMAdapter] | None = None,
    ) -> None:
        self._default = default
        self._routes: dict[str, LLMAdapter] = dict(routes) if routes else {}
        self.current_capsule: str = ""  # set by executor before each call

    # ------------------------------------------------------------------
    # LLMAdapter Protocol
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        """Return the smallest context window across all registered adapters
        (conservative: the executor uses this to enforce Rule 6)."""
        windows = [self._default.context_window] + [
            a.context_window for a in self._routes.values()
        ]
        return min(windows)

    def complete(self, messages: list[LLMMessage]) -> str:
        """Route the call to the adapter registered for `current_capsule`."""
        adapter = self._routes.get(self.current_capsule, self._default)
        return adapter.complete(messages)

    def count_tokens(self, text: str) -> int:
        """Delegate to the adapter for `current_capsule` (or default)."""
        adapter = self._routes.get(self.current_capsule, self._default)
        return adapter.count_tokens(text)

    # ------------------------------------------------------------------
    # Router management
    # ------------------------------------------------------------------

    def register(self, capsule_name: str, adapter: LLMAdapter) -> None:
        """Register or replace a route at runtime."""
        self._routes[capsule_name] = adapter

    def adapter_for(self, capsule_name: str) -> LLMAdapter:
        """Return the adapter that would be used for *capsule_name*."""
        return self._routes.get(capsule_name, self._default)

    def __repr__(self) -> str:
        return (
            f"ModelRouter(default={self._default!r}, "
            f"routes={list(self._routes.keys())})"
        )
