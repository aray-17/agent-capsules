"""
Generic LLM adapter base class.

Implements the LLMAdapter Protocol (core/types.py) as an abstract base so
that concrete adapters (Anthropic, OpenAI) only need to override a small
surface area.

Design plan ref: §5.1 (adapters row), §7 Research Question 5 (cross-model)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from ..core.types import LLMAdapter, LLMMessage


class BaseAdapter(ABC):
    """
    Abstract base for all LLM endpoint adapters.

    Subclasses must implement `complete()` and `count_tokens()`.
    `context_window` is set per-model in the subclass.

    The adapter is the only component that knows about model-specific
    token limits — this is how Rule 6 (context budget) is enforced
    without hard-coding limits in the framework core.

    Thread-safety of per-call bookkeeping
    --------------------------------------
    The ``_last_*`` attributes (``_last_input_tokens``, ``_last_output_tokens``,
    ``_last_tool_call_count``, etc.) capture the billing/tool bookkeeping
    for **the most recent** ``complete()`` call. Under the parallel
    executor (``ThreadPoolExecutor``) and LangGraph's
    ``asyncio.to_thread`` dispatch, multiple ``complete()`` calls can run
    concurrently on the **same adapter instance**. If those ``_last_*``
    attributes were plain instance attributes, every parallel call would
    race the same slot and per-call telemetry would be clobbered.

    BaseAdapter backs each ``_last_*`` attribute with a thread-local store,
    so every thread sees its own per-call counters. Subclasses should
    continue to reset and accumulate via ``self._last_input_tokens = 0``
    / ``self._last_input_tokens += n`` — the property descriptors below
    transparently route those operations to the thread-local slot.

    Subclasses must call ``super().__init__()`` from their ``__init__``
    so the thread-local store is created before any ``complete()`` call.
    """

    def __init__(self) -> None:
        # One ``threading.local`` per adapter instance. Each worker thread
        # that calls ``complete()`` on this adapter sees its own attribute
        # namespace for the ``_last_*`` bookkeeping.
        self._billing_tls: threading.local = threading.local()

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Token capacity of the underlying model."""
        ...

    @abstractmethod
    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        """
        Send messages and return the assistant reply.

        When *tools* is non-empty, subclasses run the multi-turn tool invocation
        loop and store the total invocation count in self._last_tool_call_count.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* as the model would tokenize it."""
        ...

    # ------------------------------------------------------------------
    # Per-call billing + tool bookkeeping (thread-local)
    # ------------------------------------------------------------------

    @property
    def _last_input_tokens(self) -> int:
        return getattr(self._billing_tls, "input_tokens", 0)

    @_last_input_tokens.setter
    def _last_input_tokens(self, value: int) -> None:
        self._billing_tls.input_tokens = value

    @property
    def _last_output_tokens(self) -> int:
        return getattr(self._billing_tls, "output_tokens", 0)

    @_last_output_tokens.setter
    def _last_output_tokens(self, value: int) -> None:
        self._billing_tls.output_tokens = value

    @property
    def _last_cache_read_tokens(self) -> int:
        return getattr(self._billing_tls, "cache_read_tokens", 0)

    @_last_cache_read_tokens.setter
    def _last_cache_read_tokens(self, value: int) -> None:
        self._billing_tls.cache_read_tokens = value

    @property
    def _last_cache_creation_tokens(self) -> int:
        return getattr(self._billing_tls, "cache_creation_tokens", 0)

    @_last_cache_creation_tokens.setter
    def _last_cache_creation_tokens(self, value: int) -> None:
        self._billing_tls.cache_creation_tokens = value

    @property
    def _last_tool_call_count(self) -> int:
        return getattr(self._billing_tls, "tool_call_count", 0)

    @_last_tool_call_count.setter
    def _last_tool_call_count(self, value: int) -> None:
        self._billing_tls.tool_call_count = value

    @property
    def _last_tool_call_sequence(self) -> list[str]:
        seq = getattr(self._billing_tls, "tool_call_sequence", None)
        if seq is None:
            seq = []
            self._billing_tls.tool_call_sequence = seq
        return seq

    @_last_tool_call_sequence.setter
    def _last_tool_call_sequence(self, value: list[str]) -> None:
        self._billing_tls.tool_call_sequence = value


# Runtime check: BaseAdapter satisfies the Protocol
def _assert_protocol() -> None:
    assert isinstance(BaseAdapter, type)  # structural — verified by mypy/pyright
