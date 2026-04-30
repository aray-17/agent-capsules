"""
Anthropic API adapter.

Wraps the Anthropic SDK to satisfy the LLMAdapter Protocol.
Default model: claude-opus-4-5-20251101.

Design plan ref: §5.1 (Anthropic adapter row), decision D-3
"""

from __future__ import annotations

import time

from ..core.types import LLMMessage
from .generic import BaseAdapter

# Retry config for transient network / overload errors.
_MAX_RETRIES  = 4
_BACKOFF_BASE = 2.0  # seconds; actual waits: 2, 4, 8, 16

# Per-HTTP-call timeout. The Anthropic SDK defaults leave read sockets open
# indefinitely, so a mid-stream TLS hang will freeze a run forever (observed
# 2026-04-06: PID 4549 wedged 9+ hours in recv() on an ESTABLISHED socket).
# 300s is well above any legitimate Messages API latency we've measured
# (~60s tail) but short enough that a stalled socket surfaces as
# APITimeoutError → caught by _call_with_retry → retried up to 4x → the
# outer run_resilient.sh wrapper picks up from the checkpoint if still stuck.
_HTTP_TIMEOUT_SECONDS = 300.0

# Known context windows for Anthropic models (tokens).
# New models not listed here are accepted with _FALLBACK_CONTEXT_WINDOW.
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}

_DEFAULT_MODEL = "claude-opus-4-6"
_FALLBACK_CONTEXT_WINDOW = 200_000  # conservative default for unknown models


class AnthropicAdapter(BaseAdapter):
    """
    LLM adapter backed by the Anthropic Messages API.

    Usage:
        adapter = AnthropicAdapter()          # uses claude-opus-4-5-20251101
        adapter = AnthropicAdapter(model="claude-sonnet-4-6")

    Token counting uses the Anthropic SDK's count_tokens utility when
    available; falls back to a character-based heuristic (chars / 3.5)
    so the framework never hard-depends on a specific SDK version.

    Design plan ref: §5.1, decision D-2 (adapter-delegated count_tokens)
    """

    def __init__(self, model: str = _DEFAULT_MODEL, max_tokens: int = 8192) -> None:
        super().__init__()
        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic(timeout=_HTTP_TIMEOUT_SECONDS)
            self._anthropic = _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
            ) from exc

        self._model = model
        self._max_tokens = max_tokens
        self._context_window = _CONTEXT_WINDOWS.get(model, _FALLBACK_CONTEXT_WINDOW)

    # ------------------------------------------------------------------
    # LLMAdapter Protocol
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def supports_prompt_caching(self) -> bool:
        return True

    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        """
        Send messages to the Anthropic API and return the text reply.

        Separates the system prompt (role="system") from the conversation turns.

        When *tools* is non-empty (list of ToolDefinition), passes tool schemas
        to the API and runs the multi-turn invocation loop until the model
        returns a final text response with no tool calls.

        Sets self._last_tool_call_count and self._last_tool_call_sequence so the
        executor can record these in TelemetryRecord (tool_calls, tool_call_sequence).
        """
        self._last_tool_call_count = 0
        self._last_tool_call_sequence: list[str] = []
        # T-047: accumulate billed tokens across all turns in the tool loop
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0
        # C-1: cache hit tokens (Anthropic prompt caching — 90% discount on hits)
        self._last_cache_read_tokens: int = 0
        self._last_cache_creation_tokens: int = 0

        system_msgs: list[LLMMessage] = []
        turns: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_msgs.append(msg)
            else:
                turns.append({"role": msg.role, "content": msg.content})

        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=turns,
        )
        if system_msgs:
            # C-1: if any system message carries cache_control, emit the multi-block
            # array format so Anthropic can cache the marked prefix.
            if any(m.cache_control for m in system_msgs):
                blocks: list[dict] = []
                for m in system_msgs:
                    block: dict = {"type": "text", "text": m.content}
                    if m.cache_control:
                        block["cache_control"] = m.cache_control
                    blocks.append(block)
                kwargs["system"] = blocks
            else:
                kwargs["system"] = "".join(m.content for m in system_msgs)
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        while True:
            response = self._call_with_retry(**kwargs)
            # T-047: accumulate billed token counts across all tool-loop turns
            if hasattr(response, "usage") and response.usage is not None:
                self._last_input_tokens         += getattr(response.usage, "input_tokens",          0) or 0
                self._last_output_tokens        += getattr(response.usage, "output_tokens",         0) or 0
                self._last_cache_read_tokens    += getattr(response.usage, "cache_read_input_tokens",    0) or 0
                self._last_cache_creation_tokens += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            if not tool_calls:
                return "\n".join(text_parts) if text_parts else ""

            # Invoke each requested tool and collect results
            tool_results = []
            for call in tool_calls:
                tool_def = next((t for t in tools if t.name == call["name"]), None)
                if tool_def is None:
                    result_content = f"Error: tool {call['name']!r} not found in registry."
                else:
                    result_content = str(tool_def.callable(call["input"]))
                self._last_tool_call_count += 1
                self._last_tool_call_sequence.append(call["name"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result_content,
                })

            # Extend conversation: assistant turn (tool_use blocks) + tool results
            kwargs["messages"] = list(kwargs["messages"]) + [
                {"role": "assistant", "content": response.content},
                {"role": "user",      "content": tool_results},
            ]

    def _call_with_retry(self, **kwargs) -> object:
        """
        Call the Anthropic Messages API with exponential backoff on transient errors.

        Retries up to _MAX_RETRIES times on:
          - APIConnectionError  (network-level failure)
          - APITimeoutError     (request timed out)
          - APIStatusError 429  (rate limited)
          - APIStatusError 529  (API overloaded)
          - APIStatusError 503  (service unavailable)

        All other errors propagate immediately.
        Waits: 2, 4, 8, 16 seconds between attempts.
        """
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.messages.create(**kwargs)
            except self._anthropic.APIConnectionError:
                pass
            except self._anthropic.APITimeoutError:
                pass
            except self._anthropic.APIStatusError as exc:
                if exc.status_code not in (429, 503, 529):
                    raise
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Anthropic API call failed after {_MAX_RETRIES + 1} attempts."
                )
            time.sleep(_BACKOFF_BASE ** (attempt + 1))

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using the Anthropic SDK's token counter.
        Falls back to a heuristic if the SDK method is unavailable.
        """
        try:
            # The Anthropic SDK exposes count_tokens on the client
            result = self._client.messages.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            # Heuristic fallback: ~3.5 characters per token
            return max(1, int(len(text) / 3.5))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AnthropicAdapter(model={self._model!r})"
