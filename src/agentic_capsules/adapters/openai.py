"""
OpenAI API adapter.

Wraps the OpenAI SDK to satisfy the LLMAdapter Protocol.
Default model: gpt-4.1 (latest capable model as of 2026).

Follows the same pattern as AnthropicAdapter:
- Known models have explicit context windows in _CONTEXT_WINDOWS
- Unknown/future models fall back to _FALLBACK_CONTEXT_WINDOW
- Token counting is adapter-delegated (decision D-2)

Design plan ref: §5.1 (OpenAI adapter row), §5.2 Phase 6 (multi-model execution)
"""

from __future__ import annotations

import time

from ..core.types import LLMMessage
from .generic import BaseAdapter

# Retry config for transient network / overload errors.
_MAX_RETRIES  = 4
_BACKOFF_BASE = 2.0  # seconds; actual waits: 2, 4, 8, 16

# Per-HTTP-call timeout — symmetry with AnthropicAdapter. See note there for
# the stalled-socket incident that motivated this. 300s is well above any
# legitimate Chat Completions latency we observe and short enough that a
# wedged socket surfaces as a retryable APITimeoutError.
_HTTP_TIMEOUT_SECONDS = 300.0

# Known context windows for OpenAI models (tokens).
# New models not listed here are accepted with _FALLBACK_CONTEXT_WINDOW.
_CONTEXT_WINDOWS: dict[str, int] = {
    # gpt-4.1 family (2025)
    "gpt-4.1":       1_047_576,
    "gpt-4.1-mini":  1_047_576,
    "gpt-4.1-nano":  1_047_576,
    # o-series reasoning models
    "o4-mini":       200_000,
    "o3":            200_000,
    "o3-mini":       200_000,
    "o1":            200_000,
    "o1-mini":       128_000,
    # legacy (still supported)
    "gpt-4o":        128_000,
    "gpt-4o-mini":   128_000,
}

_DEFAULT_MODEL = "gpt-4.1"
_FALLBACK_CONTEXT_WINDOW = 128_000  # conservative default for unknown models


class OpenAIAdapter(BaseAdapter):
    """
    LLM adapter backed by the OpenAI Chat Completions API.

    Usage:
        adapter = OpenAIAdapter()                   # uses gpt-4.1
        adapter = OpenAIAdapter(model="o3")
        adapter = OpenAIAdapter(model="gpt-5")      # unknown model — uses fallback window

    Token counting uses tiktoken when available (exact counts); falls back
    to a character-based heuristic so the framework never hard-depends on
    a specific SDK or tokenizer version.

    Design plan ref: §5.1, decision D-2 (adapter-delegated count_tokens)
    """

    def __init__(self, model: str = _DEFAULT_MODEL, max_tokens: int = 8192) -> None:
        super().__init__()
        try:
            import openai as _openai
            self._openai = _openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            ) from exc

        self._model = model
        self._max_tokens = max_tokens
        self._context_window = _CONTEXT_WINDOWS.get(model, _FALLBACK_CONTEXT_WINDOW)
        # Client is created lazily on first complete() call (T-001 fix).
        # This allows constructing the adapter without an API key present,
        # which is required for unit tests and schema-inspection use cases.
        self._client = None

        # tiktoken is optional — used for exact token counts if available
        self._tiktoken = None
        self._encoding = None
        try:
            import tiktoken
            self._tiktoken = tiktoken
            # cl100k_base covers gpt-4.1 and gpt-4o; o-series uses the same encoding
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            pass  # will fall back to heuristic

    # ------------------------------------------------------------------
    # LLMAdapter Protocol
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        """
        Send messages to the OpenAI Chat Completions API and return the reply.

        The OpenAI API accepts system messages inline in the messages list.

        When *tools* is non-empty (list of ToolDefinition), passes function
        definitions to the API and runs the multi-turn invocation loop until
        the model returns a final text response with no tool calls.

        Sets self._last_tool_call_count and self._last_tool_call_sequence so the
        executor can record these in TelemetryRecord (tool_calls, tool_call_sequence).
        """
        import json

        if self._client is None:
            self._client = self._openai.OpenAI(timeout=_HTTP_TIMEOUT_SECONDS)

        self._last_tool_call_count = 0
        self._last_tool_call_sequence: list[str] = []
        # T-047: accumulate billed tokens across all turns in the tool loop
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0
        turn_messages: list[dict] = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=turn_messages,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        while True:
            response = self._call_with_retry(**kwargs)
            # T-047: accumulate billed token counts across all tool-loop turns
            if response.usage is not None:
                self._last_input_tokens  += response.usage.prompt_tokens or 0
                self._last_output_tokens += response.usage.completion_tokens or 0
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            # Invoke each requested tool
            tool_result_messages: list[dict] = []
            for call in message.tool_calls:
                tool_def = next((t for t in tools if t.name == call.function.name), None)
                if tool_def is None:
                    result_content = f"Error: tool {call.function.name!r} not found."
                else:
                    args = json.loads(call.function.arguments)
                    result_content = str(tool_def.callable(args))
                self._last_tool_call_count += 1
                self._last_tool_call_sequence.append(call.function.name)
                tool_result_messages.append({
                    "role":         "tool",
                    "tool_call_id": call.id,
                    "content":      result_content,
                })

            # Append assistant message (with tool_calls) and tool results
            kwargs["messages"] = list(kwargs["messages"]) + [message] + tool_result_messages

    def _call_with_retry(self, **kwargs) -> object:
        """
        Call the OpenAI Chat Completions API with exponential backoff on transient errors.

        Retries up to _MAX_RETRIES times on:
          - APIConnectionError  (network-level failure)
          - APITimeoutError     (request timed out)
          - RateLimitError      (429 rate limited)
          - APIStatusError 503  (service unavailable)
          - APIStatusError 529  (overloaded)

        All other errors propagate immediately.
        Waits: 2, 4, 8, 16 seconds between attempts.
        """
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except self._openai.APIConnectionError:
                pass
            except self._openai.APITimeoutError:
                pass
            except self._openai.RateLimitError:
                pass
            except self._openai.APIStatusError as exc:
                if exc.status_code not in (503, 529):
                    raise
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"OpenAI API call failed after {_MAX_RETRIES + 1} attempts."
                )
            time.sleep(_BACKOFF_BASE ** (attempt + 1))

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken when available; heuristic fallback otherwise.

        Design plan ref: decision D-2 (adapter-delegated count_tokens)
        """
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        # Heuristic fallback: ~3.5 characters per token
        return max(1, int(len(text) / 3.5))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"OpenAIAdapter(model={self._model!r})"
