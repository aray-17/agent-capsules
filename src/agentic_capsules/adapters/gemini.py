"""
Google Gemini API adapter.

Wraps the google-genai SDK to satisfy the LLMAdapter Protocol.
Default model: gemini-2.5-flash (lightweight, cost-efficient).

Follows the same pattern as OpenAIAdapter:
- Known models have explicit context windows in _CONTEXT_WINDOWS
- Unknown/future models fall back to _FALLBACK_CONTEXT_WINDOW
- Token counting uses a character-based heuristic (no official tiktoken equivalent)

Gemini-specific notes:

- System messages are passed via GenerateContentConfig.system_instruction,
  not inline in the message list.
- Assistant role in Gemini is "model" not "assistant".
- Tool results are sent as function_response Parts (not role="tool" messages).

Design plan ref: §5.1 (adapters row), §7 Research Question 5 (cross-model)
"""

from __future__ import annotations

import json
import time

from ..core.types import LLMMessage
from .generic import BaseAdapter

# Retry config for transient 503 errors (high demand / throttling).
_MAX_RETRIES   = 4
_BACKOFF_BASE  = 2.0   # seconds; actual waits: 2, 4, 8, 16

# Per-HTTP-call timeout — symmetry with Anthropic/OpenAI adapters. See the
# Anthropic adapter comment for the stalled-socket incident that motivated
# this. google-genai's HttpOptions.timeout field is in **milliseconds**.
# 300s == 300_000 ms, well above any legitimate Gemini response time and
# short enough that a wedged httpx socket surfaces as an httpx.ReadTimeout
# → caught by _generate_with_backoff → retried up to 4x → outer
# run_resilient.sh wrapper picks up from the checkpoint if still stuck.
_HTTP_TIMEOUT_MS = 300_000

# Known context windows for Gemini models (tokens).
_CONTEXT_WINDOWS: dict[str, int] = {
    # Gemini 2.5 family (2025–2026)
    "gemini-2.5-flash":             1_048_576,
    "gemini-2.5-flash-lite":        1_048_576,
    "gemini-2.5-pro":               1_048_576,
    # Gemini 2.0 family
    "gemini-2.0-flash":             1_048_576,
    "gemini-2.0-flash-lite":        1_048_576,
    # Aliases
    "gemini-flash-latest":          1_048_576,
    "gemini-flash-lite-latest":     1_048_576,
    "gemini-pro-latest":            1_048_576,
}

_DEFAULT_MODEL = "gemini-2.5-flash"
_FALLBACK_CONTEXT_WINDOW = 1_048_576  # Gemini models have large contexts


class GeminiAdapter(BaseAdapter):
    """
    LLM adapter backed by the Google Gemini API (google-genai SDK).

    Usage:
        adapter = GeminiAdapter()                         # uses gemini-2.5-flash
        adapter = GeminiAdapter(model="gemini-2.5-pro")

    Requires GOOGLE_API_KEY environment variable or explicit api_key parameter.
    Install dependency: pip install google-genai
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 8192,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self._types = genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for GeminiAdapter. "
                "Install it with: pip install google-genai"
            ) from exc

        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key  # if None, SDK reads GOOGLE_API_KEY from env
        self._context_window = _CONTEXT_WINDOWS.get(model, _FALLBACK_CONTEXT_WINDOW)
        # Client is created lazily on first complete() call.
        self._client = None
        # Per-call billing + tool bookkeeping live on the thread-local store
        # managed by BaseAdapter so parallel ``complete()`` calls on the same
        # adapter instance do not race each other's counters.

    # ------------------------------------------------------------------
    # LLMAdapter Protocol
    # ------------------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        """
        Send messages to the Gemini API and return the reply.

        Extracts system messages (role="system") and passes them via
        system_instruction config. Remaining messages are converted to
        Gemini Content objects (role "assistant" → "model").

        When *tools* is non-empty (list of ToolDefinition), runs the
        multi-turn tool invocation loop until the model returns a final
        text response.

        Sets self._last_tool_call_count and self._last_tool_call_sequence.
        """
        if self._client is None:
            kwargs = {
                "http_options": self._types.HttpOptions(timeout=_HTTP_TIMEOUT_MS),
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = self._genai.Client(**kwargs)

        self._last_tool_call_count = 0
        self._last_tool_call_sequence = []
        self._last_input_tokens = 0
        self._last_output_tokens = 0

        # Separate system prompt from conversation messages
        system_parts: list[str] = []
        conversation: list = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                gemini_role = "model" if m.role == "assistant" else "user"
                conversation.append(
                    self._types.Content(
                        role=gemini_role,
                        parts=[self._types.Part(text=m.content)],
                    )
                )

        # Build config
        config_kwargs: dict = {"max_output_tokens": self._max_tokens}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)

        if tools:
            function_declarations = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=self._schema_to_genai(t.input_schema),
                )
                for t in tools
            ]
            config_kwargs["tools"] = [
                self._types.Tool(function_declarations=function_declarations)
            ]

        config = self._types.GenerateContentConfig(**config_kwargs)

        # Multi-turn tool loop
        while True:
            response = self._generate_with_backoff(conversation, config)
            # T-047: accumulate billed token counts across all tool-loop turns
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self._last_input_tokens  += getattr(usage, "prompt_token_count",     0) or 0
                self._last_output_tokens += getattr(usage, "candidates_token_count", 0) or 0

            candidate = response.candidates[0]
            content = candidate.content if candidate.content else None
            parts = (content.parts or []) if content else []

            # Check for function calls
            function_calls = [p for p in parts if p.function_call is not None]
            if not function_calls:
                # Final text response
                text_parts = [p.text for p in parts if p.text is not None]
                return "".join(text_parts)

            # Append the model's response (with function calls) to conversation
            conversation.append(candidate.content)

            # Invoke each tool and collect results
            result_parts: list = []
            for part in function_calls:
                call = part.function_call
                tool_def = next((t for t in tools if t.name == call.name), None)
                if tool_def is None:
                    result_content = {"error": f"tool {call.name!r} not found"}
                else:
                    args = dict(call.args) if call.args else {}
                    result_content = {"result": str(tool_def.callable(args))}

                self._last_tool_call_count += 1
                self._last_tool_call_sequence.append(call.name)
                result_parts.append(
                    self._types.Part(
                        function_response=self._types.FunctionResponse(
                            name=call.name,
                            response=result_content,
                        )
                    )
                )

            # Append tool results as a user turn
            conversation.append(
                self._types.Content(role="user", parts=result_parts)
            )

    def _generate_with_backoff(self, conversation: list, config) -> object:
        """
        Call generate_content with exponential backoff on transient errors.

        Retries up to _MAX_RETRIES times with waits of 2, 4, 8, 16 seconds on:
          - genai ServerError (5xx — throttling, UNAVAILABLE, INTERNAL)
          - httpx.TimeoutException (Connect/Read/Write/Pool timeout from the
            300s per-call deadline set via HttpOptions in complete())
          - httpx.ConnectError (network-level failure, e.g. DNS flap)

        All other errors propagate immediately.
        """
        import httpx
        from google.genai import errors as genai_errors

        retryable = (
            genai_errors.ServerError,
            httpx.TimeoutException,
            httpx.ConnectError,
        )

        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.models.generate_content(
                    model=self._model,
                    contents=conversation,
                    config=config,
                )
            except retryable as exc:
                if attempt == _MAX_RETRIES:
                    raise
                wait = _BACKOFF_BASE ** (attempt + 1)
                last_exc = exc
                time.sleep(wait)
        raise last_exc  # unreachable, but satisfies type checkers

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using a character-based heuristic.

        Gemini uses SentencePiece tokenization; no public Python tokenizer
        is available. The ~4 chars/token ratio is conservative for English text.
        """
        return max(1, int(len(text) / 4))

    def _schema_to_genai(self, schema: dict) -> "google.genai.types.Schema":
        """
        Convert a JSON Schema dict (ToolDefinition.input_schema) to a
        google.genai.types.Schema object.

        Only handles the subset used by the framework: object type with
        string/number properties and required lists.
        """
        return self._build_schema(schema)

    def _build_schema(self, schema: dict):
        """Recursively build a genai Schema from a JSON Schema dict."""
        type_map = {
            "string":  "STRING",
            "number":  "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array":   "ARRAY",
            "object":  "OBJECT",
        }
        json_type = schema.get("type", "string")
        genai_type = type_map.get(json_type, "STRING")

        kwargs: dict = {"type": genai_type}

        if "description" in schema:
            kwargs["description"] = schema["description"]

        if "enum" in schema:
            kwargs["enum"] = schema["enum"]

        if genai_type == "OBJECT" and "properties" in schema:
            kwargs["properties"] = {
                k: self._build_schema(v)
                for k, v in schema["properties"].items()
            }
            if "required" in schema:
                kwargs["required"] = schema["required"]

        if genai_type == "ARRAY" and "items" in schema:
            kwargs["items"] = self._build_schema(schema["items"])

        return self._types.Schema(**kwargs)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GeminiAdapter(model={self._model!r})"
