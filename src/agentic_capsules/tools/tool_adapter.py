"""
ToolAdapter — protocol and mock implementation for tool invocation.

The ToolAdapter protocol is the tool-space analog of LLMAdapter: it abstracts
the underlying tool execution mechanism so the ToolOrchestrator and tests
are independent of the actual transport (MCP, HTTP, subprocess, etc.).

A real MCP adapter will be a thin wrapper around this protocol in Phase 6.

Design plan ref: §3.2.4, §5.1 (Tool Orchestrator row)
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolAdapter(Protocol):
    """
    Protocol for tool execution backends.

    Implementations:
      MockToolAdapter  — scripted responses for tests and offline benchmarks
      (Phase 6) McpToolAdapter — real MCP server over stdio/JSON-RPC
    """

    def invoke(self, tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute *tool_name* with *input_data* and return its output dict."""
        ...

    def get_schema(self, tool_name: str) -> dict[str, Any]:
        """
        Return the schema for *tool_name* as a dict with keys:
          "input":  {field_name: type_str, ...}
          "output": {field_name: type_str, ...}
        """
        ...


class MockToolAdapter:
    """
    Scripted tool adapter for tests and offline benchmarks.

    Responses are registered via register() before use. If no response is
    registered for a tool_name, a generic echo response is returned.

    Tracks call_count per tool for benchmark assertions.
    """

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self.call_counts: dict[str, int] = {}

    def register(
        self,
        tool_name: str,
        response: dict[str, Any],
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a scripted response (and optional schema) for *tool_name*."""
        self._responses[tool_name] = response
        if schema:
            self._schemas[tool_name] = schema

    def invoke(self, tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        if tool_name in self._responses:
            return dict(self._responses[tool_name])
        # Generic echo: return input wrapped under "result"
        return {"result": f"mock output from {tool_name}", **input_data}

    def get_schema(self, tool_name: str) -> dict[str, Any]:
        if tool_name in self._schemas:
            return dict(self._schemas[tool_name])
        return {
            "input": {"query": "str"},
            "output": {"result": "str"},
        }

    @property
    def total_calls(self) -> int:
        return sum(self.call_counts.values())

    def reset(self) -> None:
        self.call_counts.clear()

    def __repr__(self) -> str:
        return f"MockToolAdapter(tools={list(self._responses)}, total_calls={self.total_calls})"
