"""
ToolRegistry — registers ToolDefinitions that agents can invoke at reasoning time.

This is distinct from ToolCapsule (pipeline-level pre-processing chains).
ToolDefinition describes a tool the LLM can call during its own generation;
ToolCapsule describes a mechanical step-chain that bypasses the LLM entirely.

Design plan ref: Phase 10 (agent tool use), §4.1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """
    Describes a tool that an LLM agent can call during its reasoning.

    Attributes:
        name         — tool identifier, must match the name used in AgentStepCapsule.tools
        description  — plain-English description sent to the LLM so it knows when to use it
        input_schema — JSON Schema object describing the tool's expected input
                       (passed verbatim to the LLM API's tool/function spec)
        callable     — the function invoked when the LLM requests this tool;
                       receives a dict matching input_schema and returns a dict
        independent  — True when this tool's inputs are fully determined from the original
                       task input alone, with no dependency on the results of other tools.
                       Only independent tools are safe for TOOL_CHAIN pre-bundling.
                       Sequential tools (tool N+1 depends on tool N's result) must stay
                       False — pre-bundling would require guessing downstream queries.
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    callable: Callable[[dict[str, Any]], dict[str, Any]]
    independent: bool = False  # T-015: safe for TOOL_CHAIN pre-bundling only when True

    def __repr__(self) -> str:
        return f"ToolDefinition(name={self.name!r})"


class ToolRegistry:
    """
    Maps tool names to their ToolDefinition objects.

    Pass a ToolRegistry to CapsuleExecutor so agents that declare
    `tools=["tool_name"]` on their AgentStepCapsule can resolve the
    definition at execution time.

    Usage::

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="web_search",
            description="Search the web for a query.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                          "required": ["query"]},
            callable=lambda inp: {"results": search(inp["query"])},
        ))

        executor = CapsuleExecutor(adapter=..., tool_registry=registry)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Overwrites any existing registration with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        """
        Return the ToolDefinition for *name*.

        Raises KeyError with a descriptive message if not found.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} is not registered. "
                f"Available tools: {sorted(self._tools)}. "
                f"Register it with ToolRegistry.register()."
            )
        return self._tools[name]

    def definitions_for(self, names: list[str]) -> list[ToolDefinition]:
        """Return ToolDefinitions for all *names*, in order. Raises KeyError on missing."""
        return [self.get(n) for n in names]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={sorted(self._tools)!r})"
