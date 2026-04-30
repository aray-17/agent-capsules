"""
Tool — developer-facing tool declaration for agent-driven tool use.

Wraps ToolDefinition internally. The developer never imports ToolDefinition
or ToolRegistry directly; the compiler builds the registry from Tool instances
at pipeline.run() time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """
    A callable tool an agent can invoke during its own reasoning.

    Example::

        search = Tool(
            name="web_search",
            description="Search the web for current information.",
            input_schema={"query": "str"},
            fn=lambda args: {"results": f"Results for: {args['query']}"},
        )
    """

    name:         str
    description:  str
    input_schema: dict[str, str]
    fn:           Callable[[dict[str, Any]], dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not callable(self.fn):
            raise ValueError(f"Tool '{self.name}': fn must be callable.")
