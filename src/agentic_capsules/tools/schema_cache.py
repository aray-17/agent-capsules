"""
ToolSchemaCache — fetches and caches tool schemas; validates chain compatibility.

Schema negotiation happens once per tool type per session, not per invocation.
This eliminates the token overhead of repeating full tool schemas on every call —
a key metric in Benchmark 3.

T-Rule 1 validation is run at chain-definition time via validate_chain().

Design plan ref: §3.2.4 (Tool Discovery and Caching), §4.2 Strategy 3
"""
from __future__ import annotations

import logging
from typing import Any

from .tool_capsule import ToolCapsule

logger = logging.getLogger(__name__)


class ToolSchemaCache:
    """
    Per-session cache of tool schemas.

    Usage:
        cache = ToolSchemaCache(adapter)
        cache.prefetch(["web_search", "fetch_page", "extract_text"])
        cache.validate_chain(my_tool_capsule)  # T-Rule 1
        schema = cache.get("web_search")
    """

    def __init__(self, adapter) -> None:
        """adapter must implement ToolAdapter.get_schema(tool_name) -> dict."""
        self._adapter = adapter
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, tool_name: str) -> dict[str, Any]:
        """Return the schema for *tool_name*, fetching once if not cached."""
        if tool_name not in self._cache:
            schema = self._adapter.get_schema(tool_name)
            self._cache[tool_name] = schema
            logger.debug("ToolSchemaCache: fetched schema for %r", tool_name)
        return self._cache[tool_name]

    def prefetch(self, tool_names: list[str]) -> None:
        """Fetch schemas for all *tool_names* in one pass (session startup)."""
        for name in tool_names:
            self.get(name)

    def validate_chain(self, capsule: ToolCapsule) -> None:
        """
        T-Rule 1 runtime validation: fetch schemas for all steps and verify
        that input_keys expected by each step are present in the output schema
        of the prior step (or available from the external input for step 0).

        Raises ValueError if a schema mismatch is detected.
        Note: ToolCapsule.validate() already checks input_from references at
        construction time; this method adds schema-level key compatibility.
        """
        # Collect available output keys starting from an open external input
        available_output_keys: set[str] = set()

        for i, step in enumerate(capsule.steps):
            schema = self.get(step.tool_name)
            output_fields = set(schema.get("output", {}).keys())

            # Determine where this step's inputs come from
            if step.input_from is not None:
                # Should have been validated structurally already; re-check schema
                prior_schema = self.get(
                    capsule.steps[
                        next(
                            j for j, s in enumerate(capsule.steps[:i])
                            if s.output_key == step.input_from
                        )
                    ].tool_name
                )
                source_fields = set(prior_schema.get("output", {}).keys())
            else:
                # Step reads from external input; treat as open (any keys allowed)
                source_fields = set(step.input_keys)

            # Verify all expected input_keys are satisfiable
            unsatisfied = set(step.input_keys) - source_fields - available_output_keys
            if unsatisfied and step.input_from is not None:
                raise ValueError(
                    f"ToolCapsule {capsule.name!r} T-Rule 1 schema violation: "
                    f"step {i} ({step.tool_name!r}) expects input keys "
                    f"{sorted(unsatisfied)} but source {step.input_from!r} "
                    f"only outputs {sorted(source_fields)}."
                )

            available_output_keys.update(output_fields)

        logger.debug(
            "ToolSchemaCache: chain %r validated (%d steps)", capsule.name, len(capsule.steps)
        )

    @property
    def cached_tools(self) -> list[str]:
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"ToolSchemaCache(cached={self.cached_tools})"
