"""
Tests for agent tool use (Phase 10).

Covers:
- AgentStepCapsule.tools field
- Executor wiring: tool_registry → adapter.complete(tools=)
- TelemetryRecord.tool_calls populated from adapter._last_tool_call_count
- Graceful handling when tool_registry is absent (existing behaviour unchanged)
"""
import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, LLMMessage, Schema
from agentic_capsules.controller.telemetry import TelemetryRecord
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.tools.registry import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Scripted adapter that accepts tools= and tracks calls
# ---------------------------------------------------------------------------

class ScriptedToolAdapter:
    """
    Scripted adapter for testing tool use without a real LLM.

    On the first complete() call it simulates making *n_tool_calls* tool
    invocations by calling each tool callable and setting _last_tool_call_count.
    It returns *final_response* as the final text.
    """

    context_window = 200_000

    def __init__(self, final_response: str, n_tool_calls: int = 0):
        self._final_response = final_response
        self._n_tool_calls = n_tool_calls
        self._last_tool_call_count = 0
        self.calls: list[dict] = []  # records each complete() invocation

    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        self.calls.append({"messages": messages, "tools": tools})
        self._last_tool_call_count = 0

        # Simulate tool invocations if tools are provided
        if tools and self._n_tool_calls > 0:
            for i in range(min(self._n_tool_calls, len(tools))):
                tools[i].callable({"query": "test"})
                self._last_tool_call_count += 1

        return self._final_response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hierarchy(agent_name: str = "analyst", tools: list[str] | None = None) -> CapsuleHierarchy:
    capsule = AgentStepCapsule(
        name=agent_name,
        system_prompt="Analyse the topic.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("output", fields={"result": "str"}),
        tools=tools or [],
    )
    leaf = AgentLeaf(capsule=capsule)
    group = CompoundCapsule(name="group", children=[leaf], dependency_edges={})
    return CapsuleHierarchy(name="test_pipeline", root=group)


def _make_registry(*tool_names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name in tool_names:
        reg.register(ToolDefinition(
            name=name,
            description=f"Tool: {name}",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            callable=lambda inp, n=name: {"result": f"{n}_result"},
        ))
    return reg


# ---------------------------------------------------------------------------
# AgentStepCapsule.tools field
# ---------------------------------------------------------------------------

class TestAgentStepCapsuleToolsField:
    def test_default_is_empty(self):
        c = AgentStepCapsule(
            name="a",
            system_prompt="p",
            input_schema=Schema("i"),
            output_schema=Schema("o"),
        )
        assert c.tools == []

    def test_tools_set_on_construction(self):
        c = AgentStepCapsule(
            name="a",
            system_prompt="p",
            input_schema=Schema("i"),
            output_schema=Schema("o"),
            tools=["search", "calc"],
        )
        assert c.tools == ["search", "calc"]

    def test_repr_includes_tools_when_set(self):
        c = AgentStepCapsule(
            name="a",
            system_prompt="p",
            input_schema=Schema("i"),
            output_schema=Schema("o"),
            tools=["search"],
        )
        assert "search" in repr(c)

    def test_repr_omits_tools_when_empty(self):
        c = AgentStepCapsule(
            name="a",
            system_prompt="p",
            input_schema=Schema("i"),
            output_schema=Schema("o"),
        )
        assert "tools" not in repr(c)


# ---------------------------------------------------------------------------
# Executor wiring — no tool_registry (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestExecutorNoToolRegistry:
    def test_complete_called_without_tools_when_no_registry(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult")
        hierarchy = _make_hierarchy(tools=["search"])  # agent declares tools
        # No registry passed → tools= should NOT be passed to adapter
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
        )
        result = executor.run(hierarchy, task_input="test")
        # complete() was called
        assert len(adapter.calls) == 1
        # tools= argument was None (registry absent)
        assert adapter.calls[0]["tools"] is None

    def test_tool_calls_zero_in_telemetry_when_no_tools(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult")
        hierarchy = _make_hierarchy()
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
        )
        result = executor.run(hierarchy, task_input="test")
        assert result.telemetry[0].tool_calls == 0


# ---------------------------------------------------------------------------
# Executor wiring — with tool_registry
# ---------------------------------------------------------------------------

class TestExecutorWithToolRegistry:
    def test_complete_called_with_tool_defs_when_agent_has_tools(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult", n_tool_calls=1)
        registry = _make_registry("search")
        hierarchy = _make_hierarchy(tools=["search"])
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        result = executor.run(hierarchy, task_input="test")
        assert len(adapter.calls) == 1
        passed_tools = adapter.calls[0]["tools"]
        assert passed_tools is not None
        assert len(passed_tools) == 1
        assert passed_tools[0].name == "search"

    def test_complete_not_passed_tools_when_agent_has_none(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult")
        registry = _make_registry("search")  # registry exists but agent doesn't use it
        hierarchy = _make_hierarchy(tools=[])
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        result = executor.run(hierarchy, task_input="test")
        assert adapter.calls[0]["tools"] is None

    def test_tool_calls_recorded_in_telemetry(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult", n_tool_calls=2)
        registry = _make_registry("search", "calc")
        hierarchy = _make_hierarchy(tools=["search", "calc"])
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        result = executor.run(hierarchy, task_input="test")
        assert result.telemetry[0].tool_calls == 2

    def test_tool_calls_zero_when_agent_has_no_tools(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult")
        registry = _make_registry("search")
        hierarchy = _make_hierarchy(tools=[])
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        result = executor.run(hierarchy, task_input="test")
        assert result.telemetry[0].tool_calls == 0

    def test_missing_tool_in_registry_raises(self):
        adapter = ScriptedToolAdapter("ANALYST_OUTPUT:\nresult")
        registry = _make_registry("search")  # 'calc' not registered
        hierarchy = _make_hierarchy(tools=["calc"])
        executor = CapsuleExecutor(
            adapter=adapter,
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        with pytest.raises(KeyError, match="calc"):
            executor.run(hierarchy, task_input="test")

    def test_multiple_agents_independent_tool_sets(self):
        """Two agents in FINE mode — each gets only its own tools."""
        calls: list[list | None] = []

        class TrackingAdapter:
            context_window = 200_000
            _last_tool_call_count = 0

            def complete(self, messages, tools=None):
                calls.append(tools)
                if tools:
                    key = tools[0].name.upper()
                else:
                    key = "UNKNOWN"
                return f"{key}_OUTPUT:\nok"

            def count_tokens(self, text):
                return max(1, len(text) // 4)

        c1 = AgentStepCapsule("researcher", "Research.", Schema("i"), Schema("o"), tools=["search"])
        c2 = AgentStepCapsule("writer",     "Write.",    Schema("i"), Schema("o"), tools=["spellcheck"])
        group = CompoundCapsule(
            name="root",
            children=[AgentLeaf(c1), AgentLeaf(c2)],
            dependency_edges={"writer": ["researcher"]},
        )
        hierarchy = CapsuleHierarchy(name="p", root=group)
        registry = _make_registry("search", "spellcheck")

        executor = CapsuleExecutor(
            adapter=TrackingAdapter(),
            composition_level=CompositionLevel.FINE,
            tool_registry=registry,
        )
        executor.run(hierarchy, task_input="test")

        assert calls[0] is not None and calls[0][0].name == "search"
        assert calls[1] is not None and calls[1][0].name == "spellcheck"


# ---------------------------------------------------------------------------
# TelemetryRecord.tool_calls field
# ---------------------------------------------------------------------------

class TestTelemetryToolCalls:
    def test_default_is_zero(self):
        r = TelemetryRecord(
            capsule_name="a", composition_mode="FINE",
            batch_size=1, total_tokens=100, coordination_tokens=10,
            latency_ms=5.0,
        )
        assert r.tool_calls == 0

    def test_set_on_construction(self):
        r = TelemetryRecord(
            capsule_name="a", composition_mode="FINE",
            batch_size=1, total_tokens=100, coordination_tokens=10,
            latency_ms=5.0, tool_calls=3,
        )
        assert r.tool_calls == 3
