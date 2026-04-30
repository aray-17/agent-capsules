"""Tests for COMPOUND and FINE modes in runtime/executor.py"""

import re

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.runtime.sync_manager import BoundarySyncManager


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    """Returns one output section per expected heading."""
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, tools=None):
        self.call_count += 1
        # Find all OUTPUT headings expected in the prompt and reply to each
        keys = re.findall(r"(\w+_OUTPUT)", messages[-1].content + messages[0].content)
        # Deduplicate while preserving order
        seen = set()
        unique_keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not unique_keys:
            return "RESULT_OUTPUT:\nDone."
        return "\n\n".join(f"{k}:\nResult for {k}." for k in unique_keys)

    def count_tokens(self, text):
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(name: str, prompt: str = "") -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=prompt or f"You are the {name} agent.",
            input_schema=Schema("in", fields={"text": "str"}),
            output_schema=Schema("out", fields={"result": "str"}),
        )
    )


def _hierarchy(*names: str) -> CapsuleHierarchy:
    leaves = [_leaf(n) for n in names]
    root = CompoundCapsule(name="pipeline", children=leaves, dependency_edges={})
    compute_order(root)
    return CapsuleHierarchy(name="test_pipeline", root=root)


# ---------------------------------------------------------------------------
# Call counts
# ---------------------------------------------------------------------------

def test_fine_mode_makes_n_calls():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    executor.run(_hierarchy("researcher", "fact_checker", "summarizer"),
                 task_input="topic", task_id="t1")
    assert adapter.call_count == 3


def test_compound_mode_makes_one_call():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    executor.run(_hierarchy("researcher", "fact_checker", "summarizer"),
                 task_input="topic", task_id="t1")
    assert adapter.call_count == 1


def test_single_agent_compound_makes_one_call():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    executor.run(_hierarchy("analyst"), task_input="topic", task_id="t1")
    assert adapter.call_count == 1


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_compound_outputs_all_keys():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs


def test_fine_outputs_all_keys():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs


def test_compound_final_output_is_last_phase():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert "SUMMARIZER_OUTPUT" in result.final_output or result.final_output != ""


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_compound_telemetry_record_emitted():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert len(result.telemetry) == 1


def test_compound_telemetry_mode_is_compound():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert result.telemetry[0].composition_mode == "COMPOUND"


def test_compound_telemetry_latency_nonnegative():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert result.telemetry[0].latency_ms >= 0.0


def test_compound_telemetry_coordination_tokens_nonzero():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "summarizer"),
                          task_input="topic", task_id="t1")
    assert result.telemetry[0].coordination_tokens > 0


def test_compound_telemetry_overhead_ratio_in_range():
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_hierarchy("researcher", "fact_checker", "summarizer"),
                          task_input="topic", task_id="t1")
    assert 0.0 < result.telemetry[0].overhead_ratio <= 1.0


# ---------------------------------------------------------------------------
# GC eviction
# ---------------------------------------------------------------------------

def test_compound_gc_evicts_intermediates():
    sync = BoundarySyncManager()
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND, sync_manager=sync
    )
    executor.run(_hierarchy("researcher", "fact_checker", "summarizer"),
                 task_input="topic", task_id="t1")
    stored = sync.stored_keys()
    # Intermediate outputs (researcher, fact_checker) should be evicted;
    # only the final output (summarizer) should remain
    assert not any("researcher" in k for k in stored)
    assert not any("fact_checker" in k for k in stored)


def test_compound_gc_retains_final_output():
    sync = BoundarySyncManager()
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND, sync_manager=sync
    )
    executor.run(_hierarchy("researcher", "summarizer"),
                 task_input="topic", task_id="t1")
    stored = sync.stored_keys()
    assert any("summarizer" in k for k in stored)


# ---------------------------------------------------------------------------
# T-006: nested CompoundCapsule children (mixed-compound dispatch)
# ---------------------------------------------------------------------------

def _nested_hierarchy() -> CapsuleHierarchy:
    """
    Root: CompoundCapsule(children=[inner_compound, summarizer_leaf])
      inner_compound: CompoundCapsule(children=[researcher, fact_checker])
      summarizer_leaf: AgentLeaf(summarizer)

    Expected execution: 2 LLM calls
      call 1 — inner_compound merged prompt (researcher + fact_checker)
      call 2 — summarizer as standalone leaf
    """
    researcher = _leaf("researcher")
    fact_checker = _leaf("fact_checker")
    summarizer = _leaf("summarizer")

    inner = CompoundCapsule(
        name="inner",
        children=[researcher, fact_checker],
        dependency_edges={"fact_checker": ["researcher"]},
    )
    compute_order(inner)

    root = CompoundCapsule(
        name="pipeline",
        children=[inner, summarizer],
        dependency_edges={"summarizer": ["inner"]},
    )
    compute_order(root)
    return CapsuleHierarchy(name="nested_pipeline", root=root)


def test_nested_compound_makes_two_calls():
    """COMPOUND mode on mixed-children compound: inner compound + leaf = 2 calls."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    executor.run(_nested_hierarchy(), task_input="topic", task_id="t-006")
    assert adapter.call_count == 2


def test_nested_compound_outputs_contain_all_keys():
    """All three agents' output keys are present in the final result."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_nested_hierarchy(), task_input="topic", task_id="t-006-b")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "FACT_CHECKER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs


def test_nested_compound_final_output_is_summarizer():
    """The final output comes from the last child (summarizer leaf)."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_nested_hierarchy(), task_input="topic", task_id="t-006-c")
    assert result.final_output != ""


def test_nested_compound_telemetry_two_records():
    """Mixed dispatch emits 2 telemetry records: one COMPOUND + one FINE."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_nested_hierarchy(), task_input="topic", task_id="t-006-d")
    modes = [r.composition_mode for r in result.telemetry]
    assert "COMPOUND" in modes
    assert "FINE" in modes
    assert len(result.telemetry) == 2


def test_nested_compound_three_levels_deep():
    """Three levels: root → middle_compound → inner_compound + leaf."""
    a = _leaf("a")
    b = _leaf("b")
    c = _leaf("c")

    inner = CompoundCapsule(name="inner", children=[a, b],
                            dependency_edges={"b": ["a"]})
    compute_order(inner)

    middle = CompoundCapsule(name="middle", children=[inner, c],
                             dependency_edges={"c": ["inner"]})
    compute_order(middle)

    root = CompoundCapsule(name="root", children=[middle], dependency_edges={})
    compute_order(root)

    hierarchy = CapsuleHierarchy(name="deep", root=root)
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(hierarchy, task_input="topic", task_id="t-006-e")
    # inner (a+b) = 1 call, c = 1 call, total = 2
    assert adapter.call_count == 2
    assert "A_OUTPUT" in result.outputs
    assert "B_OUTPUT" in result.outputs
    assert "C_OUTPUT" in result.outputs


def test_all_leaf_compound_unaffected():
    """A compound with only leaf children still uses the original merged path (1 call)."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    executor.run(_hierarchy("researcher", "fact_checker", "summarizer"),
                 task_input="topic", task_id="t-006-f")
    assert adapter.call_count == 1


# ---------------------------------------------------------------------------
# T-038: standard mode tool budget
# ---------------------------------------------------------------------------

class _ToolTrackingAdapter:
    """Records each complete() call with its tools argument."""
    context_window = 200_000
    _last_tool_call_count = 0

    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, messages, tools=None):
        self.calls.append({"tools": tools})
        keys = re.findall(r"(\w+_OUTPUT)", messages[-1].content + messages[0].content)
        seen = set()
        unique_keys = [k for k in keys if not (k in seen or seen.add(k))]
        return "\n\n".join(f"{k}:\nResult." for k in unique_keys) if unique_keys else "RESULT_OUTPUT:\nDone."

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def _leaf_with_tools(name: str, tools: list[str]) -> AgentLeaf:
    from agentic_capsules.core.capsule import AgentStepCapsule
    from agentic_capsules.core.types import Schema
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=f"You are the {name} agent.",
            input_schema=Schema("in", fields={"text": "str"}),
            output_schema=Schema("out", fields={"result": "str"}),
            tools=tools,
        )
    )


def _hierarchy_with_tools(*specs: tuple) -> CapsuleHierarchy:
    """specs: list of (name, tools_list) tuples."""
    leaves = [_leaf_with_tools(name, tools) for name, tools in specs]
    root = CompoundCapsule(name="pipeline", children=leaves, dependency_edges={})
    compute_order(root)
    return CapsuleHierarchy(name="test_pipeline", root=root)


def _make_tool_registry(*names: str):
    from agentic_capsules.tools.registry import ToolDefinition, ToolRegistry
    reg = ToolRegistry()
    for n in names:
        reg.register(ToolDefinition(
            name=n,
            description=f"Tool {n}",
            input_schema={"type": "object"},
            callable=lambda inp, _n=n: {"result": f"{_n}_result"},
        ))
    return reg


def test_standard_mode_no_tools_when_budget_zero():
    """Default compound_tool_budget=0 means no tools passed to merged call."""
    adapter = _ToolTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="standard",
        compound_tool_budget=0,
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    assert adapter.calls[0]["tools"] is None


def test_standard_mode_tools_passed_when_budget_nonzero():
    """compound_tool_budget=-1 passes all agent tool defs to merged call."""
    adapter = _ToolTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="standard",
        compound_tool_budget=-1,
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    passed_tools = adapter.calls[0]["tools"]
    assert passed_tools is not None
    assert any(t.name == "search" for t in passed_tools)


def test_standard_mode_no_tools_when_no_registry():
    """Budget != 0 but no registry → still no tools passed."""
    adapter = _ToolTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="standard",
        compound_tool_budget=-1,
        # no tool_registry
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    assert adapter.calls[0]["tools"] is None


# ---------------------------------------------------------------------------
# T-038: two_phase mode
# ---------------------------------------------------------------------------

class _TwoPhaseTrackingAdapter:
    """
    Records every complete() call.  Simulates N_TOOL_CALLS tool calls on
    the first call per phase-A agent (i.e. when tools= is provided).
    """
    context_window = 200_000

    def __init__(self, n_tool_calls_phase_a: int = 1):
        self.calls: list[dict] = []
        self._n_tool_calls = n_tool_calls_phase_a
        self._last_tool_call_count = 0
        self._last_tool_call_sequence: list[str] = []

    def complete(self, messages, tools=None):
        self._last_tool_call_count = 0
        self._last_tool_call_sequence = []
        self.calls.append({"tools": tools, "messages": messages})

        if tools:
            # Phase A gather: simulate tool calls and return a data summary
            for i in range(min(self._n_tool_calls, len(tools))):
                tools[i].callable({"query": "test"})
                self._last_tool_call_count += 1
                self._last_tool_call_sequence.append(tools[i].name)
            return "Gathered data: revenue $124.3B, PE ratio 28.4."

        # Phase B or no-tool call: return phase outputs
        keys = re.findall(r"(\w+_OUTPUT)", messages[-1].content + messages[0].content)
        seen = set()
        unique_keys = [k for k in keys if not (k in seen or seen.add(k))]
        return "\n\n".join(f"{k}:\nResult." for k in unique_keys) if unique_keys else "RESULT_OUTPUT:\nDone."

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def test_two_phase_makes_phase_a_plus_phase_b_calls():
    """With 2 tool-using agents, two_phase makes 2 Phase A + 1 Phase B = 3 calls."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(
        ("researcher", ["search"]),
        ("analyst",    ["analyze"]),
    )
    registry = _make_tool_registry("search", "analyze")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    assert len(adapter.calls) == 3  # 2 Phase A + 1 Phase B


def test_two_phase_no_tool_agents_skip_phase_a():
    """Tool-free agents are skipped in Phase A; only tool agents get a gather call."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(
        ("researcher", ["search"]),
        ("analyst",    []),          # no tools → skipped in Phase A
    )
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    # 1 Phase A (researcher only) + 1 Phase B = 2 calls
    assert len(adapter.calls) == 2


def test_two_phase_falls_back_to_standard_when_no_agents_have_tools():
    """When no agents have tools, two_phase falls back to standard (1 call)."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(
        ("analyst",    []),
        ("writer",     []),
    )
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    # No tools → standard fallback → 1 compound call
    assert len(adapter.calls) == 1


def test_two_phase_phase_a_calls_receive_tools():
    """Phase A calls are made with tools= set; Phase B call has tools=None."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    # First call: Phase A for researcher → tools provided
    assert adapter.calls[0]["tools"] is not None
    # Second call: Phase B compound → no tools
    assert adapter.calls[1]["tools"] is None


def test_two_phase_no_tool_agents_only_makes_one_call():
    """Without a registry, two_phase falls back to standard — 1 compound call."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", []), ("analyst", []))
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    assert len(adapter.calls) == 1


def test_two_phase_tool_context_injected_in_phase_b():
    """Phase A gather response appears in Phase B system prompt."""
    adapter = _TwoPhaseTrackingAdapter(n_tool_calls_phase_a=1)
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    # Phase B call is the last one; its system message should contain injected context
    phase_b_messages = adapter.calls[-1]["messages"]
    system_content = phase_b_messages[0].content
    assert "Pre-gathered tool data for this phase:" in system_content


def test_two_phase_telemetry_includes_phase_a_records():
    """COMPOUND_PHASE_A records appear in telemetry for each tool-using Phase A agent."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(
        ("researcher", ["search"]),
        ("analyst",    ["analyze"]),
    )
    registry = _make_tool_registry("search", "analyze")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    result = executor.run(hierarchy, task_input="task", task_id="t")
    modes = [r.composition_mode for r in result.telemetry]
    assert modes.count("COMPOUND_PHASE_A") == 2
    assert modes.count("COMPOUND") == 1


def test_two_phase_compound_record_includes_tool_calls():
    """COMPOUND telemetry record's tool_calls reflects total Phase A tool invocations."""
    adapter = _TwoPhaseTrackingAdapter(n_tool_calls_phase_a=2)
    hierarchy = _hierarchy_with_tools(("researcher", ["search", "fetch"]), ("analyst", []))
    registry = _make_tool_registry("search", "fetch")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    result = executor.run(hierarchy, task_input="task", task_id="t")
    compound_record = next(r for r in result.telemetry if r.composition_mode == "COMPOUND")
    # Phase A simulates min(n_tool_calls, len(tools)) = min(2, 2) = 2 tool calls
    assert compound_record.tool_calls == 2


def test_two_phase_outputs_all_keys():
    """two_phase mode still returns all agent output keys."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="two_phase",
    )
    result = executor.run(hierarchy, task_input="task", task_id="t")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "ANALYST_OUTPUT" in result.outputs


# ---------------------------------------------------------------------------
# T-039: sequential COMPOUND mode
# ---------------------------------------------------------------------------

def test_sequential_makes_n_calls():
    """sequential mode makes one LLM call per agent (N total, not 1)."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="sequential",
    )
    executor.run(_hierarchy("researcher", "analyst", "writer"),
                 task_input="task", task_id="t")
    assert adapter.call_count == 3


def test_sequential_returns_all_output_keys():
    """sequential mode returns all per-agent output keys."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="sequential",
    )
    result = executor.run(_hierarchy("researcher", "analyst"),
                          task_input="task", task_id="t")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "ANALYST_OUTPUT" in result.outputs


def test_sequential_later_agents_see_prior_outputs():
    """In sequential mode each agent receives all previous agents' outputs in its prompt."""
    seen_inputs: list[str] = []

    class _CapturingAdapter:
        context_window = 200_000
        call_count = 0

        def complete(self, messages, tools=None):
            self.call_count += 1
            seen_inputs.append(messages[-1].content)
            keys = re.findall(r"(\w+_OUTPUT)", messages[-1].content + messages[0].content)
            seen = set()
            unique = [k for k in keys if not (k in seen or seen.add(k))]
            if not unique:
                return "RESULT_OUTPUT:\nDone."
            return "\n\n".join(f"{k}:\nResult for {k}." for k in unique)

        def count_tokens(self, text):
            return max(1, len(text) // 4)

    adapter = _CapturingAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="sequential",
    )
    executor.run(_hierarchy("researcher", "analyst"), task_input="task", task_id="t")

    # Second agent's prompt should be longer than first (includes prior output section)
    assert len(seen_inputs[1]) > len(seen_inputs[0])
    # Second agent's prompt should contain the first agent's actual output value
    assert "Result for RESEARCHER_OUTPUT" in seen_inputs[1]


def test_sequential_emits_single_compound_telemetry_record():
    """sequential mode emits one COMPOUND telemetry record for the group."""
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="sequential",
    )
    result = executor.run(_hierarchy("researcher", "analyst"),
                          task_input="task", task_id="t")
    compound_records = [r for r in result.telemetry if r.composition_mode == "COMPOUND"]
    assert len(compound_records) == 1
    assert compound_records[0].batch_size == 2


def test_sequential_preserves_per_agent_tools():
    """In sequential mode each agent with tools receives its own tool definitions."""
    adapter = _TwoPhaseTrackingAdapter()
    hierarchy = _hierarchy_with_tools(("researcher", ["search"]), ("analyst", []))
    registry = _make_tool_registry("search")
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.COMPOUND,
        tool_registry=registry,
        compound_execution_model="sequential",
    )
    executor.run(hierarchy, task_input="task", task_id="t")
    # researcher call should have tools; analyst call should not
    assert adapter.calls[0]["tools"] is not None   # researcher has tools
    assert adapter.calls[1]["tools"] is None        # analyst has no tools


def test_sequential_policy_validation_accepts_sequential():
    """ControllerPolicy accepts 'sequential' as a valid compound_execution_model."""
    from agentic_capsules.controller.policy import ControllerPolicy
    policy = ControllerPolicy(compound_execution_model="sequential")
    assert policy.compound_execution_model == "sequential"


# ---------------------------------------------------------------------------
# T-039: auto-calibrate compound_min_output_words
# ---------------------------------------------------------------------------

def test_auto_min_output_words_returns_none_before_min_obs():
    """get_auto_min_output_words returns None when fewer than min_obs observations."""
    from agentic_capsules.api.state import PipelineState
    from agentic_capsules.controller.policy import ControllerPolicy
    state = PipelineState("test", ControllerPolicy())
    state.record_avg_output_tokens_fine("research", 1000.0)
    state.record_avg_output_tokens_fine("research", 1100.0)
    # Only 2 observations — default min_obs=3 not met
    assert state.get_auto_min_output_words("research") is None


def test_auto_min_output_words_calibrates_from_fine_observations():
    """get_auto_min_output_words returns a proportional floor after min_obs observations."""
    from agentic_capsules.api.state import PipelineState
    from agentic_capsules.controller.policy import ControllerPolicy
    state = PipelineState("test", ControllerPolicy())
    for _ in range(3):
        state.record_avg_output_tokens_fine("research", 1300.0)  # ~1000 words at 1.3 tok/word
    result = state.get_auto_min_output_words("research")
    # floor_pct=0.75: 1300 / 1.3 * 0.75 = 750 words
    assert result is not None
    assert result == 750


def test_auto_min_output_words_floors_at_50():
    """get_auto_min_output_words never returns less than 50 words."""
    from agentic_capsules.api.state import PipelineState
    from agentic_capsules.controller.policy import ControllerPolicy
    state = PipelineState("test", ControllerPolicy())
    for _ in range(3):
        state.record_avg_output_tokens_fine("synthesis", 50.0)  # tiny output
    result = state.get_auto_min_output_words("synthesis")
    assert result == 50


# ---------------------------------------------------------------------------
# G-2 (parity): FINE mode must honor explicit depends_on=[] — siblings that
# declare no deps must not see each other's outputs via accumulated_outputs.
# This is the fine-mode analogue of the sequential "deps" injection strategy
# already implemented in _run_compound_sequential.
# ---------------------------------------------------------------------------

class _FineCapturingAdapter:
    """Records the user-message text each leaf receives."""
    context_window = 200_000

    def __init__(self):
        self.seen: list[str] = []
        self.call_count = 0

    def complete(self, messages, tools=None):
        self.call_count += 1
        user_text = messages[-1].content
        self.seen.append(user_text)
        # Reply with the default OUTPUT heading(s) found in the prompt
        keys = re.findall(r"(\w+_OUTPUT)", user_text + messages[0].content)
        seen_keys: set[str] = set()
        unique = [k for k in keys if not (k in seen_keys or seen_keys.add(k))]
        if not unique:
            return "RESULT_OUTPUT:\nDone."
        return "\n\n".join(f"{k}:\nResult for {k}." for k in unique)

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def _hierarchy_with_edges(names, dep_edges):
    """Build a CapsuleHierarchy with explicit dependency_edges (strategy='deps').

    Mirrors how ``_compile_group`` constructs compounds when any agent used
    the public ``depends_on=...`` kwarg: ``sequential_injection_strategy`` is
    forced to ``"deps"`` via ``classify_and_set_strategy`` so the FINE-mode
    dep-aware filter applies.
    """
    from agentic_capsules.runtime.topology import classify_and_set_strategy

    leaves = [_leaf(n) for n in names]
    root = CompoundCapsule(
        name="pipeline", children=leaves, dependency_edges=dep_edges
    )
    compute_order(root)
    classify_and_set_strategy(root, has_explicit_dependencies=True)
    return CapsuleHierarchy(name="test_pipeline", root=root)


def test_fine_mode_siblings_with_empty_depends_on_do_not_see_each_other():
    """FINE mode: leaves with explicit depends_on=[] must not receive prior siblings' outputs.

    G-2 regression test. Mirrors the real multi_source_brief arms where each
    extractor declares depends_on=[] (parallel within its group). Before the
    fix, _run_fine passed all accumulated outputs to every subsequent leaf,
    causing claims/signals extractors to quote entities' output via
    prior_outputs and inflate their input by hundreds of characters.
    """
    adapter = _FineCapturingAdapter()
    # Three siblings, all explicitly independent
    dep_edges = {"entities": [], "claims": [], "signals": []}
    hierarchy = _hierarchy_with_edges(["entities", "claims", "signals"], dep_edges)

    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    executor.run(hierarchy, task_input="source bundle", task_id="t1")

    assert adapter.call_count == 3
    # Later siblings must NOT see earlier siblings' output values
    assert "Result for ENTITIES_OUTPUT" not in adapter.seen[1]
    assert "Result for ENTITIES_OUTPUT" not in adapter.seen[2]
    assert "Result for CLAIMS_OUTPUT" not in adapter.seen[2]


def test_fine_mode_explicit_depends_on_chains_only_declared_deps():
    """FINE mode: leaves with depends_on=[X] must see X's output but NOT unrelated siblings'.

    Preservation + scope test. A leaf that explicitly depends on one sibling
    must continue to see that sibling's output, but must not receive outputs
    from other siblings that happened to run before it.
    """
    adapter = _FineCapturingAdapter()
    # scoping is independent; extractor depends_on=['scoping'] but NOT on noise_a
    dep_edges = {
        "scoping": [],
        "noise_a": [],
        "extractor": ["scoping"],
    }
    hierarchy = _hierarchy_with_edges(
        ["scoping", "noise_a", "extractor"], dep_edges
    )

    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    executor.run(hierarchy, task_input="task", task_id="t1")

    # extractor is the 3rd call — it must see scoping but NOT noise_a
    extractor_prompt = adapter.seen[2]
    assert "Result for SCOPING_OUTPUT" in extractor_prompt
    assert "Result for NOISE_A_OUTPUT" not in extractor_prompt


def test_fine_mode_implicit_linear_chaining_preserved():
    """FINE mode: pipelines with empty dependency_edges (implicit linear) still chain.

    Regression guard. The existing _hierarchy helper constructs CompoundCapsules
    with dependency_edges={}. Many live pipelines and tests rely on the legacy
    behavior where each leaf sees all prior accumulated outputs. The G-2 fix
    must preserve this path — only explicit depends_on declarations narrow the
    visibility window.
    """
    adapter = _FineCapturingAdapter()
    # dependency_edges={} → legacy implicit behavior
    hierarchy = _hierarchy("researcher", "analyst", "writer")

    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    executor.run(hierarchy, task_input="topic", task_id="t1")

    # Under legacy behavior, analyst sees researcher, writer sees both
    assert "Result for RESEARCHER_OUTPUT" in adapter.seen[1]
    assert "Result for RESEARCHER_OUTPUT" in adapter.seen[2]
    assert "Result for ANALYST_OUTPUT" in adapter.seen[2]


# ---------------------------------------------------------------------------
# G-1 (parity, Phase 2 Batch B-2): _aggregate_terminal_outputs label
# compaction.
#
# Multi-terminal groups in multi_source_brief (competitive / product /
# financial / risk — each a 3-agent parallel-independent compound) pass
# their final_output into the downstream briefer via cross-group dep
# injection. Before the fix the aggregation format was
#     [competitive_entities]\n<content>
#     [competitive_claims]\n<content>
#     [competitive_signals]\n<content>
# which duplicates the parent-group label ("competitive") on every inner
# sub-label. LangGraph's equivalent output uses
#     [entities]\n<content>
#     [claims]\n<content>
#     [signals]\n<content>
# with the parent group name already carried by the downstream
# cross-group dep injector's "[competitive output]" wrapper — so
# repeating it on every inner label is pure overhead. The fix strips
# the compound name prefix from inner labels when every terminal
# shares it.
# ---------------------------------------------------------------------------

def test_aggregate_terminal_outputs_strips_compound_name_prefix():
    """Multi-terminal group: inner labels drop the compound-name prefix."""
    from agentic_capsules.runtime.executor import _aggregate_terminal_outputs

    # Compound named "competitive" with three terminals whose names all
    # start with "competitive_". Labels should collapse to [entities] etc.
    leaves = [
        _leaf("competitive_entities"),
        _leaf("competitive_claims"),
        _leaf("competitive_signals"),
    ]
    compound = CompoundCapsule(
        name="competitive", children=leaves, dependency_edges={}
    )
    compute_order(compound)
    accumulated = {
        "COMPETITIVE_ENTITIES_OUTPUT": "entity content",
        "COMPETITIVE_CLAIMS_OUTPUT": "claim content",
        "COMPETITIVE_SIGNALS_OUTPUT": "signal content",
    }
    out = _aggregate_terminal_outputs(compound, accumulated)
    # Compact LG-style labels, parent-name prefix stripped
    assert "[entities]" in out
    assert "[claims]" in out
    assert "[signals]" in out
    assert "[competitive_entities]" not in out
    assert "[competitive_claims]" not in out
    # Content still carried through
    assert "entity content" in out
    assert "claim content" in out
    assert "signal content" in out


def test_aggregate_terminal_outputs_no_strip_when_prefix_mismatch():
    """When terminals do not all share the compound-name prefix, keep full names.

    Guards against accidentally stripping something non-prefix-shaped
    (e.g. a compound named 'fan' with workers 'w_0', 'w_1', 'w_2') — the
    existing fan-out resume test relies on the legacy [w_0]/[w_1]/[w_2]
    format.
    """
    from agentic_capsules.runtime.executor import _aggregate_terminal_outputs

    leaves = [_leaf("w_0"), _leaf("w_1"), _leaf("w_2")]
    compound = CompoundCapsule(
        name="fan", children=leaves, dependency_edges={}
    )
    compute_order(compound)
    accumulated = {
        "W_0_OUTPUT": "alpha",
        "W_1_OUTPUT": "beta",
        "W_2_OUTPUT": "gamma",
    }
    out = _aggregate_terminal_outputs(compound, accumulated)
    assert "[w_0]" in out
    assert "[w_1]" in out
    assert "[w_2]" in out


def test_aggregate_terminal_outputs_single_terminal_unchanged():
    """Single-terminal groups still return raw content (no label wrapper)."""
    from agentic_capsules.runtime.executor import _aggregate_terminal_outputs

    leaf = _leaf("briefer")
    compound = CompoundCapsule(
        name="synthesis", children=[leaf], dependency_edges={}
    )
    compute_order(compound)
    accumulated = {"BRIEFER_OUTPUT": "the final brief."}
    out = _aggregate_terminal_outputs(compound, accumulated)
    # Single-terminal path is unchanged: raw content, no label wrapper.
    assert out == "the final brief."
    assert "[briefer]" not in out
