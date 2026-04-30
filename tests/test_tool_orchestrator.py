"""Tests for tools/orchestrator.py and ToolLeaf executor dispatch"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule, ToolLeaf
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.tools.orchestrator import ToolExecutionError, ToolOrchestrator
from agentic_capsules.tools.tool_adapter import MockToolAdapter
from agentic_capsules.tools.tool_capsule import ToolCapsule, ToolStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_chain() -> ToolCapsule:
    return ToolCapsule(
        name="search_pipeline",
        steps=[
            ToolStep("web_search", ["query"], "search_results"),
            ToolStep("fetch_page", ["url"], "page_content", input_from="search_results"),
            ToolStep("extract_text", ["html"], "extracted_text", input_from="page_content"),
        ],
    )


def _make_adapter() -> MockToolAdapter:
    adapter = MockToolAdapter()
    adapter.register("web_search", {"url": "https://example.com", "snippet": "AI safety research"})
    adapter.register("fetch_page", {"html": "<html>AI safety content</html>"})
    adapter.register("extract_text", {"text": "AI safety content extracted."})
    return adapter


# ---------------------------------------------------------------------------
# ToolOrchestrator — basic execution
# ---------------------------------------------------------------------------

def test_orchestrator_runs_all_steps():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "AI safety"})
    assert adapter.total_calls == 3


def test_orchestrator_final_output_is_last_step():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "AI safety"})
    assert result.final_output == {"text": "AI safety content extracted."}


def test_orchestrator_outputs_contain_all_step_keys():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "AI safety"})
    assert "search_results" in result.outputs
    assert "page_content" in result.outputs
    assert "extracted_text" in result.outputs


def test_orchestrator_step_latencies_length():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "test"})
    assert len(result.step_latencies_ms) == 3


def test_orchestrator_total_latency_nonnegative():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "test"})
    assert result.total_latency_ms >= 0.0


def test_orchestrator_total_calls():
    adapter = _make_adapter()
    orch = ToolOrchestrator(adapter)
    result = orch.run(_search_chain(), initial_input={"query": "test"})
    assert result.total_calls == 3


def test_orchestrator_single_step():
    adapter = MockToolAdapter()
    adapter.register("lookup", {"result": "found"})
    orch = ToolOrchestrator(adapter)
    tc = ToolCapsule(name="lookup", steps=[ToolStep("lookup", ["id"], "lookup_result")])
    result = orch.run(tc, initial_input={"id": "123"})
    assert result.final_output == {"result": "found"}
    assert result.total_calls == 1


def test_orchestrator_threads_input_from_correctly():
    """Step 2 should receive step 1's output, not the initial input."""
    adapter = MockToolAdapter()
    adapter.register("step1", {"processed": "step1_output"})
    adapter.register("step2", {"final": "step2_output"})

    # Track what step2 actually received
    received = {}
    orig_invoke = adapter.invoke
    def tracking_invoke(tool_name, input_data):
        if tool_name == "step2":
            received.update(input_data)
        return orig_invoke(tool_name, input_data)
    adapter.invoke = tracking_invoke

    orch = ToolOrchestrator(adapter)
    tc = ToolCapsule(name="chain", steps=[
        ToolStep("step1", ["query"], "step1_out"),
        ToolStep("step2", ["processed"], "step2_out", input_from="step1_out"),
    ])
    orch.run(tc, initial_input={"query": "hello"})
    # step2 received step1's output dict
    assert received == {"processed": "step1_output"}


# ---------------------------------------------------------------------------
# ToolOrchestrator — error handling
# ---------------------------------------------------------------------------

def test_orchestrator_raises_on_non_dict_response():
    class BadAdapter:
        def invoke(self, tool_name, input_data):
            return "not a dict"
        def get_schema(self, tool_name):
            return {}

    orch = ToolOrchestrator(BadAdapter())
    tc = ToolCapsule(name="bad", steps=[ToolStep("bad_tool", ["q"], "out")])
    with pytest.raises(ToolExecutionError):
        orch.run(tc, initial_input={"q": "test"})


# ---------------------------------------------------------------------------
# ToolLeaf in CapsuleHierarchy
# ---------------------------------------------------------------------------

def _agent_leaf(name: str) -> AgentLeaf:
    return AgentLeaf(capsule=AgentStepCapsule(
        name=name,
        system_prompt=f"You are the {name} agent.",
        input_schema=Schema("in", fields={"text": "str"}),
        output_schema=Schema("out", fields={"result": "str"}),
    ))


class ScriptedLLMAdapter:
    context_window = 200_000
    def __init__(self): self.call_count = 0
    def complete(self, messages, tools=None):
        self.call_count += 1
        import re
        keys = re.findall(r"(\w+_OUTPUT)", messages[0].content + messages[-1].content)
        seen = set(); unique = [k for k in keys if not (k in seen or seen.add(k))]
        return "\n\n".join(f"{k}:\nResult." for k in unique) if unique else "OUTPUT:\nDone."
    def count_tokens(self, text): return max(1, len(text) // 4)


def test_tool_leaf_in_hierarchy_no_orchestrator_raises():
    tool_leaf = ToolLeaf(tool_capsule=_search_chain())
    root = CompoundCapsule(name="pipeline", children=[tool_leaf], dependency_edges={})
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="test", root=root)

    llm_adapter = ScriptedLLMAdapter()
    executor = CapsuleExecutor(llm_adapter, composition_level=CompositionLevel.FINE)
    from agentic_capsules.core.types import CapsuleExecutionError
    with pytest.raises(CapsuleExecutionError, match="tool_orchestrator"):
        executor.run(hierarchy, task_input="test")


def test_tool_leaf_executes_via_orchestrator():
    tool_adapter = _make_adapter()
    orch = ToolOrchestrator(tool_adapter)
    tool_leaf = ToolLeaf(tool_capsule=_search_chain())
    root = CompoundCapsule(name="pipeline", children=[tool_leaf], dependency_edges={})
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="test", root=root)

    llm_adapter = ScriptedLLMAdapter()
    executor = CapsuleExecutor(
        llm_adapter, composition_level=CompositionLevel.FINE, tool_orchestrator=orch
    )
    result = executor.run(hierarchy, task_input="AI safety", task_id="t1")
    assert "SEARCH_PIPELINE_OUTPUT" in result.outputs
    assert tool_adapter.total_calls == 3
    assert llm_adapter.call_count == 0  # no LLM calls for pure tool pipeline


def test_mixed_agent_tool_pipeline():
    """ToolLeaf followed by AgentLeaf: tool result flows into agent context."""
    tool_adapter = _make_adapter()
    orch = ToolOrchestrator(tool_adapter)

    tool_leaf = ToolLeaf(tool_capsule=_search_chain())
    agent_leaf = _agent_leaf("summarizer")
    root = CompoundCapsule(
        name="pipeline",
        children=[tool_leaf, agent_leaf],
        dependency_edges={"summarizer": ["search_pipeline"]},
    )
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="test", root=root)

    llm_adapter = ScriptedLLMAdapter()
    executor = CapsuleExecutor(
        llm_adapter, composition_level=CompositionLevel.FINE, tool_orchestrator=orch
    )
    result = executor.run(hierarchy, task_input="AI safety", task_id="t1")
    assert tool_adapter.total_calls == 3
    assert llm_adapter.call_count == 1   # one LLM call for the summarizer
    assert "SUMMARIZER_OUTPUT" in result.outputs


def test_tool_leaf_telemetry_mode_is_tool():
    tool_adapter = _make_adapter()
    orch = ToolOrchestrator(tool_adapter)
    tool_leaf = ToolLeaf(tool_capsule=_search_chain())
    root = CompoundCapsule(name="pipeline", children=[tool_leaf], dependency_edges={})
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="test", root=root)

    llm_adapter = ScriptedLLMAdapter()
    executor = CapsuleExecutor(
        llm_adapter, composition_level=CompositionLevel.FINE, tool_orchestrator=orch
    )
    result = executor.run(hierarchy, task_input="test", task_id="t1")
    assert len(result.telemetry) == 1
    assert result.telemetry[0].composition_mode == "TOOL"
    assert result.telemetry[0].total_tokens == 0   # tool calls don't consume LLM tokens
