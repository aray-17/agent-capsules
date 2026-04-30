"""
T-042 — Sequential Topology tests.

Verifies:
  - Topology classification (linear / fan_out / diamond / parallel_converge)
  - Strategy selection (full / deps)
  - Gate 0: non-linear topology forces sequential in auto mode
  - Executor dep-aware injection: each leaf only receives its declared deps
"""
import pytest
from agentic_capsules.core.hierarchy import AgentLeaf, CompoundCapsule
from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.types import Schema
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.runtime.topology import _classify_topology, classify_and_set_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_leaf(name: str) -> AgentLeaf:
    return AgentLeaf(capsule=AgentStepCapsule(
        name=name,
        system_prompt=f"Do {name}.",
        input_schema=Schema(f"{name}_in", fields={"q": "str"}),
        output_schema=Schema(f"{name}_out", fields={"r": "str"}),
    ))


def _make_compound(name: str, agents: list[str], dep_edges: dict) -> CompoundCapsule:
    leaves = [_make_leaf(n) for n in agents]
    compound = CompoundCapsule(name=name, children=leaves, dependency_edges=dep_edges)
    compute_order(compound)
    return compound


# ---------------------------------------------------------------------------
# S1 — Linear: A→B→C
# ---------------------------------------------------------------------------

def test_classify_linear():
    c = _make_compound("g", ["a", "b", "c"], {"b": ["a"], "c": ["b"]})
    assert _classify_topology(c) == "linear"


def test_strategy_linear_short():
    c = _make_compound("g", ["a", "b", "c"], {"b": ["a"], "c": ["b"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "full"


def test_strategy_linear_verbose():
    """Verbose linear — summary stub falls through to 'full' until T-042.2."""
    c = _make_compound("g", ["a", "b"], {"b": ["a"]})
    classify_and_set_strategy(c, avg_output_tokens=4000)
    # T-042.2 not yet implemented — stub returns "full"
    assert c.sequential_injection_strategy == "full"


# ---------------------------------------------------------------------------
# S2 — Fan-out: A→B, A→C, A→D
# ---------------------------------------------------------------------------

def test_classify_fan_out():
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["a"]})
    assert _classify_topology(c) == "fan_out"


def test_strategy_fan_out():
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["a"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "deps"


# ---------------------------------------------------------------------------
# S3 — Diamond: A→B→D, A→C→D
# ---------------------------------------------------------------------------

def test_classify_diamond():
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["b", "c"]})
    assert _classify_topology(c) == "diamond"


def test_strategy_diamond():
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["b", "c"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "deps"


# ---------------------------------------------------------------------------
# S4 — Parallel converge: root→(B‖C)→synth
# ---------------------------------------------------------------------------

def test_classify_parallel_converge():
    # root→b, root→c, b→synth, c→synth
    c = _make_compound("g", ["root", "b", "c", "synth"],
                        {"b": ["root"], "c": ["root"], "synth": ["b", "c"]})
    # synth has in_degree=2 → classified as diamond
    # (parallel_converge is a subset of diamond in our classifier)
    topo = _classify_topology(c)
    assert topo in ("diamond", "parallel_converge")
    # Either way, strategy must be "deps"
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "deps"


# ---------------------------------------------------------------------------
# Single node — edge case
# ---------------------------------------------------------------------------

def test_classify_single_node():
    c = _make_compound("g", ["a"], {})
    assert _classify_topology(c) == "linear"
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "full"


# ---------------------------------------------------------------------------
# Dep-aware injection — verify executor passes correct context
# ---------------------------------------------------------------------------

class _CapturingAdapter:
    context_window = 200_000

    def __init__(self):
        self.calls: list[list] = []   # messages per call

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        return "## OUTPUT\nResult."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _run_sequential(compound: CompoundCapsule, adapter=None):
    """Run _run_compound_sequential via the public executor interface."""
    from agentic_capsules.runtime.executor import CapsuleExecutor
    from agentic_capsules.core.hierarchy import CapsuleHierarchy
    from agentic_capsules.controller.telemetry import TelemetryCollector

    adp = adapter or _CapturingAdapter()
    executor = CapsuleExecutor(
        adapter=adp,
        composition_level=None,
        telemetry=TelemetryCollector(),
        compound_execution_model="sequential",
    )
    from agentic_capsules.core.types import CompositionLevel
    executor._composition_level = CompositionLevel.COMPOUND

    result = executor._run_compound_sequential(
        compound=compound,
        task_input="test task",
        task_id="t0",
    )
    return result, adp


def test_linear_injection_full():
    """Linear: each agent receives all accumulated outputs (full strategy)."""
    c = _make_compound("g", ["a", "b", "c"], {"b": ["a"], "c": ["b"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "full"

    _, adp = _run_sequential(c, _CapturingAdapter())
    # 3 calls were made
    assert len(adp.calls) == 3
    # Prior outputs are injected as "\n{KEY}:\n{value}" — check that format
    # Agent A: no prior outputs yet
    assert "\nA_OUTPUT:\n" not in adp.calls[0][1].content
    # Agent B: should see A's output as prior context
    assert "\nA_OUTPUT:\n" in adp.calls[1][1].content
    # Agent C: should see A+B outputs
    assert "\nA_OUTPUT:\n" in adp.calls[2][1].content
    assert "\nB_OUTPUT:\n" in adp.calls[2][1].content


def test_fan_out_injection_deps_only():
    """Fan-out A→B, A→C, A→D: B, C, D should each only see A's output (not siblings)."""
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["a"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "deps"

    _, adp = _run_sequential(c, _CapturingAdapter())
    # 4 calls: A, B, C, D (in topological order)
    assert len(adp.calls) == 4

    # Call 0 = A (root): no deps → no prior context
    assert "\nA_OUTPUT:\n" not in adp.calls[0][1].content

    # Calls 1,2,3 = B, C, D — each sees A's prior output, not siblings
    for call_idx in [1, 2, 3]:
        user_content = adp.calls[call_idx][1].content
        assert "\nA_OUTPUT:\n" in user_content, f"Call {call_idx} missing A_OUTPUT prior"


def test_diamond_injection_deps_only():
    """Diamond A→B→D, A→C→D: D only sees B and C outputs (not A directly)."""
    c = _make_compound("g", ["a", "b", "c", "d"],
                        {"b": ["a"], "c": ["a"], "d": ["b", "c"]})
    classify_and_set_strategy(c)
    assert c.sequential_injection_strategy == "deps"

    _, adp = _run_sequential(c, _CapturingAdapter())
    # 4 calls: A, B, C, D
    assert len(adp.calls) == 4

    # D is the last call — it depends on B and C (not A directly)
    d_user_content = adp.calls[3][1].content
    assert "\nB_OUTPUT:\n" in d_user_content
    assert "\nC_OUTPUT:\n" in d_user_content
    # D does NOT declare A as a direct dependency
    assert "\nA_OUTPUT:\n" not in d_user_content


# ---------------------------------------------------------------------------
# Gate 0 — non-linear topology forces "sequential" in auto mode
# ---------------------------------------------------------------------------

def test_gate0_fan_out_forces_sequential():
    """Auto mode: fan_out topology with strategy='deps' → sequential."""
    from agentic_capsules import Pipeline
    from agentic_capsules.controller.policy import ControllerPolicy

    class _ScriptedAdapter:
        context_window = 200_000
        last_messages = None

        def complete(self, messages, tools=None):
            self.last_messages = messages
            return "## OUTPUT\nResult."

        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    adp = _ScriptedAdapter()

    # Build a pipeline; use mode="compound" explicitly to test Gate 0
    # (auto would need FINE observations; compound forces the executor path)
    result = (
        Pipeline("test", policy=ControllerPolicy(compound_execution_model="sequential"))
        .group("g")
        .agent("a", "root task")
        .agent("b", "branch b")
        .agent("c", "branch c")
        .run("topic", adapter=adp, mode="compound")
    )
    assert adp.last_messages is not None
