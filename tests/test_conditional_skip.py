"""
Tests for G-2 — conditional edges (runtime agent skipping).

The `condition=` kwarg on `.agent()` lets a pipeline declare a runtime
predicate that the executor consults before dispatching the agent's LLM
call. False → skip the call, propagate "" as the agent's output, emit a
zero-cost SKIPPED telemetry record. This closes the in-run recovery story
that AC was missing relative to LangGraph's `add_conditional_edges`
(see evals/langgraph_gap_phase.md G-2 for the gap entry).

Coverage:
  * Builder accepts `condition=` (callable or None) and rejects garbage
  * Compiler propagates the predicate to AgentStepCapsule.skip_condition
  * FINE executor: True → runs normally; False → no LLM call, SKIPPED
    telemetry, empty output propagates to dependents
  * Predicate sees prior agents' outputs in the same group
  * COMPOUND mode: unanimous-skip short-circuit fires; partial-skip
    proceeds normally
  * Parallel executor honours skip predicates per worker thread
  * Predicate exceptions surface as CapsuleExecutionError

These tests use stub adapters and never touch live APIs.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from agentic_capsules import Pipeline, PipelineResult
from agentic_capsules.api.compiler import _PipelineCompiler
from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import (
    AgentLeaf, CapsuleHierarchy, CompoundCapsule,
)
from agentic_capsules.core.types import (
    CapsuleExecutionError, CapsuleState, CompositionLevel, Schema,
)
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order


def _leaf(name: str, condition=None) -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=f"You are the {name} agent.",
            input_schema=Schema("in", fields={"text": "str"}),
            output_schema=Schema("out", fields={"result": "str"}),
            skip_condition=condition,
        )
    )


def _hierarchy(*leaves: AgentLeaf) -> CapsuleHierarchy:
    root = CompoundCapsule(name="pipeline", children=list(leaves), dependency_edges={})
    compute_order(root)
    return CapsuleHierarchy(name="test_pipeline", root=root)


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

class _CountingAdapter:
    """Records every complete() call so we can assert call counts."""
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nstub."):
        self._response  = response
        self._lock      = threading.Lock()
        self.call_count = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------------

def test_builder_accepts_callable_condition():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "do it", condition=lambda outputs: True)
    )
    spec = p._groups[0].agents[0]
    assert callable(spec.condition)


def test_builder_accepts_none_condition_default():
    p = Pipeline("t").group("g").agent("a", "do it")
    assert p._groups[0].agents[0].condition is None


def test_builder_rejects_non_callable_condition():
    with pytest.raises(ValueError, match="condition must be callable"):
        Pipeline("t").group("g").agent("a", "do it", condition="not a callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Compiler propagation: condition → AgentStepCapsule.skip_condition
# ---------------------------------------------------------------------------

def test_compiler_propagates_condition_to_capsule():
    pred = lambda outputs: bool(outputs)
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "second", condition=pred)
    )
    compiler = _PipelineCompiler(p, "task", MagicMock(), "fine", None)
    compound = compiler._compile_group(p._groups[0])
    leaves = {l.capsule.name: l.capsule for l in compound.serialization_order}
    assert leaves["a"].skip_condition is None
    assert leaves["b"].skip_condition is pred


# ---------------------------------------------------------------------------
# FINE mode — primary G-2 path
# ---------------------------------------------------------------------------

def test_fine_condition_true_runs_agent():
    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "always run", condition=lambda outputs: True)
    )
    result = p.run("task", adapter=adapter, mode="fine")
    assert adapter.call_count == 1
    assert result.step_outputs["a"]


def test_fine_condition_false_skips_llm_call():
    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "never run", condition=lambda outputs: False)
    )
    result = p.run("task", adapter=adapter, mode="fine")
    assert adapter.call_count == 0  # no LLM call made
    assert result.step_outputs["a"] == ""  # empty output propagates


def test_fine_skipped_agent_emits_skipped_telemetry():
    """Use the lower-level CapsuleExecutor to inspect telemetry directly."""
    adapter   = _CountingAdapter()
    hierarchy = _hierarchy(_leaf("a", condition=lambda outputs: False))
    executor  = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result    = executor.run(hierarchy, task_input="topic", task_id="t1")
    assert adapter.call_count == 0  # no LLM call
    skipped = [t for t in result.telemetry if t.composition_mode == "SKIPPED"]
    assert len(skipped) == 1
    assert skipped[0].capsule_name == "a"
    assert skipped[0].llm_call_count == 0
    assert skipped[0].total_tokens  == 0
    assert skipped[0].latency_ms    == 0.0


def test_fine_condition_sees_prior_agent_outputs_in_group():
    """The predicate must receive the in-group accumulated outputs dict."""
    seen: dict[str, str] = {}

    def capture(outputs):
        seen.update(outputs)
        return True

    adapter = _CountingAdapter(response="## OUTPUT\nfrom A")
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "second", condition=capture)
    )
    p.run("task", adapter=adapter, mode="fine")
    # By the time `capture` runs, agent A has finished and its output
    # should be in the snapshot keyed by output_key (A_OUTPUT).
    assert "A_OUTPUT" in seen
    assert seen["A_OUTPUT"]  # non-empty


def test_fine_skipped_agent_blocks_downstream_dependents_with_empty():
    """Skipped agent emits "" — downstream sees an empty prior output."""
    seen: dict[str, str] = {}

    def capture(outputs):
        seen.update(outputs)
        return True

    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "skip me", condition=lambda outputs: False)
            .agent("b", "downstream", condition=capture)
    )
    p.run("task", adapter=adapter, mode="fine")
    assert seen.get("A_OUTPUT") == ""


def test_fine_condition_exception_raises_capsule_execution_error():
    def boom(outputs):
        raise RuntimeError("kaboom")

    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "explosive", condition=boom)
    )
    with pytest.raises(CapsuleExecutionError, match="skip_condition"):
        p.run("task", adapter=adapter, mode="fine")


# ---------------------------------------------------------------------------
# COMPOUND mode — unanimous-skip short-circuit
# ---------------------------------------------------------------------------

def test_compound_unanimous_skip_short_circuits_llm_call():
    """All agents skipped → executor must not dispatch the compound call."""
    adapter   = _CountingAdapter()
    hierarchy = _hierarchy(
        _leaf("a", condition=lambda outputs: False),
        _leaf("b", condition=lambda outputs: False),
    )
    executor  = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result    = executor.run(hierarchy, task_input="topic", task_id="t1")
    assert adapter.call_count == 0
    assert result.outputs["A_OUTPUT"] == ""
    assert result.outputs["B_OUTPUT"] == ""
    skipped = [t for t in result.telemetry if t.composition_mode == "SKIPPED"]
    assert {r.capsule_name for r in skipped} == {"a", "b"}


def test_compound_partial_skip_proceeds_normally():
    """If only some agents have conditions, the compound runs normally."""
    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "always",     condition=lambda outputs: True)
            .agent("b", "no predicate")
    )
    result = p.run("task", adapter=adapter, mode="compound")
    # Compound runs as one LLM call — not zero, not two.
    assert adapter.call_count >= 1
    assert isinstance(result, PipelineResult)


def test_compound_unanimous_predicates_returning_true_proceed_normally():
    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first",  condition=lambda outputs: True)
            .agent("b", "second", condition=lambda outputs: True)
    )
    result = p.run("task", adapter=adapter, mode="compound")
    # All conditions returned True → no short-circuit → normal compound run
    assert adapter.call_count >= 1
    assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# Parallel executor — predicates run inside worker threads
# ---------------------------------------------------------------------------

def test_parallel_executor_honours_skip_condition():
    """The parallel compiler dispatches each group through CapsuleExecutor;
    the FINE-mode skip path runs inside each worker thread."""
    adapter = _CountingAdapter()
    p = (
        Pipeline("t")
        .group("ingest", depends_on=[])
            .agent("ingest_agent", "ingest")
        .group("arm_a", depends_on=["ingest"])
            .agent("a_agent", "arm A")
        .group("arm_b", depends_on=["ingest"])
            .agent(
                "b_agent", "arm B",
                condition=lambda outputs: False,  # always skip
            )
    )
    result = p.run("task", adapter=adapter, mode="fine", parallel=True)
    # Three groups, but b_agent is skipped → only 2 LLM calls
    assert adapter.call_count == 2
    assert result.step_outputs["b_agent"] == ""


# ---------------------------------------------------------------------------
# CapsuleState.SKIPPED enum value
# ---------------------------------------------------------------------------

def test_capsule_state_skipped_enum_value_exists():
    assert hasattr(CapsuleState, "SKIPPED")
    # Distinct from COMPLETE / FAILED so observers can route on it
    assert CapsuleState.SKIPPED is not CapsuleState.COMPLETE
    assert CapsuleState.SKIPPED is not CapsuleState.FAILED
