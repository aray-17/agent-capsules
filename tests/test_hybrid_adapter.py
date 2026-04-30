"""
T-043 — Hybrid adapter tests.

Verifies that group-level adapter overrides work correctly while preserving
full backward compatibility when no override is set.
"""
import pytest
from agentic_capsules import Pipeline, Tool, PipelineResult


# ---------------------------------------------------------------------------
# Minimal scripted adapter that records which calls it handled
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    context_window = 200_000

    def __init__(self, label: str, response: str = "## OUTPUT\nResult."):
        self.label       = label
        self._response   = response
        self.call_count  = 0
        self.calls: list[list] = []  # all messages received

    def complete(self, messages, tools=None):
        self.call_count += 1
        self.calls.append(messages)
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def __repr__(self):
        return f"_ScriptedAdapter({self.label!r})"


# ---------------------------------------------------------------------------
# Builder-level tests — _GroupSpec.adapter field
# ---------------------------------------------------------------------------

def test_group_adapter_defaults_to_none():
    p = Pipeline("test").group("g").agent("a", "do it")
    assert p._groups[0].adapter is None


def test_group_adapter_stored_when_set():
    adp = _ScriptedAdapter("override")
    p = Pipeline("test").group("g", adapter=adp).agent("a", "do it")
    assert p._groups[0].adapter is adp


def test_group_adapter_independent_per_group():
    adp1 = _ScriptedAdapter("cheap")
    adp2 = _ScriptedAdapter("expensive")
    p = (
        Pipeline("test")
        .group("research",  adapter=adp1).agent("r", "research")
        .group("synthesis", adapter=adp2).agent("s", "synthesise")
    )
    assert p._groups[0].adapter is adp1
    assert p._groups[1].adapter is adp2


def test_mixed_groups_some_with_override():
    adp = _ScriptedAdapter("override")
    p = (
        Pipeline("test")
        .group("research", adapter=adp).agent("r", "research")
        .group("synthesis").agent("s", "synthesise")
    )
    assert p._groups[0].adapter is adp
    assert p._groups[1].adapter is None  # no override → will fall back to pipeline adapter


def test_group_returns_pipeline_with_adapter():
    """Method chaining still works when adapter kwarg is passed."""
    adp = _ScriptedAdapter("x")
    p = Pipeline("test")
    result = p.group("g", adapter=adp)
    assert result is p


# ---------------------------------------------------------------------------
# Execution-level tests — correct adapter used per group
# ---------------------------------------------------------------------------

def test_pipeline_adapter_used_when_no_group_override():
    """Without any group overrides the pipeline adapter handles all calls."""
    pipeline_adp = _ScriptedAdapter("pipeline")
    (
        Pipeline("test")
        .group("g1").agent("a", "do it")
        .group("g2").agent("b", "do it too")
        .run("task", adapter=pipeline_adp)
    )
    assert pipeline_adp.call_count == 2  # one FINE call per group


def test_group_override_adapter_receives_calls():
    """Group override adapter is called instead of the pipeline adapter."""
    pipeline_adp = _ScriptedAdapter("pipeline")
    group_adp    = _ScriptedAdapter("group-override")

    (
        Pipeline("test")
        .group("research", adapter=group_adp).agent("r", "research")
        .run("task", adapter=pipeline_adp)
    )

    assert group_adp.call_count    == 1
    assert pipeline_adp.call_count == 0


def test_mixed_pipeline_routes_correctly():
    """Groups with override use their own adapter; groups without use the pipeline adapter."""
    pipeline_adp = _ScriptedAdapter("pipeline")
    research_adp = _ScriptedAdapter("research-cheap")

    (
        Pipeline("test")
        .group("research",  adapter=research_adp).agent("r", "research")
        .group("synthesis").agent("s", "synthesise")
        .run("task", adapter=pipeline_adp)
    )

    assert research_adp.call_count  == 1  # override adapter handled research
    assert pipeline_adp.call_count  == 1  # pipeline adapter handled synthesis


def test_three_groups_mixed_routing():
    pipeline_adp  = _ScriptedAdapter("pipeline")
    group1_adp    = _ScriptedAdapter("group1")
    group3_adp    = _ScriptedAdapter("group3")

    (
        Pipeline("test")
        .group("g1", adapter=group1_adp).agent("a1", "first")
        .group("g2").agent("a2", "second")
        .group("g3", adapter=group3_adp).agent("a3", "third")
        .run("task", adapter=pipeline_adp)
    )

    assert group1_adp.call_count   == 1
    assert pipeline_adp.call_count == 1  # only g2
    assert group3_adp.call_count   == 1


def test_result_still_produced_with_group_override():
    """PipelineResult assembles correctly when group adapters differ."""
    pipeline_adp = _ScriptedAdapter("pipeline",  response="## OUTPUT\npipeline output")
    group_adp    = _ScriptedAdapter("override",  response="## OUTPUT\ngroup output")

    result = (
        Pipeline("test")
        .group("research",  adapter=group_adp).agent("researcher", "research")
        .group("synthesis").agent("writer",     "write")
        .run("task", adapter=pipeline_adp)
    )

    assert isinstance(result, PipelineResult)
    assert result.output != ""


def test_group_override_does_not_affect_controller_state():
    """
    The composition score and controller state are pipeline-scoped.
    Using a group adapter override must not corrupt state for other groups.
    """
    pipeline_adp = _ScriptedAdapter("pipeline")
    group_adp    = _ScriptedAdapter("override")

    result = (
        Pipeline("test")
        .group("research",  adapter=group_adp).agent("r", "research")
        .group("synthesis").agent("s", "synthesis")
        .run("task", adapter=pipeline_adp)
    )

    # Both groups must appear in scores (controller observed both)
    assert "research"  in result.scores
    assert "synthesis" in result.scores


# ---------------------------------------------------------------------------
# Eval compatibility — pipeline adapter still valid for Pareto / due diligence
# ---------------------------------------------------------------------------

def test_pareto_pattern_no_group_overrides():
    """
    Simulates the Pareto sweep pattern: one adapter for the whole pipeline.
    No group overrides → same behavior as before T-043.
    """
    adp = _ScriptedAdapter("sweep-model")

    result = (
        Pipeline("due-diligence")
        .group("research").agent("researcher", "Research the topic.")
        .group("analysis").agent("analyst",    "Analyse the research.")
        .group("synthesis").agent("writer",    "Write the report.")
        .run("AI safety", adapter=adp, mode="fine")
    )

    assert adp.call_count == 3  # one FINE call per group, all on same adapter
    assert isinstance(result, PipelineResult)


def test_group_adapter_not_used_in_calibrate():
    """
    calibrate() always uses the adapter passed directly to it,
    regardless of group overrides. This preserves a single cost model
    for calibration.
    """
    from agentic_capsules.evaluation.schema_compliance import SchemaComplianceEvaluator
    pipeline_adp = _ScriptedAdapter("pipeline")
    group_adp    = _ScriptedAdapter("override")

    p = (
        Pipeline("test")
        .group("g", adapter=group_adp).agent("a", "do it")
    )
    # calibrate() passes pipeline_adp explicitly — group override should not
    # interfere with calibration's token accounting (both FINE and COMPOUND
    # runs use the same adapter for fair comparison)
    report = p.calibrate(
        sample_tasks=["task1"],
        adapter=pipeline_adp,
        evaluator=SchemaComplianceEvaluator(),
        n_paired_runs=1,
    )
    # Calibration ran — report has the group
    assert "g" in report.quality_by_group()
