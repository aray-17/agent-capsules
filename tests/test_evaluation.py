"""
Tests for the evaluation module — P12-1 through P12-6.

All tests are offline (no API key required) unless marked with
@pytest.mark.integration (which are skipped in CI without API keys).
"""
from __future__ import annotations

import json
import pytest
from agentic_capsules.evaluation.base import QualityScore, QualityEvaluator
from agentic_capsules.evaluation.schema_compliance import SchemaComplianceEvaluator
from agentic_capsules.evaluation.llm_judge import LLMJudgeEvaluator
from agentic_capsules.evaluation.consistency import ConsistencyEvaluator
from agentic_capsules.evaluation.calibration import CalibrationReport, _GroupCalibration
from agentic_capsules.api.state import PipelineState, GroupControllerState
from agentic_capsules.controller.policy import ControllerPolicy
from agentic_capsules.core.types import CompositionLevel


# ---------------------------------------------------------------------------
# P12-1 — QualityScore and QualityEvaluator protocol
# ---------------------------------------------------------------------------

class _MockEvaluator:
    def evaluate(self, task_input, fine_output, compound_output) -> QualityScore:
        return QualityScore(score=0.9, confidence=0.8, evaluator="MockEvaluator")


def test_quality_score_clamps_to_unit_interval():
    s = QualityScore(score=1.5, confidence=0.5)
    assert s.score == 1.0

    s2 = QualityScore(score=-0.2, confidence=0.5)
    assert s2.score == 0.0


def test_quality_score_confidence_clamps():
    s = QualityScore(score=0.8, confidence=2.0)
    assert s.confidence == 1.0


def test_quality_evaluator_protocol_isinstance():
    """MockEvaluator is recognized as a QualityEvaluator via runtime_checkable Protocol."""
    assert isinstance(_MockEvaluator(), QualityEvaluator)


def test_quality_score_defaults():
    s = QualityScore(score=0.7, confidence=0.8)
    assert s.details == {}
    assert s.evaluator == ""


# ---------------------------------------------------------------------------
# P12-2 — SchemaComplianceEvaluator
# ---------------------------------------------------------------------------

def test_schema_compliance_identical_inputs():
    ev = SchemaComplianceEvaluator()
    s  = ev.evaluate("task", "Hello world analysis complete.", "Hello world analysis complete.")
    assert s.score >= 0.90
    assert s.evaluator == "SchemaComplianceEvaluator"
    assert s.confidence == 0.70


def test_schema_compliance_empty_compound():
    ev = SchemaComplianceEvaluator()
    s  = ev.evaluate("task", "Detailed analysis with many key terms and concepts.", "")
    assert s.score < 0.50


def test_schema_compliance_short_compound_below_floor():
    ev = SchemaComplianceEvaluator(completeness_floor=0.80)
    fine     = "A" * 1000
    compound = "A" * 100   # only 10% of fine — well below 80% floor
    s = ev.evaluate("task", fine, compound)
    details = s.details
    assert details["completeness"] < 1.0


def test_schema_compliance_completeness_above_floor():
    ev = SchemaComplianceEvaluator(completeness_floor=0.60)
    fine     = "The market analysis for payments infrastructure is comprehensive."
    compound = "The market analysis for payments infrastructure is complete."
    s = ev.evaluate("task", fine, compound)
    assert s.details["completeness"] == 1.0


def test_schema_compliance_format_match_both_json():
    ev = SchemaComplianceEvaluator()
    fine     = json.dumps({"score": 8, "analysis": "good"})
    compound = json.dumps({"score": 7, "analysis": "good"})
    s = ev.evaluate("task", fine, compound)
    assert s.details["format_match"] == 1.0


def test_schema_compliance_format_mismatch():
    ev = SchemaComplianceEvaluator()
    fine     = json.dumps({"score": 8})
    compound = "score is 7"   # not JSON
    s = ev.evaluate("task", fine, compound)
    assert s.details["format_match"] == 0.0


def test_schema_compliance_non_json_fine_format_score_one():
    ev = SchemaComplianceEvaluator()
    fine     = "plain text output without JSON"
    compound = "plain text output"
    s = ev.evaluate("task", fine, compound)
    assert s.details["format_match"] == 1.0


def test_schema_compliance_field_coverage_partial():
    ev = SchemaComplianceEvaluator()
    # fine has specific terms that compound omits
    fine     = "Revenue retention metrics indicate strong financial performance growth trajectory."
    compound = "Financial performance is good."
    s = ev.evaluate("task", fine, compound)
    assert s.details["field_coverage"] < 1.0


def test_schema_compliance_invalid_floor():
    with pytest.raises(ValueError):
        SchemaComplianceEvaluator(completeness_floor=0.0)
    with pytest.raises(ValueError):
        SchemaComplianceEvaluator(completeness_floor=1.5)


def test_schema_compliance_score_bounds():
    ev = SchemaComplianceEvaluator()
    for fine, compound in [("", ""), ("abc", ""), ("", "abc"), ("abc", "abc")]:
        s = ev.evaluate("task", fine, compound)
        assert 0.0 <= s.score <= 1.0


# ---------------------------------------------------------------------------
# P12-3 — LLMJudgeEvaluator (mock adapter tests)
# ---------------------------------------------------------------------------

class _MockJudgeAdapter:
    """Returns preset JSON or junk on demand."""

    def __init__(self, response: str):
        self._response = response

    def complete(self, messages, tools=None) -> str:
        return self._response


def test_llm_judge_valid_json():
    adapter = _MockJudgeAdapter('{"completeness": 8, "accuracy": 9, "structure": 7, "conciseness": 8}')
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine output", "compound output")
    assert 0.0 < s.score <= 1.0
    assert s.confidence == 0.85
    assert s.evaluator == "LLMJudgeEvaluator"


def test_llm_judge_perfect_scores():
    adapter = _MockJudgeAdapter('{"completeness": 10, "accuracy": 10, "structure": 10, "conciseness": 10}')
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine output", "compound output")
    assert abs(s.score - 1.0) < 1e-6


def test_llm_judge_zero_scores():
    adapter = _MockJudgeAdapter('{"completeness": 0, "accuracy": 0, "structure": 0, "conciseness": 0}')
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine output", "compound output")
    assert s.score == 0.0


def test_llm_judge_malformed_json_returns_zero_confidence():
    adapter = _MockJudgeAdapter("This is not JSON at all")
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine output", "compound output")
    assert s.score == 0.0
    assert s.confidence == 0.0
    assert "parse_error" in s.details


def test_llm_judge_partial_keys_uses_defaults():
    """Partial JSON response — missing keys default to mid-score (5/10 = 0.5)."""
    adapter = _MockJudgeAdapter('{"completeness": 8}')
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine output", "compound output")
    # completeness=8/10=0.8, others default to 5/10=0.5 → mean = (0.8+0.5+0.5+0.5)/4 = 0.575
    assert 0.5 < s.score < 0.8


def test_llm_judge_markdown_code_fence_stripped():
    response = '```json\n{"completeness": 8, "accuracy": 8, "structure": 8, "conciseness": 8}\n```'
    adapter  = _MockJudgeAdapter(response)
    ev = LLMJudgeEvaluator(judge_adapter=adapter)
    s  = ev.evaluate("task", "fine", "compound")
    assert s.score > 0.0
    assert s.confidence == 0.85


def test_llm_judge_adapter_exception_returns_zero_confidence():
    class _ErrorAdapter:
        def complete(self, messages, tools=None):
            raise RuntimeError("network error")
    ev = LLMJudgeEvaluator(judge_adapter=_ErrorAdapter())
    s  = ev.evaluate("task", "fine", "compound")
    assert s.score == 0.0
    assert s.confidence == 0.0
    assert "error" in s.details


# ---------------------------------------------------------------------------
# P12-4 — ConsistencyEvaluator (protocol-compliant evaluate())
# ---------------------------------------------------------------------------

def test_consistency_evaluate_identical_strings():
    ev = ConsistencyEvaluator()
    s  = ev.evaluate("task", "hello world analysis", "hello world analysis")
    assert s.score == 1.0
    assert s.confidence == 0.60


def test_consistency_evaluate_disjoint_strings():
    ev = ConsistencyEvaluator()
    s  = ev.evaluate("task", "alpha beta gamma delta", "zeta theta iota kappa")
    assert s.score == 0.0


def test_consistency_evaluate_partial_overlap():
    ev = ConsistencyEvaluator()
    s  = ev.evaluate("task", "market analysis payments infrastructure growth", "market analysis competitive landscape")
    assert 0.0 < s.score < 1.0


def test_consistency_evaluator_score_bounds():
    ev = ConsistencyEvaluator()
    s  = ev.evaluate("task", "abc", "xyz")
    assert 0.0 <= s.score <= 1.0


# ---------------------------------------------------------------------------
# P12-5 — Quality gate in ControllerPolicy and PipelineState
# ---------------------------------------------------------------------------

def _make_quality_policy(quality_floor: float | None = 0.75):
    return ControllerPolicy(
        compose_at=0.40,
        decompose_at=0.15,
        confidence=0.80,
        min_observations=3,
        window_size=5,
        quality_floor=quality_floor,
    )


def test_quality_floor_validation_out_of_range():
    with pytest.raises(ValueError, match="quality_floor"):
        ControllerPolicy(compose_at=0.40, decompose_at=0.15, quality_floor=1.5)

    with pytest.raises(ValueError, match="quality_floor"):
        ControllerPolicy(compose_at=0.40, decompose_at=0.15, quality_floor=-0.1)


def test_quality_floor_zero_is_valid():
    p = ControllerPolicy(compose_at=0.40, decompose_at=0.15, quality_floor=0.0)
    assert p.quality_floor == 0.0


def test_record_quality_appends_score():
    ps = PipelineState("p", _make_quality_policy())
    from agentic_capsules.evaluation.base import QualityScore
    ps.record_quality("research", QualityScore(score=0.85, confidence=0.70))
    ps.record_quality("research", QualityScore(score=0.80, confidence=0.70))
    s = ps.snapshot()["research"]
    assert len(s.quality_scores) == 2
    assert abs(s.quality_scores[0] - 0.85) < 1e-6


def test_get_quality_none_when_no_data():
    ps = PipelineState("p", _make_quality_policy())
    assert ps.get_quality("research") is None


def test_get_quality_returns_rolling_mean():
    ps = PipelineState("p", _make_quality_policy())
    from agentic_capsules.evaluation.base import QualityScore
    ps.record_quality("g", QualityScore(score=0.80, confidence=0.70))
    ps.record_quality("g", QualityScore(score=0.90, confidence=0.70))
    q = ps.get_quality("g")
    assert q is not None
    assert abs(q - 0.85) < 1e-6


def test_quality_gate_triggers_decompose_when_below_floor():
    """
    Quality gate: get_recommendation() returns DECOMPOSE when rolling mean
    quality < quality_floor, even when mode is FINE and composition confidence is low.
    """
    ps = PipelineState("p", _make_quality_policy(quality_floor=0.75))
    from agentic_capsules.evaluation.base import QualityScore
    # Record 2 low-quality scores
    ps.record_quality("g", QualityScore(score=0.50, confidence=0.70))
    ps.record_quality("g", QualityScore(score=0.45, confidence=0.70))
    assert ps.get_recommendation("g") == "DECOMPOSE"


def test_quality_gate_no_override_above_floor():
    """Quality above floor does not change MAINTAIN recommendation."""
    ps = PipelineState("p", _make_quality_policy(quality_floor=0.75))
    from agentic_capsules.evaluation.base import QualityScore
    ps.record_quality("g", QualityScore(score=0.88, confidence=0.70))
    ps.record_quality("g", QualityScore(score=0.90, confidence=0.70))
    # Only 2 observations total — below min_observations=3 for composition gate
    assert ps.get_recommendation("g") == "MAINTAIN"


def test_quality_gate_requires_two_scores():
    """Quality gate requires >= 2 quality scores before firing."""
    ps = PipelineState("p", _make_quality_policy(quality_floor=0.75))
    from agentic_capsules.evaluation.base import QualityScore
    ps.record_quality("g", QualityScore(score=0.10, confidence=0.70))  # only 1
    # Should NOT fire with only 1 quality score
    assert ps.get_recommendation("g") != "DECOMPOSE"


def test_quality_gate_disabled_when_floor_none():
    """When quality_floor=None, quality scores never trigger DECOMPOSE."""
    ps = PipelineState("p", _make_quality_policy(quality_floor=None))
    from agentic_capsules.evaluation.base import QualityScore
    ps.record_quality("g", QualityScore(score=0.00, confidence=0.70))
    ps.record_quality("g", QualityScore(score=0.00, confidence=0.70))
    assert ps.get_recommendation("g") != "DECOMPOSE"


def test_quality_scores_json_round_trip():
    s = GroupControllerState(
        name="g",
        quality_scores=[0.80, 0.85, 0.78],
    )
    restored = GroupControllerState.from_json(s.to_json())
    assert restored.quality_scores == [0.80, 0.85, 0.78]


def test_quality_scores_json_backward_compatible():
    """Old JSON without quality_scores field deserialises with empty list."""
    old_json = json.dumps({
        "name": "g", "observations": [], "current_mode": "fine",
        "confidence": 0.0, "last_score": 0.0,
        "latency_fine_ms": [], "latency_compound_ms": [],
        "tokens_fine": [], "tokens_compound": [],
        # no "quality_scores" key
    })
    s = GroupControllerState.from_json(old_json)
    assert s.quality_scores == []


# ---------------------------------------------------------------------------
# P12-5 — Pipeline evaluator integration (using ScriptedAdapter + mock evaluator)
# ---------------------------------------------------------------------------

def test_pipeline_run_with_evaluator_populates_quality():
    """Pipeline.run(evaluator=...) populates PipelineResult.quality."""
    from evals.shared.pipeline import ScriptedAdapter, build_pipeline
    from agentic_capsules.evaluation.schema_compliance import SchemaComplianceEvaluator

    pipeline  = build_pipeline(sensitivity="balanced")
    adapter   = ScriptedAdapter()
    evaluator = SchemaComplianceEvaluator()

    # Run enough times to switch to COMPOUND (min_observations=3)
    for _ in range(3):
        result = pipeline.run("Analyse Acme Corp", adapter=adapter, evaluator=evaluator)

    # Quality dict may or may not be populated depending on whether switch occurred,
    # but it must always be a dict (never None)
    assert isinstance(result.quality, dict)
    assert isinstance(result.quality_details, dict)


def test_pipeline_run_without_evaluator_empty_quality():
    """Pipeline.run() without evaluator leaves quality empty."""
    from evals.shared.pipeline import ScriptedAdapter, build_pipeline

    pipeline = build_pipeline(sensitivity="balanced")
    adapter  = ScriptedAdapter()
    result   = pipeline.run("Analyse Acme Corp", adapter=adapter)
    assert result.quality == {}


# ---------------------------------------------------------------------------
# P12-6 — PipelineResult quality fields
# ---------------------------------------------------------------------------

def test_pipeline_result_repr_includes_quality_when_set():
    from agentic_capsules.api.result import PipelineResult
    r = PipelineResult(
        output="test",
        quality={"research": 0.88},
    )
    assert "quality" in repr(r)


def test_pipeline_result_repr_omits_quality_when_empty():
    from agentic_capsules.api.result import PipelineResult
    r = PipelineResult(output="test")
    assert "quality" not in repr(r)


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------

def test_calibration_report_quality_by_group():
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=0.75,
        _groups={
            "research": _GroupCalibration(group="research", quality_scores=[0.80, 0.85, 0.82]),
            "analysis": _GroupCalibration(group="analysis", quality_scores=[0.70, 0.72]),
        },
    )
    q = report.quality_by_group()
    assert abs(q["research"] - 0.82333) < 0.001
    assert abs(q["analysis"] - 0.71) < 0.001


def test_calibration_report_passes_quality_floor():
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=0.75,
        _groups={
            "research": _GroupCalibration(group="research", quality_scores=[0.80, 0.85]),
        },
    )
    assert report.passes_quality_floor() is True


def test_calibration_report_fails_quality_floor():
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=0.75,
        _groups={
            "research": _GroupCalibration(group="research", quality_scores=[0.60, 0.65]),
        },
    )
    assert report.passes_quality_floor() is False
    assert report.recommend_compose_at() is None


def test_calibration_report_recommend_compose_at_when_passing():
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=0.75,
        _groups={
            "research": _GroupCalibration(group="research", quality_scores=[0.85, 0.88]),
            "analysis": _GroupCalibration(group="analysis", quality_scores=[0.80, 0.82]),
        },
    )
    assert report.recommend_compose_at() == 0.36


def test_calibration_report_no_floor_recommend_none():
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=None,
        _groups={},
    )
    assert report.recommend_compose_at() is None


def test_calibration_report_save(tmp_path):
    report = CalibrationReport(
        pipeline_name="test",
        quality_floor=0.75,
        _groups={
            "research": _GroupCalibration(
                group="research",
                quality_scores=[0.85, 0.88],
                latency_fine_ms=[4000.0, 4200.0],
                latency_compound_ms=[2800.0, 3000.0],
                tokens_fine=[200, 210],
                tokens_compound=[80, 85],
            ),
        },
    )
    out = str(tmp_path / "report.md")
    report.save(out)
    content = open(out).read()
    assert "Calibration Report" in content
    assert "research" in content
    assert "0.865" in content  # mean quality ≈ 0.865


# ---------------------------------------------------------------------------
# Pipeline.calibrate() integration (offline, ScriptedAdapter)
# ---------------------------------------------------------------------------

def test_pipeline_calibrate_returns_report():
    from evals.shared.pipeline import ScriptedAdapter, build_pipeline
    from agentic_capsules.evaluation.schema_compliance import SchemaComplianceEvaluator
    from agentic_capsules.evaluation.calibration import CalibrationReport

    pipeline  = build_pipeline(sensitivity="balanced")
    adapter   = ScriptedAdapter()
    evaluator = SchemaComplianceEvaluator()

    report = pipeline.calibrate(
        sample_tasks=["Conduct due diligence on Acme Corp"],
        adapter=adapter,
        evaluator=evaluator,
        n_paired_runs=1,
    )

    assert isinstance(report, CalibrationReport)
    quality = report.quality_by_group()
    assert len(quality) == 3  # research, analysis, synthesis
    for name, q in quality.items():
        assert q is not None
        assert 0.0 <= q <= 1.0


# ---------------------------------------------------------------------------
# CalibrationReport.recommended_policy()
# ---------------------------------------------------------------------------

def _report_with(quality_scores=None, comp_scores=None, avg_out=None, floor=0.75):
    """Build a CalibrationReport with synthetic per-group observations.

    Three groups ("g1"/"g2"/"g3") are created with the per-group lists given.
    Missing lists default to empty so tests can exercise 'insufficient data'
    paths selectively.
    """
    quality_scores = quality_scores or {}
    comp_scores = comp_scores or {}
    avg_out = avg_out or {}
    names = set(quality_scores) | set(comp_scores) | set(avg_out) or {"g1", "g2", "g3"}
    return CalibrationReport(
        pipeline_name="t",
        quality_floor=floor,
        _groups={
            name: _GroupCalibration(
                group=name,
                quality_scores=list(quality_scores.get(name, [])),
                composition_scores=list(comp_scores.get(name, [])),
                avg_output_tokens_fine=list(avg_out.get(name, [])),
            )
            for name in names
        },
    )


def test_recommended_policy_returns_controller_policy():
    rec = _report_with().recommended_policy()
    assert isinstance(rec, ControllerPolicy)


def test_recommend_compose_at_falls_back_when_no_observations():
    """Without composition scores we can't recommend — return base policy value."""
    base = ControllerPolicy(compose_at=0.23)
    rec = _report_with().recommended_policy(base)
    assert rec.compose_at == 0.23


def test_recommend_compose_at_uses_median_of_observed_scores():
    """With enough observations, recommend the median (clamped)."""
    report = _report_with(comp_scores={
        "g1": [0.20, 0.22, 0.24],
        "g2": [0.25, 0.26, 0.28],
        "g3": [0.30, 0.32, 0.33],
    })
    rec = report.recommended_policy(ControllerPolicy(compose_at=0.23))
    # Median of 9 scores = 0.26 (fifth sorted element)
    assert abs(rec.compose_at - 0.26) < 0.01


def test_recommend_compose_at_clamps_to_bounds():
    """Even when observed scores are very high, don't recommend absurd values."""
    report = _report_with(comp_scores={
        "g1": [0.80, 0.82, 0.84],  # would give median 0.82
    })
    rec = report.recommended_policy(ControllerPolicy(compose_at=0.23))
    assert rec.compose_at <= 0.45  # upper bound


def test_recommend_quality_floor_rounds_down_to_step():
    report = _report_with(quality_scores={
        "g1": [0.88, 0.90, 0.87],
        "g2": [0.82, 0.84, 0.80],
        "g3": [0.78, 0.76, 0.81],
    })
    rec = report.recommended_policy(ControllerPolicy(quality_floor=0.75))
    # 5th-percentile-ish of the 9 values = around 0.76; rounded DOWN to 0.05
    # step gives 0.75.
    assert rec.quality_floor is not None
    assert 0.50 <= rec.quality_floor <= 0.85
    # Quality floor must be a multiple of 0.05
    assert abs(rec.quality_floor / 0.05 - round(rec.quality_floor / 0.05)) < 1e-9


def test_recommend_quality_floor_falls_back_with_few_samples():
    """<5 quality observations → use base policy's floor."""
    base = ControllerPolicy(quality_floor=0.75)
    rec = _report_with(quality_scores={"g1": [0.88, 0.85]}).recommended_policy(base)
    assert rec.quality_floor == 0.75


def test_recommend_verbosity_threshold_keeps_default_when_observations_straddle():
    """When observed groups sit on both sides of the current threshold, keep it."""
    report = _report_with(avg_out={
        "g1": [300],           # below 1500
        "g2": [3000, 3100],    # above 1500
    })
    rec = report.recommended_policy(ControllerPolicy(verbosity_guidance_threshold=1500))
    assert rec.verbosity_guidance_threshold == 1500


def test_recommend_verbosity_threshold_moves_when_all_below():
    """All groups below the default — move threshold into the observed cluster."""
    report = _report_with(avg_out={
        "g1": [400], "g2": [600], "g3": [800],
    })
    rec = report.recommended_policy(ControllerPolicy(verbosity_guidance_threshold=1500))
    # Midpoint of min 400 and max 800 = 600
    assert rec.verbosity_guidance_threshold == 600


def test_recommend_verbosity_threshold_moves_when_all_above():
    """All groups above the default — move threshold up to midpoint."""
    report = _report_with(avg_out={
        "g1": [3000], "g2": [5000], "g3": [7000],
    })
    rec = report.recommended_policy(ControllerPolicy(verbosity_guidance_threshold=1500))
    # Midpoint of min 3000 and max 7000 = 5000
    assert rec.verbosity_guidance_threshold == 5000


def test_recommend_verbosity_threshold_falls_back_with_single_group():
    """Need at least 2 groups to reason about where the threshold should split."""
    base = ControllerPolicy(verbosity_guidance_threshold=1500)
    rec = _report_with(avg_out={"g1": [4000]}).recommended_policy(base)
    assert rec.verbosity_guidance_threshold == 1500


def test_recommended_policy_preserves_tier_ordering():
    """Sequential threshold must stay > two-phase threshold (policy validation)."""
    # Recommend small values that would collapse ordering on a naive pass
    report = _report_with(avg_out={
        "g1": [200], "g2": [400], "g3": [600],
    })
    base = ControllerPolicy(
        verbosity_two_phase_threshold=1500,
        verbosity_sequential_threshold=3500,
    )
    rec = report.recommended_policy(base)
    # If both would have collapsed to the same observed midpoint, recommender
    # should fall back to base values to preserve the ordering invariant.
    assert rec.verbosity_sequential_threshold > rec.verbosity_two_phase_threshold


def test_recommended_policy_is_valid_policy():
    """The recommendation must itself pass ControllerPolicy validation."""
    report = _report_with(
        quality_scores={"g1": [0.9, 0.88, 0.85, 0.82, 0.80]},
        comp_scores={"g1": [0.22, 0.24, 0.26]},
        avg_out={"g1": [400], "g2": [5000]},
    )
    rec = report.recommended_policy()
    # Validates by virtue of __post_init__ — constructing replace() runs it.
    assert isinstance(rec, ControllerPolicy)


# ---------------------------------------------------------------------------
# E-1 — Quality-Driven Escalation Ladder
# ---------------------------------------------------------------------------

def _make_escalation_policy(escalation_min_failures: int = 2, **kw):
    return ControllerPolicy(
        compose_at=0.40,
        decompose_at=0.15,
        confidence=0.80,
        min_observations=3,
        window_size=5,
        quality_floor=0.75,
        escalation_enabled=True,
        escalation_min_failures=escalation_min_failures,
        **kw,
    )


def test_escalation_policy_validation_min_failures_zero():
    with pytest.raises(ValueError, match="escalation_min_failures"):
        ControllerPolicy(
            compose_at=0.40, decompose_at=0.15,
            escalation_min_failures=0,
        )


def test_escalation_policy_defaults_on():
    """escalation_enabled=True by default (E-1 win, 2026-04-06)."""
    p = ControllerPolicy(compose_at=0.40, decompose_at=0.15)
    assert p.escalation_enabled is True
    assert p.escalation_min_failures == 2


def test_quality_failure_streak_serialised():
    """quality_failure_streak survives to_json/from_json round-trip."""
    gs = GroupControllerState(name="g", quality_failure_streak=3)
    restored = GroupControllerState.from_json(gs.to_json())
    assert restored.quality_failure_streak == 3


def test_quality_failure_streak_backward_compatible():
    """Old JSON without quality_failure_streak field deserialises to 0."""
    old_json = json.dumps({
        "name": "g", "observations": [], "current_mode": "fine",
        "confidence": 0.0, "last_score": 0.0,
        "latency_fine_ms": [], "latency_compound_ms": [],
        "tokens_fine": [], "tokens_compound": [],
        "quality_scores": [], "avg_output_tokens_fine": [],
        # no "quality_failure_streak" key
    })
    gs = GroupControllerState.from_json(old_json)
    assert gs.quality_failure_streak == 0


class _FixedQualityEvaluator:
    """Returns a fixed quality score for all evaluations."""
    def __init__(self, score: float):
        self._score = score

    def evaluate(self, task: str, fine_output: str, compound_output: str) -> QualityScore:
        return QualityScore(score=self._score, confidence=0.90)


def _make_compound_pipeline(execution_model: str = "standard"):
    """Pipeline in COMPOUND mode with escalation_enabled policy."""
    from agentic_capsules import Pipeline
    policy = _make_escalation_policy(compound_execution_model=execution_model)
    p = (
        Pipeline("test_e1", policy=policy)
        .group("g").agent("a", "do the work")
    )
    # Manually set group to COMPOUND mode and seed a fine baseline output
    gs = p._pipeline_state._load("g")
    gs.current_mode = "compound"
    gs.last_fine_output = "fine output baseline"
    p._pipeline_state._save("g", gs)
    return p


class _MinimalAdapter:
    context_window = 200_000

    def complete(self, messages, tools=None):
        return "## OUTPUT\nCompound result."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_e1_streak_increments_on_quality_failure():
    """H3: quality_failure_streak increments each time quality < floor (before threshold)."""
    # Use min_failures=3 so we can observe streak=1,2 without escalation firing
    from agentic_capsules import Pipeline
    policy = _make_escalation_policy(escalation_min_failures=3)
    p = Pipeline("streak_test", policy=policy).group("g").agent("a", "work")
    gs = p._pipeline_state._load("g")
    gs.current_mode = "compound"
    gs.last_fine_output = "fine output baseline"
    p._pipeline_state._save("g", gs)

    evaluator = _FixedQualityEvaluator(score=0.50)  # below floor=0.75
    adapter = _MinimalAdapter()

    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    gs = p._pipeline_state._load("g")
    assert gs.quality_failure_streak == 1

    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    gs = p._pipeline_state._load("g")
    assert gs.quality_failure_streak == 2
    assert p._pipeline_state.get_execution_model_override("g") is None  # not yet


def test_e1_streak_resets_on_quality_pass():
    """H3: quality_failure_streak resets to 0 when quality >= floor."""
    p = _make_compound_pipeline()
    low_eval  = _FixedQualityEvaluator(score=0.50)
    high_eval = _FixedQualityEvaluator(score=0.90)
    adapter = _MinimalAdapter()

    p.run("task", adapter=adapter, mode="auto", evaluator=low_eval)
    gs = p._pipeline_state._load("g")
    assert gs.quality_failure_streak == 1

    p.run("task", adapter=adapter, mode="auto", evaluator=high_eval)
    gs = p._pipeline_state._load("g")
    assert gs.quality_failure_streak == 0


def test_e1_escalates_standard_to_two_phase_after_min_failures():
    """After escalation_min_failures=2 consecutive failures, override set to two_phase."""
    p = _make_compound_pipeline(execution_model="standard")
    evaluator = _FixedQualityEvaluator(score=0.50)
    adapter = _MinimalAdapter()

    # First failure: streak=1, no escalation yet
    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    assert p._pipeline_state.get_execution_model_override("g") is None

    # Second failure: streak=2 >= min_failures=2 → escalate
    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    assert p._pipeline_state.get_execution_model_override("g") == "two_phase"

    # Streak resets after escalation
    gs = p._pipeline_state._load("g")
    assert gs.quality_failure_streak == 0


def test_e1_escalates_two_phase_to_sequential():
    """Escalation chain: two_phase → sequential on continued quality failures."""
    p = _make_compound_pipeline(execution_model="two_phase")
    # Seed override as two_phase (simulate previous escalation from standard)
    p._pipeline_state.set_execution_model_override("g", "two_phase")
    evaluator = _FixedQualityEvaluator(score=0.50)
    adapter = _MinimalAdapter()

    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    assert p._pipeline_state.get_execution_model_override("g") == "sequential"


def test_e1_no_escalation_at_top_of_ladder():
    """At sequential (top of ladder): no override change; quality gate fires DECOMPOSE."""
    p = _make_compound_pipeline(execution_model="sequential")
    p._pipeline_state.set_execution_model_override("g", "sequential")
    evaluator = _FixedQualityEvaluator(score=0.50)
    adapter = _MinimalAdapter()

    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)
    p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)

    # Override stays at sequential — no further escalation possible
    assert p._pipeline_state.get_execution_model_override("g") == "sequential"
    # Quality gate fires DECOMPOSE via get_recommendation
    assert p._pipeline_state.get_recommendation("g") == "DECOMPOSE"


def test_e1_disabled_explicitly_no_escalation():
    """When escalation_enabled=False (opt-out), quality failures never touch override."""
    from agentic_capsules import Pipeline
    policy = ControllerPolicy(
        compose_at=0.40, decompose_at=0.15,
        quality_floor=0.75,
        compound_execution_model="standard",
        escalation_enabled=False,  # explicit opt-out (default is now True)
    )
    p = Pipeline("no_e1", policy=policy).group("g").agent("a", "work")
    gs = p._pipeline_state._load("g")
    gs.current_mode = "compound"
    gs.last_fine_output = "baseline"
    p._pipeline_state._save("g", gs)

    evaluator = _FixedQualityEvaluator(score=0.50)
    adapter = _MinimalAdapter()

    for _ in range(3):
        p.run("task", adapter=adapter, mode="auto", evaluator=evaluator)

    assert p._pipeline_state.get_execution_model_override("g") is None
    gs2 = p._pipeline_state._load("g")
    assert gs2.quality_failure_streak == 0
