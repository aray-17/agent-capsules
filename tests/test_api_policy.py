"""Tests for controller/policy.py."""
import pytest
from agentic_capsules.controller.policy import (
    ControllerPolicy,
    SENSITIVITY_PRESETS,
    policy_for,
)


def test_balanced_preset_values():
    p = policy_for("balanced")
    assert p.compose_at == 0.23    # T-021/T-032 (2026-03-29): lowered from 0.36 after removing w3 bias
    assert p.decompose_at == 0.10  # T-021/T-032 (2026-03-29): lowered from 0.12
    assert p.confidence == 0.80
    assert p.min_observations == 3


def test_conservative_higher_thresholds():
    c = policy_for("conservative")
    b = policy_for("balanced")
    assert c.compose_at > b.compose_at
    assert c.confidence > b.confidence
    assert c.min_observations > b.min_observations


def test_aggressive_lower_thresholds():
    a = policy_for("aggressive")
    b = policy_for("balanced")
    assert a.compose_at < b.compose_at
    assert a.confidence < b.confidence
    assert a.min_observations < b.min_observations


def test_policy_for_invalid_raises():
    with pytest.raises(ValueError, match="sensitivity must be one of"):
        policy_for("unknown")


def test_decompose_always_below_compose():
    for name, p in SENSITIVITY_PRESETS.items():
        assert p.decompose_at < p.compose_at, f"{name}: decompose_at must be < compose_at"


def test_all_presets_present():
    assert set(SENSITIVITY_PRESETS) == {"conservative", "balanced", "aggressive"}


def test_controller_policy_validation():
    with pytest.raises(ValueError):
        ControllerPolicy(compose_at=0.10, decompose_at=0.20)   # decompose > compose

    with pytest.raises(ValueError):
        ControllerPolicy(compose_at=1.5)                        # out of (0,1]

    with pytest.raises(ValueError):
        ControllerPolicy(min_observations=0)                    # must be >= 1


def test_custom_policy_overrides():
    p = ControllerPolicy(compose_at=0.35, confidence=0.75, min_observations=5)
    assert p.compose_at == 0.35
    assert p.confidence == 0.75
    assert p.min_observations == 5


# ---------------------------------------------------------------------------
# T-017 — error_rate_threshold and context_util_threshold defaults and validation
# ---------------------------------------------------------------------------

def test_default_override_thresholds_match_v1():
    """Defaults must match v1 ControllerThresholds (§3.2.2)."""
    p = ControllerPolicy()
    assert p.error_rate_threshold == 0.15
    assert p.context_util_threshold == 0.85


def test_all_presets_have_override_thresholds():
    for name, p in SENSITIVITY_PRESETS.items():
        assert hasattr(p, "error_rate_threshold"), f"{name} missing error_rate_threshold"
        assert hasattr(p, "context_util_threshold"), f"{name} missing context_util_threshold"


def test_error_rate_threshold_validation():
    with pytest.raises(ValueError, match="error_rate_threshold"):
        ControllerPolicy(error_rate_threshold=0.0)   # must be in (0, 1]
    with pytest.raises(ValueError, match="error_rate_threshold"):
        ControllerPolicy(error_rate_threshold=1.1)


def test_context_util_threshold_validation():
    with pytest.raises(ValueError, match="context_util_threshold"):
        ControllerPolicy(context_util_threshold=0.0)  # must be in (0, 1]
    with pytest.raises(ValueError, match="context_util_threshold"):
        ControllerPolicy(context_util_threshold=1.2)


# ---------------------------------------------------------------------------
# T-038: compound_execution_model, compound_tool_budget, compound_min_output_words
# ---------------------------------------------------------------------------

def test_compound_execution_model_default_standard():
    p = ControllerPolicy()
    assert p.compound_execution_model == "standard"


def test_compound_tool_budget_default_zero():
    p = ControllerPolicy()
    assert p.compound_tool_budget == 0


def test_compound_min_output_words_default_none():
    p = ControllerPolicy()
    assert p.compound_min_output_words is None


def test_compound_execution_model_two_phase_valid():
    p = ControllerPolicy(compound_execution_model="two_phase")
    assert p.compound_execution_model == "two_phase"


def test_compound_execution_model_invalid_raises():
    with pytest.raises(ValueError, match="compound_execution_model"):
        ControllerPolicy(compound_execution_model="batch")


def test_compound_tool_budget_unlimited_valid():
    p = ControllerPolicy(compound_tool_budget=-1)
    assert p.compound_tool_budget == -1


def test_compound_tool_budget_positive_valid():
    p = ControllerPolicy(compound_tool_budget=4)
    assert p.compound_tool_budget == 4


def test_compound_tool_budget_below_minus_one_raises():
    with pytest.raises(ValueError, match="compound_tool_budget"):
        ControllerPolicy(compound_tool_budget=-2)


def test_compound_min_output_words_positive_valid():
    p = ControllerPolicy(compound_min_output_words=200)
    assert p.compound_min_output_words == 200


def test_compound_min_output_words_zero_raises():
    with pytest.raises(ValueError, match="compound_min_output_words"):
        ControllerPolicy(compound_min_output_words=0)


def test_compound_min_output_words_negative_raises():
    with pytest.raises(ValueError, match="compound_min_output_words"):
        ControllerPolicy(compound_min_output_words=-10)


# ---------------------------------------------------------------------------
# T-058 — output_guidance="auto" + verbosity_guidance_threshold
# ---------------------------------------------------------------------------

def test_output_guidance_default_is_auto():
    """T-058: default flipped from 'concise' to 'auto' so the gate runs
    out of the box on every deployment."""
    assert ControllerPolicy().output_guidance == "auto"


def test_output_guidance_rejects_removed_adaptive():
    """T-058 deleted the 'adaptive' variant (tested, −0.079 Haiku; continuous
    budget reinforced verbosity). Validator must reject it explicitly so old
    configs surface as errors rather than silent fallbacks."""
    with pytest.raises(ValueError, match="output_guidance"):
        ControllerPolicy(output_guidance="adaptive")


def test_output_guidance_accepts_auto_concise_none_moderate_brief():
    for variant in ("auto", "concise", "none", "moderate", "brief"):
        p = ControllerPolicy(output_guidance=variant)
        assert p.output_guidance == variant


def test_verbosity_guidance_threshold_default():
    """T-058: 1,500 tok/agent calibrated against §15.2 cluster boundaries
    (Gemini-flash 960 below, Haiku 3,616 above)."""
    assert ControllerPolicy().verbosity_guidance_threshold == 1_500


def test_verbosity_guidance_threshold_custom_valid():
    p = ControllerPolicy(verbosity_guidance_threshold=6_000)
    assert p.verbosity_guidance_threshold == 6_000


def test_verbosity_guidance_threshold_zero_raises():
    with pytest.raises(ValueError, match="verbosity_guidance_threshold"):
        ControllerPolicy(verbosity_guidance_threshold=0)


def test_verbosity_guidance_threshold_negative_raises():
    with pytest.raises(ValueError, match="verbosity_guidance_threshold"):
        ControllerPolicy(verbosity_guidance_threshold=-100)
