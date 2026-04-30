"""Tests for api/state.py — confidence math and switching logic."""
import pytest
from agentic_capsules.api.state import (
    PipelineState, GroupControllerState, CompositionSignal, compute_composition_score,
)
from agentic_capsules.controller.policy import ControllerPolicy
from agentic_capsules.core.types import CompositionLevel


def _state(compose_at=0.40, decompose_at=0.15, confidence=0.80,
           min_observations=3, window_size=10):
    policy = ControllerPolicy(
        compose_at=compose_at,
        decompose_at=decompose_at,
        confidence=confidence,
        min_observations=min_observations,
        window_size=window_size,
    )
    return PipelineState("test_pipeline", policy)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_new_group_starts_fine():
    ps = _state()
    assert ps.get_mode("research") == CompositionLevel.FINE


def test_new_group_confidence_zero():
    ps = _state()
    assert ps.get_confidence("research") == 0.0


# ---------------------------------------------------------------------------
# No switch before min_observations
# ---------------------------------------------------------------------------

def test_no_switch_before_min_observations():
    ps = _state(min_observations=3)
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    # Only 2 observations — not enough
    assert ps.get_mode("g") == CompositionLevel.FINE


# ---------------------------------------------------------------------------
# Switch at confidence threshold
# ---------------------------------------------------------------------------

def test_switch_when_confidence_met():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=5)
    # 4 out of 5 above threshold → confidence = 0.80
    for oh in [0.55, 0.50, 0.45, 0.55, 0.18]:
        ps.record_and_maybe_switch("g", oh)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_no_switch_below_confidence_threshold():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=3)
    # 2 out of 3 above threshold → confidence = 0.67 < 0.80
    for oh in [0.55, 0.50, 0.18]:
        ps.record_and_maybe_switch("g", oh)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_exactly_at_confidence_threshold_switches():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=5)
    # Exactly 4/5 = 0.80 → meets threshold
    for oh in [0.55, 0.50, 0.55, 0.55, 0.18]:
        ps.record_and_maybe_switch("g", oh)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


# ---------------------------------------------------------------------------
# Switch back to fine from compound
# ---------------------------------------------------------------------------

def test_switch_back_to_fine():
    ps = _state(compose_at=0.40, decompose_at=0.15, confidence=0.80,
                min_observations=3, window_size=3)
    # Switch to compound first
    for oh in [0.55, 0.60, 0.65]:
        ps.record_and_maybe_switch("g", oh)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Now observations drop — switch back to fine
    for oh in [0.10, 0.08, 0.12]:
        ps.record_and_maybe_switch("g", oh)
    assert ps.get_mode("g") == CompositionLevel.FINE


# ---------------------------------------------------------------------------
# Observe mode — never switches
# ---------------------------------------------------------------------------

def test_observe_mode_never_switches():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3)
    for oh in [0.55, 0.60, 0.65, 0.70, 0.55]:
        ps.record_and_maybe_switch("g", oh, apply_switch=False)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_observe_mode_still_records_observations():
    ps = _state(min_observations=3)
    for oh in [0.55, 0.60, 0.65]:
        ps.record_and_maybe_switch("g", oh, apply_switch=False)
    s = ps.snapshot()["g"]
    assert len(s.observations) == 3


def test_observe_then_auto_uses_accumulated_confidence():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=3)
    # Observe 3 high-overhead runs — confidence builds but mode doesn't switch
    for oh in [0.55, 0.60, 0.65]:
        ps.record_and_maybe_switch("g", oh, apply_switch=False)
    assert ps.get_mode("g") == CompositionLevel.FINE

    # Now switch to auto — one more high observation → already at confidence
    ps.record_and_maybe_switch("g", 0.55, apply_switch=True)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


# ---------------------------------------------------------------------------
# Confidence score correctness
# ---------------------------------------------------------------------------

def test_confidence_score_correct():
    ps = _state(compose_at=0.40, window_size=5, min_observations=3)
    for oh in [0.55, 0.50, 0.20, 0.55, 0.45]:
        ps.record_and_maybe_switch("g", oh, apply_switch=False)
    # above threshold: 0.55, 0.50, 0.55, 0.45 = 4/5 = 0.80
    assert abs(ps.get_confidence("g") - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# Window size limits
# ---------------------------------------------------------------------------

def test_window_size_limits_observations_used():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=3)
    # First 7 observations are low
    for _ in range(7):
        ps.record_and_maybe_switch("g", 0.10, apply_switch=False)
    # Last 3 are high — window=3 means only these count
    for oh in [0.55, 0.60, 0.65]:
        ps.record_and_maybe_switch("g", oh, apply_switch=True)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


# ---------------------------------------------------------------------------
# Hysteresis — no oscillation
# ---------------------------------------------------------------------------

def test_no_oscillation_with_noisy_signal():
    ps = _state(compose_at=0.40, decompose_at=0.15, confidence=0.80,
                min_observations=3, window_size=4)
    # Alternating above/below compose_at — never consistent enough
    for oh in [0.45, 0.35, 0.45, 0.35, 0.45, 0.35]:
        ps.record_and_maybe_switch("g", oh)
    # 50% confidence — below 80% threshold — should not switch
    assert ps.get_mode("g") == CompositionLevel.FINE


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_recommendation_maintain_before_threshold():
    ps = _state(min_observations=5)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_recommendation("g") == "MAINTAIN"


def test_recommendation_compose_when_confident():
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=3)
    for oh in [0.55, 0.60, 0.65]:
        ps.record_and_maybe_switch("g", oh, apply_switch=False)
    assert ps.get_recommendation("g") == "COMPOSE"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_group_controller_state_json_round_trip():
    s = GroupControllerState(
        name="research",
        observations=[0.52, 0.48, 0.55],
        current_mode="compound",
        confidence=0.80,
    )
    restored = GroupControllerState.from_json(s.to_json())
    assert restored.name == s.name
    assert restored.observations == s.observations
    assert restored.current_mode == s.current_mode
    assert abs(restored.confidence - s.confidence) < 1e-9


# ---------------------------------------------------------------------------
# In-memory persistence between calls
# ---------------------------------------------------------------------------

def test_in_memory_state_persists_between_calls():
    ps = _state(min_observations=3)
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.50)
    s = ps.snapshot()
    assert len(s["g"].observations) == 2


# ---------------------------------------------------------------------------
# Phase 11 — CompositionSignal and compute_composition_score
# ---------------------------------------------------------------------------

_W = (0.35, 0.25, 0.25, 0.10, 0.05)  # default weights


def test_score_single_agent_minimal_output():
    """Single agent, short output, no tools → should score low (below balanced threshold)."""
    sig = CompositionSignal(
        overhead_ratio=0.08,
        agent_count=1,
        avg_output_tokens=200.0,
        tool_calls_per_agent=0.0,
        dependency_depth=0,
    )
    score = compute_composition_score(sig, _W)
    # 0.35*0.08 + 0.25*0.25 + 0.25*0.67 + 0 - 0 ≈ 0.257 (below balanced 0.40)
    assert score < 0.40


def test_score_two_agent_tool_heavy():
    """2-agent tool-heavy group with real LLM output → fires at balanced threshold."""
    sig = CompositionSignal(
        overhead_ratio=0.10,
        agent_count=2,
        avg_output_tokens=400.0,
        tool_calls_per_agent=2.0,
        dependency_depth=1,
    )
    score = compute_composition_score(sig, _W)
    # 0.35*0.10 + 0.25*0.50 + 0.25*1.0 + 0.10*0.67 - 0.05*1.0 ≈ 0.427 (above 0.40)
    assert score >= 0.40


def test_score_five_agent_group():
    """5-agent pipeline group → scores above balanced threshold even with low overhead."""
    sig = CompositionSignal(
        overhead_ratio=0.08,
        agent_count=5,
        avg_output_tokens=350.0,
        tool_calls_per_agent=1.0,
        dependency_depth=4,
    )
    score = compute_composition_score(sig, _W)
    # 0.35*0.08 + 0.25*1.0 + 0.25*1.0 + 0.10*0.33 - 0.05*1.0 ≈ 0.511 (above 0.40)
    assert score >= 0.40


def test_score_ordering_more_agents_scores_higher():
    """More agents → higher score (all else equal)."""
    base = dict(overhead_ratio=0.10, avg_output_tokens=300.0,
                tool_calls_per_agent=1.0, dependency_depth=0)
    small = compute_composition_score(CompositionSignal(agent_count=1, **base), _W)
    large = compute_composition_score(CompositionSignal(agent_count=4, **base), _W)
    assert large > small


def test_score_clamped_to_unit_interval():
    """Score must always be in [0, 1]."""
    # Extreme high
    sig_hi = CompositionSignal(
        overhead_ratio=1.0, agent_count=10, avg_output_tokens=10000.0,
        tool_calls_per_agent=100.0, dependency_depth=0,
    )
    assert compute_composition_score(sig_hi, _W) <= 1.0

    # Extreme low (deep sequential chain penalises hard)
    sig_lo = CompositionSignal(
        overhead_ratio=0.0, agent_count=1, avg_output_tokens=0.0,
        tool_calls_per_agent=0.0, dependency_depth=0,
    )
    assert compute_composition_score(sig_lo, _W) >= 0.0


def test_deeper_chain_penalises_score():
    """Deeper sequential dependency_depth → lower score (depth penalty w5)."""
    base = dict(overhead_ratio=0.10, agent_count=4,
                avg_output_tokens=300.0, tool_calls_per_agent=1.0)
    shallow = compute_composition_score(
        CompositionSignal(dependency_depth=0, **base), _W
    )
    deep = compute_composition_score(
        CompositionSignal(dependency_depth=3, **base), _W
    )
    assert deep < shallow


# ---------------------------------------------------------------------------
# Phase 11 — multi-signal path in PipelineState.record_and_maybe_switch
# ---------------------------------------------------------------------------

def _state_with_weights(compose_at=0.40, decompose_at=0.15, confidence=0.80,
                        min_observations=3, window_size=5):
    policy = ControllerPolicy(
        compose_at=compose_at,
        decompose_at=decompose_at,
        confidence=confidence,
        min_observations=min_observations,
        window_size=window_size,
        score_weights=_W,
    )
    return PipelineState("test_pipeline", policy)


def test_multisignal_switches_fine_to_compound():
    """Multi-signal score for a 2-agent tool-heavy group fires at balanced threshold."""
    ps = _state_with_weights(compose_at=0.40, confidence=0.80, min_observations=3)
    sig = CompositionSignal(
        overhead_ratio=0.10,
        agent_count=2,
        avg_output_tokens=400.0,
        tool_calls_per_agent=2.0,
        dependency_depth=1,
    )
    for _ in range(5):
        ps.record_and_maybe_switch("g", overhead=0.10, signal=sig)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_multisignal_low_score_does_not_switch():
    """Single-agent minimal group stays FINE under balanced threshold."""
    ps = _state_with_weights(compose_at=0.40, confidence=0.80, min_observations=3)
    sig = CompositionSignal(
        overhead_ratio=0.08,
        agent_count=1,
        avg_output_tokens=150.0,
        tool_calls_per_agent=0.0,
        dependency_depth=0,
    )
    for _ in range(5):
        ps.record_and_maybe_switch("g", overhead=0.08, signal=sig)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_multisignal_last_score_stored():
    """last_score is updated on each call when signal and weights are present."""
    ps = _state_with_weights()
    sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=400.0,
        tool_calls_per_agent=2.0, dependency_depth=1,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=sig)
    s = ps.snapshot()["g"]
    assert s.last_score > 0.0


def test_no_weights_falls_back_to_raw_overhead():
    """When score_weights is None, signal is ignored and raw overhead is used."""
    policy = ControllerPolicy(
        compose_at=0.40, confidence=0.80, min_observations=3, score_weights=None
    )
    ps = PipelineState("p", policy)
    sig = CompositionSignal(
        overhead_ratio=0.01, agent_count=10, avg_output_tokens=999.0,
        tool_calls_per_agent=10.0, dependency_depth=0,
    )
    # Pass a signal but no weights — should use raw overhead=0.50 instead
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.50, signal=sig)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_json_round_trip_includes_last_score():
    """to_json / from_json preserves last_score field."""
    s = GroupControllerState(
        name="research", observations=[0.42, 0.45], current_mode="fine",
        confidence=0.5, last_score=0.427,
    )
    restored = GroupControllerState.from_json(s.to_json())
    assert abs(restored.last_score - 0.427) < 1e-9


# ---------------------------------------------------------------------------
# T-017 — error_rate and context_utilization hard-override paths
# ---------------------------------------------------------------------------

def test_high_error_rate_clamps_score_to_one():
    """error_rate >= threshold → obs_value clamped to 1.0 regardless of weighted score."""
    ps = _state_with_weights(compose_at=0.40, confidence=0.80, min_observations=3)
    # Signal that would score LOW on its own (1 agent, tiny output)
    sig = CompositionSignal(
        overhead_ratio=0.05, agent_count=1, avg_output_tokens=50.0,
        tool_calls_per_agent=0.0, dependency_depth=0,
        error_rate=0.20,          # above default threshold of 0.15
        context_utilization=0.0,
    )
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.05, signal=sig)
    # All obs clamped to 1.0 → well above compose_at → should switch
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_high_context_util_clamps_score_to_zero():
    """context_utilization >= threshold → obs_value clamped to 0.0."""
    # Start compound, then context pressure should drive DECOMPOSE
    policy = ControllerPolicy(
        compose_at=0.40, decompose_at=0.15, confidence=0.80,
        min_observations=3, window_size=5, score_weights=_W,
    )
    ps = PipelineState("p", policy)
    # Force into compound first via high overhead
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.60)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Now send signals with high context_utilization — score clamped to 0.0 → DECOMPOSE
    sig = CompositionSignal(
        overhead_ratio=0.50, agent_count=4, avg_output_tokens=500.0,
        tool_calls_per_agent=2.0, dependency_depth=3,
        error_rate=0.0,
        context_utilization=0.90,   # above default threshold of 0.85
    )
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.50, signal=sig)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_context_util_override_takes_precedence_over_error_rate():
    """When both error_rate and context_util exceed thresholds, context_util wins (→ DECOMPOSE)."""
    ps = _state_with_weights(
        compose_at=0.40, decompose_at=0.15, confidence=0.80, min_observations=3
    )
    # Force compound first
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.60)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Both overrides active — context_util takes precedence → score = 0.0 → DECOMPOSE
    sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=1, avg_output_tokens=100.0,
        tool_calls_per_agent=0.0, dependency_depth=0,
        error_rate=0.30,          # above error threshold
        context_utilization=0.90, # above context threshold — wins
    )
    for _ in range(4):
        ps.record_and_maybe_switch("g", overhead=0.10, signal=sig)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_error_rate_and_context_util_below_threshold_no_override():
    """Values below thresholds leave the weighted score unchanged."""
    ps = _state_with_weights(compose_at=0.40, confidence=0.80, min_observations=3)
    sig = CompositionSignal(
        overhead_ratio=0.08, agent_count=1, avg_output_tokens=100.0,
        tool_calls_per_agent=0.0, dependency_depth=0,
        error_rate=0.05,           # below 0.15 threshold
        context_utilization=0.50,  # below 0.85 threshold
    )
    for _ in range(5):
        ps.record_and_maybe_switch("g", overhead=0.08, signal=sig)
    # Weighted score for this signal is low — should NOT switch
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_composition_signal_new_fields_default_to_zero():
    """CompositionSignal can be constructed without error_rate / context_utilization."""
    sig = CompositionSignal(overhead_ratio=0.1, agent_count=2, avg_output_tokens=300.0)
    assert sig.error_rate == 0.0
    assert sig.context_utilization == 0.0


# ---------------------------------------------------------------------------
# T-026 — stale confidence reset on mode switch
# ---------------------------------------------------------------------------

def test_recommendation_is_maintain_on_switch_run_fine_to_compound():
    """After FINE→COMPOUND switch, recommendation must be MAINTAIN not DECOMPOSE (T-026)."""
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=10)
    # Three observations above compose_at trigger the switch
    for _ in range(3):
        ps.record_and_maybe_switch("g", overhead=0.50, apply_switch=True)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND
    # On the switch run, recommendation must NOT be DECOMPOSE
    assert ps.get_recommendation("g") == "MAINTAIN"


def test_confidence_reset_to_zero_after_fine_to_compound_switch():
    """Confidence is reset to 0.0 when FINE→COMPOUND switch fires (T-026)."""
    ps = _state(compose_at=0.40, confidence=0.80, min_observations=3, window_size=10)
    for _ in range(3):
        ps.record_and_maybe_switch("g", overhead=0.50, apply_switch=True)
    assert ps.get_confidence("g") == 0.0


def test_recommendation_is_maintain_on_switch_run_compound_to_fine():
    """After COMPOUND→FINE switch, recommendation must be MAINTAIN not COMPOSE (T-026).

    Uses window_size=3 so that 3 below-threshold observations fill the entire window
    and reach 100% confidence (3/3), triggering the switch.
    """
    ps = _state(compose_at=0.40, decompose_at=0.15, confidence=0.80,
                min_observations=3, window_size=3)
    # First get into COMPOUND (3 high observations fill window → 100% confidence)
    for _ in range(3):
        ps.record_and_maybe_switch("g", overhead=0.50, apply_switch=True)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND
    # Now push three below-decompose_at observations; window=[0.05,0.05,0.05] → 100% → FINE
    for _ in range(3):
        ps.record_and_maybe_switch("g", overhead=0.05, apply_switch=True)
    assert ps.get_mode("g") == CompositionLevel.FINE
    assert ps.get_recommendation("g") == "MAINTAIN"


# ---------------------------------------------------------------------------
# Phase 12 — Gate 3: latency gate
# ---------------------------------------------------------------------------

def _state_with_latency_gate(latency_threshold_ms: float):
    policy = ControllerPolicy(
        compose_at=0.40,
        decompose_at=0.15,
        confidence=0.80,
        min_observations=3,
        window_size=5,
        latency_threshold_ms=latency_threshold_ms,
    )
    return PipelineState("test_pipeline", policy)


def _switch_to_compound(ps: PipelineState, group: str = "g") -> None:
    """Push enough above-threshold observations to reach COMPOUND mode."""
    for _ in range(3):
        ps.record_and_maybe_switch(group, overhead=0.50, apply_switch=True)
    assert ps.get_mode(group) == CompositionLevel.COMPOUND


def test_latency_gate_reverts_to_fine_when_exceeded():
    """Gate 3: COMPOUND → FINE when rolling mean compound latency > threshold."""
    ps = _state_with_latency_gate(latency_threshold_ms=1000.0)
    _switch_to_compound(ps)

    # Feed compound-mode signals with latency above threshold
    high_latency_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        latency_ms=2000.0,  # 2 s — above 1000 ms threshold
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=high_latency_sig)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_latency_gate_no_revert_below_threshold():
    """Gate 3 does not fire when rolling mean compound latency is within threshold."""
    ps = _state_with_latency_gate(latency_threshold_ms=5000.0)
    _switch_to_compound(ps)

    fast_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        latency_ms=800.0,   # well below 5000 ms threshold
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=fast_sig)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_latency_gate_disabled_when_none():
    """Gate 3 is disabled when latency_threshold_ms is None (default)."""
    ps = _state()  # default policy — latency_threshold_ms=None
    _switch_to_compound(ps)

    # Even extreme latency should not trigger revert
    extreme_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        latency_ms=999_999.0,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=extreme_sig)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_latency_gate_confidence_reset_on_revert():
    """Gate 3 revert also resets confidence to 0.0."""
    ps = _state_with_latency_gate(latency_threshold_ms=500.0)
    _switch_to_compound(ps)

    slow_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        latency_ms=1500.0,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=slow_sig)
    assert ps.get_mode("g") == CompositionLevel.FINE
    assert ps.get_confidence("g") == 0.0


def test_latency_only_recorded_in_compound_mode():
    """Latency signals recorded in FINE mode go to latency_fine_ms, not latency_compound_ms."""
    ps = _state()
    fine_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        latency_ms=500.0,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=fine_sig)
    s = ps.snapshot()["g"]
    assert len(s.latency_fine_ms) == 1
    assert len(s.latency_compound_ms) == 0
    assert s.latency_fine_ms[0] == 500.0


# ---------------------------------------------------------------------------
# Phase 12 — Gate 4: token-reduction gate
# ---------------------------------------------------------------------------

def test_token_gate_reverts_when_compound_uses_more_tokens():
    """Gate 4: COMPOUND → FINE when rolling mean COMPOUND tokens >= FINE tokens."""
    ps = _state()
    _switch_to_compound(ps)

    # Record enough FINE mode observations first (need >= 2)
    # Trick: temporarily record 2 fine-mode token observations via a raw state edit
    s = ps.snapshot()["g"]
    s.tokens_fine.extend([100, 120])       # FINE: mean 110 tokens
    s.tokens_compound.extend([200, 220])   # COMPOUND: mean 210 tokens — no saving
    ps._memory["g"] = s

    # One more record_and_maybe_switch call triggers gate evaluation
    compound_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        total_tokens=210,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=compound_sig)
    assert ps.get_mode("g") == CompositionLevel.FINE


def test_token_gate_no_revert_when_compound_uses_fewer_tokens():
    """Gate 4 does not fire when COMPOUND uses fewer tokens than FINE."""
    ps = _state()
    _switch_to_compound(ps)

    s = ps.snapshot()["g"]
    s.tokens_fine.extend([200, 220])       # FINE: mean 210 tokens
    s.tokens_compound.extend([80, 90])     # COMPOUND: mean 85 tokens — clear saving
    ps._memory["g"] = s

    compound_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        total_tokens=85,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=compound_sig)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_token_gate_requires_two_observations_each_mode():
    """Gate 4 does not fire if either mode has fewer than 2 token observations."""
    ps = _state()
    _switch_to_compound(ps)

    # Only 1 fine token observation — gate should not fire
    s = ps.snapshot()["g"]
    s.tokens_fine.extend([100])            # only 1 fine observation
    s.tokens_compound.extend([9999])       # compound using many more tokens
    ps._memory["g"] = s

    compound_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        total_tokens=9999,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=compound_sig)
    # Gate 4 should NOT fire — only 1 fine observation
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_token_gate_confidence_reset_on_revert():
    """Gate 4 revert resets confidence to 0.0."""
    ps = _state()
    _switch_to_compound(ps)

    s = ps.snapshot()["g"]
    s.tokens_fine.extend([100, 110])
    s.tokens_compound.extend([300, 320])
    ps._memory["g"] = s

    compound_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        total_tokens=310,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=compound_sig)
    assert ps.get_mode("g") == CompositionLevel.FINE
    assert ps.get_confidence("g") == 0.0


def test_tokens_recorded_by_current_mode():
    """total_tokens in signal are routed to tokens_fine or tokens_compound by current mode."""
    ps = _state()
    # Record in FINE mode
    fine_sig = CompositionSignal(
        overhead_ratio=0.10, agent_count=2, avg_output_tokens=300.0,
        total_tokens=150,
    )
    ps.record_and_maybe_switch("g", overhead=0.10, signal=fine_sig)
    s = ps.snapshot()["g"]
    assert s.tokens_fine == [150]
    assert s.tokens_compound == []


# ---------------------------------------------------------------------------
# Phase 12 — GroupControllerState helper methods
# ---------------------------------------------------------------------------

def test_mean_latency_ms_none_when_no_data():
    s = GroupControllerState(name="g")
    assert s.mean_latency_ms("fine") is None
    assert s.mean_latency_ms("compound") is None


def test_mean_tokens_none_when_no_data():
    s = GroupControllerState(name="g")
    assert s.mean_tokens("fine") is None
    assert s.mean_tokens("compound") is None


def test_token_reduction_pct_positive_when_compound_saves():
    s = GroupControllerState(name="g", tokens_fine=[200, 200], tokens_compound=[100, 100])
    pct = s.token_reduction_pct()
    assert pct is not None
    assert abs(pct - 50.0) < 1e-6  # 50% reduction


def test_token_reduction_pct_negative_when_compound_costs_more():
    s = GroupControllerState(name="g", tokens_fine=[100, 100], tokens_compound=[150, 150])
    pct = s.token_reduction_pct()
    assert pct is not None
    assert pct < 0.0  # negative = COMPOUND uses more tokens


def test_token_reduction_pct_none_when_missing_mode():
    s = GroupControllerState(name="g", tokens_fine=[100, 100])  # no compound data
    assert s.token_reduction_pct() is None


def test_json_round_trip_includes_phase12_fields():
    """to_json / from_json preserves latency and token rolling lists."""
    s = GroupControllerState(
        name="g",
        observations=[0.40],
        current_mode="compound",
        confidence=0.0,
        last_score=0.42,
        latency_fine_ms=[500.0, 480.0],
        latency_compound_ms=[700.0],
        tokens_fine=[100, 110],
        tokens_compound=[90],
    )
    restored = GroupControllerState.from_json(s.to_json())
    assert restored.latency_fine_ms == [500.0, 480.0]
    assert restored.latency_compound_ms == [700.0]
    assert restored.tokens_fine == [100, 110]
    assert restored.tokens_compound == [90]


def test_policy_latency_threshold_ms_validation():
    """latency_threshold_ms must be > 0 when set."""
    import pytest
    with pytest.raises(ValueError, match="latency_threshold_ms"):
        ControllerPolicy(
            compose_at=0.40, decompose_at=0.15,
            latency_threshold_ms=0.0,
        )


# ---------------------------------------------------------------------------
# T-049 — Revert cooldown (oscillation prevention)
# ---------------------------------------------------------------------------

def _aggressive_state(min_obs: int = 2, window: int = 5):
    """Policy that switches quickly — compose_at=0.18, min_obs=2, confidence=0.65."""
    policy = ControllerPolicy(
        compose_at=0.18,
        decompose_at=0.05,
        confidence=0.65,
        min_observations=min_obs,
        window_size=window,
    )
    return PipelineState("osc_test", policy)


def test_no_cooldown_on_first_switch():
    """Normal first switch: no prior revert, so required = min_observations × 1."""
    ps = _aggressive_state(min_obs=2)
    # Two observations above threshold → enough=True, switch to compound
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_revert_sets_floor_and_count():
    """After a Gate-1 DECOMPOSE revert, revert_obs_floor and revert_count are set."""
    ps = _aggressive_state(min_obs=2, window=5)
    # Switch to compound
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Force Gate-1 DECOMPOSE: feed low scores until decompose confidence fires.
    # decompose_at=0.05 → scores ≤ 0.05 count; confidence=0.65.
    # Window=5; need ≥4 below threshold (4/5=0.80 ≥ 0.65).
    for _ in range(4):
        ps.record_and_maybe_switch("g", 0.03)

    s = ps._load("g")
    assert s.current_mode == "fine"
    assert s.revert_count == 1
    assert s.revert_obs_floor == len(s.observations)


def test_cooldown_prevents_immediate_reswitch():
    """After one revert, controller requires 2× min_observations fresh evidence."""
    ps = _aggressive_state(min_obs=2, window=5)

    # Initial switch
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Trigger Gate-1 DECOMPOSE revert (4 low obs → 4/5=0.80 ≥ confidence=0.65)
    for _ in range(4):
        ps.record_and_maybe_switch("g", 0.03)
    assert ps.get_mode("g") == CompositionLevel.FINE

    # With revert_count=1, required = min_obs × 2 = 4 fresh observations.
    # After only 2 high-score obs, the controller must NOT switch.
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.FINE, (
        "Should not switch after only 2 fresh obs when cooldown requires 4"
    )


def test_cooldown_allows_switch_after_required_fresh_obs():
    """After one revert, controller switches once 2× min_obs fresh evidence accumulates."""
    ps = _aggressive_state(min_obs=2, window=5)

    # Initial switch
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    # Revert via Gate-1 DECOMPOSE (4 low obs → 4/5=0.80 ≥ 0.65)
    for _ in range(4):
        ps.record_and_maybe_switch("g", 0.03)
    assert ps.get_mode("g") == CompositionLevel.FINE

    # Provide 4 fresh high-score observations (required = 2 × 2 = 4)
    for _ in range(4):
        ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND, (
        "Should switch after 4 fresh obs (2× min_obs) following first revert"
    )


def test_cooldown_multiplier_caps_at_five():
    """After 5+ reverts, required = min_observations × 5 (cap) — not higher."""
    policy = ControllerPolicy(
        compose_at=0.18, decompose_at=0.05, confidence=0.65,
        min_observations=2, window_size=10,
    )
    ps = PipelineState("cap_test", policy)
    # Simulate 6 reverts by directly manipulating state
    s = ps._load("g")
    s.revert_count = 6
    s.revert_obs_floor = 0
    ps._save("g", s)

    # required = 2 × min(1 + 6, 5) = 2 × 5 = 10
    # Provide 9 fresh obs — still not enough
    for _ in range(9):
        ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.FINE

    # 10th fresh obs — now enough
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND


def test_gate4_revert_increments_cooldown():
    """Gate 4 (token-reduction) reverts use _do_revert — revert_count increments."""
    from agentic_capsules.api.state import CompositionSignal

    policy = ControllerPolicy(
        compose_at=0.18, decompose_at=0.05, confidence=0.65,
        min_observations=2, window_size=10,
    )
    ps = PipelineState("gate4_test", policy)

    # Switch to compound first via raw overhead
    ps.record_and_maybe_switch("g", 0.55)
    ps.record_and_maybe_switch("g", 0.55)
    assert ps.get_mode("g") == CompositionLevel.COMPOUND

    # Inject FINE token records
    s = ps._load("g")
    s.tokens_fine = [100, 110]
    ps._save("g", s)

    # Inject two COMPOUND token records that are worse than FINE → Gate 4 fires
    sig = CompositionSignal(overhead_ratio=0.50, agent_count=3,
                            avg_output_tokens=50.0, total_tokens=200)
    ps.record_and_maybe_switch("g", 0.50, signal=sig)
    ps.record_and_maybe_switch("g", 0.50, signal=sig)

    s = ps._load("g")
    assert s.current_mode == "fine"
    assert s.revert_count == 1
    assert s.revert_obs_floor == len(s.observations)


def test_revert_cooldown_serialisation_roundtrip():
    """revert_obs_floor and revert_count survive to_json / from_json."""
    from agentic_capsules.api.state import GroupControllerState

    s = GroupControllerState(
        name="g",
        revert_obs_floor=7,
        revert_count=3,
    )
    restored = GroupControllerState.from_json(s.to_json())
    assert restored.revert_obs_floor == 7
    assert restored.revert_count == 3


def test_revert_cooldown_backward_compat():
    """JSON without revert fields deserialises with defaults of 0."""
    import json
    from agentic_capsules.api.state import GroupControllerState

    legacy = json.dumps({
        "name": "g",
        "observations": [0.5],
        "current_mode": "fine",
        "confidence": 0.0,
    })
    s = GroupControllerState.from_json(legacy)
    assert s.revert_obs_floor == 0
    assert s.revert_count == 0
