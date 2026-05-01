"""
PipelineState — per-group controller state manager.

Holds one GroupControllerState per CompoundCapsule (group) and manages:
  - Observation accumulation (rolling overhead window)
  - Confidence-based switching (fraction of window above threshold)
  - Latency and token-efficiency tracking per mode (Phase 12)
  - Persistence via SyncBackend (in-memory default, Redis for production)

The confidence model::

    confidence = (observations in window that meet threshold) / len(window)
    Switch only when: len(observations) >= min_observations
                  AND confidence >= policy.confidence

Phase 11 — multi-signal scoring:

When ControllerPolicy.score_weights is set, observations are multi-signal
composition scores rather than raw overhead_ratio values. Score formula::

    score = w1 * overhead_ratio
          + w2 * min(agent_count / 4, 1.0)
          + w3 * min(avg_output_tokens / 300, 1.0)
          + w4 * min(tool_calls_per_agent / 3, 1.0)
          - w5 * min(depth / max(agent_count - 1, 1), 1.0)

This fires correctly with real LLMs where overhead_ratio alone is ~5-15%.

Phase 12 — latency and token-efficiency gates:

After switching to COMPOUND, two additional gates can trigger DECOMPOSE::

    Latency gate:       mean(latency_compound_ms) > latency_threshold_ms
    Token-reduction:    mean(tokens_compound) >= mean(tokens_fine)
                        (COMPOUND using more tokens than FINE — no efficiency gain)

Phase 12 — quality gate (T-033):

Proactive shadow comparison on first FINE → COMPOUND switch:
on the run where confidence first reaches the threshold, the compiler
runs both FINE and COMPOUND (one extra LLM call set), evaluates quality
via QualityEvaluator, and only commits the switch if
``quality_score >= policy.quality_floor``.

Reactive rolling quality gate (post-switch):
each COMPOUND run records a quality score in ``quality_scores``.
``get_recommendation()`` returns DECOMPOSE if
``mean(quality_scores[-window:]) < policy.quality_floor``.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..controller.policy import ControllerPolicy
from ..core.types import CompositionLevel

if TYPE_CHECKING:
    from ..runtime.sync_manager import SyncBackend
    from ..evaluation.base import QualityScore


# ---------------------------------------------------------------------------
# CompositionSignal — rich per-group observation (Phase 11)
# ---------------------------------------------------------------------------

@dataclass
class CompositionSignal:
    """
    Multi-signal observation for one group execution.

    Passed to PipelineState.record_and_maybe_switch() in place of a bare
    overhead_ratio so the controller can fire on real LLM output.

    Attributes:
        overhead_ratio:       coordination_tokens / total_tokens (primary signal).
        agent_count:          Number of agents in the group.
        avg_output_tokens:    Average tokens per agent response this run.
        tool_calls_per_agent: Average tool invocations per agent this run.
        dependency_depth:     Sequential chain depth (agent_count - 1 for a
                              fully sequential group; 0 for a single agent).
        error_rate:           errors / total items across all agents this run.
                              When >= ControllerPolicy.error_rate_threshold the
                              composition score is hard-clamped to 1.0 (force
                              COMPOSE — agent errors benefit from batching).
        context_utilization:  Average fraction of the adapter context window
                              consumed across agents this run.  When >=
                              ControllerPolicy.context_util_threshold the score
                              is hard-clamped to 0.0 (force DECOMPOSE — context
                              pressure means capsules must shrink).
        latency_ms:           Wall-clock time for this group's execution (ms).
                              Tracked per mode; used by latency gate (Phase 12).
        total_tokens:         Total tokens consumed by this group this run.
                              Tracked per mode; used by token-reduction gate
                              (Phase 12).  Success = COMPOUND uses fewer tokens
                              than FINE over the rolling window.
    """
    overhead_ratio:        float
    agent_count:           int
    avg_output_tokens:     float
    tool_calls_per_agent:  float = 0.0
    dependency_depth:      int   = 0
    error_rate:            float = 0.0
    context_utilization:   float = 0.0
    latency_ms:            float = 0.0   # Phase 12: per-group wall-clock time
    total_tokens:          int   = 0     # Phase 12: per-group token usage


def compute_composition_score(
    signal: CompositionSignal,
    weights: tuple[float, float, float, float, float],
) -> float:
    """
    Compute a weighted composition score from a :class:`CompositionSignal`.

    Score ranges roughly 0–1.  Values above ``ControllerPolicy.compose_at``
    count as COMPOSE evidence; values below ``decompose_at`` count as DECOMPOSE.

    Normalization:
      - agent_count        normalised to 4  (typical multi-agent group)
      - avg_output_tokens  normalised to 300 tokens (kept for backward compat;
                           default w3=0.0 so this term contributes nothing —
                           see T-021: real LLMs always exceed 300 tokens)
      - tool_calls         normalised to 3  (typical tool-heavy agent)
      - depth              normalised to max(agent_count - 1, 1)

    Default weights (T-021/T-032, 2026-03-29): (0.45, 0.25, 0.00, 0.25, 0.05).
    w3 removed from defaults — avg_output_tokens/300 saturates for real LLMs.
    Phase 11 ref: T-016 — calibration for real LLM output.
    """
    w1, w2, w3, w4, w5 = weights
    max_depth = max(signal.agent_count - 1, 1)
    score = (
          w1 * signal.overhead_ratio
        + w2 * min(signal.agent_count / 4.0, 1.0)
        + w3 * min(signal.avg_output_tokens / 300.0, 1.0)
        + w4 * min(signal.tool_calls_per_agent / 3.0, 1.0)
        - w5 * min(signal.dependency_depth / max_depth, 1.0)
    )
    return max(0.0, min(score, 1.0))  # clamp to [0, 1]


# ---------------------------------------------------------------------------
# GroupControllerState
# ---------------------------------------------------------------------------

@dataclass
class GroupControllerState:
    """
    Persisted state for one group's controller.

    Tracks composition observations, mode, confidence, and — from Phase 12 —
    per-mode latency and token usage so the controller can verify that
    COMPOUND mode is actually delivering efficiency gains.
    """
    name:                str
    observations:        list[float] = field(default_factory=list)
    current_mode:        str         = "fine"   # "fine" | "compound"
    confidence:          float       = 0.0
    last_score:          float       = 0.0
    # Phase 12: latency (ms) and token counts recorded separately per mode.
    # Used by latency gate and token-reduction gate in record_and_maybe_switch.
    latency_fine_ms:     list[float] = field(default_factory=list)
    latency_compound_ms: list[float] = field(default_factory=list)
    tokens_fine:         list[int]   = field(default_factory=list)
    tokens_compound:     list[int]   = field(default_factory=list)
    # Phase 12: rolling quality scores recorded when an evaluator is active.
    # Serialised to JSON for Redis persistence; last_quality_score and
    # last_fine_output are transient (not serialised).
    quality_scores:      list[float]          = field(default_factory=list)
    # T-039: rolling avg_output_tokens observed during FINE runs.
    # Used to auto-calibrate compound_min_output_words before COMPOUND switch.
    avg_output_tokens_fine: list[float]       = field(default_factory=list)
    # T-040: execution model resolved by quality escalation ladder.
    # None  = use auto gate logic each time.
    # str   = a previous quality escalation found this mode works; use it directly.
    # Serialised so the escalated mode persists across runs.
    execution_model_override: "str | None"    = field(default=None)
    # T-049: revert cooldown — prevents indefinite oscillation after gate-triggered reverts.
    # revert_obs_floor: len(observations) at the time of the last revert.
    # revert_count:     cumulative number of gate-triggered reverts (capped at 5 for enough calc).
    # Serialised; backward-compatible (from_json defaults both to 0).
    revert_obs_floor:    int                  = 0
    revert_count:        int                  = 0
    # E-1: consecutive quality-below-floor observations while in COMPOUND mode.
    # Incremented by H3 on each below-floor quality score; reset to 0 on quality
    # pass or execution model escalation.  Serialised; backward-compatible (defaults 0).
    quality_failure_streak: int               = 0
    # E-1 de-escalation: consecutive above-floor observations at current escalated tier.
    # When >= escalation_decay_window, de-escalate one tier (or clear override if at base).
    escalation_success_streak: int            = 0
    # Transient — not serialised to JSON.
    last_signal:         "CompositionSignal | None" = field(default=None, compare=False, repr=False)
    last_quality_score:  "float | None"            = field(default=None, compare=False, repr=False)
    last_fine_output:    "str | None"               = field(default=None, compare=False, repr=False)

    def to_json(self) -> str:
        """Serialise state to a JSON string (used by SyncBackend persistence)."""
        return json.dumps({
            "name":                    self.name,
            "observations":            self.observations,
            "current_mode":            self.current_mode,
            "confidence":              self.confidence,
            "last_score":              self.last_score,
            "latency_fine_ms":         self.latency_fine_ms,
            "latency_compound_ms":     self.latency_compound_ms,
            "tokens_fine":             self.tokens_fine,
            "tokens_compound":         self.tokens_compound,
            "quality_scores":           self.quality_scores,
            "avg_output_tokens_fine":   self.avg_output_tokens_fine,
            "execution_model_override":  self.execution_model_override,
            "revert_obs_floor":          self.revert_obs_floor,
            "revert_count":              self.revert_count,
            "quality_failure_streak":    self.quality_failure_streak,
            "escalation_success_streak": self.escalation_success_streak,
        })

    @classmethod
    def from_json(cls, data: str) -> GroupControllerState:
        """Deserialise state from a JSON string produced by :meth:`to_json`."""
        d = json.loads(data)
        return cls(
            name=d["name"],
            observations=d["observations"],
            current_mode=d["current_mode"],
            confidence=d["confidence"],
            last_score=d.get("last_score", 0.0),
            latency_fine_ms=d.get("latency_fine_ms", []),
            latency_compound_ms=d.get("latency_compound_ms", []),
            tokens_fine=d.get("tokens_fine", []),
            tokens_compound=d.get("tokens_compound", []),
            quality_scores=d.get("quality_scores", []),
            avg_output_tokens_fine=d.get("avg_output_tokens_fine", []),
            execution_model_override=d.get("execution_model_override", None),
            revert_obs_floor=d.get("revert_obs_floor", 0),
            revert_count=d.get("revert_count", 0),
            quality_failure_streak=d.get("quality_failure_streak", 0),
            escalation_success_streak=d.get("escalation_success_streak", 0),
        )

    def mean_latency_ms(self, mode: str, window: int = 10) -> float | None:
        """Rolling mean latency for the given mode ('fine' or 'compound'). None if no data."""
        series = self.latency_compound_ms if mode == "compound" else self.latency_fine_ms
        if not series:
            return None
        recent = series[-window:]
        return sum(recent) / len(recent)

    def mean_tokens(self, mode: str, window: int = 10) -> float | None:
        """Rolling mean token usage for the given mode. None if no data."""
        series = self.tokens_compound if mode == "compound" else self.tokens_fine
        if not series:
            return None
        recent = series[-window:]
        return sum(recent) / len(recent)

    def mean_quality(self, window: int = 10) -> float | None:
        """Rolling mean quality score.  None when no quality scores recorded."""
        if not self.quality_scores:
            return None
        recent = self.quality_scores[-window:]
        return sum(recent) / len(recent)

    def mean_avg_output_tokens_fine(self, window: int = 10, min_obs: int = 3) -> float | None:
        """
        Rolling mean avg_output_tokens observed during FINE runs.

        Returns None when fewer than *min_obs* FINE observations have been
        recorded — avoids calibrating on a single noisy sample.
        """
        if len(self.avg_output_tokens_fine) < min_obs:
            return None
        recent = self.avg_output_tokens_fine[-window:]
        return sum(recent) / len(recent)

    def token_reduction_pct(self, window: int = 10) -> float | None:
        """
        Percentage reduction in tokens when using COMPOUND vs FINE.

        Positive = COMPOUND uses fewer tokens (efficiency gain).
        Negative = COMPOUND uses more tokens (no gain).
        None when either mode has no data.
        """
        mean_f = self.mean_tokens("fine",     window)
        mean_c = self.mean_tokens("compound", window)
        if mean_f is None or mean_c is None or mean_f == 0:
            return None
        return (mean_f - mean_c) / mean_f * 100.0


def _do_revert(s: GroupControllerState) -> None:
    """
    T-049: Atomically revert a group to FINE mode and record the cooldown floor.

    Sets the revert_obs_floor to the current observation count so the controller
    requires fresh post-revert evidence before the next COMPOUND attempt, and
    increments revert_count to impose a linear cooldown multiplier.  Apply to
    every gate-triggered revert path (Gates 1-DECOMPOSE, 3, 4, H2, H3).
    """
    s.current_mode     = "fine"
    s.confidence       = 0.0
    s.revert_obs_floor = len(s.observations)
    s.revert_count    += 1


class PipelineState:
    """
    Manages per-group GranularityController state for a Pipeline.

    Usage::

        state = PipelineState("my_pipeline", policy, store=None)
        level = state.get_mode("research")           # CompositionLevel.FINE first run
        state.record_and_maybe_switch("research", overhead=0.52, apply_switch=True)
        confidence = state.get_confidence("research")
    """

    def __init__(
        self,
        pipeline_name: str,
        policy: ControllerPolicy,
        store: SyncBackend | None = None,
    ) -> None:
        self._pipeline_name = pipeline_name
        self._policy        = policy
        self._store         = store
        self._memory: dict[str, GroupControllerState] = {}   # fallback when no store
        # Per-group policy overrides. Populated by
        # ``register_group_policy(name, policy)`` whenever the builder sees
        # ``.group(..., policy=X)``. Empty dict means every group uses the
        # pipeline default.
        self._group_policies: dict[str, ControllerPolicy] = {}
        # G-7: Reentrant lock guarding all reads/writes to per-group state.
        # Reentrant because record_and_maybe_switch() etc. internally call
        # _load() / _save(), which themselves take the lock. The parallel
        # executor (_ParallelPipelineCompiler) needs this lock so that the
        # composition controller and quality gate can run concurrently from
        # multiple worker threads without corrupting rolling windows,
        # quality_scores, or escalation bookkeeping.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Policy resolution — per-group overrides
    # ------------------------------------------------------------------

    def register_group_policy(
        self, group_name: str, policy: ControllerPolicy
    ) -> None:
        """Record an override so ``_effective_policy(group_name)`` returns it
        instead of the pipeline-wide policy."""
        with self._lock:
            self._group_policies[group_name] = policy

    def _effective_policy(self, group_name: str) -> ControllerPolicy:
        """Return the policy that governs ``group_name``: the registered
        override if one exists, otherwise the pipeline-wide policy."""
        return self._group_policies.get(group_name, self._policy)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_mode(self, group_name: str) -> CompositionLevel:
        """Return the current composition level for this group."""
        with self._lock:
            s = self._load(group_name)
            return CompositionLevel.COMPOUND if s.current_mode == "compound" else CompositionLevel.FINE

    def record_and_maybe_switch(
        self,
        group_name:   str,
        overhead:     float,
        apply_switch: bool = True,
        signal:       CompositionSignal | None = None,
    ) -> GroupControllerState:
        """
        Record an observation and optionally switch composition mode.

        Switching gates (all must pass for FINE → COMPOUND):
          1. Composition score confidence >= policy.confidence over rolling window.
          2. (Phase 12) Quality gate — proactive shadow comparison (T-033).

        Post-switch DECOMPOSE gates (any can revert COMPOUND → FINE):
          3. Latency gate:      rolling mean COMPOUND latency > latency_threshold_ms.
          4. Token-reduction:   rolling mean COMPOUND tokens >= rolling mean FINE tokens
                                (COMPOUND delivering no efficiency gain).

        Args:
            group_name:   Name of the group that was just executed.
            overhead:     Observed coordination overhead ratio (0.0–1.0).
                          Used as the observation value when ``signal`` is None
                          or ``policy.score_weights`` is not configured.
            apply_switch: If False, observations are recorded and confidence
                          computed but current_mode never changes (observe mode).
            signal:       Optional multi-signal observation (Phase 11/12).  When
                          provided and ``policy.score_weights`` is set, the
                          weighted composition score replaces ``overhead`` as the
                          observation value.  Also carries latency_ms and
                          total_tokens for efficiency tracking.

        Returns:
            Updated GroupControllerState (confidence, mode, and efficiency data).
        """
        with self._lock:
            return self._record_and_maybe_switch_locked(
                group_name, overhead, apply_switch, signal,
            )

    def _record_and_maybe_switch_locked(
        self,
        group_name:   str,
        overhead:     float,
        apply_switch: bool,
        signal:       CompositionSignal | None,
    ) -> GroupControllerState:
        s      = self._load(group_name)
        policy = self._effective_policy(group_name)

        # ------------------------------------------------------------------
        # Phase 11: compute composition score
        # ------------------------------------------------------------------
        if signal is not None and policy.score_weights is not None:
            obs_value = compute_composition_score(signal, policy.score_weights)
            # Hard overrides: context pressure > error rate.
            if signal.context_utilization >= policy.context_util_threshold:
                obs_value = 0.0
            elif signal.error_rate >= policy.error_rate_threshold:
                obs_value = 1.0
            s.last_score  = obs_value
            s.last_signal = signal
        else:
            obs_value = overhead  # legacy: raw overhead_ratio

        s.observations.append(obs_value)

        # ------------------------------------------------------------------
        # Phase 12: record latency and tokens keyed by the mode just executed
        # ------------------------------------------------------------------
        if signal is not None:
            if signal.latency_ms > 0:
                if s.current_mode == "compound":
                    s.latency_compound_ms.append(signal.latency_ms)
                else:
                    s.latency_fine_ms.append(signal.latency_ms)
            if signal.total_tokens > 0:
                if s.current_mode == "compound":
                    s.tokens_compound.append(signal.total_tokens)
                else:
                    s.tokens_fine.append(signal.total_tokens)

        # ------------------------------------------------------------------
        # Gate 1: composition-score confidence switching
        # ------------------------------------------------------------------
        window    = s.observations[-policy.window_size:]
        # T-049: require fresh post-revert observations — linear cooldown after each revert.
        # First attempt: min_observations × 1 (unchanged).  After k reverts: × min(1+k, 5).
        fresh_obs = len(s.observations) - s.revert_obs_floor
        required  = policy.min_observations * min(1 + s.revert_count, 5)
        enough    = fresh_obs >= required

        if enough:
            if s.current_mode == "fine":
                above = sum(1 for oh in window if oh >= policy.compose_at)
                s.confidence = above / len(window)
                if apply_switch and s.confidence >= policy.confidence:
                    s.current_mode = "compound"
                    s.confidence = 0.0  # reset — next get_recommendation returns MAINTAIN
            else:  # currently compound
                below = sum(1 for oh in window if oh <= policy.decompose_at)
                s.confidence = below / len(window)
                if apply_switch and s.confidence >= policy.confidence:
                    _do_revert(s)  # Gate 1 DECOMPOSE (T-049)

        else:
            s.confidence = 0.0

        # ------------------------------------------------------------------
        # Gates 3 & 4: post-switch efficiency gates (only fire in COMPOUND mode)
        # These act as additional DECOMPOSE triggers independent of score.
        # ------------------------------------------------------------------
        if apply_switch and s.current_mode == "compound":
            w = policy.window_size

            # Gate 3 — latency: revert if COMPOUND is slower than threshold
            if policy.latency_threshold_ms is not None:
                mean_lat = s.mean_latency_ms("compound", w)
                if mean_lat is not None and mean_lat > policy.latency_threshold_ms:
                    _do_revert(s)  # Gate 3 latency (T-049)

            # Gate 4 — token reduction: revert if COMPOUND uses ≥ as many tokens as FINE
            # Requires at least 2 observations in each mode to avoid noise from single runs.
            if s.current_mode == "compound":   # re-check — gate 3 may have already reverted
                mean_f = s.mean_tokens("fine",     w)
                mean_c = s.mean_tokens("compound", w)
                if (
                    mean_f is not None and mean_c is not None
                    and len(s.tokens_fine) >= 2 and len(s.tokens_compound) >= 2
                    and mean_c >= mean_f
                ):
                    _do_revert(s)  # Gate 4 token-reduction (T-049)

        self._save(group_name, s)
        return s

    def get_confidence(self, group_name: str) -> float:
        """Return the last computed confidence score (0.0–1.0) for a group."""
        with self._lock:
            return self._load(group_name).confidence

    def get_recommendation(self, group_name: str) -> str:
        """
        Return the controller's current recommendation for this group.
        Does not change state.

        Quality gate (Phase 12): if policy.quality_floor is set and the rolling
        mean quality score falls below the floor, returns DECOMPOSE regardless
        of the composition-score confidence.
        """
        with self._lock:
            s      = self._load(group_name)
            policy = self._effective_policy(group_name)

            # Quality gate override — fires in either mode when quality degrades
            if (
                policy.quality_floor is not None
                and len(s.quality_scores) >= 2
            ):
                mean_q = s.mean_quality(policy.window_size)
                if mean_q is not None and mean_q < policy.quality_floor:
                    return "DECOMPOSE"

            if s.current_mode == "fine":
                if s.confidence >= policy.confidence and len(s.observations) >= policy.min_observations:
                    return "COMPOSE"
                return "MAINTAIN"
            else:  # compound
                if s.confidence >= policy.confidence and len(s.observations) >= policy.min_observations:
                    return "DECOMPOSE"
                return "MAINTAIN"

    def set_execution_model_override(self, group_name: str, model: str | None) -> None:
        """
        T-040: Persist the execution model resolved by quality escalation.

        Called by _PipelineCompiler when the escalation ladder finds a mode
        that passes the quality gate.  Subsequent runs skip the gate logic
        and use this mode directly, avoiding redundant shadow comparisons.

        Pass model=None to clear the override and return to auto gate logic.
        """
        with self._lock:
            s = self._load(group_name)
            s.execution_model_override = model
            self._save(group_name, s)

    def get_execution_model_override(self, group_name: str) -> str | None:
        """T-040: Return the persisted execution model override for this group, or None."""
        with self._lock:
            return self._load(group_name).execution_model_override

    def record_avg_output_tokens_fine(self, group_name: str, avg_tokens: float) -> None:
        """
        Record the avg_output_tokens observed for this group during a FINE run.

        Called by _PipelineCompiler after each FINE group execution so the
        controller can auto-calibrate compound_min_output_words (T-039).
        """
        with self._lock:
            s = self._load(group_name)
            s.avg_output_tokens_fine.append(avg_tokens)
            self._save(group_name, s)

    def get_auto_min_output_words(
        self,
        group_name: str,
        window: int = 10,
        min_obs: int = 3,
        floor_pct: float = 0.75,
    ) -> int | None:
        """
        Derive a compound_min_output_words hint from FINE observations (T-039).

        Computes the rolling mean avg_output_tokens across FINE runs and
        converts to a word floor: words ≈ tokens / 1.3 (English avg), then
        takes *floor_pct* of that as the minimum depth hint.

        Returns None when fewer than *min_obs* observations are available.
        The caller should fall back to policy.compound_min_output_words or None.
        """
        with self._lock:
            s = self._load(group_name)
            mean_tokens = s.mean_avg_output_tokens_fine(window=window, min_obs=min_obs)
            if mean_tokens is None:
                return None
            words = mean_tokens / 1.3 * floor_pct
            return max(50, int(words))

    def record_quality(self, group_name: str, score: "QualityScore") -> None:
        """
        Record a quality score observation for a group.

        Called by _PipelineCompiler after evaluating COMPOUND output quality.
        Updates the rolling quality_scores list and last_quality_score.
        """
        with self._lock:
            s = self._load(group_name)
            s.quality_scores.append(score.score)
            s.last_quality_score = score.score
            self._save(group_name, s)

    def get_quality(self, group_name: str) -> float | None:
        """Return the rolling mean quality score for a group.  None if no data."""
        with self._lock:
            s = self._load(group_name)
            return s.mean_quality(self._effective_policy(group_name).window_size)

    def snapshot(self) -> dict[str, GroupControllerState]:
        """Return a copy of all known group states (for PipelineResult assembly)."""
        with self._lock:
            return dict(self._memory)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _redis_key(self, group_name: str) -> str:
        return f"agentic_capsules:{self._pipeline_name}:{group_name}:controller_state"

    def _load(self, group_name: str) -> GroupControllerState:
        with self._lock:
            if self._store is not None:
                raw = self._store.get_sync(self._redis_key(group_name))
                if raw:
                    return GroupControllerState.from_json(raw)
            # Initialise if not yet seen
            if group_name not in self._memory:
                self._memory[group_name] = GroupControllerState(name=group_name)
            return self._memory[group_name]

    def _save(self, group_name: str, state: GroupControllerState) -> None:
        with self._lock:
            self._memory[group_name] = state
            if self._store is not None:
                self._store.put_sync(self._redis_key(group_name), state.to_json())
