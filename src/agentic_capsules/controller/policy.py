"""
Controller policy — thresholds and execution knobs for the per-group granularity
controller.

Most applications should construct a pipeline with a sensitivity preset
(``Pipeline(name, sensitivity="balanced")``) rather than assembling a policy
directly. ``ControllerPolicy`` is the escape hatch for deployments that need
to override specific thresholds — quality floor, verbosity thresholds,
escalation behavior, or the default compound execution strategy.

The policy is applied per-pipeline and shared by all groups in that pipeline.
Per-group overrides are a future capability; for now, assemble multiple
pipelines if you need heterogeneous policies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControllerPolicy:
    """
    Configuration for the per-group granularity controller.

    Sensible defaults ship with the three sensitivity presets
    (``conservative``, ``balanced``, ``aggressive``). Override individual
    fields for specialized deployments — e.g., a premium customer tier might
    raise ``quality_floor`` while a cost-optimized tier lowers it.

    Controller behavior — thresholds that govern fine ↔ compound switching:

    Attributes:
        compose_at:
            Composition score above which the controller treats a group as a
            candidate for compound execution. The score weights coordination
            overhead, agent count, tool-call density, and dependency depth
            into a 0–1 signal; values above ``compose_at`` count as
            "consider merging."
        decompose_at:
            Composition score below which the controller reverts a group to
            fine mode. Below-floor scores signal that merging would be noise.
        confidence:
            Fraction of the rolling window that must exceed ``compose_at``
            (or fall below ``decompose_at``) before the controller commits to
            a mode switch. Higher values delay switching until the signal is
            consistent; lower values let the controller react faster.
        min_observations:
            Minimum number of fine-mode runs before the controller will
            consider any compound switch. Protects against single-sample
            overreactions during warmup.
        window_size:
            Number of recent observations kept in the rolling window for
            score, latency, and quality averages.
        score_weights:
            Per-signal weights for the composition score as a tuple of five
            floats: ``(overhead_ratio, agent_count, avg_output_tokens,
            tool_calls, depth_penalty)``. ``avg_output_tokens`` carries zero
            weight by default because the signal saturates for real LLMs and
            provides no group-differentiating information; tool-call density
            dominates the partition across providers. ``None`` falls back to
            overhead-ratio-only scoring.

    Hard overrides — clamp the score to 0 or 1 when operating conditions
    demand it regardless of the learned signal:

        error_rate_threshold:
            When the group's error rate meets or exceeds this, the
            composition score is forced to 1.0 (batch agents to reduce retry
            surface). Set higher for tolerant workloads, lower for
            error-sensitive ones.
        context_util_threshold:
            When context utilization meets or exceeds this, the score is
            forced to 0.0 (keep agents separate to avoid context overflow).

    Quality and latency gates — runtime safety nets for compound mode:

        quality_floor:
            Minimum acceptable rolling-mean quality score in compound mode
            (0.0–1.0). When set, the controller (a) shadow-evaluates the
            first fine→compound switch and blocks it if quality would fall
            below this floor, and (b) reverts compound→fine whenever the
            rolling-mean quality drops below the floor. Requires a
            ``QualityEvaluator`` to be passed to ``pipeline.run()`` — no-op
            otherwise. ``None`` disables quality gating entirely.
        latency_threshold_ms:
            Reverts compound→fine when the rolling-mean compound latency
            exceeds this bound (milliseconds). ``None`` disables the latency
            gate (default — most deployments care about quality more than
            raw latency).

    Compound execution strategy — what a "compound" group actually does when
    the controller commits to it:

        compound_execution_model:
            Which compound implementation to use when a group runs in
            compound mode:

            - ``"standard"`` — merge all agents into one LLM call. Best
              token savings; highest risk of output compression on small
              models and tool loss on tool-using groups.
            - ``"two_phase"`` — first do per-agent tool-gathering calls,
              then a single merged reasoning call with the gathered tool
              results pre-injected. Preserves tool access; cost sits
              between standard and fine.
            - ``"sequential"`` — per-agent calls with shared accumulated
              context. No merging, no compression. Each agent sees prior
              agents' outputs as context. Highest compound quality ceiling;
              smallest token savings.
            - ``"auto"`` — choose per-group at runtime: tools → ``two_phase``;
              high observed output verbosity → ``sequential``; otherwise
              ``standard``. Quality failures escalate up this ladder
              automatically (see ``escalation_enabled``).

        compound_tool_budget:
            When ``compound_execution_model="standard"``, controls how many
            tool calls the single merged call may issue. ``0`` disables
            tools in standard compound (recommended — tool-using groups
            should use ``two_phase``). ``-1`` or any positive integer passes
            every agent's tool definitions through.
        compound_min_output_words:
            Depth hint appended to each section in the compound prompt
            ("Aim for a comprehensive response of at least N words."). Useful
            for small models that tend to abbreviate later phases of a
            compound prompt. ``None`` emits no hint.
        compound_prompt_style:
            Which prompt structure to use when assembling the merged
            compound call:

            - ``"standard"`` — ``== PHASE N: agent_name ==`` headers with
              explicit role labels. Clear to the reader; on some small
              models the role labels cue a shift away from the agent's
              actual task.
            - ``"compact"`` — ``--- Step N ---`` anonymous headers. Same
              structural information, less role-framing to argue with.

    Verbosity-driven selection — signals used by ``compound_execution_model="auto"``
    and the auto output-guidance selector:

        verbosity_two_phase_threshold:
            When ``compound_execution_model="auto"``, a group whose mean
            fine-mode output per agent meets or exceeds this (tokens/agent)
            is routed to ``two_phase`` rather than ``standard``. Calibrated
            against mid-tier frontier models on information-retrieval
            pipelines; retune for creative-writing or visible-reasoning
            workloads.
        verbosity_sequential_threshold:
            Same signal; when the group meets or exceeds this, the
            controller routes to ``sequential`` instead of ``two_phase``.
            Must be strictly greater than ``verbosity_two_phase_threshold``.
        verbosity_guidance_threshold:
            When ``output_guidance="auto"``, a group whose mean fine-mode
            output per agent meets or exceeds this gets the concise length
            hint; otherwise the hint is suppressed. Same scale as the
            two-phase threshold because the same behavioral inflection —
            "is this group naturally verbose?" — drives both decisions.

    Prompt-shaping for compound execution — nudges that affect output
    quality or length without changing the call structure:

        merged_output_structure:
            Instruction added after each agent's output heading in the
            single merged compound call (i.e., ``compound_execution_model=
            "standard"``). Fights the model's tendency to abbreviate later
            sections when many agents share one call.

            - ``"none"`` — no hint.
            - ``"budgeted"`` — target roughly 800 words per section and keep
              every section equally detailed. When multiple agents are
              merged into one call, keep the sections equally detailed so
              later ones don't get abbreviated.
            - ``"budgeted_adaptive"`` — per-agent target derived from
              observed fine-mode output (80% of the agent's mean), clamped
              sensibly. Uses the framework's own observations instead of a
              fixed word count.
            - ``"reinforced"`` — "Fully address all requirements from your
              instructions above; do not omit any requested dimension." No
              length directive, just a no-drop instruction.

            Applies only to the standard compound path. Sequential and
            two-phase are unaffected.

        output_guidance:
            Length hint added to each agent's prompt when that agent gets
            its own call (fine mode and sequential compound — the per-agent
            call path, not the merged call).

            - ``"auto"`` — observations-based: apply the concise hint only
              on groups whose fine-mode output exceeds
              ``verbosity_guidance_threshold``. Avoids compressing
              already-terse models. This is the recommended default.
            - ``"none"`` — no hint.
            - ``"concise"`` — "Be concise. Aim for 300–400 words." Always
              applied. Use when you know every group in the pipeline
              produces long outputs that benefit from compression.
            - ``"moderate"`` — "Aim for 500–600 words."
            - ``"brief"`` — "Be brief. Aim for 200 words."

            Applies only to the per-agent call path. The merged-call
            ``merged_output_structure`` field is the analogous knob for
            the standard compound path.

        sequential_context_strategy:
            In sequential compound, how much accumulated context each agent
            receives:

            - ``"full"`` — every agent sees all prior agents' outputs.
              Safest for quality, most expensive in input tokens.
            - ``"predecessor_only"`` — each agent sees only its immediate
              predecessor's output. Cuts accumulated-context input tokens
              on long chains; quality-neutral in measured pipelines.

            For non-linear topologies (fan-out, diamond), the topology
            classifier may set a dependency-aware strategy at runtime that
            supersedes this field.

        cache_aligned_prompts:
            Reorders message blocks so consecutive per-agent calls share an
            identical prefix, enabling providers that support prompt
            caching (e.g., Anthropic's ephemeral cache) to discount the
            shared portion. No-op on providers without prompt caching;
            quality-neutral by design.

    Escalation ladder — automatic recovery when compound quality regresses:

        escalation_enabled:
            When a group in compound mode records
            ``escalation_min_failures`` consecutive quality scores below
            ``quality_floor``, step its execution model up the ladder
            (``standard`` → ``two_phase`` → ``sequential``) rather than
            immediately reverting to fine. If already at ``sequential`` and
            quality still fails, fall back to fine via the quality gate.
            Requires ``quality_floor`` and an evaluator to have any effect.
        escalation_min_failures:
            Consecutive below-floor observations required to trigger
            escalation. Two is a reasonable default; increase for noisier
            judges, decrease for more responsive correction.
        escalation_decay_window:
            After escalation, how many consecutive above-floor observations
            at the current tier are needed before stepping back down. Lets
            the controller recover efficiency when an upstream prompt
            change lifts quality back to a lower tier's operating range.
    """
    compose_at:             float = 0.40
    decompose_at:           float = 0.15
    confidence:             float = 0.80
    min_observations:       int   = 3
    window_size:            int   = 10
    score_weights:          tuple[float, float, float, float, float] | None = None

    # Hard overrides — clamp the composition score to 0 or 1 when the
    # learned signal would be unsafe.
    error_rate_threshold:   float = 0.15
    context_util_threshold: float = 0.85

    # Runtime safety gates — inactive by default (opt-in once an evaluator
    # is supplied to pipeline.run() or latency bounds are calibrated).
    latency_threshold_ms:   float | None = None
    quality_floor:          float | None = None

    # Which compound implementation to use when a group commits to compound
    # mode. See the class docstring for semantics of each value.
    compound_execution_model: str = "standard"
    compound_tool_budget:   int = 0
    compound_min_output_words: int | None = None
    compound_prompt_style: str = "standard"

    # Verbosity thresholds (tokens per agent, measured from fine-mode
    # observations). Used by the auto compound-mode selector and the auto
    # output-guidance selector. Defaults calibrated on mid-tier frontier
    # models against information-retrieval pipelines; retune per-domain for
    # creative writing or visible-reasoning workloads.
    verbosity_two_phase_threshold:  int = 1_500
    verbosity_sequential_threshold: int = 3_500
    verbosity_guidance_threshold:   int = 1_500

    # Prompt-shaping knobs. See the class docstring for the distinction
    # between merged_output_structure (standard compound's single merged
    # call) and output_guidance (the per-agent call path used by fine mode
    # and sequential compound).
    merged_output_structure: str = "budgeted"
    output_guidance: str = "auto"
    sequential_context_strategy: str = "predecessor_only"
    cache_aligned_prompts: bool = True

    # Escalation ladder — when compound quality drifts below the floor,
    # step the execution model up the ladder (standard → two_phase →
    # sequential) before falling back to fine. Requires quality_floor and
    # an evaluator to have any effect.
    escalation_enabled:      bool = True
    escalation_min_failures: int  = 2
    escalation_decay_window: int  = 5

    def __post_init__(self) -> None:
        if not 0 < self.compose_at <= 1:
            raise ValueError(f"compose_at must be in (0, 1], got {self.compose_at}")
        if not 0 <= self.decompose_at < self.compose_at:
            raise ValueError(
                f"decompose_at ({self.decompose_at}) must be < compose_at ({self.compose_at})"
            )
        if not 0 < self.confidence <= 1:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {self.min_observations}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.score_weights is not None:
            if len(self.score_weights) != 5:
                raise ValueError(f"score_weights must have exactly 5 elements, got {len(self.score_weights)}")
            if any(w < 0 for w in self.score_weights):
                raise ValueError(f"all score_weights must be >= 0, got {self.score_weights}")
        if not 0 < self.error_rate_threshold <= 1:
            raise ValueError(f"error_rate_threshold must be in (0, 1], got {self.error_rate_threshold}")
        if not 0 < self.context_util_threshold <= 1:
            raise ValueError(f"context_util_threshold must be in (0, 1], got {self.context_util_threshold}")
        if self.latency_threshold_ms is not None and self.latency_threshold_ms <= 0:
            raise ValueError(f"latency_threshold_ms must be > 0 when set, got {self.latency_threshold_ms}")
        if self.quality_floor is not None and not 0 <= self.quality_floor <= 1:
            raise ValueError(f"quality_floor must be in [0, 1] when set, got {self.quality_floor}")
        if self.compound_execution_model not in ("standard", "two_phase", "sequential", "auto"):
            raise ValueError(
                f"compound_execution_model must be 'standard', 'two_phase', 'sequential', "
                f"or 'auto', got {self.compound_execution_model!r}"
            )
        if self.compound_tool_budget < -1:
            raise ValueError(
                f"compound_tool_budget must be >= -1, got {self.compound_tool_budget}"
            )
        if self.compound_min_output_words is not None and self.compound_min_output_words <= 0:
            raise ValueError(
                f"compound_min_output_words must be > 0 when set, "
                f"got {self.compound_min_output_words}"
            )
        if self.compound_prompt_style not in ("standard", "compact"):
            raise ValueError(
                f"compound_prompt_style must be 'standard' or 'compact', "
                f"got {self.compound_prompt_style!r}"
            )
        if self.verbosity_two_phase_threshold <= 0:
            raise ValueError(
                f"verbosity_two_phase_threshold must be > 0, "
                f"got {self.verbosity_two_phase_threshold}"
            )
        if self.verbosity_sequential_threshold <= self.verbosity_two_phase_threshold:
            raise ValueError(
                f"verbosity_sequential_threshold ({self.verbosity_sequential_threshold}) "
                f"must be > verbosity_two_phase_threshold ({self.verbosity_two_phase_threshold})"
            )
        if self.verbosity_guidance_threshold <= 0:
            raise ValueError(
                f"verbosity_guidance_threshold must be > 0, "
                f"got {self.verbosity_guidance_threshold}"
            )
        if self.merged_output_structure not in ("none", "budgeted", "budgeted_adaptive", "reinforced"):
            raise ValueError(
                f"merged_output_structure must be 'none', 'budgeted', 'budgeted_adaptive', "
                f"or 'reinforced', got {self.merged_output_structure!r}"
            )
        if self.output_guidance not in ("none", "auto", "concise", "moderate", "brief"):
            raise ValueError(
                f"output_guidance must be 'none', 'auto', 'concise', 'moderate', "
                f"or 'brief', got {self.output_guidance!r}"
            )
        if self.sequential_context_strategy not in ("full", "predecessor_only"):
            raise ValueError(
                f"sequential_context_strategy must be 'full' or 'predecessor_only', "
                f"got {self.sequential_context_strategy!r}"
            )
        if self.escalation_min_failures < 1:
            raise ValueError(
                f"escalation_min_failures must be >= 1, got {self.escalation_min_failures}"
            )
        if self.escalation_decay_window < 1:
            raise ValueError(
                f"escalation_decay_window must be >= 1, got {self.escalation_decay_window}"
            )


# ---------------------------------------------------------------------------
# Sensitivity presets
# ---------------------------------------------------------------------------
#
# Default score weights for multi-signal composition scoring. The weights
# balance four structural signals:
#   w1 overhead_ratio     — coordination waste as fraction of total tokens
#   w2 agent_count        — more agents → more redundant system-prompt
#                            repetition in fine mode
#   w3 avg_output_tokens  — zero weight: the signal saturates for real LLMs
#                            (typical output range 381–7510 tokens) and
#                            contributes no group-differentiating information
#   w4 tool_calls         — tool-heavy agents benefit most from batching;
#                            strongest group differentiator across providers
#   w5 depth_penalty      — deep sequential chains benefit less from merging
#
# Normalization denominators: agent_count/4, tool_calls/3, depth/(agents-1).
#
_DEFAULT_SCORE_WEIGHTS = (0.45, 0.25, 0.00, 0.25, 0.05)

SENSITIVITY_PRESETS: dict[str, ControllerPolicy] = {
    "conservative": ControllerPolicy(
        # Higher threshold + more confidence + more observations required.
        # Use for stable pipelines where minimizing mode churn matters more
        # than fast adaptation.
        compose_at=0.35,
        decompose_at=0.10,
        confidence=0.90,
        min_observations=5,
        window_size=10,
        score_weights=_DEFAULT_SCORE_WEIGHTS,
    ),
    "balanced": ControllerPolicy(
        # Production default. Switches when the signal is clear but doesn't
        # wait for statistical certainty.
        compose_at=0.23,
        decompose_at=0.10,
        confidence=0.80,
        min_observations=3,
        window_size=10,
        score_weights=_DEFAULT_SCORE_WEIGHTS,
    ),
    "aggressive": ControllerPolicy(
        # Lower threshold + smaller window + less confidence. Use for fast
        # optimization when you can tolerate more mode switching and want
        # the controller to pick up borderline cases that balanced would
        # leave in fine mode.
        compose_at=0.18,
        decompose_at=0.14,
        confidence=0.65,
        min_observations=2,
        window_size=5,
        score_weights=_DEFAULT_SCORE_WEIGHTS,
    ),
}


def policy_for(sensitivity: str) -> ControllerPolicy:
    """
    Return the :class:`ControllerPolicy` preset for the given sensitivity name.

    Args:
        sensitivity: One of ``"conservative"``, ``"balanced"``, or ``"aggressive"``.

    Returns:
        A pre-configured :class:`ControllerPolicy` instance.

    Raises:
        ValueError: If the sensitivity name is not recognised.
    """
    if sensitivity not in SENSITIVITY_PRESETS:
        valid = list(SENSITIVITY_PRESETS)
        raise ValueError(
            f"sensitivity must be one of {valid!r}, got {sensitivity!r}."
        )
    return SENSITIVITY_PRESETS[sensitivity]
