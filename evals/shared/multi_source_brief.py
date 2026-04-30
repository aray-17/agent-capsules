"""
Multi-source competitive intelligence brief — T-054 v2 pipeline.

A 14-agent, 6-group pipeline that produces a one-page competitive intelligence
brief on a target company by combining four parallel research arms (one per
intelligence lens) with a final synthesis step. Designed to exercise both
inter-group parallelism (the four arms run concurrently) and compound merging
within each parallel arm (three extractors per arm collapse into one LLM call
in COMPOUND mode).

Pipeline shape::

                            ┌──────────────────┐
                            │ G1: scoping      │   1 agent (FINE)
                            │  (target_scoper) │
                            └────────┬─────────┘
                                     │
                  ┌────────┬─────────┼─────────┬────────┐    ← 4-way fan-out
                  ▼        ▼         ▼         ▼
            ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
            │  G2:    ││  G3:    ││  G4:    ││  G5:    │     Each arm = 1 group
            │ compete ││ product ││ finance ││  risk   │     = 3 extractors
            └────┬────┘└────┬────┘└────┬────┘└────┬────┘     COMPOUND-friendly
                 │          │          │          │
                 └──────────┴────┬─────┴──────────┘          ← 4-way join
                                 ▼
                        ┌──────────────────┐
                        │ G6: synthesis    │   1 agent (FINE)
                        │   (briefer)      │
                        └──────────────────┘

Total: 1 + (4 × 3) + 1 = 14 agents, 6 groups.

Why this shape:

- **Inter-group parallelism is structural.** Groups G2–G5 declare
  ``depends_on=["scoping"]`` and have no edges between them, so under the
  threaded parallel executor (``Pipeline.run(parallel=True)``) they run
  concurrently. This is the parallelism axis of the v2 LangGraph comparison.

- **Each arm is a compound-merging candidate.** Within each arm, the three
  extractors (``entities``, ``claims``, ``signals``) all declare
  ``depends_on=[]`` and process the same source bundle. Under COMPOUND mode
  the three calls collapse into one merged LLM call with three extraction
  sub-tasks; under FINE mode they run as three independent calls. The
  ratio of (FINE tokens / COMPOUND tokens) per arm is the compound-merging
  win, and it stacks with the parallelism win across arms.

- **Track A defaults exercised.** This pipeline benefits from M-1
  (budgeted_adaptive — keeps the three extractor sub-outputs balanced),
  O-1 (concise — every extractor produces structured short output), C-1
  (cache_aligned_prompts — the source bundle is cacheable across the three
  extractors in each arm), and E-1 (escalation_enabled — synthesizer
  quality drives ladder escalation if a model struggles). S-1 does NOT
  apply because no within-group sequential chain exists.

- **Source bundles are fixed.** This pipeline tests execution strategies,
  not retrieval. The four bundles per target are hand-curated from public
  materials (see evals/data/multi_source_bundles.py). Both Agentic Capsules
  and the LangGraph baseline see identical input.

When to run:
  Smoke test: ``pipeline.run(target, adapter=ScriptedAdapter, mode='fine')``
  Real eval:  use the parallel executor with ``parallel=True`` and forced
              ``mode='fine'`` or ``mode='compound'`` (forced-mode invariant
              of the parallel executor — controller is bypassed).

Tracking ref: T-054 (paper v2 — peer-review readiness work, parallel + LangGraph)
"""
from __future__ import annotations

from agentic_capsules import Pipeline

from evals.data.multi_source_bundles import LENSES, TARGETS


PIPELINE_NAME = "multi_source_brief"
PIPELINE_DESCRIPTION = (
    "scoping (1 agent) → 4 parallel arms × (entities + claims + signals) → "
    "synthesis (1 agent) = 14 agents, 6 groups"
)


# ---------------------------------------------------------------------------
# Per-arm extractor system prompts
# ---------------------------------------------------------------------------

# Each extractor processes the *same* source bundle for its arm. The bundle is
# embedded as the cacheable prefix; the extraction instruction sits at the
# tail. Identical prefix across the three extractors in an arm is what makes
# C-1 (cache_aligned_prompts) effective on this pipeline.

_ENTITIES_INSTRUCTION = (
    "Extract concrete entities from the source above: company names, product "
    "names, dates, dollar amounts, percentages, locations, and named "
    "individuals. Output as a numbered list. Include only entities explicitly "
    "stated in the source — do not infer or invent. Be exhaustive but concise."
)

_CLAIMS_INSTRUCTION = (
    "Extract qualitative claims and positioning statements from the source "
    "above: 'X is the leader in Y', 'X is known for Z', sentiment about the "
    "company or its products, comparisons to competitors. Output as a "
    "numbered list. Each claim must be paraphrased from the source — do not "
    "invent claims that are not supported."
)

_SIGNALS_INSTRUCTION = (
    "Extract forward-looking signals from the source above: stated risks, "
    "opportunities, planned initiatives, expansion or contraction signals, "
    "hiring trends, regulatory exposure, competitive threats. Output as a "
    "numbered list. Distinguish between explicit signals (stated outright) "
    "and implicit signals (clearly inferable from facts in the source)."
)

_SCOPING_INSTRUCTION = (
    "Read the source material above and extract a 2–3 sentence scope: the "
    "target company's name, sector, headquarters location, founding year, "
    "and a one-line characterization of what the company does. This scope "
    "is the context block used by all four downstream research arms."
)

_BRIEFER_INSTRUCTION = (
    "You are writing a one-page competitive intelligence brief on the target "
    "company. You will receive the scoping context plus four arm outputs "
    "(competitive landscape, product portfolio, financial signals, and "
    "risk/regulatory). Produce a brief with: (1) a 3–4 sentence executive "
    "summary, (2) one paragraph per lens with the most important entities, "
    "claims, and signals from that arm, and (3) a 'Key Takeaways' section of "
    "3 bullet points. Cite only facts present in the input — do not invent."
)


def _make_arm_prompt(bundle_text: str, instruction: str) -> str:
    """
    Build the system prompt for one extractor in one arm.

    The bundle goes first (cacheable prefix shared by all three extractors in
    the arm) and the extraction instruction goes at the tail. C-1 prompt
    caching keys on the prefix, so the bundle is read once per arm rather
    than three times.
    """
    return (
        "SOURCE MATERIAL:\n"
        f"{bundle_text}\n\n"
        "TASK:\n"
        f"{instruction}"
    )


def _make_scoping_prompt(overview_text: str) -> str:
    return (
        "SOURCE MATERIAL:\n"
        f"{overview_text}\n\n"
        "TASK:\n"
        f"{_SCOPING_INSTRUCTION}"
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(target: str, sensitivity: str = "balanced") -> Pipeline:
    """
    Build the multi-source brief pipeline for a specific target company.

    The bundles for each lens are baked into the agent system prompts at
    build time. The DSL does not currently support per-group task input,
    so the per-arm bundles live in the agents' system prompts rather than
    in the run-time task argument. The shared-prefix structure makes C-1
    (cache_aligned_prompts) effective across the three extractors in each
    arm.

    Args:
        target:      Target company name. Must be a key in
                     ``evals.data.multi_source_bundles.TARGETS``.
        sensitivity: One of ``"conservative"``, ``"balanced"``, ``"aggressive"``.

    Returns:
        A configured ``Pipeline`` ready to run with either the serial or
        parallel executor. For real evals use ``run(parallel=True)`` with
        ``mode="fine"`` or ``mode="compound"``.

    Raises:
        KeyError: if ``target`` is not in ``TARGETS``.
    """
    if target not in TARGETS:
        raise KeyError(
            f"Unknown target {target!r}. Available targets: {sorted(TARGETS)}"
        )
    bundles = TARGETS[target]

    p = Pipeline(PIPELINE_NAME, sensitivity=sensitivity)

    # ------------------------------------------------------------------
    # G1 — scoping (root, no deps)
    # ------------------------------------------------------------------
    p.group("scoping", depends_on=[]).agent(
        "target_scoper",
        _make_scoping_prompt(bundles["overview"]),
    )

    # ------------------------------------------------------------------
    # G2–G5 — four research arms in parallel (depends_on=["scoping"])
    #
    # Within each arm: three extractors with depends_on=[] (no internal
    # edges) so the agents are independent. Under FINE mode they run as
    # three separate calls; under COMPOUND mode they collapse into one
    # merged call with three sub-tasks.
    # ------------------------------------------------------------------
    for lens in LENSES:
        bundle_text = bundles[lens]
        p.group(lens, depends_on=["scoping"])
        p.agent(
            f"{lens}_entities",
            _make_arm_prompt(bundle_text, _ENTITIES_INSTRUCTION),
            depends_on=[],
        )
        p.agent(
            f"{lens}_claims",
            _make_arm_prompt(bundle_text, _CLAIMS_INSTRUCTION),
            depends_on=[],
        )
        p.agent(
            f"{lens}_signals",
            _make_arm_prompt(bundle_text, _SIGNALS_INSTRUCTION),
            depends_on=[],
        )

    # ------------------------------------------------------------------
    # G6 — synthesis (depends on all four arms)
    # ------------------------------------------------------------------
    p.group("synthesis", depends_on=list(LENSES)).agent(
        "briefer",
        _BRIEFER_INSTRUCTION,
    )

    return p


def build_pipeline_with_judge(
    target:                       str,
    sensitivity:                  str   = "balanced",
    judge_adapter                       = None,
    quality_floor:                float = 0.75,
    compound_execution_model:     str   = "standard",
    compound_min_output_words:    int | None = None,
    merged_output_structure:      str   = "none",
    output_guidance:              str   = "none",
    sequential_context_strategy:  str   = "full",
    cache_aligned_prompts:        bool  = False,
    escalation_enabled:           bool  = False,
):
    """
    Build the multi-source brief pipeline with an optional LLM judge evaluator.

    Mirrors ``evals.shared.long_chain_research.build_pipeline_with_judge``.
    Returns ``(pipeline, evaluator)``.

    Note: when running this pipeline through the parallel executor
    (``run(parallel=True)``), the evaluator must be ``None`` — the parallel
    executor's forced-mode invariant rejects quality gates because the H2/H3
    code paths write to ControllerState. Pass ``judge_adapter=None`` for
    parallel runs and use the serial executor for quality-gated experiments.
    """
    from agentic_capsules import LLMJudgeEvaluator, ControllerPolicy
    from agentic_capsules.controller.policy import policy_for

    base_policy = policy_for(sensitivity)
    policy = ControllerPolicy(
        compose_at=base_policy.compose_at,
        decompose_at=base_policy.decompose_at,
        confidence=base_policy.confidence,
        min_observations=base_policy.min_observations,
        window_size=base_policy.window_size,
        score_weights=base_policy.score_weights,
        error_rate_threshold=base_policy.error_rate_threshold,
        context_util_threshold=base_policy.context_util_threshold,
        latency_threshold_ms=base_policy.latency_threshold_ms,
        quality_floor=quality_floor if judge_adapter is not None else None,
        compound_execution_model=compound_execution_model,
        compound_min_output_words=compound_min_output_words,
        merged_output_structure=merged_output_structure,
        output_guidance=output_guidance,
        sequential_context_strategy=sequential_context_strategy,
        cache_aligned_prompts=cache_aligned_prompts,
        escalation_enabled=escalation_enabled,
    )

    evaluator = LLMJudgeEvaluator(judge_adapter) if judge_adapter is not None else None
    pipeline  = build_pipeline(target=target, sensitivity=sensitivity)
    pipeline._policy = policy
    pipeline._pipeline_state._policy = policy

    return pipeline, evaluator
