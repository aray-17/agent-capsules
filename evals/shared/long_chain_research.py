"""
Long-chain research pipeline — P-3 eval template.

Exercises topology gaps left by P-1 (due_diligence) and P-2 (code_review):
  - 4-agent sequential tool chain in gather group (S-1 primary signal source)
  - 3-agent sequential reasoning chain in analyze group (M-1/O-1 at depth > 2)
  - Tools present throughout gather (all 4 agents use intelligence_search)
  - Different domain: competitive intelligence vs financial DD and code review

Pipeline shape:
  gather    (4 agents, 1 tool each) — primary_researcher → competitor_analyst
             → source_validator → expert_contextualizer
             Sequential: each agent builds on leads from the prior agent.
             Long chain is the primary signal source for S-1 (predecessor-only
             context): agent 4 under 'full' receives 3 prior outputs; under
             'predecessor_only' it receives only agent 3's output.
  analyze   (3 agents, no tools) — trend_analyst → strategic_analyst
             → vulnerability_assessor
             Sequential: each agent refines the prior analysis step.
             Borderline composition score — exercises controller threshold behavior.
  report    (1 agent, no tools) — intelligence_writer
             Stays FINE (score too low to compose).

Score profile (balanced preset, compose_at=0.23):
  gather:   4 agents (w2=1.0), 1 tool/agent (w4=0.333), overhead ~0.15
            ≈ 0.45×0.15 + 0.25×1.0 + 0.25×0.333 − 0.05×1.0
            ≈ 0.068 + 0.25 + 0.083 − 0.05 = 0.35 → composes ✓
  analyze:  3 agents (w2=0.75), 0 tools, overhead ~0.10
            ≈ 0.45×0.10 + 0.25×0.75 − 0.05×1.0
            ≈ 0.045 + 0.1875 − 0.05 = 0.18 → borderline (model-dependent)
  report:   1 agent, 0 tools → ~0.06 → stays FINE ✓

Model recommendations:
  haiku       — primary S-1 signal (most verbose; context accumulates fastest)
  sonnet      — quality-credible result (passes 0.75 floor on all groups, T-052)
  gpt-4o-mini — cross-provider check (confirms S-1 isn't Anthropic-only)
  gemini      — skip initial run (same class as gpt-4o-mini; won't add signal)

When to run:
  Create code:  during Phase 1 analysis window (free, no API)
  Run baseline: after S-1 on P-2 is complete
  Hard gate:    must be baselined before Phase 2 (E-1/E-2) launches

Task: competitive intelligence on a named company — market position,
key competitors, strategic vulnerabilities, growth trajectory.
"""
from __future__ import annotations

import re

from agentic_capsules import Pipeline, Tool


# ---------------------------------------------------------------------------
# Scripted adapter — offline / CI mode, no API key needed
# ---------------------------------------------------------------------------

class LongChainScriptedAdapter:
    """
    Deterministic adapter for the long-chain research pipeline.

    Responses are verbose (~100–160 words each) to produce meaningful token
    counts and context accumulation, making the S-1 predecessor-only signal
    measurable in offline runs.
    """
    context_window = 200_000

    _RESPONSES: dict[str, str] = {
        "PRIMARY_RESEARCHER": (
            "Initial landscape scan for the target company:\n"
            "Market segment: B2B SaaS infrastructure, total addressable market $42B growing "
            "at 18% CAGR. Identified 6 direct competitors: two venture-backed scale-ups "
            "(Series C+), three bootstrapped niche players, one public incumbent with "
            "declining market share. Target holds approximately 4% market share in its "
            "primary vertical (mid-market logistics). Key differentiation: proprietary "
            "real-time routing algorithm with 23% latency advantage over nearest competitor "
            "in independent benchmarks (2025 Gartner peer insights). intelligence_search "
            "returned 14 sources; 3 analyst reports, 4 customer case studies, 7 news items. "
            "Critical lead: incumbent (FleetCore) announced acquisition of adjacent player "
            "(RouteIQ) in Q4 2025 — consolidation signal. Follow-up priority: FleetCore "
            "integration timeline and whether RouteIQ customers are at risk of churn."
        ),
        "COMPETITOR_ANALYST": (
            "Deep competitor analysis following primary research leads:\n"
            "FleetCore/RouteIQ consolidation: acquisition closed December 2025 at $340M. "
            "RouteIQ had 2,200 mid-market customers — direct overlap with target's ICP. "
            "Integration timeline per SEC filings: full platform migration 18–24 months. "
            "Historical FleetCore acquisitions (3 prior) show 30–40% customer churn during "
            "migration windows. Estimated at-risk RouteIQ customers: 660–880. "
            "Second competitor (TrackWise): raised $85M Series C in January 2026, "
            "hiring 60 engineers focused on AI dispatch features. Product gap: no "
            "real-time multi-modal optimization (target's core strength). "
            "Third competitor (LogiPath): bootstrapped, $18M ARR, strong in cold-chain "
            "vertical — not direct overlap. intelligence_search cross-referenced "
            "LinkedIn hiring data, Crunchbase, and G2 review velocity."
        ),
        "SOURCE_VALIDATOR": (
            "Cross-reference and source validation:\n"
            "FleetCore acquisition claims: validated against SEC Form 8-K filing "
            "(December 12, 2025), PR Newswire announcement, and three independent "
            "analyst notes (Forrester, IDC, Bloomberg Intelligence). Customer count "
            "from RouteIQ's last public filing (Q3 2025 10-Q): 2,187 active customers. "
            "Churn estimate (30–40%) extrapolated from FleetCore's 2021 acquisition of "
            "DispatchOne — validated via two customer testimonials in G2 and 1 confirmed "
            "customer win cited in target's most recent board deck excerpt (leaked via "
            "Crunchbase funding announcement). TrackWise Series C: Crunchbase, PitchBook, "
            "and press release corroborate the $85M figure. One discrepancy found: "
            "target's own website claims 4.2% market share; Gartner report used 3.8%. "
            "Using conservative 3.8% for analysis. All primary claims: HIGH confidence. "
            "Market sizing figures: MEDIUM confidence (analyst variance ±15%)."
        ),
        "EXPERT_CONTEXTUALIZER": (
            "Domain and market context:\n"
            "Logistics SaaS consolidation cycle historically runs 3–5 years (analogous: "
            "2018–2022 ERP mid-market consolidation where challengers gained 2.1× share "
            "during incumbent merger distraction windows). Current cycle began Q3 2024 "
            "with TMS megadeals; we are approximately 18 months in. Target's window of "
            "competitive opportunity from FleetCore/RouteIQ integration distraction: "
            "estimated 12–18 months starting Q1 2026. Strategic context: AI dispatch "
            "features (TrackWise investment focus) will commoditize by 2027 per Gartner "
            "Hype Cycle; real-time multi-modal optimization (target's moat) is a longer "
            "differentiation horizon. Expert signal from intelligence_search: 2 ex-RouteIQ "
            "senior engineers joined target in Q4 2025 — talent signal of consolidation "
            "awareness. Key risk: target's 18-month runway (per prior research) overlaps "
            "exactly with the competitive window; capital timing is the execution constraint."
        ),
        "TREND_ANALYST": (
            "Pattern analysis across gathered intelligence:\n"
            "Three converging trends identified:\n"
            "  1. Consolidation-driven churn opportunity: 660–880 RouteIQ customers "
            "facing 18–24 month migration disruption align with target's ICP exactly. "
            "Historical pattern (3 FleetCore prior acquisitions) shows peak churn at "
            "months 8–14 of integration. Target must accelerate sales motion now.\n"
            "  2. AI feature commoditization timeline: TrackWise's $85M AI investment "
            "suggests 24–36 month window before AI dispatch is table stakes. Target's "
            "multi-modal optimization moat is durable for that period but needs to expand "
            "scope before commoditization hits.\n"
            "  3. Capital timing constraint: 18-month runway creates forced decision — "
            "raise Series B within 12 months or constrain go-to-market during the "
            "optimal competitive window. This is the dominant strategic tension."
        ),
        "STRATEGIC_ANALYST": (
            "Strategic implications:\n"
            "The intelligence converges on a narrow but high-confidence opportunity: "
            "a 12–18 month window to capture RouteIQ customer churn at scale. "
            "Recommended posture: aggressive outbound targeting of RouteIQ's 2,187 "
            "customers immediately, with a migration incentive program (free onboarding, "
            "90-day price lock) to accelerate switching decisions before FleetCore "
            "completes integration and stabilizes the base.\n"
            "Capital strategy: the Series B should be positioned around the RouteIQ "
            "opportunity — a concrete, time-bounded, quantifiable market event makes "
            "the raise thesis unusually crisp for investors. Target $30–40M at "
            "$180–220M pre-money valuation (based on $12M ARR × 15–18× forward revenue "
            "multiple for high-growth logistics SaaS). Window: close by Q3 2026 to "
            "fund the go-to-market expansion during peak RouteIQ churn. Risk: if raise "
            "slips to Q4 2026, the churn window closes before capital is deployed."
        ),
        "VULNERABILITY_ASSESSOR": (
            "Strategic vulnerabilities and gaps:\n"
            "  PRIMARY VULNERABILITY: Capital timing mismatch. 18-month runway expires "
            "at the same time the RouteIQ churn window peaks. If Series B takes longer "
            "than 9 months to close (realistic given current venture market), the company "
            "will be in fundraising mode during its best competitive quarter. Severity: HIGH.\n"
            "  SECONDARY VULNERABILITY: Sales capacity. Targeting 660–880 potential "
            "churning customers requires outbound capacity the company likely does not have "
            "today (no VP Sales per earlier signals). Hiring lag of 3–6 months means "
            "sales investment must begin immediately. Severity: HIGH.\n"
            "  TERTIARY VULNERABILITY: TrackWise AI investment. If AI dispatch reaches "
            "parity faster than the 24–36 month estimate, the multi-modal moat narrows. "
            "Mitigation: file continuation patents on optimization algorithm and invest "
            "in customer integrations that create switching costs. Severity: MEDIUM.\n"
            "  GAP IN INTELLIGENCE: No data on target's current NRR or logo churn rate. "
            "NRR above 115% would significantly strengthen the investment thesis."
        ),
        "INTELLIGENCE_WRITER": (
            "COMPETITIVE INTELLIGENCE REPORT\n\n"
            "Subject: Target company competitive position and strategic outlook\n"
            "Confidence: HIGH (primary sources validated; one market share discrepancy noted)\n\n"
            "HEADLINE FINDING:\n"
            "A 12–18 month window of competitive opportunity has opened due to the "
            "FleetCore/RouteIQ acquisition. An estimated 660–880 RouteIQ customers face "
            "platform disruption — these customers map directly to the target's ICP. "
            "Historical precedent suggests peak churn at months 8–14 of integration "
            "(starting Q1 2026). Capital timing is the primary execution constraint.\n\n"
            "KEY RECOMMENDATIONS:\n"
            "1. Launch RouteIQ migration program immediately (outbound + migration incentive)\n"
            "2. Accelerate Series B timeline — close by Q3 2026 before churn window closes\n"
            "3. Prioritize VP Sales hire; outbound capacity is the binding constraint\n"
            "4. File continuation patents on multi-modal optimization algorithm\n\n"
            "RISKS: Capital/window timing mismatch (HIGH), sales capacity gap (HIGH), "
            "AI commoditization accelerating (MEDIUM)."
        ),
    }

    def complete(self, messages: list, tools=None) -> str:
        combined = " ".join(
            getattr(m, "content", str(m)) for m in messages
        )
        keys = re.findall(r"([A-Z][A-Z_]+_OUTPUT)", combined)
        seen: set[str] = set()
        parts: list[str] = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:  # type: ignore[func-returns-value]
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{self._RESPONSES.get(agent, 'Research complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

def _tools() -> tuple[Tool]:
    """
    Stub tool for the long-chain research pipeline.

    intelligence_search: returns structured search results for a given query.
    Each gather-group agent calls this once, producing realistic tool output
    that inflates agent outputs and drives context accumulation in the
    sequential chain — the key driver of S-1 signal.
    """
    intelligence_search = Tool(
        name="intelligence_search",
        description=(
            "Search for competitive intelligence on a company, market, or topic. "
            "Returns structured results: news, filings, analyst reports, and social signals."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query":   {"type": "string",  "description": "Search query"},
                "sources": {"type": "array",   "items": {"type": "string"},
                            "description": "Source types to search (news, filings, analyst, social)"},
                "limit":   {"type": "integer", "description": "Maximum results to return"},
            },
            "required": ["query"],
        },
        fn=lambda args: {
            "query":          args.get("query", ""),
            "results_count":  12,
            "sources_hit":    ["news", "sec_filings", "analyst_reports", "linkedin"],
            "top_results": [
                {"source": "Gartner Peer Insights 2025",
                 "snippet": "Target company rated 4.6/5 on real-time routing accuracy; "
                            "23% latency advantage cited by 8 of 12 reviewed customers."},
                {"source": "SEC Form 8-K (FleetCore, Dec 2025)",
                 "snippet": "Acquisition of RouteIQ Inc. for $340M cash, expected to close Q1 2026. "
                            "RouteIQ had 2,187 active customers as of Q3 2025."},
                {"source": "Crunchbase funding data",
                 "snippet": "TrackWise raised $85M Series C (January 2026). "
                            "Stated use of proceeds: AI dispatch feature development."},
                {"source": "G2 review delta (last 90 days)",
                 "snippet": "Target: +18 new reviews (avg 4.5). "
                            "FleetCore: −4 reviews (avg 3.8, declining). "
                            "RouteIQ: 0 new reviews since acquisition announcement."},
            ],
        },
    )

    return (intelligence_search,)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_pipeline_with_judge(
    sensitivity:                str   = "balanced",
    judge_adapter                     = None,
    quality_floor:              float = 0.75,
    compound_execution_model:   str   = "standard",
    compound_min_output_words:  int | None = None,
    merged_output_structure:    str   = "none",
    output_guidance:            str   = "none",
    sequential_context_strategy: str  = "full",
    cache_aligned_prompts:      bool  = False,
    escalation_enabled:         bool  = False,
):
    """
    Build the long-chain research pipeline with an optional LLM judge evaluator.

    Returns ``(pipeline, evaluator)``.  ``evaluator`` is None when
    ``judge_adapter`` is None.
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
    pipeline  = build_pipeline(sensitivity=sensitivity)
    pipeline._policy = policy
    pipeline._pipeline_state._policy = policy

    return pipeline, evaluator

TASK_TEMPLATE = (
    "Produce a competitive intelligence report on {company}: "
    "market position, key competitors, strategic vulnerabilities, and growth trajectory."
)

PIPELINE_DESCRIPTION = (
    "gather (4 agents, 1 tool) → analyze (3 agents) → report (1 agent)"
)


def build_pipeline(sensitivity: str = "balanced") -> Pipeline:
    """
    Build the long-chain research evaluation pipeline.

    Sequential 4-agent gather group with tools — each agent builds on leads
    from the prior agent. Primary signal source for S-1 (predecessor-only
    context strategy). Longer sequential chains and greater chain depth than
    P-1 (due_diligence) and P-2 (code_review).

    Args:
        sensitivity: One of "conservative", "balanced", "aggressive".
    """
    (intelligence_search,) = _tools()

    return (
        Pipeline("long_chain_research", sensitivity=sensitivity)
        .group("gather")
            .agent(
                "primary_researcher",
                "Conduct an initial broad landscape scan on the target company. "
                "Use intelligence_search to gather market sizing, competitor counts, "
                "key differentiators, and any recent market events (M&A, funding rounds). "
                "Return: market segment, TAM, identified competitors (with tier), "
                "target's estimated market share, and 2–3 high-priority follow-up leads "
                "for deeper investigation.",
                tools=[intelligence_search],
            )
            .agent(
                "competitor_analyst",
                "Conduct deep competitor analysis using the leads identified by the "
                "primary researcher. Use intelligence_search to investigate the top "
                "2–3 competitors in detail: funding, product gaps, customer overlap, "
                "recent strategic moves, and hiring signals. "
                "Return: per-competitor breakdown with evidence sources and estimated "
                "competitive threat level (HIGH/MEDIUM/LOW).",
                tools=[intelligence_search],
            )
            .agent(
                "source_validator",
                "Cross-reference and validate the claims made by the primary researcher "
                "and competitor analyst. Use intelligence_search to verify key facts "
                "against primary sources (SEC filings, press releases, analyst reports). "
                "Flag any discrepancies. Assign confidence levels (HIGH/MEDIUM/LOW) "
                "to each major claim and note any intelligence gaps.",
                tools=[intelligence_search],
            )
            .agent(
                "expert_contextualizer",
                "Add domain and market cycle context to the gathered intelligence. "
                "Use intelligence_search to find historical analogues, industry benchmarks, "
                "and expert commentary relevant to the target's situation. "
                "Return: market cycle positioning, timing of competitive windows, "
                "and any talent or cultural signals that validate or contradict the "
                "strategic picture assembled by prior agents.",
                tools=[intelligence_search],
            )
        .group("analyze")
            .agent(
                "trend_analyst",
                "Identify the 2–4 most significant converging trends in the gathered "
                "intelligence. For each trend, state the supporting evidence, the "
                "likely timeline, and what it means for the target's competitive position. "
                "Prioritise trends by strategic relevance, not recency.",
            )
            .agent(
                "strategic_analyst",
                "Derive the strategic implications of the identified trends. "
                "What should the target company do, and by when? "
                "State the recommended strategic posture, the key decisions to make "
                "in the next 90 days, and the estimated value at stake for each. "
                "Be specific: name actions, owners, and timelines.",
            )
            .agent(
                "vulnerability_assessor",
                "Identify the target company's strategic vulnerabilities — "
                "gaps between the recommended strategy and current capabilities. "
                "Rate each vulnerability HIGH/MEDIUM/LOW. For HIGH items, state "
                "the earliest point at which the vulnerability becomes irreversible. "
                "Also flag any intelligence gaps that, if filled, could materially "
                "change the strategic picture.",
            )
        .group("report")
            .agent(
                "intelligence_writer",
                "Write a concise competitive intelligence report (250–350 words): "
                "headline finding, confidence level, 3–4 prioritised recommendations "
                "with owners and timelines, and key risks. "
                "Lead with the most time-sensitive finding. "
                "Do not repeat analysis already covered — synthesise into decisions.",
            )
    )
