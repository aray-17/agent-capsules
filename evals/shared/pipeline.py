"""
Canonical evaluation pipeline — Startup Due Diligence.

This is the shared pipeline definition used by all evals and tuning runs.
Both OpenAI and Anthropic evals run the exact same pipeline so that
controller behaviour is directly comparable across providers and models.

Pipeline shape:
  research  (2 agents, 2 tools) — market_researcher → team_analyst
  analysis  (2 agents, no tools) — financial_modeler → risk_assessor
  synthesis (1 agent,  no tools) — investment_writer

This layout deliberately exercises three distinct group profiles:
  - research:  multi-agent, tool-heavy, sequential dependency (highest expected score)
  - analysis:  multi-agent, no tools, sequential dependency (mid score)
  - synthesis: single-agent, no tools (lowest score — should stay FINE)

These score differences let the controller demonstrate per-group adaptation.

Adapters (T-028):
  ScriptedAdapter        — deterministic, fixed-length responses (offline / CI)
  VariableToolAdapter    — realistic response size variance per call (simulates
                           real tool output: short on cache hit, long on fresh fetch)
  ErrorInjectionAdapter  — wraps another adapter and injects agent failures at
                           a configurable rate (used to verify error_rate signal)
"""
from __future__ import annotations

import random
import re

from agentic_capsules import Pipeline, Tool


# ---------------------------------------------------------------------------
# Scripted adapter — offline / CI mode, no API key needed
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    """
    Deterministic adapter that returns realistic fixed responses.

    Use for:
      - Offline eval runs (no API cost)
      - CI / regression tests
      - Verifying eval harness logic before spending on live API calls

    Responses are verbose enough (~80–120 words each) to approximate real
    LLM output patterns and produce meaningful token counts for the controller.
    """
    context_window = 200_000

    _RESPONSES: dict[str, str] = {
        "MARKET_RESEARCHER": (
            "Market analysis complete. The payments infrastructure market is valued at $2.1T "
            "globally, growing 12% YoY driven by embedded finance and API-first platforms. "
            "Key competitors: Stripe (42% market share), Adyen (18%), Square (11%). "
            "The target company occupies the SMB segment with a differentiated embedded "
            "finance approach, targeting merchants below the $1M GMV threshold where no "
            "dominant player has established lock-in. web_search returned 8 high-authority "
            "sources; competitor_lookup identified 4 direct competitors and 2 adjacent "
            "players entering the space. White space confirmed: vertical SaaS bundles "
            "for SMB remain underpenetrated."
        ),
        "TEAM_ANALYST": (
            "Team assessment: founding team has 3 members with prior exits (Plaid, "
            "Braintree). CTO holds 2 patents in payment tokenisation covering novel "
            "approaches to embedded credential vaulting. Key risk: VP Sales role unfilled "
            "for 6 months — new logo pipeline growth slowing despite strong expansion "
            "revenue from existing accounts. Advisors include 2 ex-Visa executives and "
            "1 former PayPal GM. Engineering team of 12 skews senior (median 8 YoE). "
            "Overall team score: 8/10 with VP Sales hire as the single critical unlock "
            "before Series A close."
        ),
        "FINANCIAL_MODELER": (
            "Financial model: ARR $4.2M (+180% YoY), net revenue retention 118%, "
            "CAC $1,200, LTV $18,500 (LTV:CAC 15.4x). Burn rate $380k/month, 18-month "
            "runway at current pace. Gross margin 71%, improving toward 75% as infra "
            "costs amortise. Rule of 40 score: 180+71=251 (exceptional). "
            "Path to profitability: 24 months at current growth rate assuming VP Sales "
            "hire accelerates new logo acquisition by 40%. Key sensitivity: churn above "
            "8% annually would compress NRR below 110% and extend breakeven by 12 months."
        ),
        "RISK_ASSESSOR": (
            "Risk register (top 3):\n"
            "  1. Regulatory — PCI-DSS Level 1 certification pending; blocks enterprise "
            "sales and two LOIs until Q3. Probability HIGH, Impact HIGH. Mitigation: "
            "QSA engaged, 90-day remediation plan in place.\n"
            "  2. Competition — Stripe launching SMB embedded finance product in Q2. "
            "Probability MEDIUM, Impact HIGH. Mitigation: vertical SaaS partnerships "
            "create switching costs before Stripe reaches market.\n"
            "  3. Key-person — 60% of technical IP concentrated in CTO. "
            "Probability LOW, Impact HIGH. Mitigation: IP assignment complete; "
            "second engineer being groomed as technical lead.\n"
            "Composite risk score: MEDIUM. Regulatory timing is the primary gating factor."
        ),
        "INVESTMENT_WRITER": (
            "INVESTMENT BRIEF\n\n"
            "Recommendation: CONDITIONAL PASS — invest at $18M pre-money valuation "
            "contingent on (1) VP Sales hire within 60 days, (2) PCI-DSS L1 "
            "certification roadmap with milestones.\n\n"
            "Thesis: The company has a rare combination of strong unit economics "
            "(LTV:CAC 15x, NRR 118%) and product differentiation in an underpenetrated "
            "SMB embedded-finance segment. Team quality and 180% ARR growth indicate "
            "genuine product-market fit with early signs of network effects in the "
            "vertical SaaS distribution channel.\n\n"
            "Key risks: regulatory timing (PCI-DSS L1 blocks 2 LOIs) and competitive "
            "response from Stripe's SMB embedded finance launch in Q2.\n\n"
            "Next steps: request data room, reference calls with 3 enterprise customers, "
            "and board observer rights as condition of investment."
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
            parts.append(f"{key}:\n{self._RESPONSES.get(agent, 'Analysis complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# VariableToolAdapter — realistic response-size variance (T-028)
# ---------------------------------------------------------------------------

class VariableToolAdapter(ScriptedAdapter):
    """
    Scripted adapter with realistic response-size variance.

    Simulates real-world tool output variance: sometimes a tool call returns
    a brief cached response (short), sometimes a full fresh result (long).

    On each complete() call, responses are randomly selected from short/long
    variants based on ``cache_hit_rate``.  The controller will see genuinely
    varying ``avg_output_tokens`` across runs, making the composition score
    less deterministic — closer to real production behaviour.

    Args:
        cache_hit_rate: Probability (0–1) that a tool call returns the short
                        cached variant.  Default 0.4 (40% cache hit rate).
        seed:           Optional random seed for reproducible runs.
    """

    _SHORT_RESPONSES: dict[str, str] = {
        "MARKET_RESEARCHER": (
            "Market data retrieved from cache. Payments market $2.1T, 12% YoY. "
            "Stripe 42%, Adyen 18%, Square 11%. White space: SMB embedded finance."
        ),
        "TEAM_ANALYST": (
            "Team cached. Founders: 3 prior exits (Plaid, Braintree). CTO 2 patents. "
            "Gap: VP Sales unfilled 6 months. Team score: 8/10."
        ),
        "FINANCIAL_MODELER": (
            "Financials cached. ARR $4.2M +180% YoY. NRR 118%. LTV:CAC 15.4x. "
            "Burn $380k/mo, 18mo runway. Rule of 40: 251."
        ),
        "RISK_ASSESSOR": (
            "Risks cached. Top risk: PCI-DSS L1 certification pending (HIGH/HIGH). "
            "Stripe SMB launch Q2 (MEDIUM/HIGH). CTO key-person (LOW/HIGH). "
            "Composite: MEDIUM."
        ),
        "INVESTMENT_WRITER": (
            "INVESTMENT BRIEF\nRecommendation: CONDITIONAL PASS at $18M pre-money. "
            "Conditions: VP Sales hire + PCI-DSS roadmap. "
            "Next steps: data room, 3 ref calls, board observer."
        ),
    }

    def __init__(self, cache_hit_rate: float = 0.40, seed: int | None = None) -> None:
        super().__init__()
        self._cache_hit_rate = max(0.0, min(1.0, cache_hit_rate))
        self._rng = random.Random(seed)

    def complete(self, messages: list, tools=None) -> str:
        combined = " ".join(
            getattr(m, "content", str(m)) for m in messages
        )
        keys = re.findall(r"([A-Z][A-Z_]+_OUTPUT)", combined)
        seen: set[str] = set()
        parts: list[str] = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:  # type: ignore[func-returns-value]
            agent = key.replace("_OUTPUT", "")
            if self._rng.random() < self._cache_hit_rate:
                response = self._SHORT_RESPONSES.get(agent, "Cached: done.")
            else:
                response = self._RESPONSES.get(agent, "Analysis complete.")
            parts.append(f"{key}:\n{response}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."


# ---------------------------------------------------------------------------
# ErrorInjectionAdapter — injects agent failures for error_rate signal testing (T-028)
# ---------------------------------------------------------------------------

class ErrorInjectionAdapter:
    """
    Wraps any adapter and raises RuntimeError on a fraction of calls.

    Used to verify that the controller's error_rate signal responds correctly
    when agents fail, and that Gate 1 (error_rate_threshold) fires.

    Args:
        inner:       The underlying adapter (ScriptedAdapter, VariableToolAdapter,
                     or a live LLM adapter).
        error_rate:  Probability (0–1) that any given complete() call raises
                     RuntimeError.  Default 0.20 (20% failure rate).
        seed:        Optional random seed for reproducible error injection.
    """

    context_window = 200_000

    def __init__(self, inner, error_rate: float = 0.20, seed: int | None = None) -> None:
        self._inner      = inner
        self._error_rate = max(0.0, min(1.0, error_rate))
        self._rng        = random.Random(seed)
        self._call_count = 0
        self._error_count = 0

    def complete(self, messages: list, tools=None) -> str:
        self._call_count += 1
        if self._rng.random() < self._error_rate:
            self._error_count += 1
            raise RuntimeError(
                f"ErrorInjectionAdapter: simulated agent failure "
                f"(injected_error #{self._error_count})"
            )
        return self._inner.complete(messages, tools=tools)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    @property
    def observed_error_rate(self) -> float:
        """Actual error rate observed so far (for test assertions)."""
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tools() -> tuple[Tool, Tool]:
    """
    Stub tools returning realistic fixed data.

    Using stubs means the eval pipeline runs without external service
    dependencies while still exercising the full tool-call path through the
    adapter and executor.  Both tools are marked independent=True so the
    TOOL_CHAIN optimisation applies.
    """
    web_search = Tool(
        name="web_search",
        description=(
            "Search the web for news, market data, and articles about a company or "
            "market.  Returns titles, URLs, and brief snippets from top results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
        fn=lambda args: {
            "results": [
                {
                    "title":   "Payments infrastructure: market size and growth 2025",
                    "url":     "https://example.com/payments-market-2025",
                    "snippet": (
                        "Global payments infrastructure market reached $2.1T in 2024, "
                        "growing 12% YoY driven by embedded finance and API-first platforms."
                    ),
                },
                {
                    "title":   "SMB embedded finance: competitive landscape Q1 2025",
                    "url":     "https://example.com/smb-embedded-finance",
                    "snippet": (
                        "Stripe dominates with 42% share; emerging players targeting "
                        "vertical SaaS bundles are gaining traction in the SMB segment."
                    ),
                },
                {
                    "title":   "Startup funding trends: fintech Q1 2025",
                    "url":     "https://example.com/fintech-funding-q1",
                    "snippet": (
                        "Fintech Series A deals averaged $12–22M pre-money in Q1 2025; "
                        "investor focus shifting toward profitability timelines."
                    ),
                },
            ],
        },
    )

    competitor_lookup = Tool(
        name="competitor_lookup",
        description=(
            "Look up direct and adjacent competitors for a company in its market segment. "
            "Returns competitor names, market share estimates, and key differentiators."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company to analyse"},
                "segment":      {"type": "string", "description": "Market segment (optional)"},
            },
            "required": ["company_name"],
        },
        fn=lambda args: {
            "direct_competitors": [
                {"name": "Stripe",       "market_share": "42%", "differentiator": "Developer experience, global reach"},
                {"name": "Adyen",        "market_share": "18%", "differentiator": "Enterprise, omnichannel"},
                {"name": "Square",       "market_share": "11%", "differentiator": "SMB hardware + software bundle"},
                {"name": "Checkout.com", "market_share": "7%",  "differentiator": "High-volume enterprise processing"},
            ],
            "adjacent_entrants": [
                {"name": "Shopify Payments",              "threat": "Captures SMB merchants within Shopify ecosystem"},
                {"name": "Stripe Financial Connections",  "threat": "Moving into embedded finance Q2 2025"},
            ],
            "white_space": "Vertical SaaS bundles for SMB — no dominant player below $1M GMV threshold",
        },
    )

    return web_search, competitor_lookup


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

TASK_TEMPLATE = "Conduct due diligence on {company}"

PIPELINE_DESCRIPTION = (
    "research (2 agents, 2 tools) → analysis (2 agents) → synthesis (1 agent)"
)


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
    Build the canonical due diligence pipeline with an optional LLM judge evaluator.

    Args:
        sensitivity:                Controller preset.
        judge_adapter:              LLMAdapter for the judge model.  When None, evaluator is
                                    not created and quality gating is disabled.
        quality_floor:              Minimum quality score (0–1) before COMPOUND is permitted.
                                    Ignored when judge_adapter is None.
        compound_execution_model:   T-038 — "standard" or "two_phase".
        compound_min_output_words:  T-038 — depth hint per phase (None = no hint).
        merged_output_structure:    M-1 — "none"|"budgeted"|"budgeted_adaptive"|"reinforced".
        output_guidance:            O-1 — "none"|"auto"|"concise"|"moderate"|"brief".
        sequential_context_strategy: S-1 — "full"|"predecessor_only".
        cache_aligned_prompts:      C-1 — Anthropic prefix caching restructure.
        escalation_enabled:         E-1 — quality-driven execution model escalation ladder.

    Returns:
        ``(pipeline, evaluator)`` tuple.  ``evaluator`` is None when
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
    # Override policy to include quality_floor and T-038 compound settings
    pipeline._policy = policy
    pipeline._pipeline_state._policy = policy

    return pipeline, evaluator


def build_pipeline(sensitivity: str = "balanced") -> Pipeline:
    """
    Build the canonical due diligence evaluation pipeline.

    The same pipeline is used across all providers so that controller
    behaviour can be directly compared.  Sensitivity can be overridden
    to evaluate how different presets affect the adaptation rate.

    Args:
        sensitivity: One of "conservative", "balanced", "aggressive".
    """
    web_search, competitor_lookup = _tools()

    return (
        Pipeline("startup_due_diligence", sensitivity=sensitivity)
        .group("research")
            .agent(
                "market_researcher",
                "Research the company's market size, growth rate, and competitive "
                "landscape.  Use web_search to find recent market data and "
                "competitor_lookup to map direct and adjacent competitors.  "
                "Synthesise findings into a concise market snapshot with quantified "
                "market share data and white-space analysis.",
                tools=[web_search, competitor_lookup],
            )
            .agent(
                "team_analyst",
                "Assess the founding team: prior exits, domain expertise, key hires, "
                "and any obvious gaps (unfilled roles, key-person concentration).  "
                "Score the team 1–10 with a brief rationale for each dimension.",
            )
        .group("analysis")
            .agent(
                "financial_modeler",
                "Analyse the company's key financial metrics: ARR, YoY growth, net "
                "revenue retention, LTV:CAC ratio, gross margin, burn rate, and runway.  "
                "State the Rule of 40 score and a clear path-to-profitability timeline "
                "with key sensitivities.",
            )
            .agent(
                "risk_assessor",
                "Identify the top 3 risks (regulatory, competitive, operational).  "
                "Rate each on probability (LOW/MEDIUM/HIGH) and impact (LOW/MEDIUM/HIGH).  "
                "Describe the current mitigation status and assign a composite risk score.",
            )
        .group("synthesis")
            .agent(
                "investment_writer",
                "Write a concise investment brief (200–300 words) with: a clear "
                "recommendation (PASS / CONDITIONAL PASS / STRONG PASS), the core "
                "investment thesis in 2–3 sentences, top 2 risks, and 3 concrete "
                "next steps before committing capital.",
            )
    )
