"""DSPy implementation of the startup_due_diligence pipeline (T-059 Phase 0).

Mirrors `evals/shared/pipeline.py::build_pipeline()` structurally:
  research  (2 agents, 2 tools) — market_researcher (ReAct) → team_analyst (CoT)
  analysis  (2 agents, no tools) — financial_modeler → risk_assessor
  synthesis (1 agent, no tools)  — investment_writer

Agent system prompts in AC become Signature docstrings in DSPy. Tool stubs
match AC's deterministic tool returns exactly so the H2H comparison isolates
orchestration/prompting overhead rather than tool-call variance.

Two callable entry points:
  - `DueDiligenceUncompiled()` — pure orchestration baseline, no optimization
  - `DueDiligenceCompiled(compiled_prompts_path)` — loads MIPRO-compiled prompts

The single-shot call site is `.forward(task=str) -> dspy.Prediction` with
one output field per AC agent (market_snapshot, team_assessment,
financial_model, risk_report, investment_brief).
"""

from __future__ import annotations

import dspy


# ---------------------------------------------------------------------------
# Tool stubs — copied verbatim from evals/shared/pipeline.py::_tools()
# ---------------------------------------------------------------------------

def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web for news, market data, and articles about a company or
    market. Returns titles, URLs, and brief snippets from top results."""
    return {
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
                    "Fintech Series A deals averaged $12-22M pre-money in Q1 2025; "
                    "investor focus shifting toward profitability timelines."
                ),
            },
        ],
    }


def competitor_lookup(company_name: str, segment: str = "") -> dict:
    """Look up direct and adjacent competitors for a company in its market
    segment. Returns competitor names, market share estimates, and key
    differentiators."""
    return {
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
    }


# ---------------------------------------------------------------------------
# Signatures — docstrings mirror AC agent system_prompts verbatim
# ---------------------------------------------------------------------------

class MarketResearch(dspy.Signature):
    """Research the company's market size, growth rate, and competitive
    landscape.  Use web_search to find recent market data and
    competitor_lookup to map direct and adjacent competitors.
    Synthesise findings into a concise market snapshot with quantified
    market share data and white-space analysis."""

    task: str = dspy.InputField(desc="Due-diligence task, e.g. 'Conduct due diligence on Acme Fintech Inc.'")
    market_snapshot: str = dspy.OutputField(desc="Concise market snapshot with quantified share + white-space analysis")


class TeamAnalysis(dspy.Signature):
    """Assess the founding team: prior exits, domain expertise, key hires,
    and any obvious gaps (unfilled roles, key-person concentration).
    Score the team 1-10 with a brief rationale for each dimension."""

    task: str = dspy.InputField()
    team_assessment: str = dspy.OutputField(desc="Team scored 1-10 with rationale per dimension")


class FinancialModeling(dspy.Signature):
    """Analyse the company's key financial metrics: ARR, YoY growth, net
    revenue retention, LTV:CAC ratio, gross margin, burn rate, and runway.
    State the Rule of 40 score and a clear path-to-profitability timeline
    with key sensitivities."""

    task: str = dspy.InputField()
    market_snapshot: str = dspy.InputField(desc="Market snapshot from the research group")
    team_assessment: str = dspy.InputField(desc="Team assessment from the research group")
    financial_model: str = dspy.OutputField(desc="Financial metrics + Rule of 40 + path to profitability")


class RiskAssessment(dspy.Signature):
    """Identify the top 3 risks (regulatory, competitive, operational).
    Rate each on probability (LOW/MEDIUM/HIGH) and impact (LOW/MEDIUM/HIGH).
    Describe the current mitigation status and assign a composite risk score."""

    task: str = dspy.InputField()
    market_snapshot: str = dspy.InputField()
    team_assessment: str = dspy.InputField()
    financial_model: str = dspy.InputField()
    risk_report: str = dspy.OutputField(desc="Top 3 risks with probability/impact/mitigation ratings")


class InvestmentWriting(dspy.Signature):
    """Write a concise investment brief (200-300 words) with: a clear
    recommendation (PASS / CONDITIONAL PASS / STRONG PASS), the core
    investment thesis in 2-3 sentences, top 2 risks, and 3 concrete
    next steps before committing capital."""

    task: str = dspy.InputField()
    market_snapshot: str = dspy.InputField()
    team_assessment: str = dspy.InputField()
    financial_model: str = dspy.InputField()
    risk_report: str = dspy.InputField()
    investment_brief: str = dspy.OutputField(desc="200-300 word investment brief with recommendation + thesis")


# ---------------------------------------------------------------------------
# Module — pipeline orchestration
# ---------------------------------------------------------------------------

class DueDiligenceUncompiled(dspy.Module):
    """Uncompiled baseline: structurally mirrors AC's due_diligence.

    ReAct for the tool-using market_researcher; ChainOfThought for all
    reasoning-only agents. No MIPRO/bootstrap optimization — this is the
    fair orchestration-only baseline the T-059 parity program targets
    with AC input ≤ 1.10× of.
    """

    def __init__(self):
        super().__init__()
        self.market_researcher = dspy.ReAct(
            MarketResearch,
            tools=[web_search, competitor_lookup],
        )
        self.team_analyst = dspy.ChainOfThought(TeamAnalysis)
        self.financial_modeler = dspy.ChainOfThought(FinancialModeling)
        self.risk_assessor = dspy.ChainOfThought(RiskAssessment)
        self.investment_writer = dspy.ChainOfThought(InvestmentWriting)

    def forward(self, task: str) -> dspy.Prediction:
        market = self.market_researcher(task=task)
        team = self.team_analyst(task=task)
        financial = self.financial_modeler(
            task=task,
            market_snapshot=market.market_snapshot,
            team_assessment=team.team_assessment,
        )
        risk = self.risk_assessor(
            task=task,
            market_snapshot=market.market_snapshot,
            team_assessment=team.team_assessment,
            financial_model=financial.financial_model,
        )
        brief = self.investment_writer(
            task=task,
            market_snapshot=market.market_snapshot,
            team_assessment=team.team_assessment,
            financial_model=financial.financial_model,
            risk_report=risk.risk_report,
        )
        return dspy.Prediction(
            market_snapshot=market.market_snapshot,
            team_assessment=team.team_assessment,
            financial_model=financial.financial_model,
            risk_report=risk.risk_report,
            investment_brief=brief.investment_brief,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_pipeline(compiled_path: str | None = None) -> dspy.Module:
    """Return a DSPy pipeline.

    Args:
        compiled_path: If given, load a MIPRO-compiled pipeline from disk
          (Phase 0 Day 2 output). If None, return the uncompiled baseline.
    """
    if compiled_path is None:
        return DueDiligenceUncompiled()
    pipeline = DueDiligenceUncompiled()
    pipeline.load(compiled_path)
    return pipeline
