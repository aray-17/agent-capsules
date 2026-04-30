"""
Competitive Analysis Pipeline — v2 SDK

Real-world example: three groups, seven agents, three tools.
Models a competitive intelligence workflow used in product strategy teams.

Pipeline:
  gathering group  — company_researcher (web_search, financial_data)
                   → news_monitor (news_search)
  analysis group   — market_analyst → trend_analyzer → risk_assessor
  reporting group  — strategist → report_writer

The controller runs in auto mode. As the pipeline accumulates evidence:
  - gathering group (tool-heavy, high coordination) will switch to COMPOUND first
  - analysis group  (3 dependent agents) likely follows
  - reporting group (2 agents) may stay FINE if overhead stays low

Use observe mode for the first N runs to baseline before letting the
controller switch automatically:
    pipeline.run(topic, adapter=adapter, mode="observe")

Run offline:
    python -m examples.competitive_analysis

Run live:
    ANTHROPIC_API_KEY=... python -m examples.competitive_analysis --live --company "OpenAI"

Run multiple times to watch mode adaptation:
    python -m examples.competitive_analysis --runs 6
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules import Pipeline, Tool, PipelineResult, ControllerPolicy


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    _RESPONSES = {
        "COMPANY_RESEARCHER": (
            "Company profile: Founded 2015, 800 employees, $4B ARR. "
            "Core products: API platform and enterprise AI suite. "
            "Recent funding: $500M Series D at $20B valuation."
        ),
        "NEWS_MONITOR": (
            "Last 30 days: 3 major product announcements, 2 executive hires from FAANG, "
            "1 acquisition (ML tooling startup for $85M). Sentiment: 78% positive."
        ),
        "MARKET_ANALYST": (
            "Market position: #2 in enterprise AI by revenue. "
            "Growing 40% YoY vs. industry avg 28%. "
            "Key differentiator: developer experience and API reliability."
        ),
        "TREND_ANALYZER": (
            "Trend signals: shifting focus from API to vertically integrated products. "
            "Increasing enterprise sales motion. Platform ecosystem investments accelerating."
        ),
        "RISK_ASSESSOR": (
            "Primary risks: regulatory scrutiny in EU (AI Act compliance unclear), "
            "talent concentration (top 5 researchers = 60% of output), "
            "customer concentration (top 10 = 35% of revenue)."
        ),
        "STRATEGIST": (
            "Strategic recommendation: target mid-market segment where competitor "
            "enterprise pricing creates an opening. Accelerate integrations with "
            "existing enterprise software stacks to reduce switching cost."
        ),
        "REPORT_WRITER": (
            "Executive Summary: The target company is a well-capitalised #2 with "
            "strong growth momentum, but faces execution risk in its platform pivot. "
            "Recommended response: differentiate on pricing and ecosystem depth. "
            "Priority actions: (1) mid-market campaign, (2) three new integrations "
            "by Q3, (3) monitor regulatory developments quarterly."
        ),
    }

    def complete(self, messages, tools=None) -> str:
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        parts = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{self._RESPONSES.get(agent, 'Analysis complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tools() -> tuple[Tool, Tool, Tool]:
    web_search = Tool(
        name="web_search",
        description="Search the web for news, articles, and public information about a company.",
        input_schema={"query": "str"},
        fn=lambda args: {"results": [{"title": "Company overview", "snippet": "Key facts..."}]},
    )
    news_search = Tool(
        name="news_search",
        description="Search recent news and press releases from the last 30 days.",
        input_schema={"company": "str", "days": "int"},
        fn=lambda args: {"articles": [{"headline": "Major announcement", "date": "2025-03-15"}]},
    )
    financial_data = Tool(
        name="financial_data",
        description="Retrieve funding rounds, revenue estimates, and valuation data.",
        input_schema={"company": "str"},
        fn=lambda args: {"funding": "$500M Series D", "valuation": "$20B", "arr": "$4B"},
    )
    return web_search, news_search, financial_data


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_pipeline(policy: ControllerPolicy | None = None) -> Pipeline:
    web_search, news_search, financial_data = _tools()

    return (
        Pipeline("competitive_analysis", policy=policy or ControllerPolicy(
            compose_at=0.35, decompose_at=0.12,
            confidence=0.75, min_observations=3, window_size=8,
        ))
        .group("gathering")
            .agent(
                "company_researcher",
                "Research the company's profile, products, financials, and recent strategic moves.",
                tools=[web_search, financial_data],
            )
            .agent(
                "news_monitor",
                "Gather the last 30 days of news, press releases, and social signals.",
                tools=[news_search],
            )
        .group("analysis")
            .agent("market_analyst",  "Analyse market position, revenue growth, and competitive differentiation.")
            .agent("trend_analyzer",  "Identify strategic trends and shifts in the company's product focus.")
            .agent("risk_assessor",   "Assess regulatory, talent, and customer concentration risks.")
        .group("reporting")
            .agent("strategist",    "Formulate a concrete competitive response strategy with prioritised actions.")
            .agent("report_writer", "Write an executive summary suitable for a board presentation.")
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_offline(company: str, n_runs: int = 1) -> None:
    adapter  = ScriptedAdapter()
    pipeline = _build_pipeline()

    print(f"\nCompetitive Analysis Pipeline — offline mode")
    print(f"Company: {company!r}")
    print(f"Groups: gathering (2 agents, 3 tools) → analysis (3 agents) → reporting (2 agents)")
    print(f"Runs: {n_runs}\n")

    for i in range(n_runs):
        mode = "observe" if i < 2 else "auto"   # baseline first 2 runs, then adapt
        result: PipelineResult = pipeline.run(
            f"Analyse {company} as a competitor", adapter=adapter, mode=mode
        )

        print(f"Run {i + 1} (mode={mode}):")
        print(f"  mode_used:      {result.mode_used}")
        print(f"  recommendation: {result.recommendation}")
        print(f"  confidence:     {{{', '.join(f'{k}: {v:.0%}' for k, v in result.confidence.items())}}}")
        print(f"  latency_ms:     {result.latency_ms}")

    print(f"\nFinal output:\n{result.output}")


def run_live(company: str) -> None:
    from agentic_capsules.adapters.anthropic import AnthropicAdapter
    pipeline = _build_pipeline()
    result: PipelineResult = pipeline.run(
        f"Analyse {company} as a competitor",
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
    )
    print(result.output)
    print(f"\nstep_outputs: {list(result.step_outputs.keys())}")
    print(f"mode_used:    {result.mode_used}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitive Analysis Pipeline — v2 SDK")
    parser.add_argument("--company", type=str, default="Acme AI Corp")
    parser.add_argument("--runs",   type=int, default=1)
    parser.add_argument("--live",   action="store_true")
    args = parser.parse_args()

    if args.live:
        run_live(args.company)
    else:
        run_offline(args.company, n_runs=args.runs)
