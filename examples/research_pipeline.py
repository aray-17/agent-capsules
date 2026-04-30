"""
Research Pipeline — v2 SDK

Entry-level example: two groups, four agents, one tool.
Demonstrates the full v2 Pipeline API including mode adaptation over
multiple runs.

Pipeline:
  research group — web_searcher (uses web_search tool) → fact_checker
  writing group  — analyst → writer

Groups run sequentially; the controller tracks overhead per group and
auto-switches to COMPOUND when confidence is sufficient.

Run offline (no API key):
    python -m examples.research_pipeline

Run live against Anthropic:
    ANTHROPIC_API_KEY=... python -m examples.research_pipeline --live

Run multiple times to see mode adaptation:
    python -m examples.research_pipeline --runs 5
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules import Pipeline, Tool, PipelineResult


# ---------------------------------------------------------------------------
# Scripted adapter — runs offline, no API key needed
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def complete(self, messages, tools=None) -> str:
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        responses = {
            "WEB_SEARCHER": "Found 5 recent articles on the topic. Key sources: Nature (2024), arXiv (2025), MIT Tech Review.",
            "FACT_CHECKER": "Verified: all 5 sources are credible. No contradictions found across sources.",
            "ANALYST":      "Key insight: the topic has accelerating research momentum with 3 major open problems.",
            "WRITER":       "Summary: This topic is at a pivotal moment. Recent research confirms strong evidence across multiple disciplines, with consensus forming around the core hypothesis.",
        }
        parts = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{responses.get(agent, 'Analysis complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _web_search_tool() -> Tool:
    return Tool(
        name="web_search",
        description="Search the web for recent articles and research on a topic.",
        input_schema={"query": "str"},
        fn=lambda args: {
            "results": [
                {"title": "Recent advances in the field", "url": "https://example.com/1", "snippet": "Key findings..."},
                {"title": "Critical analysis", "url": "https://example.com/2", "snippet": "Expert perspectives..."},
            ]
        },
    )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

def _build_pipeline(sensitivity: str = "balanced") -> Pipeline:
    web_search = _web_search_tool()

    return (
        Pipeline("research", sensitivity=sensitivity)
        .group("research")
            .agent("web_searcher", "Search for recent, credible sources on the topic.", tools=[web_search])
            .agent("fact_checker", "Verify the credibility of each source and flag any contradictions.")
        .group("writing")
            .agent("analyst", "Identify the three most important insights from the verified research.")
            .agent("writer",  "Write a clear, well-structured 150-word summary for a general audience.")
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_offline(topic: str, n_runs: int = 1) -> None:
    adapter  = ScriptedAdapter()
    pipeline = _build_pipeline(sensitivity="aggressive")   # faster convergence for demo

    print(f"\nResearch Pipeline — offline mode")
    print(f"Topic: {topic!r}")
    print(f"Groups: research (web_searcher → fact_checker), writing (analyst → writer)")
    print(f"Runs: {n_runs}\n")

    for i in range(n_runs):
        result: PipelineResult = pipeline.run(topic, adapter=adapter)

        print(f"Run {i + 1}:")
        print(f"  mode_used:      {result.mode_used}")
        conf_str = {k: f"{v:.0%}" for k, v in result.confidence.items()}
        print(f"  confidence:     {conf_str}")
        print(f"  recommendation: {result.recommendation}")
        print(f"  token_usage:    {result.token_usage}")
        if i == n_runs - 1:
            print(f"\nFinal output:\n{result.output}")


def run_live(topic: str) -> None:
    from agentic_capsules.adapters.anthropic import AnthropicAdapter
    adapter  = AnthropicAdapter(model="claude-sonnet-4-6")
    pipeline = _build_pipeline(sensitivity="balanced")

    print(f"\nResearch Pipeline — live mode (Anthropic)")
    print(f"Topic: {topic!r}\n")

    result: PipelineResult = pipeline.run(topic, adapter=adapter)
    print(result.output)
    print(f"\nmode_used:  {result.mode_used}")
    print(f"confidence: {result.confidence}")
    print(f"tokens:     {result.token_usage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Pipeline — v2 SDK example")
    parser.add_argument("--topic", type=str, default="AI safety challenges at scale")
    parser.add_argument("--runs",  type=int, default=1,
                        help="Number of runs (>1 shows mode adaptation)")
    parser.add_argument("--live",  action="store_true",
                        help="Use real Anthropic API (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    if args.live:
        run_live(args.topic)
    else:
        run_offline(args.topic, n_runs=args.runs)
