"""
Benchmark 2 — Research Pipeline (Computation Space Composition)

Measures the overhead-reduction benefit of merging tightly-coupled agents
into a single compound LLM call (computation-space composition) vs. running
each agent as a separate call (fine-grained baseline).

Pipeline: Researcher → Fact-checker → Summarizer  (3 agents, sequential)

Two configurations:
  FINE       — 3 separate LLM calls (baseline)
  MONOLITHIC — All 3 merged into one compound call (1 call)

Note: A PARTIAL configuration (some groups compound, some fine) is not
expressible through the public Pipeline API, which applies mode= uniformly
across all groups. Use the adaptive controller (Benchmark 4) for per-group
mode control.

Metrics per configuration:
  total_calls    — number of LLM calls made
  total_tokens   — total tokens consumed (from PipelineResult.token_usage)
  total_latency_ms — cumulative wall-clock time

Usage:
    # Offline (no API calls — uses ScriptedAdapter):
    python -m benchmarks.research_pipeline.benchmark

    # Live (real API calls — requires ANTHROPIC_API_KEY):
    python -m benchmarks.research_pipeline.benchmark --live

Design plan ref: §5.2 Phase 3, §6.2 Benchmark 2
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from agentic_capsules import Pipeline
from agentic_capsules.core.types import LLMMessage


# ---------------------------------------------------------------------------
# Synthetic topics corpus
# ---------------------------------------------------------------------------

def make_topics(n: int) -> list[str]:
    """Generate N synthetic research topic strings."""
    domains = [
        "the impact of large language models on scientific publishing",
        "renewable energy storage technologies and grid integration",
        "CRISPR gene editing in agricultural applications",
        "central bank digital currencies and monetary policy",
        "autonomous vehicle safety certification frameworks",
    ]
    return [domains[i % len(domains)] + f" (variant {i})" for i in range(n)]


# ---------------------------------------------------------------------------
# Scripted adapter (offline mode)
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    """
    Offline adapter that returns deterministic responses with the correct
    output headings so the parser works without API calls.
    """
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages: list[LLMMessage], tools=None) -> str:
        self.call_count += 1
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        unique_keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not unique_keys:
            return "OUTPUT:\nDone."
        parts = []
        for key in unique_keys:
            if "RESEARCHER" in key:
                parts.append(f"{key}:\nKey findings: topic is well-documented. Sources: [1], [2], [3].")
            elif "FACT_CHECKER" in key:
                parts.append(f"{key}:\nVerified: all 3 sources are credible. No contradictions found.")
            elif "SUMMARIZER" in key:
                parts.append(f"{key}:\nSummary: The topic has strong evidence with verified sources.")
            else:
                parts.append(f"{key}:\nOutput for {key}.")
        return "\n\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = (
    "Given a topic, identify key findings, list 3 relevant sources, "
    "and note any open questions."
)
FACT_CHECKER_PROMPT = (
    "Given research findings and sources, verify credibility of each source "
    "and flag any contradictions."
)
SUMMARIZER_PROMPT = (
    "Given verified research findings, produce a concise 2-sentence summary "
    "suitable for an executive briefing."
)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _make_pipeline(name: str) -> Pipeline:
    return (
        Pipeline(name)
        .group("pipeline")
            .agent("researcher",   RESEARCHER_PROMPT)
            .agent("fact_checker", FACT_CHECKER_PROMPT)
            .agent("summarizer",   SUMMARIZER_PROMPT)
    )


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    config:          str
    total_calls:     int
    total_tokens:    int
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Single run per configuration
# ---------------------------------------------------------------------------

def run_fine(topic: str, adapter: ScriptedAdapter) -> BenchmarkResult:
    adapter.call_count = 0
    pipeline = _make_pipeline("fine")
    result = pipeline.run(topic, adapter=adapter, mode="fine")
    return BenchmarkResult(
        config="FINE",
        total_calls=adapter.call_count,
        total_tokens=result.token_usage,
        total_latency_ms=result.latency_ms or 0.0,
    )


def run_monolithic(topic: str, adapter: ScriptedAdapter) -> BenchmarkResult:
    adapter.call_count = 0
    pipeline = _make_pipeline("compound")
    result = pipeline.run(topic, adapter=adapter, mode="compound")
    return BenchmarkResult(
        config="MONOLITHIC",
        total_calls=adapter.call_count,
        total_tokens=result.token_usage,
        total_latency_ms=result.latency_ms or 0.0,
    )


def run_sweep(topics: list[str], live: bool = False) -> list[BenchmarkResult]:
    results = []
    for topic in topics:
        if live:
            from agentic_capsules.adapters.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter()
        else:
            adapter = ScriptedAdapter()
        results.append(run_fine(topic, adapter))
        results.append(run_monolithic(topic, adapter))
    return results


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"\n{'Config':<12} {'Calls':>6} {'Tokens':>10} {'Latency(ms)':>12}"
    )
    print("-" * 44)
    for r in results:
        print(
            f"{r.config:<12} {r.total_calls:>6} {r.total_tokens:>10} "
            f"{r.total_latency_ms:>12.1f}"
        )

    # Highlight savings
    fine = next((r for r in results if r.config == "FINE"), None)
    mono = next((r for r in results if r.config == "MONOLITHIC"), None)
    if fine and mono and fine.total_calls > 0:
        call_reduction  = (fine.total_calls - mono.total_calls) / fine.total_calls
        token_reduction = (
            (fine.total_tokens - mono.total_tokens) / fine.total_tokens
            if fine.total_tokens > 0 else 0.0
        )
        print(
            f"\nMONOLITHIC vs FINE: "
            f"{call_reduction:.0%} fewer LLM calls, "
            f"{token_reduction:.0%} token reduction"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 2 — Research Pipeline")
    parser.add_argument("--live", action="store_true", help="Use real API calls")
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of topics to run (default: 1)"
    )
    args = parser.parse_args()

    topics = make_topics(args.n)
    print(f"Benchmark 2 — Research Pipeline  ({'LIVE' if args.live else 'OFFLINE'})")
    print(f"Topics: {args.n}, Pipeline: Researcher → Fact-checker → Summarizer")
    results = run_sweep(topics, live=args.live)
    print_results(results)
