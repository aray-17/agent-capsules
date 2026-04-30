"""
Benchmark 3 — Tool-Augmented Pipeline

Compares a naive multi-agent pipeline (each step is an LLM call, including
steps that only need data retrieval) against a tool-augmented single-agent
pipeline where the LLM invokes tools for data gathering and focuses its
reasoning on synthesis.

Pipeline shapes:
  NAIVE         — 3 agents (searcher, fetcher, summarizer), no tools.
                  Each step makes one LLM call even for deterministic tasks.
                  → 3 LLM calls

  TOOL_AUGMENTED — 1 agent with web_search + web_fetch declared as tools.
                  The LLM calls tools for data retrieval, then synthesizes.
                  → 1 LLM call + 2 tool dispatches

Metrics per configuration:
  total_llm_calls  — number of LLM adapter.complete() invocations
  total_tool_calls — number of tool fn invocations
  total_tokens     — total tokens consumed (from PipelineResult.token_usage)
  total_latency_ms — cumulative wall-clock time

Success criterion:
  TOOL_AUGMENTED.total_llm_calls < NAIVE.total_llm_calls   (fewer LLM calls)
  TOOL_AUGMENTED.total_tool_calls == 2                      (tools dispatched)
  Round-trip reduction ≥ 50%

Note: The prior Benchmark 3 tested ToolLeaf/ToolOrchestrator — internal
pipeline-level tool chains with no LLM involvement (TOOL_ONLY config: 0 LLM
calls). That pattern is an internal implementation detail not exposed through
the public Tool API. This benchmark tests the correct public pattern: agent-
declared Tool objects where the LLM decides when to invoke them.

Usage:
    python -m benchmarks.tool_heavy.benchmark

Design plan ref: §5.2 Phase 4, §6.2 Benchmark 3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from agentic_capsules import Pipeline, Tool
from agentic_capsules.core.types import LLMMessage


# ---------------------------------------------------------------------------
# Scripted LLM adapter — simulates tool invocations when tools= is passed
# ---------------------------------------------------------------------------

class ScriptedToolAdapter:
    """
    Returns deterministic responses and calls tool callables when the agent
    has tools declared (simulating the LLM choosing to call each tool once).
    """
    context_window = 200_000

    def __init__(self, n_tool_calls: int = 0):
        self.call_count       = 0
        self.total_tool_calls = 0
        self._n_tool_calls    = n_tool_calls

    def complete(self, messages: list[LLMMessage], tools=None) -> str:
        self.call_count += 1
        # Simulate tool invocations if the agent has tools and we're told to call them
        if tools and self._n_tool_calls > 0:
            for tool in tools[:self._n_tool_calls]:
                tool.callable({"query": "AI safety"})
                self.total_tool_calls += 1
        # Build response with expected output keys
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        unique = [k for k in keys if not (k in seen or seen.add(k))]
        parts = [
            f"{k}:\nSummary: AI safety research covers alignment, "
            "interpretability, and robustness."
            for k in unique
        ]
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    config:          str
    total_llm_calls: int
    total_tool_calls: int
    total_tokens:    int
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

SUMMARIZER_PROMPT = (
    "Given extracted web content, produce a concise 2-sentence summary "
    "for an executive briefing."
)


def run_naive(query: str) -> BenchmarkResult:
    """3-agent pipeline — all steps via LLM calls, no tools declared."""
    adapter = ScriptedToolAdapter(n_tool_calls=0)
    pipeline = (
        Pipeline("naive")
        .group("pipeline")
            .agent("web_searcher",  "Search for relevant URLs and snippets about the query.")
            .agent("page_fetcher",  "Fetch and extract the main text from the search results.")
            .agent("summarizer",    SUMMARIZER_PROMPT)
    )
    result = pipeline.run(query, adapter=adapter, mode="fine")
    return BenchmarkResult(
        config="NAIVE",
        total_llm_calls=adapter.call_count,
        total_tool_calls=0,
        total_tokens=result.token_usage,
        total_latency_ms=result.latency_ms or 0.0,
    )


def run_tool_augmented(query: str) -> BenchmarkResult:
    """Single agent with web_search + web_fetch tools — 1 LLM call, 2 tool dispatches."""
    adapter = ScriptedToolAdapter(n_tool_calls=2)

    # Tool fns are wired through the public Tool API; the adapter calls
    # tool.callable() which invokes these fns (tracked via adapter.total_tool_calls).
    search = Tool(
        name="web_search",
        description="Search the web for information.",
        input_schema={"query": "str"},
        fn=lambda args: {"url": "https://example.com", "snippet": "AI safety content."},
    )
    fetch = Tool(
        name="web_fetch",
        description="Fetch and extract content from a URL.",
        input_schema={"url": "str"},
        fn=lambda args: {"content": "AI safety: alignment, interpretability, robustness."},
    )
    pipeline = (
        Pipeline("tool_augmented")
        .group("pipeline")
            .agent(
                "analyst",
                "Research the topic using web_search and web_fetch. "
                "Summarise the findings in 2 sentences.",
                tools=[search, fetch],
            )
    )
    result = pipeline.run(query, adapter=adapter, mode="fine")
    return BenchmarkResult(
        config="TOOL_AUGMENTED",
        total_llm_calls=adapter.call_count,
        total_tool_calls=adapter.total_tool_calls,
        total_tokens=result.token_usage,
        total_latency_ms=result.latency_ms or 0.0,
    )


def run_sweep(queries: list[str]) -> list[BenchmarkResult]:
    results = []
    for query in queries:
        results.append(run_naive(query))
        results.append(run_tool_augmented(query))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"\n{'Config':<18} {'LLM Calls':>10} {'Tool Calls':>11} "
        f"{'Tokens':>8} {'Latency(ms)':>12}"
    )
    print("-" * 62)
    for r in results:
        print(
            f"{r.config:<18} {r.total_llm_calls:>10} {r.total_tool_calls:>11} "
            f"{r.total_tokens:>8} {r.total_latency_ms:>12.1f}"
        )

    naive = next((r for r in results if r.config == "NAIVE"),          None)
    tool  = next((r for r in results if r.config == "TOOL_AUGMENTED"), None)
    if naive and tool and naive.total_llm_calls > 0:
        reduction = (naive.total_llm_calls - tool.total_llm_calls) / naive.total_llm_calls
        print(
            f"\nTOOL_AUGMENTED vs NAIVE: "
            f"{reduction:.0%} fewer LLM calls, "
            f"tool_calls={tool.total_tool_calls} dispatched"
        )
        c1 = tool.total_llm_calls < naive.total_llm_calls
        c2 = tool.total_tool_calls == 2
        c3 = reduction >= 0.50
        print(f"\n  {'✓' if c1 else '✗'} Fewer LLM calls: NAIVE={naive.total_llm_calls}, TOOL_AUGMENTED={tool.total_llm_calls}")
        print(f"  {'✓' if c2 else '✗'} Tools dispatched: {tool.total_tool_calls} (expected 2)")
        print(f"  {'✓' if c3 else '✗'} Round-trip reduction ≥ 50%: {reduction:.0%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 3 — Tool-Augmented Pipeline")
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of queries to run (default: 1)"
    )
    args = parser.parse_args()

    queries = [f"AI safety research landscape (query {i})" for i in range(args.n)]
    print(f"Benchmark 3 — Tool-Augmented Pipeline  (OFFLINE)")
    print(f"Queries: {args.n},  NAIVE: 3 agents / TOOL_AUGMENTED: 1 agent + 2 tools")
    results = run_sweep(queries)
    print_results(results)
