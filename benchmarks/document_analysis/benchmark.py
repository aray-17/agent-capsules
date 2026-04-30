"""
Benchmark 1 — Document Analysis (Iteration Space Composition)

Measures the overhead-reduction benefit of batching multiple documents
into a single LLM call (iteration-space composition) vs. one call per
document (fine-grained baseline).

Sweep: K (batch size) = 1, 5, 10, 25, 50  ×  N (corpus size) = 10, 50

Metrics per (N, K) configuration (design plan §6.1):
  total_tokens     — sum of estimated_tokens across all capsule calls
  total_calls      — number of LLM calls made
  avg_overhead_ratio — coordination tokens / total tokens (lower = better)
  total_latency_ms   — cumulative wall-clock time

Usage:
    # Offline (no API calls — uses ScriptedAdapter):
    python -m benchmarks.document_analysis.benchmark

    # Live (real API calls — requires ANTHROPIC_API_KEY):
    python -m benchmarks.document_analysis.benchmark --live

Note on internal API usage:
  This benchmark uses internal types (AgentStepCapsule, CapsuleHierarchy,
  CapsuleExecutor, TagSpace) because it tests CompositionLevel.ITERATION —
  iteration-space composition that batches multiple documents into one LLM
  call. The public Pipeline API does not expose iteration-space composition
  (Pipeline.run() takes a single task string). Internal APIs are the only
  way to exercise this execution path.

Design plan ref: §5.2 Phase 2, §6.2 Benchmark 1
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.tag import TagDimension, TagSpace
from agentic_capsules.core.types import CompositionLevel, LLMMessage, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.controller.telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

def make_corpus(n: int) -> list[str]:
    """
    Generate N synthetic document snippets.
    In a real benchmark these would be CNN/DailyMail or arXiv abstracts.
    """
    templates = [
        "Scientists at {inst} have published findings on {topic}. "
        "The study involved {n} participants over {y} years and concluded that {finding}.",
        "The government announced new policies regarding {topic}. "
        "Officials stated that {claim}. Critics argue that {counter}.",
        "Tech company {co} released {product} yesterday. "
        "The device features {spec} and is priced at ${price}. Analysts expect {forecast}.",
        "A new report from {org} highlights concerns about {topic}. "
        "Key findings include {f1}, {f2}, and {f3}.",
    ]
    corpus = []
    for i in range(n):
        t = templates[i % len(templates)]
        doc = t.format(
            inst=f"University-{i}", topic=f"topic-{i}", n=100 + i,
            y=3 + (i % 5), finding=f"finding-{i}", claim=f"claim-{i}",
            counter=f"counter-{i}", co=f"Corp-{i}", product=f"Product-{i}",
            spec=f"spec-{i}", price=100 * (i + 1), forecast=f"forecast-{i}",
            org=f"Org-{i}", f1=f"fact1-{i}", f2=f"fact2-{i}", f3=f"fact3-{i}",
        )
        corpus.append(doc)
    return corpus


# ---------------------------------------------------------------------------
# Scripted adapter (offline mode)
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    """
    Offline adapter that returns deterministic responses without API calls.
    Responses include the expected ITEM_N_OUTPUT headings so the parser works.
    """
    context_window = 200_000
    call_count = 0

    def complete(self, messages: list[LLMMessage]) -> str:
        self.call_count += 1
        # Count how many ITEM_N_OUTPUT headings are expected
        user_content = messages[-1].content
        import re
        keys = re.findall(r"ITEM_(\d+)_OUTPUT", user_content)
        if not keys:
            return "ANALYST_OUTPUT:\nAnalysis complete."
        parts = []
        for n in keys:
            parts.append(
                f"ITEM_{n}_OUTPUT:\n"
                f"Sentiment: neutral. Entities: [entity-{n}]. "
                f"Summary: Document {n} discusses the topic briefly."
            )
        return "\n\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    n_docs: int
    batch_size: int
    total_calls: int
    total_tokens: int
    total_coordination_tokens: int
    avg_overhead_ratio: float
    total_latency_ms: float

    @property
    def efficiency_gain(self) -> float:
        """Calls saved vs. K=1 baseline (set externally after all runs)."""
        return getattr(self, "_efficiency_gain", 0.0)


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    corpus: list[str],
    batch_size: int,
    adapter,
    reset_adapter_count: bool = True,
) -> BenchmarkResult:
    """Run one (N, K) configuration and return its BenchmarkResult."""

    n = len(corpus)
    if reset_adapter_count and hasattr(adapter, "call_count"):
        adapter.call_count = 0

    # Build the analysis agent
    analyst = AgentStepCapsule(
        name="analyst",
        system_prompt=(
            "You are a document analyst. For each document provided, produce:\n"
            "1. Sentiment (positive / neutral / negative)\n"
            "2. Key entities (comma-separated list)\n"
            "3. One-sentence summary\n"
        ),
        input_schema=Schema("document", fields={"text": "str"}),
        output_schema=Schema("analysis", fields={
            "sentiment": "str", "entities": "str", "summary": "str"
        }),
    )
    leaf = AgentLeaf(capsule=analyst)
    root = CompoundCapsule(name="pipeline", children=[leaf], dependency_edges={})
    tag_space = TagSpace(
        agent_name="analyst",
        dimensions=[TagDimension("doc_id", list(range(n)))],
    )
    hierarchy = CapsuleHierarchy(name="doc_analysis", root=root, tag_space=tag_space)

    telemetry = TelemetryCollector()
    executor = CapsuleExecutor(
        adapter=adapter,
        composition_level=CompositionLevel.ITERATION,
        telemetry=telemetry,
        batch_size=batch_size,
    )

    # Run — pass full corpus so each item receives its own document text (T-002).
    executor.run(hierarchy, task_input=corpus[0], task_inputs=corpus, task_id="bench")

    summary = telemetry.summary()
    total_calls = getattr(adapter, "call_count", len(telemetry.records))

    return BenchmarkResult(
        n_docs=n,
        batch_size=batch_size,
        total_calls=total_calls,
        total_tokens=summary.get("total_tokens", 0),
        total_coordination_tokens=summary.get("total_coordination_tokens", 0),
        avg_overhead_ratio=summary.get("avg_overhead_ratio", 0.0),
        total_latency_ms=sum(r.latency_ms for r in telemetry.records),
    )


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(
    n_values: list[int],
    k_values: list[int],
    live: bool = False,
) -> list[BenchmarkResult]:
    results = []
    for n in n_values:
        corpus = make_corpus(n)
        for k in k_values:
            if live:
                from agentic_capsules.adapters.anthropic import AnthropicAdapter
                adapter = AnthropicAdapter()
            else:
                adapter = ScriptedAdapter()
            result = run_benchmark(corpus, batch_size=k, adapter=adapter)
            results.append(result)
    return results


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"\n{'N':>6} {'K':>6} {'Calls':>8} {'Tokens':>10} "
        f"{'Coord':>10} {'Overhead':>10} {'Latency(ms)':>12}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r.n_docs:>6} {r.batch_size:>6} {r.total_calls:>8} "
            f"{r.total_tokens:>10} {r.total_coordination_tokens:>10} "
            f"{r.avg_overhead_ratio:>9.1%} {r.total_latency_ms:>12.1f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 1 — Document Analysis")
    parser.add_argument("--live", action="store_true", help="Use real API calls")
    parser.add_argument(
        "--n", nargs="+", type=int, default=[10, 50],
        help="Corpus sizes to sweep (default: 10 50)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 5, 10, 25, 50],
        help="Batch sizes to sweep (default: 1 5 10 25 50)",
    )
    args = parser.parse_args()

    print(f"Benchmark 1 — Document Analysis  ({'LIVE' if args.live else 'OFFLINE'})")
    results = run_sweep(args.n, args.k, live=args.live)
    print_results(results)
