"""
Benchmark 4 — Adaptive Granularity

Demonstrates confidence-based mode switching using ControllerPolicy and
Pipeline.run() in auto mode. The controller observes a composition score
per run and switches from FINE to COMPOUND once it has sufficient evidence.

Pipeline: Researcher → Fact-checker → Summarizer (same as Benchmark 2)

Offline overhead dynamics (scripted adapter):
  FINE mode:     overhead_ratio ≈ 20%
  COMPOUND mode: overhead_ratio ≈ 64%

Three configurations:
  STATIC_FINE     — always FINE (3 LLM calls/run, ~20% overhead)
  STATIC_COMPOUND — always COMPOUND (1 call/run, ~64% overhead)
  ADAPTIVE        — starts FINE with compose_at=0.15 (below FINE's ~20% score);
                    controller accumulates confidence and switches to COMPOUND.

Sensitivity sweep: varies compose_at × confidence × min_observations and
reports the run index at which the controller first switches.

Success criterion: controller switches within 5 runs for aggressive preset.

Usage:
    python -m benchmarks.adaptive.benchmark
    python -m benchmarks.adaptive.benchmark --sweep
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from agentic_capsules import Pipeline, ControllerPolicy
from agentic_capsules.controller.policy import policy_for
from agentic_capsules.core.types import LLMMessage


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages: list[LLMMessage], tools=None) -> str:
        self.call_count += 1
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        unique = [k for k in keys if not (k in seen or seen.add(k))]
        parts = []
        for key in unique:
            if "RESEARCHER" in key:
                parts.append(f"{key}:\nKey findings. Sources: [1], [2], [3].")
            elif "FACT_CHECKER" in key:
                parts.append(f"{key}:\nVerified: all sources credible.")
            elif "SUMMARIZER" in key:
                parts.append(f"{key}:\nSummary: Topic well-supported.")
            else:
                parts.append(f"{key}:\nOutput.")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = (
    "Given a topic, identify key findings, list 3 relevant sources, "
    "and note any open questions."
)
FACT_CHECKER_PROMPT = "Verify credibility of the sources in the research findings."
SUMMARIZER_PROMPT   = "Produce a concise 2-sentence summary of the verified findings."

GROUP_NAME = "pipeline"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_idx:    int
    mode_used:  str    # "fine" or "compound" this run
    score:      float  # composition score (replaces overhead_ratio)
    llm_calls:  int    # calls made this run
    confidence: float  # controller confidence after this run


@dataclass
class AdaptiveBenchmarkResult:
    config:     str
    runs:       list[RunRecord] = field(default_factory=list)
    switch_run: int | None      = None   # first run where mode changed to compound

    @property
    def switched(self) -> bool:
        return self.switch_run is not None

    @property
    def avg_score(self) -> float:
        return sum(r.score for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def total_llm_calls(self) -> int:
        return sum(r.llm_calls for r in self.runs)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_static(mode: str, config_name: str, n: int) -> AdaptiveBenchmarkResult:
    """Run n iterations locked to a fixed mode (fine or compound)."""
    pipeline = (
        Pipeline("bench_static")
        .group(GROUP_NAME)
            .agent("researcher",   RESEARCHER_PROMPT)
            .agent("fact_checker", FACT_CHECKER_PROMPT)
            .agent("summarizer",   SUMMARIZER_PROMPT)
    )
    result_obj = AdaptiveBenchmarkResult(config=config_name)
    for i in range(n):
        adapter = ScriptedAdapter()
        result  = pipeline.run(f"AI safety run {i}", adapter=adapter, mode=mode)
        result_obj.runs.append(RunRecord(
            run_idx=i,
            mode_used=result.mode_used.get(GROUP_NAME, mode),
            score=result.scores.get(GROUP_NAME, 0.0),
            llm_calls=adapter.call_count,
            confidence=result.confidence.get(GROUP_NAME, 0.0),
        ))
    return result_obj


def run_adaptive(n_runs: int, policy: ControllerPolicy | None = None) -> AdaptiveBenchmarkResult:
    """
    Start FINE. Use compose_at=0.15 so the controller's composition score
    (which includes agent_count and overhead_ratio terms) exceeds the threshold
    and the controller switches to COMPOUND once confidence is met.
    """
    if policy is None:
        policy = ControllerPolicy(
            compose_at=0.15, decompose_at=0.05,
            confidence=0.80, min_observations=3, window_size=5,
        )
    pipeline = (
        Pipeline("bench_adaptive", policy=policy)
        .group(GROUP_NAME)
            .agent("researcher",   RESEARCHER_PROMPT)
            .agent("fact_checker", FACT_CHECKER_PROMPT)
            .agent("summarizer",   SUMMARIZER_PROMPT)
    )
    result_obj  = AdaptiveBenchmarkResult(config="ADAPTIVE")
    prev_mode   = "fine"

    for i in range(n_runs):
        adapter = ScriptedAdapter()
        result  = pipeline.run(f"AI safety run {i}", adapter=adapter, mode="auto")
        mode    = result.mode_used.get(GROUP_NAME, "fine")
        if mode == "compound" and prev_mode == "fine" and result_obj.switch_run is None:
            result_obj.switch_run = i
        prev_mode = mode
        result_obj.runs.append(RunRecord(
            run_idx=i,
            mode_used=mode,
            score=result.scores.get(GROUP_NAME, 0.0),
            llm_calls=adapter.call_count,
            confidence=result.confidence.get(GROUP_NAME, 0.0),
        ))
    return result_obj


# ---------------------------------------------------------------------------
# Sensitivity sweep — varies compose_at × confidence × min_observations
# ---------------------------------------------------------------------------

@dataclass
class SweepEntry:
    compose_at:       float
    confidence:       float
    min_observations: int
    switch_run:       int | None

    @property
    def switched_within_5(self) -> bool:
        return self.switch_run is not None and self.switch_run <= 4


def run_sensitivity_sweep(n_runs: int = 10) -> list[SweepEntry]:
    entries = []
    for compose_at in [0.10, 0.15, 0.20, 0.30, 0.40]:
        for confidence in [0.65, 0.80, 0.90]:
            for min_obs in [2, 3, 5]:
                try:
                    policy = ControllerPolicy(
                        compose_at=compose_at,
                        decompose_at=round(compose_at * 0.4, 2),
                        confidence=confidence,
                        min_observations=min_obs,
                        window_size=max(min_obs, 5),
                    )
                except ValueError:
                    continue
                result = run_adaptive(n_runs, policy=policy)
                entries.append(SweepEntry(
                    compose_at=compose_at,
                    confidence=confidence,
                    min_observations=min_obs,
                    switch_run=result.switch_run,
                ))
    return entries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[AdaptiveBenchmarkResult]) -> None:
    for r in results:
        print(f"\n--- {r.config} ---")
        print(f"{'Run':>4} {'Mode':<10} {'Score':>7} {'Calls':>6} {'Confidence':>11}")
        print("-" * 44)
        for rec in r.runs:
            print(
                f"{rec.run_idx:>4} {rec.mode_used:<10} {rec.score:>7.3f} "
                f"{rec.llm_calls:>6} {rec.confidence:>10.2f}"
            )
        print(f"  avg score: {r.avg_score:.3f} | total LLM calls: {r.total_llm_calls}")
        if r.switched:
            within = r.switch_run <= 4
            print(
                f"  {'✓' if within else '~'} Switched at run {r.switch_run} "
                f"({'within' if within else 'after'} 5-run target)"
            )
        elif r.config not in ("STATIC_FINE", "STATIC_COMPOUND"):
            print(f"  ✗ Did not switch within {len(r.runs)} runs")


def print_sweep(entries: list[SweepEntry]) -> None:
    print(f"\n--- Sensitivity Sweep ---")
    print(f"{'compose_at':>11} {'confidence':>11} {'min_obs':>8} {'switch_run':>11} {'≤5?':>5}")
    print("-" * 50)
    for e in entries:
        run  = str(e.switch_run) if e.switch_run is not None else "—"
        mark = "✓" if e.switched_within_5 else ("—" if e.switch_run is None else "✗")
        print(f"{e.compose_at:>10.0%} {e.confidence:>10.0%} {e.min_observations:>8} {run:>11} {mark:>5}")


def summarize(results: list[AdaptiveBenchmarkResult]) -> None:
    fine     = next((r for r in results if r.config == "STATIC_FINE"),     None)
    compound = next((r for r in results if r.config == "STATIC_COMPOUND"), None)
    adaptive = next((r for r in results if r.config == "ADAPTIVE"),        None)

    print("\n=== Summary ===")
    print(f"{'Config':<18} {'Avg Score':>10} {'Total Calls':>12} {'Switched':>10}")
    print("-" * 54)
    for r in [r for r in [fine, compound, adaptive] if r]:
        sw = f"run {r.switch_run}" if r.switched else ("N/A" if r.config.startswith("STATIC") else "no")
        print(f"{r.config:<18} {r.avg_score:>10.3f} {r.total_llm_calls:>12} {sw:>10}")

    if adaptive:
        criterion = adaptive.switched and adaptive.switch_run <= 4
        print(
            f"\nCriterion (switch ≤5 runs): "
            + ("✓ PASS" if criterion else "✗ FAIL")
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 4 — Adaptive Granularity")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="Include sensitivity sweep")
    args = parser.parse_args()

    n = args.runs
    print(f"Benchmark 4 — Adaptive Granularity")
    print(f"Pipeline: Researcher → Fact-checker → Summarizer  ({n} runs)\n")

    results = [
        run_static("fine",     "STATIC_FINE",     n),
        run_static("compound", "STATIC_COMPOUND", n),
        run_adaptive(n),
    ]
    print_results(results)
    summarize(results)

    if args.sweep:
        entries = run_sensitivity_sweep(n_runs=n)
        print_sweep(entries)
