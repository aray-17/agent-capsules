"""
Eval result data types and reporting helpers.

EvalResult captures structured data from multiple pipeline runs and provides
aggregation, comparison, and calibration reporting utilities.

Data hierarchy:
  EvalResult
    └── EvalRun (one per pipeline.run() call)
          └── RunRecord (one per group per run)

Signal data (when available from last_signal):
  RunRecord.signal → SignalSnapshot — the individual component values that
  produced the composition score.  Used by tuning.py for weight calibration.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_capsules.api.state import CompositionSignal
    from agentic_capsules.controller.policy import ControllerPolicy


# ---------------------------------------------------------------------------
# SignalSnapshot — individual signal values for one group run
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    """
    The raw signal values that produced a composition score.

    Sourced from GroupControllerState.last_signal (transient, in-process only).
    Used by tuning reports to show which signals are carrying the score and
    suggest weight adjustments.
    """
    overhead_ratio:        float
    agent_count:           int
    avg_output_tokens:     float
    tool_calls_per_agent:  float
    dependency_depth:      int
    error_rate:            float
    context_utilization:   float

    @classmethod
    def from_composition_signal(cls, sig: "CompositionSignal") -> "SignalSnapshot":
        return cls(
            overhead_ratio=sig.overhead_ratio,
            agent_count=sig.agent_count,
            avg_output_tokens=sig.avg_output_tokens,
            tool_calls_per_agent=sig.tool_calls_per_agent,
            dependency_depth=sig.dependency_depth,
            error_rate=sig.error_rate,
            context_utilization=sig.context_utilization,
        )


# ---------------------------------------------------------------------------
# RunRecord — per-group result for one pipeline.run() call
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Result for one group within one pipeline run."""
    run_index:       int
    group:           str
    mode_used:       str    # "fine" | "compound"
    confidence:      float  # 0.0–1.0
    recommendation:  str    # "COMPOSE" | "DECOMPOSE" | "MAINTAIN"
    score:           float  # multi-signal composition score (0.0–1.0); 0 if not available
    signal:          SignalSnapshot | None = None  # None when last_signal not available
    quality_score:   float | None          = None  # Phase 12: rolling mean quality (0–1)
    quality_details: dict                  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvalRun — all groups for one pipeline.run() call
# ---------------------------------------------------------------------------

@dataclass
class EvalRun:
    """Aggregated result of a single pipeline.run() call."""
    run_index:   int
    token_usage: int
    latency_ms:  float
    records:     list[RunRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EvalResult — multiple EvalRuns for one (provider, model, sensitivity) combo
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Aggregated result of N pipeline.run() calls for one eval configuration.

    Captures enough data to:
      - Assess whether the controller adapted within N runs
      - Identify which groups switched and when
      - Compute score distributions for weight calibration
      - Compare across providers and sensitivity presets
    """
    provider:    str   # "openai" | "anthropic" | "scripted"
    model:       str   # e.g. "gpt-4o-mini", "claude-haiku-4-5-20251001"
    sensitivity: str   # "conservative" | "balanced" | "aggressive"
    company:     str
    timestamp:   str   # ISO-8601
    policy:      "ControllerPolicy | None"
    runs:        list[EvalRun] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def groups(self) -> list[str]:
        """Return group names in pipeline order."""
        if not self.runs:
            return []
        return [r.group for r in self.runs[0].records]

    def records_for(self, group: str) -> list[RunRecord]:
        """Return all RunRecords for a given group, in run order."""
        return [
            rec
            for run in self.runs
            for rec in run.records
            if rec.group == group
        ]

    def mean_score(self, group: str) -> float:
        scores = [r.score for r in self.records_for(group)]
        return statistics.mean(scores) if scores else 0.0

    def score_std(self, group: str) -> float:
        scores = [r.score for r in self.records_for(group)]
        return statistics.stdev(scores) if len(scores) > 1 else 0.0

    def mean_confidence(self, group: str) -> float:
        confs = [r.confidence for r in self.records_for(group)]
        return statistics.mean(confs) if confs else 0.0

    def switch_count(self, group: str) -> int:
        """Number of runs where mode_used == 'compound'."""
        return sum(1 for r in self.records_for(group) if r.mode_used == "compound")

    def first_switch_run(self, group: str) -> int | None:
        """1-based run index of the first run where mode_used == 'compound'."""
        for rec in self.records_for(group):
            if rec.mode_used == "compound":
                return rec.run_index + 1
        return None

    def last_signal(self, group: str) -> SignalSnapshot | None:
        """Signal values from the most recent run for a group."""
        recs = self.records_for(group)
        return recs[-1].signal if recs else None

    def mean_quality(self, group: str) -> float | None:
        """Mean rolling quality score for a group.  None when no evaluator was used."""
        scores = [r.quality_score for r in self.records_for(group) if r.quality_score is not None]
        return statistics.mean(scores) if scores else None

    def quality_std(self, group: str) -> float | None:
        """Std dev of rolling quality scores for a group."""
        scores = [r.quality_score for r in self.records_for(group) if r.quality_score is not None]
        return statistics.stdev(scores) if len(scores) > 1 else None

    def quality_vs_score(self, group: str) -> list[tuple[float, float]]:
        """(composition_score, quality_score) pairs for Pareto analysis."""
        return [
            (r.score, r.quality_score)
            for r in self.records_for(group)
            if r.quality_score is not None
        ]


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def collect_run(
    run_index: int,
    result,          # PipelineResult
    pipeline,        # Pipeline — to access _pipeline_state.snapshot()
) -> EvalRun:
    """
    Build an EvalRun from a PipelineResult and the pipeline's live state.

    Accesses pipeline._pipeline_state.snapshot() to retrieve last_signal
    (transient — only available within the same process as the run).
    """
    snapshot = pipeline._pipeline_state.snapshot()
    records: list[RunRecord] = []

    for group in result.mode_used:
        gs  = snapshot.get(group)
        sig = None
        if gs is not None and gs.last_signal is not None:
            sig = SignalSnapshot.from_composition_signal(gs.last_signal)

        # Phase 12: quality fields from PipelineResult
        quality_score   = result.quality.get(group)        if hasattr(result, "quality") else None
        quality_details = result.quality_details.get(group, {}) if hasattr(result, "quality_details") else {}

        records.append(RunRecord(
            run_index=run_index,
            group=group,
            mode_used=result.mode_used.get(group, "fine"),
            confidence=result.confidence.get(group, 0.0),
            recommendation=result.recommendation.get(group, "MAINTAIN"),
            score=result.scores.get(group, 0.0),
            signal=sig,
            quality_score=quality_score,
            quality_details=quality_details,
        ))

    return EvalRun(
        run_index=run_index,
        token_usage=result.token_usage,
        latency_ms=result.latency_ms or 0.0,
        records=records,
    )


def make_eval_result(
    provider: str,
    model: str,
    sensitivity: str,
    company: str,
    policy,
) -> EvalResult:
    return EvalResult(
        provider=provider,
        model=model,
        sensitivity=sensitivity,
        company=company,
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _score_colour(score: float, compose_at: float) -> str:
    if score >= compose_at:
        return _GREEN
    if score >= compose_at * 0.85:
        return _YELLOW
    return ""


def print_run_header(provider: str, model: str, sensitivity: str, company: str, n_runs: int) -> None:
    print(f"\n{_BOLD}{'='*70}{_RESET}")
    print(f"{_BOLD}  Eval: {provider} / {model}{_RESET}")
    print(f"  Sensitivity : {sensitivity}")
    print(f"  Company     : {company!r}")
    print(f"  Runs        : {n_runs}")
    print(f"{_BOLD}{'='*70}{_RESET}\n")


def print_run(run: EvalRun, policy=None) -> None:
    """Print a one-line summary per group for a single run."""
    compose_at = getattr(policy, "compose_at", 0.40) if policy else 0.40
    print(f"  Run {run.run_index + 1}  [tokens={run.token_usage:,}  latency={run.latency_ms:.0f}ms]")
    for rec in run.records:
        col = _score_colour(rec.score, compose_at)
        switch = f"{_GREEN}● COMPOUND{_RESET}" if rec.mode_used == "compound" else "  fine    "
        print(
            f"    {rec.group:<14} {switch}  "
            f"conf={rec.confidence:4.0%}  "
            f"score={col}{rec.score:.3f}{_RESET}  "
            f"{_DIM}{rec.recommendation}{_RESET}"
        )


def print_summary(result: EvalResult) -> None:
    """Print aggregate statistics across all runs."""
    if not result.runs:
        print("  (no runs recorded)")
        return

    n = len(result.runs)
    compose_at = getattr(result.policy, "compose_at", 0.40) if result.policy else 0.40

    print(f"\n{_BOLD}--- Summary: {n} runs / {result.provider} / {result.model} / {result.sensitivity} ---{_RESET}")
    print(f"  {'Group':<14} {'Switches':>8} {'Mean score':>11} {'±std':>6} {'First switch':>13} {'Mean conf':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*11} {'-'*6} {'-'*13} {'-'*10}")

    for group in result.groups():
        sw  = result.switch_count(group)
        ms  = result.mean_score(group)
        std = result.score_std(group)
        mc  = result.mean_confidence(group)
        fs  = result.first_switch_run(group)

        col   = _score_colour(ms, compose_at)
        fs_str = f"Run {fs}" if fs else "—"
        print(
            f"  {group:<14} {sw:>3}/{n:<4}  "
            f"{col}{ms:>9.3f}{_RESET}  "
            f"{std:>5.3f}  "
            f"{fs_str:>13}  "
            f"{mc:>9.0%}"
        )

    # Total token and latency stats
    total_tokens  = sum(r.token_usage for r in result.runs)
    avg_latency   = statistics.mean(r.latency_ms for r in result.runs)
    print(f"\n  Total tokens: {total_tokens:,}  |  Avg latency: {avg_latency:.0f}ms/run")


def print_signal_breakdown(result: EvalResult) -> None:
    """
    Print the weighted signal breakdown for the last run of each group.

    Reveals which signals are driving the composition score and where to
    adjust weights or thresholds for a given model's output patterns.
    """
    if not result.runs:
        return

    weights = getattr(result.policy, "score_weights", None) if result.policy else None
    compose_at = getattr(result.policy, "compose_at", 0.40) if result.policy else 0.40

    print(f"\n{_BOLD}--- Signal breakdown (last run) ---{_RESET}")

    for group in result.groups():
        sig = result.last_signal(group)
        score = result.mean_score(group)

        print(f"\n  {_BOLD}{group}{_RESET}  (mean score {score:.3f}, compose_at={compose_at})")

        if sig is None:
            print("    (no signal data — run with score_weights configured)")
            continue

        # Normalised values
        norm_agents = min(sig.agent_count / 4.0, 1.0)
        norm_tokens = min(sig.avg_output_tokens / 300.0, 1.0)
        norm_tools  = min(sig.tool_calls_per_agent / 3.0, 1.0)
        max_depth   = max(sig.agent_count - 1, 1)
        norm_depth  = min(sig.dependency_depth / max_depth, 1.0)

        rows = [
            ("overhead_ratio",       sig.overhead_ratio,       None,       1.0),
            ("agent_count",          sig.agent_count,           norm_agents, None),
            ("avg_output_tokens",    sig.avg_output_tokens,     norm_tokens, None),
            ("tool_calls_per_agent", sig.tool_calls_per_agent,  norm_tools,  None),
            ("dependency_depth",     sig.dependency_depth,      norm_depth,  None),  # penalty
        ]

        if weights:
            w1, w2, w3, w4, w5 = weights
            w_vals = [w1, w2, w3, w4, w5]
        else:
            w_vals = [None] * 5

        for i, (name, raw, norm, _) in enumerate(rows):
            w    = w_vals[i]
            norm_str = f" → {norm:.2f}" if norm is not None else ""
            sign = "−" if i == 4 else "+"  # depth is a penalty
            if w is not None:
                contribution = w * (norm if norm is not None else raw)
                w_str = f"  (×{w:.2f} {sign}= {contribution:+.3f})"
            else:
                w_str = ""
            print(f"    {name:<24} {str(raw):<8}{norm_str}{w_str}")

        if sig.error_rate > 0 or sig.context_utilization > 0:
            print(f"    {'error_rate':<24} {sig.error_rate:.3f}")
            print(f"    {'context_utilization':<24} {sig.context_utilization:.3f}")

        if weights:
            w1, w2, w3, w4, w5 = weights
            computed = (
                  w1 * sig.overhead_ratio
                + w2 * min(sig.agent_count / 4.0, 1.0)
                + w3 * min(sig.avg_output_tokens / 300.0, 1.0)
                + w4 * min(sig.tool_calls_per_agent / 3.0, 1.0)
                - w5 * min(sig.dependency_depth / max(sig.agent_count - 1, 1), 1.0)
            )
            computed = max(0.0, min(computed, 1.0))
            gap = compose_at - computed
            print(f"    {'─'*40}")
            print(f"    Weighted score: {computed:.3f}  (compose_at={compose_at},  gap={gap:+.3f})")


def print_quality_report(result: EvalResult, quality_floor: float = 0.75) -> None:
    """
    Print per-group rolling quality score summary.

    Only prints when at least one group has quality data (evaluator was used).
    """
    if not result.runs:
        return

    groups_with_quality = [g for g in result.groups() if result.mean_quality(g) is not None]
    if not groups_with_quality:
        return

    print(f"\n{_BOLD}--- Quality report ---{_RESET}")
    print(f"  {'Group':<14} {'Mean quality':>13} {'±std':>6}  vs floor ({quality_floor:.2f})")
    print(f"  {'-'*14} {'-'*13} {'-'*6}  {'-'*20}")

    for group in result.groups():
        mq  = result.mean_quality(group)
        std = result.quality_std(group)
        if mq is None:
            continue
        if mq >= quality_floor:
            status = f"{_GREEN}above floor ✓{_RESET}"
        elif mq >= quality_floor * 0.90:
            status = f"{_YELLOW}borderline{_RESET}"
        else:
            status = f"\033[31mbelow floor ✗{_RESET}"
        std_str = f"{std:.3f}" if std is not None else "  —  "
        print(f"  {group:<14} {mq:>12.3f}  {std_str:>6}  {status}")


def print_calibration_notes(result: EvalResult) -> None:
    """
    Print actionable calibration suggestions based on observed scores.

    Helps developers tune compose_at thresholds or score_weights when
    groups are not switching at the expected rate.
    """
    if not result.runs:
        return

    compose_at   = getattr(result.policy, "compose_at", 0.40) if result.policy else 0.40
    decompose_at = getattr(result.policy, "decompose_at", 0.15) if result.policy else 0.15
    n = len(result.runs)

    notes: list[str] = []

    for group in result.groups():
        ms  = result.mean_score(group)
        sw  = result.switch_count(group)
        std = result.score_std(group)

        if sw == 0 and ms > 0:
            gap = compose_at - ms
            if gap <= 0.05:
                notes.append(
                    f"  • '{group}' scoring {ms:.3f} (gap={gap:+.3f} from compose_at={compose_at}). "
                    f"Reduce compose_at to ≤{ms - 0.01:.2f} or increase w2/w3 to trigger switching."
                )
            elif gap <= 0.15:
                notes.append(
                    f"  • '{group}' scoring {ms:.3f} — {gap:.2f} below compose_at. "
                    f"Try sensitivity='aggressive' (compose_at=0.25) to observe switching."
                )
            else:
                notes.append(
                    f"  • '{group}' scoring {ms:.3f} — well below compose_at={compose_at}. "
                    f"This group may be intentionally single-level (check agent_count and tool_calls)."
                )
        elif sw > 0 and sw < n:
            notes.append(
                f"  • '{group}' switched in {sw}/{n} runs (inconsistent). "
                f"Score std={std:.3f} — consider wider window_size for stability."
            )
        elif sw == n:
            notes.append(
                f"  • '{group}' switched every run — consider sensitivity='conservative' "
                f"if switching is too frequent for this group's pattern."
            )

    if notes:
        print(f"\n{_BOLD}--- Calibration notes ---{_RESET}")
        for note in notes:
            print(note)
    else:
        print(f"\n{_BOLD}--- Calibration notes ---{_RESET}")
        print("  All groups behaving as expected for this sensitivity preset.")
