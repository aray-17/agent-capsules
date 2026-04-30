"""
CalibrationReport — pre-deployment FINE vs COMPOUND quality comparison (T-034).

Produced by Pipeline.calibrate().  Runs paired FINE and COMPOUND executions
for each group and summarises quality and latency so developers can answer
"is COMPOUND mode safe for my pipeline?" before enabling auto-switching in
production.

Pipeline.calibrate() never writes to GroupControllerState — it is a read-only
dry run that does not interfere with the live controller.

Usage::

    report = pipeline.calibrate(
        sample_tasks  = ["Analyse Acme Corp", "Analyse Widget Inc."],
        adapter       = adapter,
        evaluator     = LLMJudgeEvaluator(judge_adapter),
        n_paired_runs = 3,
    )
    print(report.recommend_compose_at())   # → 0.36 or None
    print(report.quality_by_group())       # → {"research": 0.88, "analysis": 0.83}
    print(report.latency_by_group())       # → {"research": {"fine_ms": 4200, ...}}
    report.save("calibration_report.md")

Phase 12 ref: P12-5 (T-034).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class _GroupCalibration:
    """Per-group calibration data collected from paired FINE / COMPOUND runs."""
    group:                 str
    quality_scores:        list[float]  = field(default_factory=list)
    latency_fine_ms:       list[float]  = field(default_factory=list)
    latency_compound_ms:   list[float]  = field(default_factory=list)
    tokens_fine:           list[int]    = field(default_factory=list)
    tokens_compound:       list[int]    = field(default_factory=list)
    # Additional signals for per-parameter threshold recommendations:
    # composition_scores — per-run composition-score observations accumulated
    #   during the FINE passes of calibrate(). Basis for recommending compose_at.
    # avg_output_tokens_fine — mean output tokens per agent in the group during
    #   FINE mode. Basis for recommending verbosity_* thresholds.
    composition_scores:    list[float]  = field(default_factory=list)
    avg_output_tokens_fine: list[float] = field(default_factory=list)


@dataclass
class CalibrationReport:
    """
    Pre-deployment quality and efficiency report comparing FINE vs COMPOUND mode.

    Produced by ``Pipeline.calibrate()``.  Never written to GroupControllerState
    or any persistent store — it is a read-only analysis artifact.

    Attributes:
        pipeline_name:      Name of the pipeline that was calibrated.
        quality_floor:      The quality floor from ControllerPolicy (0.0–1.0 or None).
        _groups:            Internal per-group calibration data.
    """
    pipeline_name:  str
    quality_floor:  float | None
    _groups:        dict[str, _GroupCalibration] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def quality_by_group(self) -> dict[str, float | None]:
        """
        Mean quality score per group (COMPOUND vs FINE comparison).

        Returns None for a group if no quality scores were recorded.
        """
        return {
            name: (statistics.mean(gc.quality_scores) if gc.quality_scores else None)
            for name, gc in self._groups.items()
        }

    def latency_by_group(self) -> dict[str, dict]:
        """
        Per-group mean latency for FINE and COMPOUND modes.

        Returns dict with keys: ``fine_ms``, ``compound_ms``, ``speedup_pct``.
        ``speedup_pct`` is positive when COMPOUND is faster.
        """
        result: dict[str, dict] = {}
        for name, gc in self._groups.items():
            fine_ms     = statistics.mean(gc.latency_fine_ms)     if gc.latency_fine_ms     else None
            compound_ms = statistics.mean(gc.latency_compound_ms) if gc.latency_compound_ms else None
            speedup = None
            if fine_ms and compound_ms and fine_ms > 0:
                speedup = (fine_ms - compound_ms) / fine_ms * 100.0
            result[name] = {
                "fine_ms":     fine_ms,
                "compound_ms": compound_ms,
                "speedup_pct": speedup,
            }
        return result

    def token_reduction_by_group(self) -> dict[str, dict]:
        """
        Per-group token usage comparison.

        Returns dict with keys: ``fine``, ``compound``, ``reduction_pct``.
        Positive ``reduction_pct`` means COMPOUND uses fewer tokens.
        """
        result: dict[str, dict] = {}
        for name, gc in self._groups.items():
            fine_mean     = statistics.mean(gc.tokens_fine)     if gc.tokens_fine     else None
            compound_mean = statistics.mean(gc.tokens_compound) if gc.tokens_compound else None
            reduction = None
            if fine_mean and compound_mean and fine_mean > 0:
                reduction = (fine_mean - compound_mean) / fine_mean * 100.0
            result[name] = {
                "fine":          fine_mean,
                "compound":      compound_mean,
                "reduction_pct": reduction,
            }
        return result

    def recommend_compose_at(self) -> float | None:
        """
        Suggest a compose_at threshold based on calibration results.

        Returns:
            The recommended compose_at value if COMPOUND quality passes the
            quality_floor for all groups; None if any group fails the floor
            (meaning COMPOUND is not yet safe and manual threshold tuning is needed).

            When quality_floor is None (not configured), always returns None
            (cannot make a quality-based recommendation without a floor).
        """
        if self.quality_floor is None:
            return None
        quality = self.quality_by_group()
        for name, q in quality.items():
            if q is None or q < self.quality_floor:
                return None  # quality check failed for at least one group
        # All groups pass — return the default balanced compose_at
        return 0.36

    def passes_quality_floor(self) -> bool:
        """True if every group with quality scores meets the quality_floor."""
        if self.quality_floor is None:
            return True  # no floor configured — treat as passing
        for q in self.quality_by_group().values():
            if q is not None and q < self.quality_floor:
                return False
        return True

    # ------------------------------------------------------------------
    # Threshold recommendations
    # ------------------------------------------------------------------

    def recommended_policy(self, base_policy=None):
        """
        Return a :class:`ControllerPolicy` populated with recommended thresholds
        derived from the calibration observations.

        Rule summary (per-parameter — see the field-level commentary inside the
        implementation for the precise cut-off and rationale for each rule):

        - ``compose_at``: median of the observed per-group composition scores,
          clamped to [0.15, 0.45]. Recommends a threshold that the observed
          groups would clear roughly half the time, letting the controller make
          a switching decision rather than locking to one mode.
        - ``quality_floor``: the 5th percentile of observed per-group quality,
          rounded down to the nearest 0.05, clamped to [0.50, 0.85]. Chooses a
          floor low enough to avoid false reverts but high enough to catch real
          regressions.
        - ``verbosity_guidance_threshold``: chosen so it separates the observed
          per-group ``avg_output_tokens_fine`` distribution. If all observations
          cluster on one side of the current threshold, keep the default (no
          signal to recalibrate); if they straddle it, pick the midpoint of the
          cluster gap.
        - ``verbosity_two_phase_threshold`` / ``verbosity_sequential_threshold``:
          same signal as ``verbosity_guidance_threshold`` but at coarser scales
          to stay consistent with the existing calibration tiers.

        When the calibration run doesn't have enough observations to make a
        confident recommendation for a field, that field falls back to
        ``base_policy`` (defaults to a fresh :class:`ControllerPolicy`). The
        report stays read-only: the caller applies the recommendation explicitly,
        typically via ``dataclasses.replace(current_policy, **changed_fields)``.

        Args:
            base_policy: :class:`ControllerPolicy` to fill in for fields the
                calibration can't recommend. Defaults to a new instance with
                framework defaults.

        Returns:
            A :class:`ControllerPolicy` reflecting the recommendations.
        """
        from ..controller.policy import ControllerPolicy
        if base_policy is None:
            base_policy = ControllerPolicy()

        rec_compose_at        = self._recommend_compose_at(base_policy.compose_at)
        rec_quality_floor     = self._recommend_quality_floor(base_policy.quality_floor)
        rec_verb_guidance     = self._recommend_verbosity_threshold(
            base_policy.verbosity_guidance_threshold
        )
        rec_verb_two_phase    = self._recommend_verbosity_threshold(
            base_policy.verbosity_two_phase_threshold
        )
        rec_verb_sequential   = self._recommend_verbosity_threshold(
            base_policy.verbosity_sequential_threshold
        )

        # Guarantee the two-tier ordering invariant required by ControllerPolicy
        # validation (sequential_threshold > two_phase_threshold). If the
        # recommendation collapses the ordering (e.g. both fell to the same
        # cluster), fall back to the base policy for both fields.
        if rec_verb_sequential <= rec_verb_two_phase:
            rec_verb_two_phase  = base_policy.verbosity_two_phase_threshold
            rec_verb_sequential = base_policy.verbosity_sequential_threshold

        from dataclasses import replace
        return replace(
            base_policy,
            compose_at=rec_compose_at,
            quality_floor=rec_quality_floor,
            verbosity_guidance_threshold=rec_verb_guidance,
            verbosity_two_phase_threshold=rec_verb_two_phase,
            verbosity_sequential_threshold=rec_verb_sequential,
        )

    # ---- Per-rule helpers ----

    _COMPOSE_AT_BOUNDS     = (0.15, 0.45)
    _QUALITY_FLOOR_BOUNDS  = (0.50, 0.85)
    _QUALITY_FLOOR_STEP    = 0.05

    def _recommend_compose_at(self, fallback: float) -> float:
        """Median of per-group composition scores, clamped to safe bounds."""
        all_scores = [
            score
            for gc in self._groups.values()
            for score in gc.composition_scores
        ]
        if len(all_scores) < 3:
            return fallback
        median = statistics.median(all_scores)
        lo, hi = self._COMPOSE_AT_BOUNDS
        return max(lo, min(hi, round(median, 2)))

    def _recommend_quality_floor(self, fallback: float | None) -> float | None:
        """5th-percentile quality, rounded down to 0.05, clamped to safe bounds.

        Returns ``fallback`` (often ``None``) when we don't have enough
        observations to compute a percentile reliably. The floor is chosen
        below every observation in the base case so that a re-run using the
        recommendation doesn't immediately fail on its own data.
        """
        all_quality = [
            q for gc in self._groups.values() for q in gc.quality_scores
        ]
        if len(all_quality) < 5:
            return fallback
        sorted_q = sorted(all_quality)
        # Nearest-rank 5th percentile. For n=5 this is the minimum; for larger
        # samples it tracks the left tail without being pinned to the worst run.
        rank = max(0, int(0.05 * len(sorted_q)) - 1)
        p5 = sorted_q[rank]
        step = self._QUALITY_FLOOR_STEP
        floor = (int(p5 / step)) * step  # round DOWN to step boundary
        lo, hi = self._QUALITY_FLOOR_BOUNDS
        return max(lo, min(hi, round(floor, 2)))

    def _recommend_verbosity_threshold(self, fallback: int) -> int:
        """Recommend a verbosity threshold (tokens/agent) from observed fine-mode
        per-agent output.

        Keep ``fallback`` unless the observations give a clear signal to move it.
        The rule: if every per-group observation sits on the same side of
        ``fallback`` (all above or all below), move the threshold to the midpoint
        between the observed extrema so future routing still distinguishes
        meaningfully across groups. Otherwise keep the default — the threshold
        is already separating the observations.
        """
        per_group_means = [
            statistics.mean(gc.avg_output_tokens_fine)
            for gc in self._groups.values()
            if gc.avg_output_tokens_fine
        ]
        if len(per_group_means) < 2:
            return fallback
        below = [m for m in per_group_means if m < fallback]
        above = [m for m in per_group_means if m >= fallback]
        if not below or not above:
            # All observations on one side — current threshold isn't gating any
            # group. Move to the midpoint of the cluster so the next run
            # produces a meaningful split (this also gives the operator a
            # concrete datapoint when recalibrating per-domain).
            lo = min(per_group_means)
            hi = max(per_group_means)
            return max(1, int(round((lo + hi) / 2)))
        return fallback

    def save(self, path: str) -> None:
        """Write a human-readable Markdown report to ``path``."""
        rec = self.recommended_policy()
        lines: list[str] = [
            f"# Calibration Report: {self.pipeline_name}",
            "",
            f"**Quality floor:** {self.quality_floor if self.quality_floor is not None else 'not configured'}",
            f"**Recommended compose_at:** {self.recommend_compose_at() or 'N/A (quality check failed)'}",
            "",
            "## Recommended policy",
            "",
            "Threshold recommendations derived from the calibration observations.",
            "Apply these explicitly with `dataclasses.replace(policy, **changed)`; ",
            "the report itself is read-only.",
            "",
            "| Field | Recommendation |",
            "|---|---|",
            f"| compose_at                    | {rec.compose_at} |",
            f"| quality_floor                 | {rec.quality_floor} |",
            f"| verbosity_guidance_threshold  | {rec.verbosity_guidance_threshold} |",
            f"| verbosity_two_phase_threshold | {rec.verbosity_two_phase_threshold} |",
            f"| verbosity_sequential_threshold| {rec.verbosity_sequential_threshold} |",
            "",
            "## Quality by group",
            "",
            "| Group | Mean quality | Passes floor |",
            "|---|---|---|",
        ]
        for name, q in self.quality_by_group().items():
            passes = "✓" if (self.quality_floor is None or (q is not None and q >= self.quality_floor)) else "✗"
            q_str = f"{q:.3f}" if q is not None else "—"
            lines.append(f"| {name} | {q_str} | {passes} |")

        lines += ["", "## Latency by group", "", "| Group | FINE ms | COMPOUND ms | Speedup |", "|---|---|---|---|"]
        for name, lat in self.latency_by_group().items():
            fine = f"{lat['fine_ms']:.0f}" if lat["fine_ms"] is not None else "—"
            comp = f"{lat['compound_ms']:.0f}" if lat["compound_ms"] is not None else "—"
            spd  = f"{lat['speedup_pct']:+.1f}%" if lat["speedup_pct"] is not None else "—"
            lines.append(f"| {name} | {fine} | {comp} | {spd} |")

        lines += ["", "## Token reduction by group", "", "| Group | FINE tokens | COMPOUND tokens | Reduction |", "|---|---|---|---|"]
        for name, tok in self.token_reduction_by_group().items():
            fine = f"{tok['fine']:.0f}" if tok["fine"] is not None else "—"
            comp = f"{tok['compound']:.0f}" if tok["compound"] is not None else "—"
            red  = f"{tok['reduction_pct']:+.1f}%" if tok["reduction_pct"] is not None else "—"
            lines.append(f"| {name} | {fine} | {comp} | {red} |")

        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
