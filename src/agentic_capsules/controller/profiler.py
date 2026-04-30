"""
GranularityProfiler — sweeps all hierarchy levels and reports the optimal config.

Used for two purposes:
  1. Establish a static-optimal baseline for Benchmark 4.
  2. Auto-calibrate GranularityController thresholds from observed data.

The profiler runs the executor at each available CompositionLevel, collects
telemetry, and ranks configurations by overhead_ratio (lower = better).
It also computes a suggested `overhead_high` threshold from the observed curve,
which can be fed back into ControllerThresholds.

Usage:
    profiler = GranularityProfiler(executor, hierarchy, task_input="...")
    report = profiler.run()
    print(report.best_level)          # CompositionLevel with lowest overhead
    print(report.suggested_thresholds)  # auto-calibrated ControllerThresholds

Design plan ref: §3.2.2, §5.2 Phase 5, §7 RQ3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.types import CompositionLevel
from .telemetry import TelemetryCollector
from .thresholds import ControllerThresholds

if TYPE_CHECKING:
    from ..runtime.executor import CapsuleExecutor
    from ..core.hierarchy import CapsuleHierarchy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProfilerEntry — one data point per level
# ---------------------------------------------------------------------------

@dataclass
class ProfilerEntry:
    """Result for a single composition level sweep."""
    level: CompositionLevel
    total_calls: int
    total_tokens: int
    coordination_tokens: int
    overhead_ratio: float
    latency_ms: float

    def __repr__(self) -> str:
        return (
            f"ProfilerEntry({self.level.name}: "
            f"calls={self.total_calls}, "
            f"tokens={self.total_tokens}, "
            f"overhead={self.overhead_ratio:.1%}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# ProfilerReport — the full sweep result
# ---------------------------------------------------------------------------

@dataclass
class ProfilerReport:
    """
    Full profiling report: one ProfilerEntry per level, ranked by overhead.

    `best_level` is the level with the lowest overhead_ratio.
    `suggested_thresholds` provides auto-calibrated decision boundaries
    derived from the observed overhead curve.
    """
    entries: list[ProfilerEntry]
    best_level: CompositionLevel
    suggested_thresholds: ControllerThresholds

    def print_table(self) -> None:
        """Print a human-readable table of results."""
        print(f"\n{'Level':<12} {'Calls':>6} {'Tokens':>8} {'Overhead':>10} {'Latency(ms)':>12}")
        print("-" * 52)
        for e in self.entries:
            marker = " ◀ best" if e.level == self.best_level else ""
            print(
                f"{e.level.name:<12} {e.total_calls:>6} {e.total_tokens:>8} "
                f"{e.overhead_ratio:>9.1%} {e.latency_ms:>12.1f}{marker}"
            )
        print(
            f"\nSuggested thresholds: "
            f"overhead_high={self.suggested_thresholds.overhead_high:.0%}, "
            f"overhead_low={self.suggested_thresholds.overhead_low:.0%}"
        )

    def __repr__(self) -> str:
        return (
            f"ProfilerReport(levels={[e.level.name for e in self.entries]}, "
            f"best={self.best_level.name})"
        )


# ---------------------------------------------------------------------------
# GranularityProfiler
# ---------------------------------------------------------------------------

class GranularityProfiler:
    """
    Sweeps all available CompositionLevels and ranks them by overhead_ratio.

    The profiler creates a fresh TelemetryCollector for each level to avoid
    cross-contamination. It does not modify the executor's internal state.

    Args:
        executor: A configured CapsuleExecutor instance.
        hierarchy: The CapsuleHierarchy to profile.
        task_input: The task string to use for all profiling runs.
        task_id: Optional task ID for sync manager scoping.
        levels: Which levels to sweep. Defaults to [FINE, COMPOUND].
                ITERATION requires a tag_space to be set on the hierarchy.
    """

    DEFAULT_LEVELS = [CompositionLevel.FINE, CompositionLevel.COMPOUND]

    def __init__(
        self,
        executor: CapsuleExecutor,
        hierarchy: CapsuleHierarchy,
        task_input: str,
        task_id: str = "profiler",
        levels: list[CompositionLevel] | None = None,
    ) -> None:
        self._executor = executor
        self._hierarchy = hierarchy
        self._task_input = task_input
        self._task_id = task_id
        self._levels = levels if levels is not None else list(self.DEFAULT_LEVELS)

    def run(self) -> ProfilerReport:
        """
        Sweep all configured levels and return a ProfilerReport.

        For each level:
          - Creates a fresh TelemetryCollector
          - Temporarily sets the executor's level and telemetry
          - Runs the hierarchy once
          - Restores the executor's original level and telemetry
        """
        entries: list[ProfilerEntry] = []
        original_level = self._executor._level
        original_telemetry = self._executor._telemetry

        for level in self._levels:
            if level == CompositionLevel.ITERATION and self._hierarchy.tag_space is None:
                logger.debug("Skipping ITERATION level: no tag_space on hierarchy.")
                continue

            telemetry = TelemetryCollector()
            self._executor._level = level
            self._executor._telemetry = telemetry

            try:
                self._executor.run(
                    self._hierarchy,
                    task_input=self._task_input,
                    task_id=self._task_id,
                )
            except Exception as exc:
                logger.warning("Profiler: level %s raised %s — skipping.", level.name, exc)
                continue
            finally:
                # Always restore
                self._executor._level = original_level
                self._executor._telemetry = original_telemetry

            summary = telemetry.summary()
            entries.append(ProfilerEntry(
                level=level,
                total_calls=len(telemetry.records),
                total_tokens=summary.get("total_tokens", 0),
                coordination_tokens=summary.get("total_coordination_tokens", 0),
                overhead_ratio=summary.get("avg_overhead_ratio", 0.0),
                latency_ms=summary.get("avg_latency_ms", 0.0),
            ))
            logger.debug(
                "Profiler: level=%s overhead=%.1f%% tokens=%d",
                level.name, entries[-1].overhead_ratio * 100, entries[-1].total_tokens,
            )

        if not entries:
            raise RuntimeError("Profiler: no levels completed successfully.")

        best = min(entries, key=lambda e: e.overhead_ratio)
        suggested = self._suggest_thresholds(entries)

        return ProfilerReport(
            entries=entries,
            best_level=best.level,
            suggested_thresholds=suggested,
        )

    # ------------------------------------------------------------------
    # Threshold auto-calibration
    # ------------------------------------------------------------------

    def _suggest_thresholds(self, entries: list[ProfilerEntry]) -> ControllerThresholds:
        """
        Derive suggested overhead thresholds from the observed overhead curve.

        Strategy:
          - overhead_high = midpoint between the highest and lowest observed ratios
            (trigger compose when above this midpoint)
          - overhead_low  = 40% of overhead_high (trigger decompose below this)

        If all levels have the same overhead (e.g., scripted offline runs), fall
        back to the design-plan defaults.
        """
        ratios = [e.overhead_ratio for e in entries]
        min_r, max_r = min(ratios), max(ratios)

        if max_r - min_r < 0.05:
            # Flat curve — not enough signal, use defaults
            return ControllerThresholds()

        overhead_high = round((min_r + max_r) / 2 + 0.05, 2)
        overhead_high = max(0.20, min(overhead_high, 0.70))
        overhead_low = round(overhead_high * 0.40, 2)
        overhead_low = max(0.05, overhead_low)

        return ControllerThresholds(
            overhead_high=overhead_high,
            overhead_low=overhead_low,
        )
