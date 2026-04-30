"""
Threshold definitions — configurable decision boundaries for the GranularityController.

Defaults (from design plan §3.2.2):
  overhead_high       = 0.40  — compose if coordination tokens exceed 40% of total
  overhead_low        = 0.15  — candidate for decompose if overhead < 15%
  error_rate_high     = 0.15  — compose if error/retry rate exceeds 15%
  context_util_high   = 0.85  — decompose if context utilization exceeds 85%
  parallelism_factor  = 2.0   — decompose if available parallelism > 2× capsules in flight
  hysteresis_window   = 3     — consecutive batches signal must hold before acting
  backoff_base        = 2     — exponential backoff multiplier after each direction change

All thresholds are overridable at construction time.

Design plan ref: §3.2.2
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControllerThresholds:
    """
    Decision boundaries for the GranularityController.

    Changing any field takes effect on the next call to `decide()`.

    Attributes:
        overhead_high: Compose if overhead_ratio exceeds this value.
        overhead_low: Consider decomposing if overhead_ratio is below this value.
        error_rate_high: Compose if error rate exceeds this value.
        context_util_high: Decompose if context utilization exceeds this value.
        parallelism_factor: Decompose if parallelism_available > factor × capsules_in_flight.
        hysteresis_window: Number of consecutive batches a signal must hold before acting.
        backoff_base: Exponential backoff multiplier applied after each direction change.
    """
    overhead_high: float = 0.40
    overhead_low: float = 0.15
    error_rate_high: float = 0.15
    context_util_high: float = 0.85
    parallelism_factor: float = 2.0
    hysteresis_window: int = 3
    backoff_base: int = 2

    def __post_init__(self) -> None:
        if not (0.0 <= self.overhead_low < self.overhead_high <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ overhead_low < overhead_high ≤ 1. "
                f"Got overhead_low={self.overhead_low}, overhead_high={self.overhead_high}"
            )
        if not (0.0 <= self.error_rate_high <= 1.0):
            raise ValueError(f"error_rate_high must be in [0, 1]. Got {self.error_rate_high}")
        if not (0.0 < self.context_util_high <= 1.0):
            raise ValueError(f"context_util_high must be in (0, 1]. Got {self.context_util_high}")
        if self.hysteresis_window < 1:
            raise ValueError(f"hysteresis_window must be >= 1. Got {self.hysteresis_window}")
        if self.backoff_base < 1:
            raise ValueError(f"backoff_base must be >= 1. Got {self.backoff_base}")

    def __repr__(self) -> str:
        return (
            f"ControllerThresholds("
            f"overhead={self.overhead_low:.0%}–{self.overhead_high:.0%}, "
            f"error={self.error_rate_high:.0%}, "
            f"ctx={self.context_util_high:.0%}, "
            f"hysteresis={self.hysteresis_window})"
        )
