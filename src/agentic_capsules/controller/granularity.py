"""
GranularityController — closed-loop runtime decision engine.

Main feedback loop (per batch):
  1. After execution, the executor calls controller.observe(record).
  2. The controller accumulates records in a WindowedCollector.
  3. On each call to decide(), the controller checks the four signals
     against configured thresholds and returns a ControllerAction.
  4. Hysteresis: a signal must exceed its threshold for `hysteresis_window`
     consecutive calls before the action fires.
  5. Exponential backoff: after a direction change (compose → decompose or
     vice versa), the required hysteresis count grows as backoff_base^n_flips
     for the opposing direction, preventing rapid oscillation.

Controller decision logic (§3.2.2)::

    overhead_ratio > overhead_high                            → COMPOSE
    error_rate > error_rate_high                              → COMPOSE
    context_util > context_util_high                          → DECOMPOSE
    parallelism_available > factor × capsules_in_flight
      AND overhead_ratio < overhead_low                       → DECOMPOSE
    else                                                      → MAINTAIN

Design plan ref: §3.1 (Granularity Controller), §3.2.2, §5.2 Phase 5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from .telemetry import TelemetryRecord, WindowedCollector
from .thresholds import ControllerThresholds

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controller action enum
# ---------------------------------------------------------------------------

class ControllerAction(Enum):
    """The three actions the controller can recommend."""
    COMPOSE = auto()    # merge adjacent agents → fewer, larger capsules
    DECOMPOSE = auto()  # split compounds → more, smaller capsules
    MAINTAIN = auto()   # keep current granularity level


# ---------------------------------------------------------------------------
# Decision reason (for observability)
# ---------------------------------------------------------------------------

@dataclass
class DecisionReason:
    """
    The reason behind a ControllerAction decision.

    Provides full transparency into which signal triggered the action and
    what the current signal values were.
    """
    action: ControllerAction
    signal: str          # e.g. "overhead_ratio", "error_rate", "MAINTAIN"
    signal_value: float  # the actual measured value
    threshold: float     # the threshold that was crossed (or 0 for MAINTAIN)
    consecutive: int     # how many consecutive windows triggered this signal

    def __repr__(self) -> str:
        if self.action == ControllerAction.MAINTAIN:
            return f"DecisionReason(MAINTAIN)"
        return (
            f"DecisionReason({self.action.name}: "
            f"{self.signal}={self.signal_value:.3f} "
            f"{'>' if self.action == ControllerAction.COMPOSE else '<'} "
            f"{self.threshold:.3f}, "
            f"consecutive={self.consecutive})"
        )


# ---------------------------------------------------------------------------
# GranularityController
# ---------------------------------------------------------------------------

class GranularityController:
    """
    Observes telemetry records and recommends granularity adjustments.

    Usage:
        thresholds = ControllerThresholds()
        controller = GranularityController(thresholds)

        # After each capsule execution:
        controller.observe(telemetry_record)

        # To get the recommended action for the next batch:
        action, reason = controller.decide()

        if action == ControllerAction.COMPOSE:
            # increase composition level
        elif action == ControllerAction.DECOMPOSE:
            # decrease composition level

    Design plan ref: §3.2.2
    """

    def __init__(
        self,
        thresholds: ControllerThresholds | None = None,
        parallelism_available: int = 1,
    ) -> None:
        self._thresholds = thresholds if thresholds is not None else ControllerThresholds()
        self._parallelism_available = parallelism_available
        self._window = WindowedCollector(window=self._thresholds.hysteresis_window)

        # Hysteresis counters: how many consecutive windows each signal has fired
        self._compose_streak: int = 0
        self._decompose_streak: int = 0

        # Backoff state: direction changes and effective required streaks
        self._direction_changes: int = 0
        self._last_action: ControllerAction = ControllerAction.MAINTAIN

        # History for observability
        self._decisions: list[DecisionReason] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def observe(self, record: TelemetryRecord) -> None:
        """Feed one TelemetryRecord into the rolling window."""
        self._window.add(record)

    def decide(self, capsules_in_flight: int = 1) -> tuple[ControllerAction, DecisionReason]:
        """
        Evaluate the current window signals and return the recommended action.

        *capsules_in_flight* is the number of capsule instances currently
        being executed (used for the parallelism decompose signal).

        Returns (ControllerAction, DecisionReason). When the window is not
        yet full, returns MAINTAIN to avoid acting on insufficient data.
        """
        signals = self._window.signals()

        if not signals:
            reason = DecisionReason(
                action=ControllerAction.MAINTAIN,
                signal="no_data", signal_value=0.0, threshold=0.0, consecutive=0,
            )
            self._decisions.append(reason)
            return ControllerAction.MAINTAIN, reason

        t = self._thresholds
        overhead = signals["overhead_ratio"]
        error_rate = signals["error_rate"]
        ctx_util = signals["context_utilization"]

        # --- Check COMPOSE signals ---
        compose_triggered = (
            overhead > t.overhead_high or
            error_rate > t.error_rate_high
        )

        # --- Check DECOMPOSE signals ---
        parallelism_signal = (
            self._parallelism_available > t.parallelism_factor * capsules_in_flight
            and overhead < t.overhead_low
        )
        decompose_triggered = ctx_util > t.context_util_high or parallelism_signal

        # Update streaks
        if compose_triggered:
            self._compose_streak += 1
            self._decompose_streak = 0
        elif decompose_triggered:
            self._decompose_streak += 1
            self._compose_streak = 0
        else:
            self._compose_streak = 0
            self._decompose_streak = 0

        # Apply backoff: after a direction change, opposing direction needs more streaks
        required = self._required_streak()

        compose_required = self._required_streak(ControllerAction.COMPOSE)
        decompose_required = self._required_streak(ControllerAction.DECOMPOSE)

        if self._compose_streak >= compose_required and compose_triggered:
            action = ControllerAction.COMPOSE
            signal = "overhead_ratio" if overhead > t.overhead_high else "error_rate"
            val = overhead if overhead > t.overhead_high else error_rate
            thresh = t.overhead_high if overhead > t.overhead_high else t.error_rate_high
            reason = DecisionReason(
                action=action, signal=signal, signal_value=val,
                threshold=thresh, consecutive=self._compose_streak,
            )
            self._on_action(action)
            self._compose_streak = 0

        elif self._decompose_streak >= decompose_required and decompose_triggered:
            action = ControllerAction.DECOMPOSE
            if ctx_util > t.context_util_high:
                signal, val, thresh = "context_utilization", ctx_util, t.context_util_high
            else:
                signal, val, thresh = "parallelism", float(self._parallelism_available), 0.0
            reason = DecisionReason(
                action=action, signal=signal, signal_value=val,
                threshold=thresh, consecutive=self._decompose_streak,
            )
            self._on_action(action)
            self._decompose_streak = 0

        else:
            action = ControllerAction.MAINTAIN
            reason = DecisionReason(
                action=action, signal="MAINTAIN",
                signal_value=overhead, threshold=0.0,
                consecutive=0,
            )

        self._decisions.append(reason)
        logger.debug(
            "Controller decide: %s | overhead=%.2f error=%.2f ctx=%.2f "
            "streak_c=%d (need %d) streak_d=%d (need %d)",
            action.name, overhead, error_rate, ctx_util,
            self._compose_streak, compose_required,
            self._decompose_streak, decompose_required,
        )
        return action, reason

    @property
    def decisions(self) -> list[DecisionReason]:
        """All decisions made so far (for observability)."""
        return list(self._decisions)

    @property
    def thresholds(self) -> ControllerThresholds:
        return self._thresholds

    def reset(self) -> None:
        """Clear window, streaks, and backoff state."""
        self._window = WindowedCollector(window=self._thresholds.hysteresis_window)
        self._compose_streak = 0
        self._decompose_streak = 0
        self._direction_changes = 0
        self._last_action = ControllerAction.MAINTAIN
        self._decisions.clear()

    def __repr__(self) -> str:
        return (
            f"GranularityController("
            f"thresholds={self._thresholds!r}, "
            f"decisions={len(self._decisions)})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _required_streak(self, candidate: ControllerAction = ControllerAction.MAINTAIN) -> int:
        """
        Minimum consecutive-window streak needed before acting on *candidate*.

        Starts at hysteresis_window. If *candidate* is the opposite direction from
        the last fired action, the required streak grows by backoff_base^(n_flips+1)
        to prevent oscillation — the +1 accounts for the pending (not-yet-recorded)
        direction change. Capped at 3× the base hysteresis window.
        """
        base = self._thresholds.hysteresis_window
        is_flip = (
            candidate != ControllerAction.MAINTAIN
            and self._last_action != ControllerAction.MAINTAIN
            and candidate != self._last_action
        )
        if not is_flip:
            return base
        backoff = self._thresholds.backoff_base ** (self._direction_changes + 1)
        return min(base * backoff, base * 3)

    def _on_action(self, action: ControllerAction) -> None:
        """Track direction changes for backoff."""
        if (
            self._last_action != ControllerAction.MAINTAIN
            and action != ControllerAction.MAINTAIN
            and action != self._last_action
        ):
            self._direction_changes += 1
        self._last_action = action
