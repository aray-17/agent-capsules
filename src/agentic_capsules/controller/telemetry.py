"""
Telemetry Collector — measures runtime signals per capsule execution.

Tracks the metrics defined in design plan §6.1 and feeds them back to
the GranularityController (Phase 5).

Metrics per capsule (§6.1)::

    tokens_coordination  — overhead tokens: delimiters, headings, instructions
    tokens_reasoning     — content tokens (total - coordination)
    overhead_ratio       — tokens_coordination / total_tokens
    latency_ms           — wall-clock time for the capsule
    batch_size           — number of items (1 for single-agent, K for iteration)
    error_count          — number of retried/failed steps (Phase 5)
    context_utilization  — tokens / adapter context_window (Phase 5)

Design plan ref: §3.1 (Telemetry Collector), §6.1
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """
    One record emitted after a capsule completes.

    Design plan ref: §6.1

    T-047 COGS fields (input_tokens, output_tokens, llm_call_count) capture
    the actual API billing signals needed to compute dollar savings and GPU
    utilisation improvements::

        input_tokens   — billed prompt tokens (prefill FLOPs on GPU)
        output_tokens  — billed completion tokens (decode steps on GPU;
                         each token = one sequential forward pass)
        llm_call_count — number of adapter.complete() invocations (each call
                         is one scheduling unit: KV alloc + auth + routing)

    COMPOUND mode reduces llm_call_count from N (FINE) to 1 and eliminates
    repeated system-prompt and task-context overhead from input_tokens.
    output_tokens should be approximately unchanged (same quality answers).
    Dollar savings = Δinput × input_price + Δoutput × output_price.
    GPU scheduling savings = (N_fine − N_compound) × per_call_overhead.
    """
    capsule_name: str
    composition_mode: str             # "FINE" | "COMPOUND" | "ITERATION" | "TOOL" | "SKIPPED" (G-2)
    batch_size: int                   # items processed in this capsule
    total_tokens: int                 # estimated input tokens (heuristic)
    coordination_tokens: int          # overhead portion
    latency_ms: float
    error_count: int = 0              # retried or failed steps (Phase 5)
    context_utilization: float = 0.0  # total_tokens / adapter context_window (Phase 5)
    tool_calls: int = 0               # tool invocations made during this agent's turn (Phase 10)
    tool_call_sequence: list[str] = field(default_factory=list)  # T-015: ordered list of tool names called

    # T-047: actual billed token counts from API responses
    input_tokens: int = 0             # billed prompt/input tokens (prefill)
    output_tokens: int = 0            # billed completion/output tokens (decode)
    llm_call_count: int = 1           # number of adapter.complete() calls for this capsule

    @property
    def reasoning_tokens(self) -> int:
        return max(0, self.total_tokens - self.coordination_tokens)

    @property
    def overhead_ratio(self) -> float:
        """
        Coordination tokens / total tokens.

        This is the primary signal the GranularityController acts on::

            > OVERHEAD_THRESHOLD (0.40)  → compose
            < OVERHEAD_LOW (0.15)        → consider decomposing

        Design plan ref: §3.2.2, §6.1
        """
        if self.total_tokens == 0:
            return 0.0
        return self.coordination_tokens / self.total_tokens

    @property
    def error_rate(self) -> float:
        """error_count / batch_size; 0.0 when batch_size == 0."""
        if self.batch_size == 0:
            return 0.0
        return self.error_count / self.batch_size

    @property
    def billed_tokens(self) -> int:
        """Total billed tokens = input + output. 0 if adapter did not report usage."""
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_call(self) -> float:
        """Billed tokens per LLM call — higher = better scheduling efficiency."""
        if self.llm_call_count == 0 or self.billed_tokens == 0:
            return 0.0
        return self.billed_tokens / self.llm_call_count

    def __repr__(self) -> str:
        return (
            f"TelemetryRecord("
            f"capsule={self.capsule_name!r}, "
            f"mode={self.composition_mode}, "
            f"batch={self.batch_size}, "
            f"tokens={self.total_tokens}, "
            f"overhead={self.overhead_ratio:.2%}, "
            f"ctx_util={self.context_utilization:.2%}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    Collects TelemetryRecords from capsule executions.

    Usage:
        collector = TelemetryCollector()

        with collector.measure("analyst", "ITERATION", batch_size=10) as ctx:
            result = adapter.complete(messages)
            ctx.total_tokens = compiled.estimated_tokens
            ctx.coordination_tokens = compiled.coordination_tokens

        records = collector.records          # list of TelemetryRecord
        summary = collector.summary()        # aggregate stats
    """

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []

    @property
    def records(self) -> list[TelemetryRecord]:
        return list(self._records)

    @contextmanager
    def measure(
        self,
        capsule_name: str,
        composition_mode: str,
        batch_size: int = 1,
    ) -> Iterator[_MeasureContext]:
        """
        Context manager that times execution and records telemetry.

        The caller sets ctx.total_tokens and ctx.coordination_tokens inside
        the with-block after the compiled prompt is available.
        """
        ctx = _MeasureContext()
        start = time.perf_counter()
        try:
            yield ctx
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            record = TelemetryRecord(
                capsule_name=capsule_name,
                composition_mode=composition_mode,
                batch_size=batch_size,
                total_tokens=ctx.total_tokens,
                coordination_tokens=ctx.coordination_tokens,
                latency_ms=latency_ms,
            )
            self._records.append(record)

    def record(self, r: TelemetryRecord) -> None:
        """Directly append a pre-built record (used by executor)."""
        self._records.append(r)

    def summary(self) -> dict:
        """
        Aggregate stats across all collected records.

        Returns the metrics the GranularityController consumes (§3.2.2).
        """
        if not self._records:
            return {}

        total_tokens = sum(r.total_tokens for r in self._records)
        total_coord = sum(r.coordination_tokens for r in self._records)
        total_errors = sum(r.error_count for r in self._records)
        total_batch = sum(r.batch_size for r in self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / len(self._records)
        avg_overhead = total_coord / total_tokens if total_tokens > 0 else 0.0
        avg_error_rate = total_errors / total_batch if total_batch > 0 else 0.0
        avg_ctx_util = (
            sum(r.context_utilization for r in self._records) / len(self._records)
        )

        total_input  = sum(r.input_tokens  for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_calls  = sum(r.llm_call_count for r in self._records)

        return {
            "capsule_count": len(self._records),
            "total_tokens": total_tokens,
            "total_coordination_tokens": total_coord,
            "total_reasoning_tokens": total_tokens - total_coord,
            "avg_overhead_ratio": avg_overhead,
            "avg_latency_ms": avg_latency,
            "avg_error_rate": avg_error_rate,
            "avg_context_utilization": avg_ctx_util,
            # T-047: COGS signals
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_llm_calls": total_calls,
            "records_by_mode": _group_by_mode(self._records),
        }

    def reset(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"TelemetryCollector(records={len(self._records)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _MeasureContext:
    """Mutable context passed into the measure() with-block."""
    total_tokens: int = 0
    coordination_tokens: int = 0


# ---------------------------------------------------------------------------
# WindowedCollector — rolling window for controller signal computation
# ---------------------------------------------------------------------------

class WindowedCollector:
    """
    A fixed-size rolling window over TelemetryRecords.

    The GranularityController uses this to compute signals over the last N
    batches rather than the entire run history, making it responsive to
    recent workload changes.

    Usage:
        wc = WindowedCollector(window=5)
        wc.add(record)
        signals = wc.signals()  # aggregated over the last 5 records

    Design plan ref: §3.2.2, §6.1
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._records: deque[TelemetryRecord] = deque(maxlen=window)

    def add(self, record: TelemetryRecord) -> None:
        self._records.append(record)

    @property
    def is_full(self) -> bool:
        """True when the window has collected at least `window` records."""
        return len(self._records) >= self._window

    def signals(self) -> dict:
        """
        Aggregate signals over the current window.

        Returns the four controller input signals:
          overhead_ratio      — avg coordination overhead
          error_rate          — avg errors per item
          context_utilization — avg fraction of context window used
          latency_ms          — avg latency

        Returns an empty dict if no records have been added.
        """
        if not self._records:
            return {}

        recs = list(self._records)
        total_tokens = sum(r.total_tokens for r in recs)
        total_coord = sum(r.coordination_tokens for r in recs)
        total_errors = sum(r.error_count for r in recs)
        total_batch = sum(r.batch_size for r in recs)

        return {
            "overhead_ratio": total_coord / total_tokens if total_tokens > 0 else 0.0,
            "error_rate": total_errors / total_batch if total_batch > 0 else 0.0,
            "context_utilization": sum(r.context_utilization for r in recs) / len(recs),
            "latency_ms": sum(r.latency_ms for r in recs) / len(recs),
            "window_size": len(recs),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"WindowedCollector(window={self._window}, filled={len(self._records)})"


def _group_by_mode(records: list[TelemetryRecord]) -> dict[str, dict]:
    groups: dict[str, list[TelemetryRecord]] = {}
    for r in records:
        groups.setdefault(r.composition_mode, []).append(r)

    result = {}
    for mode, recs in groups.items():
        t = sum(r.total_tokens for r in recs)
        c = sum(r.coordination_tokens for r in recs)
        result[mode] = {
            "count": len(recs),
            "total_tokens": t,
            "avg_overhead_ratio": c / t if t > 0 else 0.0,
            "avg_latency_ms": sum(r.latency_ms for r in recs) / len(recs),
            # T-047
            "total_input_tokens": sum(r.input_tokens for r in recs),
            "total_output_tokens": sum(r.output_tokens for r in recs),
            "total_llm_calls": sum(r.llm_call_count for r in recs),
        }
    return result
