"""Tests for controller/telemetry.py"""

import pytest

from agentic_capsules.controller.telemetry import TelemetryCollector, TelemetryRecord


def _record(total=100, coord=30, mode="ITERATION", batch=5, latency=50.0):
    return TelemetryRecord(
        capsule_name="analyst",
        composition_mode=mode,
        batch_size=batch,
        total_tokens=total,
        coordination_tokens=coord,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------

def test_overhead_ratio():
    r = _record(total=100, coord=40)
    assert r.overhead_ratio == pytest.approx(0.40)


def test_reasoning_tokens():
    r = _record(total=100, coord=30)
    assert r.reasoning_tokens == 70


def test_zero_total_tokens_no_division_error():
    r = _record(total=0, coord=0)
    assert r.overhead_ratio == 0.0


def test_repr_contains_key_info():
    r = _record()
    s = repr(r)
    assert "ITERATION" in s
    assert "overhead" in s.lower()


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

def test_collect_records():
    tc = TelemetryCollector()
    tc.record(_record())
    tc.record(_record())
    assert len(tc) == 2


def test_summary_empty():
    tc = TelemetryCollector()
    assert tc.summary() == {}


def test_summary_avg_overhead():
    tc = TelemetryCollector()
    tc.record(_record(total=100, coord=20))   # 20%
    tc.record(_record(total=100, coord=40))   # 40%
    s = tc.summary()
    # avg = (20+40) / (100+100) = 30%
    assert s["avg_overhead_ratio"] == pytest.approx(0.30)


def test_summary_total_tokens():
    tc = TelemetryCollector()
    tc.record(_record(total=100, coord=10))
    tc.record(_record(total=200, coord=20))
    s = tc.summary()
    assert s["total_tokens"] == 300
    assert s["total_coordination_tokens"] == 30
    assert s["total_reasoning_tokens"] == 270


def test_summary_groups_by_mode():
    tc = TelemetryCollector()
    tc.record(_record(mode="ITERATION", total=100, coord=20))
    tc.record(_record(mode="FINE", total=50, coord=5))
    s = tc.summary()
    assert "ITERATION" in s["records_by_mode"]
    assert "FINE" in s["records_by_mode"]


def test_reset_clears_records():
    tc = TelemetryCollector()
    tc.record(_record())
    tc.reset()
    assert len(tc) == 0
    assert tc.summary() == {}


def test_measure_context_manager():
    tc = TelemetryCollector()
    with tc.measure("analyst", "ITERATION", batch_size=5) as ctx:
        ctx.total_tokens = 200
        ctx.coordination_tokens = 50
    assert len(tc) == 1
    r = tc.records[0]
    assert r.total_tokens == 200
    assert r.coordination_tokens == 50
    assert r.batch_size == 5
    assert r.latency_ms >= 0
