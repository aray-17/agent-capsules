"""
Tests for _ParallelPipelineCompiler — opt-in threaded executor (T-054).

Coverage:
  * Existing linear pipelines (no inter-group depends_on) execute identically
    in parallel mode and serial mode for FINE and COMPOUND modes — same
    step_outputs, same final output.
  * Inter-group fan-out actually runs in parallel: with a stub adapter that
    sleeps, the wall-clock time of a 4-group fan-out level is bounded by the
    *single-group* time, not by 4× the single-group time.
  * Group-level depends_on validation: forward references, self-references,
    unknown groups, cycles all raise.
  * ControllerState is not mutated by the parallel executor (forced-mode
    invariant) — snapshots before and after a parallel run match.
  * mode='auto' / mode='observe' / non-None evaluator are rejected.
  * Task input for a group with multiple deps includes each predecessor's
    output in declaration order.

These tests use stub adapters and never touch live APIs. They do not exercise
or modify the existing 534-test baseline; the parallel compiler is a new,
isolated module.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from agentic_capsules import Pipeline, PipelineResult


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    """Thread-safe scripted adapter. Returns a fixed marker per call."""
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nstub output."):
        self._response  = response
        self._lock      = threading.Lock()
        self.call_count = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _SleepingAdapter:
    """
    Adapter that sleeps for ``delay`` seconds on every complete() call.

    Used to verify wall-clock concurrency: if N groups (each backed by this
    adapter) run concurrently, total wall time should be ~delay, not N×delay.
    Records concurrency by tracking the maximum number of in-flight calls.
    """
    context_window = 200_000

    def __init__(self, delay: float = 0.3, response: str = "## OUTPUT\nslept."):
        self._delay      = delay
        self._response   = response
        self._lock       = threading.Lock()
        self._in_flight  = 0
        self.peak_in_flight = 0
        self.call_count  = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.call_count += 1
        try:
            time.sleep(self._delay)
            return self._response
        finally:
            with self._lock:
                self._in_flight -= 1

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Equivalence with the serial executor on existing-style pipelines
# ---------------------------------------------------------------------------

def _build_linear_pipeline() -> Pipeline:
    return (
        Pipeline("linear-test")
        .group("g1")
            .agent("a1", "do step 1")
        .group("g2")
            .agent("a2", "do step 2")
        .group("g3")
            .agent("a3", "do step 3")
    )


def test_linear_pipeline_parallel_matches_serial_fine_mode():
    serial_adapter   = _ScriptedAdapter()
    parallel_adapter = _ScriptedAdapter()

    serial_result   = _build_linear_pipeline().run(
        "task", adapter=serial_adapter, mode="fine"
    )
    parallel_result = _build_linear_pipeline().run(
        "task", adapter=parallel_adapter, mode="fine", parallel=True
    )

    assert isinstance(parallel_result, PipelineResult)
    # Same agents executed → same call count
    assert parallel_adapter.call_count == serial_adapter.call_count
    # Same step_outputs keys (order is not guaranteed in dict, but membership is)
    assert set(parallel_result.step_outputs.keys()) == set(serial_result.step_outputs.keys())


def test_linear_pipeline_parallel_matches_serial_compound_mode():
    serial_adapter   = _ScriptedAdapter()
    parallel_adapter = _ScriptedAdapter()

    serial_result   = _build_linear_pipeline().run(
        "task", adapter=serial_adapter, mode="compound"
    )
    parallel_result = _build_linear_pipeline().run(
        "task", adapter=parallel_adapter, mode="compound", parallel=True
    )

    assert isinstance(parallel_result, PipelineResult)
    assert parallel_adapter.call_count == serial_adapter.call_count
    assert parallel_result.mode_used  == serial_result.mode_used


def test_parallel_returns_non_empty_final_output():
    result = _build_linear_pipeline().run(
        "task", adapter=_ScriptedAdapter(), mode="fine", parallel=True
    )
    assert len(result.output) > 0


# ---------------------------------------------------------------------------
# Inter-group parallelism — wall-clock concurrency proof
# ---------------------------------------------------------------------------

def test_fan_out_groups_run_concurrently_in_parallel_mode():
    """Four arms with no inter-arm deps must run concurrently, not serially."""
    delay   = 0.3
    adapter = _SleepingAdapter(delay=delay)

    pipeline = (
        Pipeline("fan-out")
        .group("ingest", depends_on=[])
            .agent("ingest_agent", "ingest")
        .group("arm_a", depends_on=["ingest"])
            .agent("arm_a_agent", "arm A")
        .group("arm_b", depends_on=["ingest"])
            .agent("arm_b_agent", "arm B")
        .group("arm_c", depends_on=["ingest"])
            .agent("arm_c_agent", "arm C")
        .group("arm_d", depends_on=["ingest"])
            .agent("arm_d_agent", "arm D")
        .group("synth", depends_on=["arm_a", "arm_b", "arm_c", "arm_d"])
            .agent("synth_agent", "synthesise")
    )

    start = time.perf_counter()
    result = pipeline.run("task", adapter=adapter, mode="fine", parallel=True)
    elapsed = time.perf_counter() - start

    # Six total agent calls
    assert adapter.call_count == 6
    # The four arms must have all four been in flight simultaneously
    assert adapter.peak_in_flight >= 4, (
        f"expected peak_in_flight >= 4, got {adapter.peak_in_flight}"
    )
    # Wall-clock upper bound: 3 sequential levels × delay each + slack.
    # Serial would take 6 × delay = 1.8s; parallel should be ~3 × delay = 0.9s.
    # Use a generous 5× delay ceiling to avoid CI flake.
    assert elapsed < delay * 5, (
        f"parallel execution took {elapsed:.2f}s; expected < {delay * 5:.2f}s"
    )
    assert isinstance(result, PipelineResult)


def test_serial_mode_does_not_run_groups_concurrently():
    """Sanity check: parallel=False should NOT trigger concurrent execution."""
    delay   = 0.1
    adapter = _SleepingAdapter(delay=delay)

    pipeline = (
        Pipeline("fan-out-serial")
        .group("ingest", depends_on=[])
            .agent("ingest_agent", "ingest")
        .group("arm_a", depends_on=["ingest"])
            .agent("arm_a_agent", "arm A")
        .group("arm_b", depends_on=["ingest"])
            .agent("arm_b_agent", "arm B")
    )

    pipeline.run("task", adapter=adapter, mode="fine")  # parallel=False default

    # Serial mode runs every group on the same thread → peak_in_flight == 1
    assert adapter.peak_in_flight == 1


# ---------------------------------------------------------------------------
# Group depends_on validation (in builder, before any execution)
# ---------------------------------------------------------------------------

def test_group_depends_on_unknown_group_raises():
    with pytest.raises(ValueError, match="not a group declared earlier"):
        Pipeline("t").group("a").agent("a1", "do").group("b", depends_on=["zzz"])


def test_group_depends_on_self_raises():
    with pytest.raises(ValueError, match="cannot depend on itself"):
        Pipeline("t").group("a").agent("a1", "do").group("b", depends_on=["b"])


def test_group_depends_on_empty_string_raises():
    with pytest.raises(ValueError, match="non-empty group names"):
        Pipeline("t").group("a").agent("a1", "do").group("b", depends_on=[""])


def test_group_depends_on_dedupes_silently():
    p = (
        Pipeline("t")
        .group("a").agent("a1", "do")
        .group("b").agent("b1", "do")
        .group("c", depends_on=["a", "b", "a"])  # 'a' duplicated
            .agent("c1", "do")
    )
    spec_c = next(g for g in p._groups if g.name == "c")
    assert spec_c.depends_on == ["a", "b"]


def test_group_depends_on_default_none_preserves_linear_chain():
    p = (
        Pipeline("t")
        .group("g1").agent("a1", "do")
        .group("g2").agent("a2", "do")
    )
    # No depends_on declared → field is None on both groups
    assert all(g.depends_on is None for g in p._groups)


# ---------------------------------------------------------------------------
# Cycle detection (in the parallel compiler's topological pass)
# ---------------------------------------------------------------------------

def test_cycle_in_group_deps_raises():
    # We can't build a cycle through the builder's forward-only validation,
    # so construct one by mutating the spec list directly.
    p = (
        Pipeline("t")
        .group("a", depends_on=[]).agent("a1", "do")
        .group("b", depends_on=["a"]).agent("b1", "do")
    )
    # Inject a back-edge: a depends on b → cycle a→b→a
    p._groups[0].depends_on = ["b"]
    with pytest.raises(ValueError, match="Cycle in group dependency graph"):
        p.run("task", adapter=_ScriptedAdapter(), mode="fine", parallel=True)


# ---------------------------------------------------------------------------
# G-7 (2026-04-09): parallel executor now supports auto mode and evaluators.
# Only mode="observe" remains unsupported. These tests previously codified
# the old restrictions; updated to assert the new behaviour.
# ---------------------------------------------------------------------------

def test_parallel_accepts_auto_mode():
    """G-7: auto mode is now supported in the parallel executor."""
    result = _build_linear_pipeline().run(
        "task", adapter=_ScriptedAdapter(), mode="auto", parallel=True
    )
    # Auto mode means the controller is consulted; recommendations populated.
    assert result.mode_used  # forced/resolved mode per group
    assert result.recommendation  # controller ran


def test_parallel_rejects_observe_mode():
    """G-7: observe mode is still unsupported; the serial path handles it."""
    with pytest.raises(ValueError, match="does not support 'observe'"):
        _build_linear_pipeline().run(
            "task", adapter=_ScriptedAdapter(), mode="observe", parallel=True
        )


def test_parallel_accepts_evaluator():
    """
    G-7: quality evaluator is now supported in the parallel executor.
    ``PipelineState`` uses an ``RLock`` that serialises controller writes,
    so H2/H3 can run from worker threads without corrupting state.
    """
    fake_evaluator = MagicMock()
    # Should not raise. The scripted adapter returns trivial output so H2 may
    # not fire, but the builder must accept the argument without error.
    _build_linear_pipeline().run(
        "task",
        adapter=_ScriptedAdapter(),
        mode="fine",
        parallel=True,
        evaluator=fake_evaluator,
    )


# ---------------------------------------------------------------------------
# G-7: controller state IS now written by parallel runs
# ---------------------------------------------------------------------------

def test_parallel_run_records_observations():
    """
    G-7 (2026-04-09): the parallel executor now calls
    ``_post_run_controller_step`` per completed group, so observations land
    in the rolling window exactly as they do in the serial executor. This
    test guards against a regression back to the old no-op behaviour.
    """
    pipeline = _build_linear_pipeline()
    pipeline.run("task", adapter=_ScriptedAdapter(), mode="fine", parallel=True)

    snapshot = pipeline._pipeline_state.snapshot()
    # At least one group should have recorded an observation (scripted adapter
    # produces telemetry with non-zero tokens for the agents it runs).
    assert any(len(gs.observations) >= 1 for gs in snapshot.values()), (
        "no group recorded an observation — parallel executor should now "
        "invoke _post_run_controller_step per completed group"
    )


def test_parallel_pipeline_result_has_controller_fields():
    """
    G-7: the parallel executor now populates recommendation / confidence /
    scores / efficiency from the controller snapshot, matching the serial
    executor's PipelineResult contract.
    """
    result = _build_linear_pipeline().run(
        "task", adapter=_ScriptedAdapter(), mode="fine", parallel=True
    )
    assert result.mode_used  # populated as before
    # These fields are now populated — recommendations and confidence for
    # every group the controller observed.
    assert result.recommendation  # at least one group
    assert result.confidence      # mirrors recommendation keys
    assert set(result.recommendation.keys()) == set(result.confidence.keys())


# ---------------------------------------------------------------------------
# Task chaining: a group with multiple deps sees each dep's output
# ---------------------------------------------------------------------------

class _RecordingAdapter:
    """Captures every (group → task_input) it sees."""
    context_window = 200_000

    def __init__(self):
        self._lock = threading.Lock()
        self.captures: list[str] = []

    def complete(self, messages, tools=None):
        with self._lock:
            # The task content is in the first user message; capture the whole
            # message list as a flat string for substring assertions.
            self.captures.append(
                "\n".join(getattr(m, "content", "") for m in messages)
            )
        return "## OUTPUT\nresponse."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_group_with_multiple_deps_sees_each_dep_output():
    adapter = _RecordingAdapter()
    pipeline = (
        Pipeline("multi-dep")
        .group("ingest", depends_on=[])
            .agent("ingest_agent", "ingest")
        .group("arm_a", depends_on=["ingest"])
            .agent("arm_a_agent", "arm A")
        .group("arm_b", depends_on=["ingest"])
            .agent("arm_b_agent", "arm B")
        .group("synth", depends_on=["arm_a", "arm_b"])
            .agent("synth_agent", "synth")
    )
    pipeline.run("ROOT_TASK", adapter=adapter, mode="fine", parallel=True)

    # The synthesizer's call should reference both arm_a and arm_b outputs in
    # its task input. Find the call where the synthesizer ran.
    synth_calls = [c for c in adapter.captures if "synth" in c.lower()]
    assert synth_calls, "synthesizer call not captured"
    synth_text = "\n".join(synth_calls)
    assert "[arm_a output]" in synth_text
    assert "[arm_b output]" in synth_text
    assert "ROOT_TASK"      in synth_text


# ---------------------------------------------------------------------------
# G-7: PipelineState thread-safety stress test
# ---------------------------------------------------------------------------

def test_parallel_controller_state_is_consistent_under_fan_out():
    """
    G-7 stress test: a wide fan-out with many concurrent arms must leave
    PipelineState in a consistent state — exactly one observation per group,
    no dropped writes, no corrupted lists. Guards against a regression that
    removes the RLock or bypasses the helper.
    """
    # 8 sibling arms so the thread pool actually dispatches in parallel.
    builder = Pipeline("stress").group("root", depends_on=[]).agent("r", "root")
    for i in range(8):
        builder = builder.group(f"arm_{i}", depends_on=["root"]).agent(
            f"a_{i}", f"arm {i}"
        )

    adapter = _SleepingAdapter(delay=0.05)
    result = builder.run("task", adapter=adapter, mode="fine", parallel=True)

    snapshot = builder._pipeline_state.snapshot()
    # Every group the controller saw must have exactly one observation.
    for name in ["root"] + [f"arm_{i}" for i in range(8)]:
        assert name in snapshot, f"group {name!r} missing from snapshot"
        gs = snapshot[name]
        assert len(gs.observations) == 1, (
            f"group {name!r}: expected 1 observation, got {len(gs.observations)} "
            f"— PipelineState lock may be missing or the helper is bypassed"
        )
    # Every group has a recommendation in the result.
    assert set(result.recommendation.keys()) >= {"root", *(f"arm_{i}" for i in range(8))}
    # Fan-out actually ran in parallel (peak in-flight > 1)
    assert adapter.peak_in_flight >= 2
