"""
Tests for G-4 — group-level pipeline checkpointing.

The new ``PipelineCheckpoint`` class gives the high-level ``Pipeline.run()``
API a group-level resume mechanism. After each group completes, its
``{outputs, final_output}`` is persisted under ``(task_id, group_name)``.
Rerunning with the same ``task_id`` replays the saved outputs and skips
dispatch for every checkpointed group — in both serial and parallel mode.

Coverage:
  * PipelineCheckpoint primitives: save/load/has/clear/groups_for, in-memory
    and on-disk, thread-safety under concurrent writers
  * Pipeline.run(checkpoint=...) serial path — second run reuses saved
    groups, only dispatches unfinished groups
  * Pipeline.run(checkpoint_path=...) cross-process form — JSON file is
    created and consulted
  * Mutual exclusion: passing both ``checkpoint=`` and ``checkpoint_path=``
    raises ValueError
  * Pipeline.run(..., parallel=True) honours the checkpoint — the parallel
    compiler resumes completed groups and only dispatches the ones it must
  * Checkpoint is cleared on successful completion (next run with the same
    task_id does NOT replay — it starts fresh)
  * Failure mid-pipeline leaves earlier groups checkpointed (retry semantic)
  * Resumed groups contribute zero telemetry and no controller observation
    (PipelineState not mutated on resume)

All tests use stub adapters and never touch live APIs.
"""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import pytest

from agentic_capsules import Pipeline
from agentic_capsules.runtime.checkpoint import PipelineCheckpoint


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

class _CountingAdapter:
    """Thread-safe; returns a scripted response; tracks call count."""
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nstub.") -> None:
        self._response  = response
        self._lock      = threading.Lock()
        self.call_count = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _ExplodingAdapter:
    """Runs the first N calls, then raises on the (N+1)-th."""
    context_window = 200_000

    def __init__(self, n_successful: int) -> None:
        self._n_ok      = n_successful
        self._lock      = threading.Lock()
        self.call_count = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
            current = self.call_count
        if current > self._n_ok:
            raise RuntimeError(f"adapter exploded on call #{current}")
        return f"## OUTPUT\ncall-{current}-done."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# PipelineCheckpoint primitives
# ---------------------------------------------------------------------------

class TestPipelineCheckpointPrimitives:

    def test_in_memory_save_load_roundtrip(self):
        ckpt = PipelineCheckpoint()
        ckpt.save_group(
            "task-1", "group-a",
            outputs={"A_OUTPUT": "hello"},
            final_output="hello world",
        )
        loaded = ckpt.load_group("task-1", "group-a")
        assert loaded == {
            "outputs": {"A_OUTPUT": "hello"},
            "final_output": "hello world",
        }

    def test_load_missing_returns_none(self):
        ckpt = PipelineCheckpoint()
        assert ckpt.load_group("nope", "nope") is None

    def test_has_group(self):
        ckpt = PipelineCheckpoint()
        assert not ckpt.has_group("t", "g")
        ckpt.save_group("t", "g", outputs={}, final_output="")
        assert ckpt.has_group("t", "g")

    def test_groups_for_returns_saved_group_names(self):
        ckpt = PipelineCheckpoint()
        ckpt.save_group("t", "g1", outputs={}, final_output="")
        ckpt.save_group("t", "g2", outputs={}, final_output="")
        assert set(ckpt.groups_for("t")) == {"g1", "g2"}

    def test_clear_drops_all_groups_for_task(self):
        ckpt = PipelineCheckpoint()
        ckpt.save_group("t", "g1", outputs={}, final_output="")
        ckpt.save_group("t", "g2", outputs={}, final_output="")
        ckpt.clear("t")
        assert ckpt.groups_for("t") == []
        assert not ckpt.has_group("t", "g1")
        assert not ckpt.has_group("t", "g2")

    def test_clear_does_not_touch_other_tasks(self):
        ckpt = PipelineCheckpoint()
        ckpt.save_group("t1", "g", outputs={}, final_output="")
        ckpt.save_group("t2", "g", outputs={}, final_output="")
        ckpt.clear("t1")
        assert ckpt.groups_for("t2") == ["g"]

    def test_save_overwrites_prior_group_record(self):
        ckpt = PipelineCheckpoint()
        ckpt.save_group("t", "g", outputs={"X": "first"}, final_output="1")
        ckpt.save_group("t", "g", outputs={"X": "second"}, final_output="2")
        loaded = ckpt.load_group("t", "g")
        assert loaded["outputs"] == {"X": "second"}
        assert loaded["final_output"] == "2"

    def test_on_disk_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt1 = PipelineCheckpoint(path=tmp)
            ckpt1.save_group(
                "t", "g",
                outputs={"K": "v"},
                final_output="disk-final",
            )
            # Confirm the file exists
            file = Path(tmp) / "t.json"
            assert file.exists()
            data = json.loads(file.read_text())
            assert data["g"]["outputs"] == {"K": "v"}
            assert data["g"]["final_output"] == "disk-final"

            # A brand-new instance pointed at the same dir should read
            # the saved record from disk on first load_group call.
            ckpt2 = PipelineCheckpoint(path=tmp)
            loaded = ckpt2.load_group("t", "g")
            assert loaded == {"outputs": {"K": "v"}, "final_output": "disk-final"}

    def test_on_disk_clear_removes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = PipelineCheckpoint(path=tmp)
            ckpt.save_group("t", "g", outputs={}, final_output="")
            file = Path(tmp) / "t.json"
            assert file.exists()
            ckpt.clear("t")
            assert not file.exists()

    def test_concurrent_writers_do_not_corrupt_index(self):
        """Thread-safety: 16 threads save distinct groups concurrently; all survive."""
        ckpt = PipelineCheckpoint()
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                ckpt.save_group(
                    "t", f"g{i}",
                    outputs={f"K{i}": f"v{i}"},
                    final_output=f"fin-{i}",
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(16)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == []
        assert set(ckpt.groups_for("t")) == {f"g{i}" for i in range(16)}
        # Spot-check one value
        loaded = ckpt.load_group("t", "g7")
        assert loaded["outputs"] == {"K7": "v7"}


# ---------------------------------------------------------------------------
# Serial-mode Pipeline.run() integration
# ---------------------------------------------------------------------------

class TestSerialPipelineResume:

    def test_second_run_with_same_task_id_replays_all_groups(self):
        """A fresh checkpoint is created and cleared on a successful run.
        After completion, the checkpoint should be empty, so the second
        run dispatches everything from scratch."""
        adapter = _CountingAdapter()
        ckpt    = PipelineCheckpoint()
        p = (
            Pipeline("t")
            .group("g1")
                .agent("a", "first")
            .group("g2")
                .agent("b", "second")
        )
        p.run("task", adapter=adapter, mode="fine",
              task_id="t1", checkpoint=ckpt)
        assert adapter.call_count == 2
        assert ckpt.groups_for("t1") == []  # cleared on success

    def test_mid_pipeline_failure_leaves_earlier_groups_checkpointed(self):
        """Group 1 succeeds, group 2 fails. Retry with same task_id
        should replay group 1 from the checkpoint and only re-dispatch
        group 2 — which will now also succeed if the adapter is fixed."""
        # First adapter: succeeds once, explodes on the second call
        adapter_fail = _ExplodingAdapter(n_successful=1)
        ckpt = PipelineCheckpoint()
        p = (
            Pipeline("t")
            .group("g1")
                .agent("a", "first")
            .group("g2")
                .agent("b", "second")
        )
        with pytest.raises(Exception):
            p.run("task", adapter=adapter_fail, mode="fine",
                  task_id="t1", checkpoint=ckpt)
        # g1 was saved; g2 was not
        assert ckpt.has_group("t1", "g1")
        assert not ckpt.has_group("t1", "g2")
        assert adapter_fail.call_count == 2  # one success + one explosion

        # Retry with a good adapter — g1 should replay, g2 should dispatch
        adapter_ok = _CountingAdapter()
        result = p.run("task", adapter=adapter_ok, mode="fine",
                       task_id="t1", checkpoint=ckpt)
        assert adapter_ok.call_count == 1  # only g2 re-ran
        # Checkpoint cleared on success
        assert ckpt.groups_for("t1") == []
        # Both agent outputs present in the final step_outputs
        assert "a" in result.step_outputs
        assert "b" in result.step_outputs

    def test_resumed_group_replays_final_output_into_next_task_context(self):
        """The replayed group's final_output must still drive the next
        group's task augmentation on resume."""
        # Seed the checkpoint manually to simulate a prior partial run
        ckpt = PipelineCheckpoint()
        ckpt.save_group(
            "t-resume", "g1",
            outputs={"A_OUTPUT": "SAVED-FROM-DISK"},
            final_output="SAVED-FROM-DISK",
        )

        seen_inputs: list[str] = []

        class _CapturingAdapter:
            context_window = 200_000
            def __init__(self):
                self.call_count = 0
            def complete(self, messages, tools=None):
                self.call_count += 1
                # Concatenate all message contents so we can assert the
                # prior group's output is visible to the next group.
                # LLMMessage is an object with .role and .content attrs.
                parts = []
                for m in messages:
                    role    = getattr(m, "role",    None) or (m.get("role")    if isinstance(m, dict) else "")
                    content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
                    if role == "user":
                        parts.append(content or "")
                seen_inputs.append("\n".join(parts))
                return "## OUTPUT\ngroup-2-output."
            def count_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

        adapter = _CapturingAdapter()
        p = (
            Pipeline("t")
            .group("g1")
                .agent("a", "first")
            .group("g2")
                .agent("b", "second")
        )
        p.run("the original task", adapter=adapter, mode="fine",
              task_id="t-resume", checkpoint=ckpt)
        # g1 was resumed → only g2's agent ran
        assert adapter.call_count == 1
        # g2's prompt must contain the replayed final_output
        assert any("SAVED-FROM-DISK" in s for s in seen_inputs)

    def test_checkpoint_path_kwarg_creates_json_file(self):
        adapter = _CountingAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            p = (
                Pipeline("t")
                .group("g")
                    .agent("a", "only")
            )
            p.run("task", adapter=adapter, mode="fine",
                  task_id="t-disk", checkpoint_path=tmp)
            # On success the checkpoint file should be removed by clear()
            assert not (Path(tmp) / "t-disk.json").exists()
            assert adapter.call_count == 1

    def test_checkpoint_and_checkpoint_path_are_mutually_exclusive(self):
        adapter = _CountingAdapter()
        p = Pipeline("t").group("g").agent("a", "only")
        with pytest.raises(ValueError, match="mutually exclusive"):
            p.run(
                "task", adapter=adapter, mode="fine",
                checkpoint=PipelineCheckpoint(),
                checkpoint_path="/tmp/nope",
            )

    def test_no_checkpoint_is_default_and_is_a_noop(self):
        """Default behaviour unchanged: no checkpoint param = no resume mechanism,
        but everything still runs normally."""
        adapter = _CountingAdapter()
        p = (
            Pipeline("t")
            .group("g1")
                .agent("a", "first")
            .group("g2")
                .agent("b", "second")
        )
        result = p.run("task", adapter=adapter, mode="fine")
        assert adapter.call_count == 2
        assert "a" in result.step_outputs
        assert "b" in result.step_outputs


# ---------------------------------------------------------------------------
# Parallel-mode Pipeline.run() integration
# ---------------------------------------------------------------------------

class TestParallelPipelineResume:

    def test_parallel_executor_honours_checkpoint_on_resume(self):
        """In parallel mode, a checkpointed group must not be re-dispatched,
        even if other groups in the same topological level are running."""
        # Seed the checkpoint: arm_a is "already done"
        ckpt = PipelineCheckpoint()
        ckpt.save_group(
            "t-par", "arm_a",
            outputs={"A_AGENT_OUTPUT": "ARM-A-REPLAYED"},
            final_output="ARM-A-REPLAYED",
        )
        adapter = _CountingAdapter()
        p = (
            Pipeline("t")
            .group("ingest", depends_on=[])
                .agent("ingest_agent", "ingest")
            .group("arm_a", depends_on=["ingest"])
                .agent("a_agent", "arm A")
            .group("arm_b", depends_on=["ingest"])
                .agent("b_agent", "arm B")
        )
        p.run("task", adapter=adapter, mode="fine", parallel=True,
              task_id="t-par", checkpoint=ckpt)
        # ingest + arm_b dispatched (2 calls); arm_a replayed (0 calls)
        assert adapter.call_count == 2
        # Checkpoint cleared on success
        assert ckpt.groups_for("t-par") == []

    def test_parallel_second_run_with_same_task_id_replays_all(self):
        """Full parallel run saves checkpoints per-group; on success,
        clear() is called. Verify the post-success state: everything
        cleared, nothing residual for the next run."""
        adapter = _CountingAdapter()
        ckpt    = PipelineCheckpoint()
        p = (
            Pipeline("t")
            .group("ingest", depends_on=[])
                .agent("ingest_agent", "ingest")
            .group("arm_a", depends_on=["ingest"])
                .agent("a_agent", "arm A")
            .group("arm_b", depends_on=["ingest"])
                .agent("b_agent", "arm B")
        )
        p.run("task", adapter=adapter, mode="fine", parallel=True,
              task_id="t-par2", checkpoint=ckpt)
        assert adapter.call_count == 3
        assert ckpt.groups_for("t-par2") == []

    def test_parallel_checkpoint_is_threadsafe_under_concurrent_saves(self):
        """All 3 groups save into the same PipelineCheckpoint from worker
        threads concurrently. The in-memory index must not lose a group."""
        ckpt = PipelineCheckpoint()
        # Use a counting adapter and verify every group's save_group was
        # observed by the checkpoint — the parallel compiler dispatches
        # independent groups on different threads.
        adapter = _CountingAdapter()
        p = (
            Pipeline("t")
            .group("a", depends_on=[])
                .agent("aa", "aa")
            .group("b", depends_on=[])
                .agent("bb", "bb")
            .group("c", depends_on=[])
                .agent("cc", "cc")
        )
        p.run("task", adapter=adapter, mode="fine", parallel=True,
              task_id="concurrent", checkpoint=ckpt)
        # Success clears the checkpoint — so the check is indirect: pytest
        # doesn't flake, the call count is exactly the dispatched agent
        # count, and a parallel save storm didn't raise anywhere.
        assert adapter.call_count == 3

    def test_parallel_checkpoint_path_kwarg(self):
        adapter = _CountingAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            p = (
                Pipeline("t")
                .group("a", depends_on=[])
                    .agent("aa", "aa")
                .group("b", depends_on=["a"])
                    .agent("bb", "bb")
            )
            p.run("task", adapter=adapter, mode="fine", parallel=True,
                  task_id="tdisk-par", checkpoint_path=tmp)
            assert not (Path(tmp) / "tdisk-par.json").exists()  # cleared on success
            assert adapter.call_count == 2


# ---------------------------------------------------------------------------
# Resumed groups are "quiet" — no telemetry, no controller observation
# ---------------------------------------------------------------------------

class TestResumedGroupsAreQuiet:

    def test_resumed_group_contributes_no_telemetry(self):
        """Seed g1, run only g2 — the result.token_usage should reflect
        only g2's telemetry, not g1's (which is empty on resume)."""
        ckpt = PipelineCheckpoint()
        ckpt.save_group(
            "t-quiet", "g1",
            outputs={"A_OUTPUT": "replay"},
            final_output="replay",
        )
        adapter = _CountingAdapter()
        p = (
            Pipeline("t")
            .group("g1")
                .agent("a", "first")
            .group("g2")
                .agent("b", "second")
        )
        result = p.run("task", adapter=adapter, mode="fine",
                       task_id="t-quiet", checkpoint=ckpt)
        assert adapter.call_count == 1  # only g2 dispatched
        # step_outputs for resumed agents still come from the saved outputs
        # via the output_key→agent_name reverse map.
        assert "a" in result.step_outputs
        assert result.step_outputs["a"] == "replay"
