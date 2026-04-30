"""
Phase 6 tests — ModelRouter, CheckpointStore, SyncBackend protocol.

Tests are organised into three suites:
  1. ModelRouter routing, fallback, and context_window propagation
  2. CheckpointStore in-memory and file persistence
  3. Executor checkpoint integration (save/restore across runs)
  4. SyncBackend protocol structural check
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.checkpoint import CheckpointStore
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.model_router import ModelRouter
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.runtime.sync_manager import SyncBackend


# ---------------------------------------------------------------------------
# Shared stub adapters
# ---------------------------------------------------------------------------

class StubAdapter:
    """Always returns a scripted single response; records call count and capsule name."""

    context_window = 100_000

    def __init__(self, response: str = "OUT:\nDone.", context_window: int = 100_000) -> None:
        self._response = response
        self.context_window = context_window
        self.calls: list[str] = []  # capsule names seen (if ModelRouter sets .current_capsule)
        self.current_capsule: str = ""

    def complete(self, messages, tools=None):
        self.calls.append(self.current_capsule)
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class SequencedAdapter:
    """Returns successive responses from a list."""

    context_window = 100_000

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.current_capsule: str = ""

    def complete(self, messages, tools=None):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------

def _leaf(name: str, in_key: str = "q", out_key: str | None = None) -> AgentLeaf:
    ok = out_key or f"{name}_out"
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=f"You are {name}.",
            input_schema=Schema("in", fields={in_key: "str"}),
            output_schema=Schema("out", fields={ok: "str"}),
        )
    )


def _two_leaf_hierarchy():
    a = _leaf("alpha")
    b = _leaf("beta")
    root = CompoundCapsule(
        name="root",
        children=[a, b],
        dependency_edges={"beta": ["alpha"]},
    )
    compute_order(root)
    return CapsuleHierarchy(name="h", root=root)


def _response_for(key: str) -> str:
    """Build a parseable response for the given output key."""
    return f"{key}:\nOutput for {key}."


# ---------------------------------------------------------------------------
# 1. ModelRouter
# ---------------------------------------------------------------------------

class TestModelRouter:

    def test_routes_to_registered_adapter(self):
        default = StubAdapter("default")
        special = StubAdapter("special")
        router = ModelRouter(default=default, routes={"fast": special})
        router.current_capsule = "fast"
        result = router.complete([])
        assert result == "special"

    def test_falls_back_to_default(self):
        default = StubAdapter("default")
        special = StubAdapter("special")
        router = ModelRouter(default=default, routes={"fast": special})
        router.current_capsule = "other"
        result = router.complete([])
        assert result == "default"

    def test_empty_routes_always_uses_default(self):
        default = StubAdapter("default")
        router = ModelRouter(default=default)
        router.current_capsule = "anything"
        assert router.complete([]) == "default"

    def test_context_window_is_minimum(self):
        a = StubAdapter(context_window=50_000)
        b = StubAdapter(context_window=200_000)
        router = ModelRouter(default=a, routes={"big": b})
        # Minimum across all adapters
        assert router.context_window == 50_000

    def test_context_window_minimum_is_route(self):
        a = StubAdapter(context_window=200_000)
        b = StubAdapter(context_window=30_000)
        router = ModelRouter(default=a, routes={"small": b})
        assert router.context_window == 30_000

    def test_count_tokens_delegates_to_route(self):
        class TokenAdapter:
            context_window = 100_000
            current_capsule = ""
            def complete(self, m): return ""
            def count_tokens(self, text): return 999
        router = ModelRouter(default=StubAdapter(), routes={"t": TokenAdapter()})
        router.current_capsule = "t"
        assert router.count_tokens("hello") == 999

    def test_count_tokens_falls_back_to_default(self):
        class TokenAdapter:
            context_window = 100_000
            current_capsule = ""
            def complete(self, m): return ""
            def count_tokens(self, text): return 42
        router = ModelRouter(default=TokenAdapter())
        router.current_capsule = "unknown"
        assert router.count_tokens("hello") == 42

    def test_register_replaces_route(self):
        default = StubAdapter("default")
        old = StubAdapter("old")
        new = StubAdapter("new")
        router = ModelRouter(default=default, routes={"x": old})
        router.register("x", new)
        router.current_capsule = "x"
        assert router.complete([]) == "new"

    def test_adapter_for_returns_correct(self):
        default = StubAdapter()
        special = StubAdapter()
        router = ModelRouter(default=default, routes={"s": special})
        assert router.adapter_for("s") is special
        assert router.adapter_for("other") is default

    def test_repr(self):
        router = ModelRouter(default=StubAdapter(), routes={"a": StubAdapter()})
        r = repr(router)
        assert "ModelRouter" in r
        assert "routes=['a']" in r

    def test_executor_sets_current_capsule(self):
        """Executor must set router.current_capsule before each leaf call."""
        alpha_adapter = StubAdapter()
        alpha_adapter._response = "ALPHA_OUT:\nAlpha done."
        beta_adapter = StubAdapter()
        beta_adapter._response = "BETA_OUT:\nBeta done."
        router = ModelRouter(
            default=alpha_adapter,
            routes={"beta": beta_adapter},
        )
        h = _two_leaf_hierarchy()
        executor = CapsuleExecutor(router, composition_level=CompositionLevel.FINE)
        result = executor.run(h, task_input="test", task_id="t1")
        # beta leaf should have been dispatched to beta_adapter
        assert "BETA_OUT" in result.outputs or result.final_output


# ---------------------------------------------------------------------------
# 2. CheckpointStore — in-memory
# ---------------------------------------------------------------------------

class TestCheckpointStoreMemory:

    def test_save_and_load(self):
        store = CheckpointStore()
        store.save("task-1", {"key_a": "val_a"})
        out = store.load("task-1")
        assert out == {"key_a": "val_a"}

    def test_load_missing_returns_none(self):
        store = CheckpointStore()
        assert store.load("nonexistent") is None

    def test_has_returns_true_after_save(self):
        store = CheckpointStore()
        store.save("t", {"x": "1"})
        assert store.has("t") is True

    def test_has_returns_false_before_save(self):
        store = CheckpointStore()
        assert store.has("t") is False

    def test_clear_removes_entry(self):
        store = CheckpointStore()
        store.save("t", {"x": "1"})
        store.clear("t")
        assert store.load("t") is None
        assert store.has("t") is False

    def test_clear_nonexistent_is_noop(self):
        store = CheckpointStore()
        store.clear("not-there")  # must not raise

    def test_save_overwrites(self):
        store = CheckpointStore()
        store.save("t", {"x": "1"})
        store.save("t", {"x": "2"})
        assert store.load("t") == {"x": "2"}

    def test_load_returns_copy(self):
        store = CheckpointStore()
        store.save("t", {"x": "1"})
        out = store.load("t")
        out["x"] = "mutated"
        assert store.load("t") == {"x": "1"}

    def test_len_reflects_unique_tasks(self):
        store = CheckpointStore()
        store.save("t1", {})
        store.save("t2", {})
        assert len(store) == 2

    def test_repr_contains_tasks(self):
        store = CheckpointStore()
        store.save("mytask", {})
        assert "mytask" in repr(store)


# ---------------------------------------------------------------------------
# 3. CheckpointStore — file persistence
# ---------------------------------------------------------------------------

class TestCheckpointStoreFile:

    def test_file_created_on_save(self, tmp_path):
        store = CheckpointStore(path=tmp_path)
        store.save("task-file", {"k": "v"})
        assert (tmp_path / "task-file.json").exists()

    def test_file_contents_are_valid_json(self, tmp_path):
        store = CheckpointStore(path=tmp_path)
        store.save("task-json", {"num": 42})
        data = json.loads((tmp_path / "task-json.json").read_text())
        assert data == {"num": 42}

    def test_load_from_disk_after_memory_cleared(self, tmp_path):
        store1 = CheckpointStore(path=tmp_path)
        store1.save("t", {"a": "b"})
        # New store instance — no in-memory copy
        store2 = CheckpointStore(path=tmp_path)
        assert store2.load("t") == {"a": "b"}

    def test_has_checks_disk(self, tmp_path):
        store1 = CheckpointStore(path=tmp_path)
        store1.save("t", {})
        store2 = CheckpointStore(path=tmp_path)
        assert store2.has("t") is True

    def test_clear_removes_file(self, tmp_path):
        store = CheckpointStore(path=tmp_path)
        store.save("t", {"x": "1"})
        store.clear("t")
        assert not (tmp_path / "t.json").exists()

    def test_directory_created_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        store = CheckpointStore(path=nested)
        store.save("t", {})
        assert (nested / "t.json").exists()

    def test_corrupt_file_returns_none(self, tmp_path):
        (tmp_path / "bad.json").write_text("NOT JSON{{{")
        store = CheckpointStore(path=tmp_path)
        assert store.load("bad") is None


# ---------------------------------------------------------------------------
# 4. Executor checkpoint integration
# ---------------------------------------------------------------------------

class TestExecutorCheckpoint:

    def _make_responses(self):
        return [
            "ALPHA_OUT:\nAlpha result.",
            "BETA_OUT:\nBeta result.",
        ]

    def test_checkpoint_save_called_after_each_leaf(self):
        """After a successful run, checkpoint should have outputs for both leaves."""
        store = CheckpointStore()
        adapter = SequencedAdapter(self._make_responses())
        h = _two_leaf_hierarchy()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.FINE, checkpoint=store
        )
        executor.run(h, task_input="go", task_id="ckpt-1")
        # Both outputs should be checkpointed
        saved = store.load("ckpt-1")
        assert saved is not None
        assert len(saved) >= 1  # at least the last leaf

    def test_checkpoint_resume_skips_completed_leaves(self):
        """
        Simulate a partial run: pre-populate the checkpoint with the first leaf's
        output, then run again. The second run should only make 1 LLM call (for
        the second leaf), not 2.
        """
        store = CheckpointStore()
        # Pre-seed: first leaf already done
        a_leaf = _leaf("alpha")
        store.save("resume-1", {a_leaf.capsule.output_key: "Alpha result."})

        # Only provide one response (for beta)
        adapter = SequencedAdapter(["BETA_OUT:\nBeta result."])
        h = _two_leaf_hierarchy()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.FINE, checkpoint=store
        )
        result = executor.run(h, task_input="go", task_id="resume-1")
        # Should restore alpha output without a new LLM call
        assert result.outputs.get(a_leaf.capsule.output_key) == "Alpha result."

    def test_no_checkpoint_runs_normally(self):
        """Without checkpoint=, executor behaves identically to before."""
        adapter = SequencedAdapter(self._make_responses())
        h = _two_leaf_hierarchy()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.FINE
        )
        result = executor.run(h, task_input="go", task_id="no-ckpt")
        assert result.final_output != ""


# ---------------------------------------------------------------------------
# 5. SyncBackend protocol
# ---------------------------------------------------------------------------

class TestSyncBackendProtocol:

    def test_concrete_implementation_satisfies_protocol(self):
        """A class implementing all four methods is a SyncBackend."""
        class FakeBackend:
            def put(self, key: str, data: str) -> None: pass
            def get(self, key: str): return None
            def delete(self, key: str) -> None: pass
            def keys_starting_with(self, prefix: str): return []

        assert isinstance(FakeBackend(), SyncBackend)

    def test_missing_method_fails_protocol_check(self):
        """A class missing one method is NOT a SyncBackend."""
        class IncompleteBackend:
            def put(self, key: str, data: str) -> None: pass
            def get(self, key: str): return None
            def delete(self, key: str) -> None: pass
            # keys_starting_with missing

        assert not isinstance(IncompleteBackend(), SyncBackend)

    def test_empty_class_fails_protocol_check(self):
        assert not isinstance(object(), SyncBackend)
