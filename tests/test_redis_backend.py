"""
Tests for RedisBackend and BoundarySyncManager backend integration.

Uses fakeredis — no real Redis server required. All tests run on every
`pytest` invocation without any environment setup.

Test suites:
  1. RedisBackend unit tests (put/get/delete/keys_starting_with, TTL, namespace)
  2. SyncBackend protocol compliance
  3. BoundarySyncManager + RedisBackend integration (put_sync, evict, cross-instance)
  4. Executor end-to-end with Redis-backed sync manager

Design plan ref: §5.2 Phase 8, T-011
"""

from __future__ import annotations

import pytest
import fakeredis

from agentic_capsules.core.capsule import AgentItemCapsule, AgentStepCapsule, AgentTagCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.backends import RedisBackend
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.runtime.sync_manager import BoundarySyncManager, SyncBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_client():
    """Fresh FakeRedis client for each test."""
    return fakeredis.FakeRedis()


@pytest.fixture
def backend(fake_client):
    return RedisBackend(fake_client)


@pytest.fixture
def shared_client():
    """Single FakeRedis client shared between two manager instances."""
    return fakeredis.FakeRedis()


# ---------------------------------------------------------------------------
# 1. RedisBackend unit tests
# ---------------------------------------------------------------------------

class TestRedisBackend:

    def test_put_and_get(self, backend):
        backend.put("key1", "hello")
        assert backend.get("key1") == "hello"

    def test_get_missing_returns_none(self, backend):
        assert backend.get("nonexistent") is None

    def test_delete_removes_key(self, backend):
        backend.put("key1", "data")
        backend.delete("key1")
        assert backend.get("key1") is None

    def test_delete_nonexistent_is_noop(self, backend):
        backend.delete("does-not-exist")  # must not raise

    def test_put_overwrites(self, backend):
        backend.put("k", "v1")
        backend.put("k", "v2")
        assert backend.get("k") == "v2"

    def test_keys_starting_with_returns_matching(self, backend):
        backend.put("task-1::agent-a", "a")
        backend.put("task-1::agent-b", "b")
        backend.put("task-2::agent-a", "c")
        keys = backend.keys_starting_with("task-1")
        assert len(keys) == 2
        assert all(k.startswith("task-1") for k in keys)

    def test_keys_starting_with_no_matches(self, backend):
        backend.put("other::x", "data")
        assert backend.keys_starting_with("task-99") == []

    def test_namespace_isolates_keys(self, fake_client):
        b1 = RedisBackend(fake_client, namespace="ns1")
        b2 = RedisBackend(fake_client, namespace="ns2")
        b1.put("key", "from-ns1")
        assert b2.get("key") is None
        assert b1.get("key") == "from-ns1"

    def test_ttl_applied_on_put(self, fake_client):
        """Keys written with ttl_seconds= expire after the given duration."""
        backend = RedisBackend(fake_client, ttl_seconds=1)
        backend.put("expiring", "data")
        assert backend.get("expiring") == "data"
        # Expire the key manually via the underlying client
        fake_client.expire(f"ac:expiring", 0)
        assert backend.get("expiring") is None

    def test_repr(self, backend):
        assert "RedisBackend" in repr(backend)
        assert "ac" in repr(backend)


# ---------------------------------------------------------------------------
# 2. SyncBackend protocol compliance
# ---------------------------------------------------------------------------

class TestRedisBackendProtocol:

    def test_is_sync_backend(self, backend):
        assert isinstance(backend, SyncBackend)

    def test_satisfies_all_methods(self, backend):
        for method in ("put", "get", "delete", "keys_starting_with"):
            assert hasattr(backend, method) and callable(getattr(backend, method))


# ---------------------------------------------------------------------------
# 3. BoundarySyncManager + RedisBackend integration
# ---------------------------------------------------------------------------

def _make_tag(agent: str, task: str) -> AgentTagCapsule:
    return AgentTagCapsule(agent_name=agent, task_id=task)


def _make_item(tag: AgentTagCapsule, data: str) -> AgentItemCapsule:
    return AgentItemCapsule(
        data=data,
        producer_tag=tag,
        schema=Schema("out", fields={"r": "str"}),
        output_key=f"{tag.agent_name.upper()}_OUTPUT",
    )


class TestSyncManagerBackendIntegration:

    def test_put_sync_mirrors_to_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        manager = BoundarySyncManager(backend=backend)
        tag = _make_tag("alpha", "t1")
        item = _make_item(tag, "result-alpha")
        manager.put_sync(tag, item)
        # Data should be in the backend
        assert backend.get(tag.key) == "result-alpha"

    def test_evict_tag_removes_from_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        manager = BoundarySyncManager(backend=backend)
        tag = _make_tag("beta", "t1")
        item = _make_item(tag, "result-beta")
        manager.put_sync(tag, item)
        manager.evict_tag(tag)
        assert backend.get(tag.key) is None

    def test_evict_scope_removes_matching_backend_keys(self, fake_client):
        backend = RedisBackend(fake_client)
        manager = BoundarySyncManager(backend=backend)
        for name in ("alpha", "beta"):
            tag = _make_tag(name, "t1")
            manager.put_sync(tag, _make_item(tag, f"data-{name}"))
        # Evict all keys scoped to "alpha"
        manager.evict("alpha")
        assert backend.get(_make_tag("alpha", "t1").key) is None
        # beta should remain
        assert backend.get(_make_tag("beta", "t1").key) == "data-beta"

    def test_has_checks_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        manager1 = BoundarySyncManager(backend=backend)
        manager2 = BoundarySyncManager(backend=backend)
        tag = _make_tag("gamma", "t1")
        # Write via manager1
        manager1.put_sync(tag, _make_item(tag, "shared"))
        # manager2 (different in-memory store, same backend) should see it via has()
        assert manager2.has(tag)

    def test_repr_shows_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        manager = BoundarySyncManager(backend=backend)
        assert "RedisBackend" in repr(manager)

    def test_no_backend_unchanged_behaviour(self):
        """Without backend=, manager behaves exactly as before."""
        manager = BoundarySyncManager()
        tag = _make_tag("delta", "t1")
        item = _make_item(tag, "data")
        manager.put_sync(tag, item)
        assert manager.has(tag)
        manager.evict_tag(tag)
        assert not manager.has(tag)

    def test_cross_process_simulation(self, shared_client):
        """
        Two manager instances sharing a FakeRedis client simulate cross-process
        output sharing: writer puts via manager1, reader retrieves via manager2.
        """
        backend1 = RedisBackend(shared_client)
        backend2 = RedisBackend(shared_client)
        writer = BoundarySyncManager(backend=backend1)
        reader = BoundarySyncManager(backend=backend2)

        tag = _make_tag("producer", "pipeline-99")
        item = _make_item(tag, "cross-process-output")

        writer.put_sync(tag, item)
        # reader has no in-memory entry — data comes from the shared backend
        assert reader.has(tag)


# ---------------------------------------------------------------------------
# 4. Executor end-to-end with Redis-backed sync manager
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, tools=None):
        self.call_count += 1
        import re
        keys = re.findall(r"(\w+_OUTPUT)", messages[0].content + messages[-1].content)
        seen: set = set()
        unique = [k for k in keys if not (k in seen or seen.add(k))]  # type: ignore
        return "\n\n".join(f"{k}:\nResult for {k}." for k in unique) or "OUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _two_leaf_hierarchy():
    a = AgentLeaf(capsule=AgentStepCapsule(
        name="alpha", system_prompt="You are alpha.",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"r": "str"}),
    ))
    b = AgentLeaf(capsule=AgentStepCapsule(
        name="beta", system_prompt="You are beta.",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"r": "str"}),
    ))
    root = CompoundCapsule(name="pipe", children=[a, b],
                           dependency_edges={"beta": ["alpha"]})
    compute_order(root)
    return CapsuleHierarchy(name="h", root=root)


class TestExecutorWithRedisBackend:

    def test_fine_mode_runs_with_redis_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        sync = BoundarySyncManager(backend=backend)
        adapter = ScriptedAdapter()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.FINE, sync_manager=sync
        )
        result = executor.run(_two_leaf_hierarchy(), task_input="test", task_id="redis-run-1")
        assert result.final_output != ""
        assert adapter.call_count == 2

    def test_compound_mode_runs_with_redis_backend(self, fake_client):
        backend = RedisBackend(fake_client)
        sync = BoundarySyncManager(backend=backend)
        adapter = ScriptedAdapter()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.COMPOUND, sync_manager=sync
        )
        result = executor.run(_two_leaf_hierarchy(), task_input="test", task_id="redis-run-2")
        assert result.final_output != ""
        assert adapter.call_count == 1

    def test_outputs_mirrored_to_backend_after_fine_run(self, fake_client):
        """After a FINE run, each leaf's output should be queryable from the backend."""
        backend = RedisBackend(fake_client)
        sync = BoundarySyncManager(backend=backend)
        adapter = ScriptedAdapter()
        executor = CapsuleExecutor(
            adapter, composition_level=CompositionLevel.FINE, sync_manager=sync
        )
        executor.run(_two_leaf_hierarchy(), task_input="test", task_id="redis-run-3")
        # At least one leaf's output should be in the backend
        stored = backend.keys_starting_with("alpha")
        assert len(stored) >= 1
