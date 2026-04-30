"""Tests for runtime/sync_manager.py"""

import threading
import time

import pytest

from agentic_capsules.core.capsule import AgentItemCapsule, AgentTagCapsule
from agentic_capsules.core.types import Schema
from agentic_capsules.runtime.sync_manager import BoundarySyncManager


def _tag(name="agent", task="t1"):
    return AgentTagCapsule(agent_name=name, task_id=task)


def _item(tag, data="result"):
    return AgentItemCapsule(
        data=data,
        producer_tag=tag,
        schema=Schema("out", fields={"r": "str"}),
        output_key="AGENT_OUTPUT",
    )


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------

def test_put_then_get():
    sm = BoundarySyncManager()
    tag = _tag()
    item = _item(tag)
    sm.put_sync(tag, item)
    result = sm.get_sync(tag)
    assert result is item


def test_has_returns_false_before_put():
    sm = BoundarySyncManager()
    tag = _tag()
    assert not sm.has(tag)


def test_has_returns_true_after_put():
    sm = BoundarySyncManager()
    tag = _tag()
    sm.put_sync(tag, _item(tag))
    assert sm.has(tag)


# ---------------------------------------------------------------------------
# Blocking get — producer arrives after consumer starts waiting
# ---------------------------------------------------------------------------

def test_get_blocks_until_put():
    sm = BoundarySyncManager()
    tag = _tag("slow_agent")
    item = _item(tag, data="late result")
    results = []

    def consumer():
        results.append(sm.get_sync(tag, timeout=2.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)  # consumer is now blocking
    sm.put_sync(tag, item)
    t.join(timeout=1.0)

    assert len(results) == 1
    assert results[0].data == "late result"


def test_get_sync_timeout_raises():
    sm = BoundarySyncManager()
    tag = _tag("missing_agent")
    with pytest.raises(TimeoutError):
        sm.get_sync(tag, timeout=0.05)


# ---------------------------------------------------------------------------
# Eviction (GC)
# ---------------------------------------------------------------------------

def test_evict_by_scope():
    sm = BoundarySyncManager()
    t1 = _tag("agent_a", "task1")
    t2 = _tag("agent_b", "task1")
    t3 = _tag("agent_c", "task2")
    for t in (t1, t2, t3):
        sm.put_sync(t, _item(t))

    evicted = sm.evict(scope="agent_a")
    assert evicted == 1
    assert not sm.has(t1)
    assert sm.has(t2)
    assert sm.has(t3)


def test_evict_tag():
    sm = BoundarySyncManager()
    tag = _tag()
    sm.put_sync(tag, _item(tag))
    assert sm.evict_tag(tag) is True
    assert not sm.has(tag)
    assert sm.evict_tag(tag) is False  # already gone


def test_len():
    sm = BoundarySyncManager()
    assert len(sm) == 0
    for i in range(3):
        t = _tag(f"agent_{i}")
        sm.put_sync(t, _item(t))
    assert len(sm) == 3
