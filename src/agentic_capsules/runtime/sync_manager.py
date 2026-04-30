"""
Boundary Sync Manager — get/put semantics at capsule boundaries.

Implements the agentic analog of the original Capsules model's
producer/consumer synchronization points (§3.2.5).

Key behaviors:
  put(tag, item)  — producer stores its output; unblocks any waiting consumers
  get(tag)        — consumer retrieves output; blocks if producer not yet done
  evict(scope)    — GC: drops all ItemCapsules scoped to a completed compound

Boundary migration: when agents are composed into a CompoundCapsule, their
internal get/put become direct in-memory dict lookups (no asyncio.Event
blocking). Only the compound's external inputs/outputs use the sync protocol.
This directly implements "moving synchronization points to the boundary" (§3.2.5).

Design plan ref: §3.2.5, decision D-4 (in-memory dict + asyncio.Event)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol, runtime_checkable

from ..core.capsule import AgentItemCapsule, AgentTagCapsule
from ..core.types import TagKey


# ---------------------------------------------------------------------------
# SyncBackend Protocol (Phase 6) — swap BoundarySyncManager's in-memory store
# with an external backend (Redis, DynamoDB, etc.) by implementing this interface.
# ---------------------------------------------------------------------------

@runtime_checkable
class SyncBackend(Protocol):
    """
    Protocol for pluggable distributed sync backends.

    Implementors can replace BoundarySyncManager's in-memory dict with
    any external store (Redis, DynamoDB, etc.) by satisfying this interface.

    Usage:
        class RedisBackend:
            def put(self, key: str, data: str) -> None: ...
            def get(self, key: str) -> str | None: ...
            def delete(self, key: str) -> None: ...
            def keys_starting_with(self, prefix: str) -> list[str]: ...

        manager = BoundarySyncManager(backend=RedisBackend())

    Design plan ref: §5.2 Phase 6 (distributed sync backend)
    """

    def put(self, key: str, data: str) -> None:
        """Store *data* under *key*. Overwrites any existing value."""
        ...

    def get(self, key: str) -> str | None:
        """Return value for *key*, or None if not present."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key* from the store. No-op if absent."""
        ...

    def keys_starting_with(self, prefix: str) -> list[str]:
        """Return all keys whose names start with *prefix* (for scope eviction)."""
        ...


class BoundarySyncManager:
    """
    Thread-safe and async-compatible in-memory store for capsule outputs.

    Two modes of access:
      - Synchronous (Phase 1): use put_sync / get_sync
      - Asynchronous (Phase 2+): use await put / await get

    The store key is AgentTagCapsule.key (agent_name::task_id).

    Phase 8 — distributed backend (T-011):
      Pass `backend=RedisBackend(client)` to mirror every put/evict to an
      external store. get_sync falls back to the backend on an in-memory miss,
      enabling workers in separate processes to share pipeline outputs.

    Example:
        import fakeredis
        from agentic_capsules.runtime.backends import RedisBackend
        client = fakeredis.FakeRedis()
        manager = BoundarySyncManager(backend=RedisBackend(client))
    """

    def __init__(self, backend: SyncBackend | None = None) -> None:
        # Completed capsule outputs
        self._store: dict[TagKey, AgentItemCapsule] = {}
        # asyncio Events: created on first get(); resolved on put()
        self._events: dict[TagKey, asyncio.Event] = {}
        # threading Event for synchronous blocking get_sync()
        self._sync_events: dict[TagKey, threading.Event] = {}
        self._lock = threading.Lock()
        # Optional external backend (Redis, etc.)
        self._backend: SyncBackend | None = backend

    # ------------------------------------------------------------------
    # Synchronous interface (Phase 1)
    # ------------------------------------------------------------------

    def put_sync(self, tag: AgentTagCapsule, item: AgentItemCapsule) -> None:
        """
        Store *item* under *tag* and unblock any waiting get_sync() callers.

        When a backend is configured, the item's data string is also written
        to the external store so workers in other processes can retrieve it.
        """
        key = tag.key
        with self._lock:
            self._store[key] = item
            if key in self._sync_events:
                self._sync_events[key].set()
        if self._backend is not None:
            self._backend.put(key, item.data)

    def get_sync(self, tag: AgentTagCapsule, timeout: float | None = None) -> AgentItemCapsule:
        """
        Return the item for *tag*, blocking until it is available.

        Raises TimeoutError if *timeout* seconds elapse before the producer
        calls put_sync().

        Within a CompoundCapsule the executor calls this directly on the
        internal store without going through the event mechanism — that's
        the boundary migration optimization (§3.2.5).
        """
        key = tag.key
        with self._lock:
            if key in self._store:
                return self._store[key]
            # Register a threading.Event before releasing the lock
            if key not in self._sync_events:
                self._sync_events[key] = threading.Event()
            event = self._sync_events[key]

        arrived = event.wait(timeout=timeout)
        if not arrived:
            raise TimeoutError(
                f"Timed out waiting for capsule output at tag {key!r}"
            )
        with self._lock:
            # Check in-memory first; fall back to backend if a remote worker
            # wrote it without going through this manager instance.
            if key in self._store:
                return self._store[key]
            if self._backend is not None:
                data = self._backend.get(key)
                if data is not None:
                    from ..core.capsule import AgentTagCapsule as _Tag
                    # Reconstruct a minimal item from the raw data string
                    stub_tag = _Tag(agent_name=tag.agent_name, task_id=tag.task_id)
                    from ..core.types import Schema
                    stub_item = AgentItemCapsule(
                        data=data,
                        producer_tag=stub_tag,
                        schema=Schema("backend", fields={"data": "str"}),
                        output_key=key,
                    )
                    self._store[key] = stub_item
                    return stub_item
            return self._store[key]  # raises KeyError if absent — timeout path

    def has(self, tag: AgentTagCapsule) -> bool:
        """Return True if the output for *tag* is in the in-memory store or backend."""
        if tag.key in self._store:
            return True
        if self._backend is not None:
            return self._backend.get(tag.key) is not None
        return False

    # ------------------------------------------------------------------
    # Async interface (Phase 2+)
    # ------------------------------------------------------------------

    async def put(self, tag: AgentTagCapsule, item: AgentItemCapsule) -> None:
        """Async put — stores item and signals any waiting coroutines."""
        key = tag.key
        self._store[key] = item
        if key in self._events:
            self._events[key].set()

    async def get(self, tag: AgentTagCapsule) -> AgentItemCapsule:
        """Async get — awaits the producer if not yet available."""
        key = tag.key
        if key not in self._store:
            if key not in self._events:
                self._events[key] = asyncio.Event()
            await self._events[key].wait()
        return self._store[key]

    # ------------------------------------------------------------------
    # Context eviction (GC)
    # ------------------------------------------------------------------

    def evict(self, scope: str) -> int:
        """
        Remove all stored ItemCapsules whose producer tag key starts with *scope*.

        Called when a CompoundCapsule completes — intermediate outputs from
        its constituent agents are no longer needed and are freed from the store.
        Returns the number of entries evicted.

        Design plan ref: §3.2.5 (Context Eviction / GC)
        """
        with self._lock:
            to_delete = [k for k in self._store if k.startswith(scope)]
            for k in to_delete:
                del self._store[k]
                self._sync_events.pop(k, None)
                self._events.pop(k, None)
        if self._backend is not None:
            for k in self._backend.keys_starting_with(scope):
                self._backend.delete(k)
        return len(to_delete)

    def evict_tag(self, tag: AgentTagCapsule) -> bool:
        """Evict a single tag. Returns True if it was present in-memory or backend."""
        key = tag.key
        found = False
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._sync_events.pop(key, None)
                self._events.pop(key, None)
                found = True
        if self._backend is not None:
            if self._backend.get(key) is not None:
                self._backend.delete(key)
                found = True
        return found

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stored_keys(self) -> list[TagKey]:
        """Return all currently stored tag keys (for debugging/testing)."""
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        backend_str = f", backend={self._backend!r}" if self._backend else ""
        return f"BoundarySyncManager(entries={len(self._store)}{backend_str})"
