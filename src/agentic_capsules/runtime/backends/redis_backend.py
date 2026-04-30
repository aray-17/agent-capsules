"""
RedisBackend — SyncBackend implementation backed by Redis (or fakeredis).

Stores BoundarySyncManager item data as UTF-8 strings in a Redis hash or
flat key namespace. Each capsule output is stored under its TagKey so that
workers in separate processes can exchange outputs without a shared in-memory
store.

Usage (production):
    import redis
    from agentic_capsules.runtime.backends import RedisBackend
    from agentic_capsules.runtime.sync_manager import BoundarySyncManager

    client = redis.Redis.from_url("redis://localhost:6379")
    manager = BoundarySyncManager(backend=RedisBackend(client))

Usage (tests — no server required):
    import fakeredis
    from agentic_capsules.runtime.backends import RedisBackend
    from agentic_capsules.runtime.sync_manager import BoundarySyncManager

    client = fakeredis.FakeRedis()
    manager = BoundarySyncManager(backend=RedisBackend(client))

Key format:
    Each item is stored as a flat Redis key: `{namespace}:{tag_key}`
    Default namespace is "ac" (agentic-capsules).

TTL:
    Optional `ttl_seconds` sets a Redis key expiry so stale pipeline
    outputs don't accumulate indefinitely. Default is None (no expiry).

Design plan ref: §5.2 Phase 8, T-011 (Redis SyncBackend)
"""

from __future__ import annotations

from typing import Any


class RedisBackend:
    """
    SyncBackend implementation backed by a Redis-compatible client.

    Args:
        client: A `redis.Redis` instance or any compatible client (e.g.
                `fakeredis.FakeRedis`). Must support `set`, `get`, `delete`,
                and `keys` / `scan_iter` commands.
        namespace: Key prefix prepended to every stored key to avoid
                   collisions with other Redis users. Default: ``"ac"``.
        ttl_seconds: Optional TTL applied to every `put()` call. When set,
                     keys expire automatically after this many seconds.
                     Useful for long-running services that process many
                     pipelines. Default: ``None`` (no expiry).
    """

    def __init__(
        self,
        client: Any,
        namespace: str = "ac",
        ttl_seconds: int | None = None,
    ) -> None:
        self._client = client
        self._ns = namespace
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # SyncBackend Protocol
    # ------------------------------------------------------------------

    def put(self, key: str, data: str) -> None:
        """Store *data* under *key*. Overwrites any existing value."""
        redis_key = self._redis_key(key)
        if self._ttl is not None:
            self._client.set(redis_key, data, ex=self._ttl)
        else:
            self._client.set(redis_key, data)

    def get(self, key: str) -> str | None:
        """Return value for *key*, or None if not present."""
        value = self._client.get(self._redis_key(key))
        if value is None:
            return None
        # redis-py returns bytes; decode to str
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def delete(self, key: str) -> None:
        """Remove *key* from the store. No-op if absent."""
        self._client.delete(self._redis_key(key))

    def keys_starting_with(self, prefix: str) -> list[str]:
        """Return all logical keys whose names start with *prefix*."""
        redis_prefix = self._redis_key(prefix)
        ns_prefix_len = len(f"{self._ns}:")
        try:
            # Prefer scan_iter (non-blocking, production-safe)
            raw_keys = list(self._client.scan_iter(match=f"{redis_prefix}*"))
        except AttributeError:
            # Fallback for clients that only implement keys()
            raw_keys = self._client.keys(f"{redis_prefix}*")
        result = []
        for k in raw_keys:
            k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            result.append(k_str[ns_prefix_len:])
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _redis_key(self, logical_key: str) -> str:
        return f"{self._ns}:{logical_key}"

    def __repr__(self) -> str:
        return f"RedisBackend(namespace={self._ns!r}, ttl={self._ttl})"
