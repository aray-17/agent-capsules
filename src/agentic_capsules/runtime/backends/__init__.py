"""
Runtime backends — pluggable storage backends for BoundarySyncManager.

Available backends:
  RedisBackend — stores capsule outputs in a Redis instance (or fakeredis
                 in tests), enabling cross-process pipeline execution.

Usage:
    import fakeredis
    from agentic_capsules.runtime.backends import RedisBackend
    from agentic_capsules.runtime.sync_manager import BoundarySyncManager

    client = fakeredis.FakeRedis()          # or redis.Redis(host="localhost")
    backend = RedisBackend(client)
    manager = BoundarySyncManager(backend=backend)

Design plan ref: §5.2 Phase 8 (T-011)
"""

from .redis_backend import RedisBackend

__all__ = ["RedisBackend"]
