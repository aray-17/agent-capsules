"""
Checkpoint / Restore — persists accumulated outputs at capsule boundaries.

Enables resuming a failed pipeline run from the last completed capsule rather
than re-executing all prior steps. This is valuable for long pipelines where
early LLM calls are expensive and a late failure would otherwise waste all
prior computation.

Design:
  - `CheckpointStore` is an in-memory store (dict); swap backend by subclassing.
  - `save(task_id, outputs)` — persist the accumulated_outputs dict at a boundary.
  - `load(task_id)` — return the last saved outputs for a task, or None.
  - `clear(task_id)` — remove checkpoints for a completed or abandoned task.

The executor calls `checkpoint.save(task_id, accumulated_outputs)` after each
leaf completes. On a new run with the same task_id, it calls `checkpoint.load()`
to restore state and skip already-completed leaves.

Usage:
    store = CheckpointStore()
    executor = CapsuleExecutor(adapter, checkpoint=store)

    # First run — fails at leaf 3
    executor.run(hierarchy, task_input="...", task_id="task-42")

    # Second run — resumes from leaf 3 (leaves 1-2 already checkpointed)
    executor.run(hierarchy, task_input="...", task_id="task-42")

Design plan ref: §5.2 Phase 6 (checkpoint/restore)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CheckpointStore — in-memory with optional file persistence
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    Stores accumulated outputs keyed by task_id.

    In-memory by default. Pass `path` to persist to JSON on disk, enabling
    cross-process resume.

    Args:
        path: Optional directory for JSON checkpoint files. Each task_id
              gets its own file: `{path}/{task_id}.json`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._store: dict[str, dict[str, Any]] = {}  # {task_id: {outputs}}
        self._path = Path(path) if path else None
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)

    def save(self, task_id: str, outputs: dict[str, Any]) -> None:
        """Persist *outputs* for *task_id*. Overwrites any prior checkpoint."""
        self._store[task_id] = dict(outputs)
        if self._path is not None:
            file = self._path / f"{task_id}.json"
            try:
                file.write_text(json.dumps(self._store[task_id], indent=2))
            except OSError as exc:
                logger.warning("Checkpoint write failed for task %r: %s", task_id, exc)

    def load(self, task_id: str) -> dict[str, Any] | None:
        """
        Return saved outputs for *task_id*, or None if no checkpoint exists.

        Checks in-memory store first; falls back to disk if a path is set.
        """
        if task_id in self._store:
            return dict(self._store[task_id])
        if self._path is not None:
            file = self._path / f"{task_id}.json"
            if file.exists():
                try:
                    data = json.loads(file.read_text())
                    self._store[task_id] = data
                    return dict(data)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Checkpoint read failed for task %r: %s", task_id, exc)
        return None

    def clear(self, task_id: str) -> None:
        """Remove checkpoint for *task_id* (call after successful completion)."""
        self._store.pop(task_id, None)
        if self._path is not None:
            file = self._path / f"{task_id}.json"
            if file.exists():
                try:
                    file.unlink()
                except OSError:
                    pass

    def has(self, task_id: str) -> bool:
        """True if a checkpoint exists for *task_id*."""
        if task_id in self._store:
            return True
        if self._path is not None:
            return (self._path / f"{task_id}.json").exists()
        return False

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"CheckpointStore(tasks={list(self._store.keys())}, path={self._path})"


# ---------------------------------------------------------------------------
# PipelineCheckpoint — group-level resume for the high-level Pipeline API (G-4)
# ---------------------------------------------------------------------------

class PipelineCheckpoint:
    """
    Group-level checkpoint/restore for the high-level Pipeline API.

    Distinct from ``CheckpointStore`` above, which is per-*leaf* (agent) and
    used by ``CapsuleExecutor`` to resume within a single group after a
    mid-group failure. ``PipelineCheckpoint`` operates one layer up: it
    saves the ``{outputs, final_output}`` of each group after it completes,
    so a later run with the same ``task_id`` can skip every already-
    completed group and resume at the first unfinished one.

    G-4 (LangGraph gap phase): LangGraph's ``Checkpointer`` saves state
    after every node. AC previously had no high-level checkpointing at all
    — neither the serial nor the parallel compiler exposed any resume
    mechanism. ``PipelineCheckpoint`` closes that gap for both executors
    with a single mechanism.

    Thread-safe: a single ``threading.Lock`` serialises all reads and
    writes so the parallel compiler can save concurrently from worker
    threads without corrupting the index.

    On-disk format: one JSON file per ``task_id`` containing
    ``{group_name: {"outputs": {...}, "final_output": "..."}}``. In-memory
    cache mirrors the file; the cache is the source of truth within one
    process and the file is the source of truth across processes.

    What is NOT checkpointed:
      * Telemetry records — resumed groups contribute zero records, so
        rolling-window controller signals effectively skip them. Documented.
      * ``PipelineState`` controller bookkeeping — resumed groups do not
        run ``_post_run_controller_step``, so their overhead/quality
        observations are not recorded on the resume run. Documented.
      * Tool invocations / side effects — if an agent called a tool that
        wrote to a database, that side effect is not re-applied on resume.
        The agent's textual output is replayed from disk.

    Args:
        path: Optional directory for JSON checkpoint files. When ``None``
              (default) the checkpoint is in-memory only — useful for
              tests and single-process retry-on-failure within one run.
              When set, each ``task_id`` gets its own file at
              ``{path}/{task_id}.json``.

    Usage:
        ckpt = PipelineCheckpoint(path="/tmp/ac-checkpoints")
        pipeline.run("topic", task_id="task-42", checkpoint=ckpt)
        # If the run fails mid-way, rerunning with the same task_id
        # will skip every group whose {outputs, final_output} is saved.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        # task_id → group_name → {"outputs": dict, "final_output": str}
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)

    # ---- single-group primitives ------------------------------------------------

    def save_group(
        self,
        task_id: str,
        group_name: str,
        outputs: dict[str, str],
        final_output: str,
    ) -> None:
        """
        Persist *outputs* and *final_output* for *(task_id, group_name)*.

        Overwrites any prior checkpoint for the same group. Thread-safe.
        """
        with self._lock:
            task_store = self._store.setdefault(task_id, {})
            task_store[group_name] = {
                "outputs": dict(outputs),
                "final_output": final_output or "",
            }
            if self._path is not None:
                self._write_task_file(task_id)

    def load_group(
        self, task_id: str, group_name: str
    ) -> dict[str, Any] | None:
        """
        Return saved ``{"outputs": dict, "final_output": str}`` for
        *(task_id, group_name)*, or ``None`` if no checkpoint exists.

        Checks the in-memory cache first; falls back to disk on miss and
        hydrates the cache if found. Thread-safe.
        """
        with self._lock:
            if task_id in self._store and group_name in self._store[task_id]:
                entry = self._store[task_id][group_name]
                return {
                    "outputs": dict(entry["outputs"]),
                    "final_output": entry["final_output"],
                }
            if self._path is not None:
                self._read_task_file(task_id)  # may populate self._store
                if task_id in self._store and group_name in self._store[task_id]:
                    entry = self._store[task_id][group_name]
                    return {
                        "outputs": dict(entry["outputs"]),
                        "final_output": entry["final_output"],
                    }
        return None

    def has_group(self, task_id: str, group_name: str) -> bool:
        """True if a checkpoint exists for *(task_id, group_name)*."""
        return self.load_group(task_id, group_name) is not None

    # ---- whole-task primitives -------------------------------------------------

    def clear(self, task_id: str) -> None:
        """
        Remove every group checkpoint for *task_id*.

        Call after successful pipeline completion so the next run with the
        same ``task_id`` starts fresh. Thread-safe.
        """
        with self._lock:
            self._store.pop(task_id, None)
            if self._path is not None:
                file = self._path / f"{task_id}.json"
                if file.exists():
                    try:
                        file.unlink()
                    except OSError:
                        pass

    def groups_for(self, task_id: str) -> list[str]:
        """Return the list of group names checkpointed for *task_id*."""
        with self._lock:
            if task_id not in self._store and self._path is not None:
                self._read_task_file(task_id)
            return list(self._store.get(task_id, {}).keys())

    # ---- private I/O -----------------------------------------------------------

    def _write_task_file(self, task_id: str) -> None:
        """Caller must hold ``self._lock``."""
        assert self._path is not None
        file = self._path / f"{task_id}.json"
        try:
            file.write_text(json.dumps(self._store[task_id], indent=2))
        except OSError as exc:
            logger.warning(
                "PipelineCheckpoint write failed for task %r: %s",
                task_id, exc,
            )

    def _read_task_file(self, task_id: str) -> None:
        """Caller must hold ``self._lock``."""
        assert self._path is not None
        file = self._path / f"{task_id}.json"
        if not file.exists():
            return
        try:
            data = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "PipelineCheckpoint read failed for task %r: %s",
                task_id, exc,
            )
            return
        if isinstance(data, dict):
            self._store[task_id] = data

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"PipelineCheckpoint(tasks={list(self._store.keys())}, "
            f"path={self._path})"
        )
