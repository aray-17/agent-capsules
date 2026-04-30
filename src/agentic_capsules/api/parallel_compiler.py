"""
_ParallelPipelineCompiler — opt-in threaded executor (T-054 v2 prerequisite).

A second executor that walks the same compiled pipeline as ``_PipelineCompiler``
but runs independent groups concurrently using a thread pool. Selected by
passing ``parallel=True`` to ``Pipeline.run()``. Default behaviour
(``parallel=False``) is unchanged and continues to use the serial executor.

Design constraints:
  * **Isolated code path.** Subclasses ``_PipelineCompiler`` to reuse compile
    helpers (``_build_tool_registry``, ``_compile_group``, ``_run_group``,
    ``_resolve_mode``, ``_post_run_controller_step``, ``_assemble_result``)
    but overrides ``execute()``. The serial executor keeps its orchestration
    loop; the parallel executor walks the group-level DAG with a thread pool.
  * **Threads, not asyncio.** LLM calls are I/O bound; the GIL releases on
    network I/O; the official adapters are thread-safe. Async migration is
    multi-week and buys nothing here. (Multiprocessing is wrong: it cannot
    share the HTTP connection pool and would force pickling adapter state.)
  * **G-7 (2026-04-09): controller + quality gate are now available in
    parallel mode.** The parallel executor previously rejected
    ``mode="auto"`` and ``evaluator=...`` because ``PipelineState`` had no
    locks and concurrent writes from worker threads could corrupt rolling
    windows / quality records / escalation bookkeeping. G-7 added a
    ``threading.RLock`` to ``PipelineState`` and extracted
    ``_post_run_controller_step`` on the parent compiler so both executors
    use the same H/H1/H2/H3 logic. The parallel executor now calls the
    helper inside the ``as_completed`` loop for each completed group.
    This is the unlock: AC's composition controller and quality gate
    finally run on the workloads that need them most (every expensive eval
    in this codebase runs in parallel mode).
  * **Group-level parallelism.** The unit of parallelism is a *group*. Within
    a group, agent execution is delegated to ``CapsuleExecutor`` unchanged
    (so COMPOUND merging within a parallel arm — the v2 headline — works
    out of the box). Intra-group agent parallelism is intentionally out of
    scope for this pass; groups are the natural unit for the 14-agent v2
    pipeline (each research arm = its own group).
  * **DAG-aware task chaining.** Each group sees the original task plus the
    final outputs of every group it declares as a dependency (``_GroupSpec``
    gains an optional ``depends_on`` field; default ``None`` falls back to
    the historical implicit linear chain).
  * **Zero impact on existing evals.** Existing pipelines that passed
    ``parallel=False`` (the default) are unchanged. Existing parallel-mode
    tests used ``mode="fine"`` / ``mode="compound"`` and no evaluator;
    those paths still work. The 608 baseline tests stay green.

Tracking ref: T-054 (paper v2 — peer-review readiness work); G-7
(LangGraph gap phase, 2026-04-09).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal

from ..core.types import CompositionLevel
from .compiler import _PipelineCompiler
from .result import PipelineResult

if TYPE_CHECKING:
    from ..core.types import LLMAdapter
    from ..evaluation.base import QualityEvaluator
    from ..runtime.checkpoint import PipelineCheckpoint
    from .builder import Pipeline


_DEFAULT_MAX_WORKERS = 8


class _ParallelPipelineCompiler(_PipelineCompiler):
    """
    Threaded executor for opt-in parallel pipeline execution.

    Subclasses ``_PipelineCompiler`` to reuse all compile helpers, the
    post-run controller step, and the final result-assembly path;
    overrides only the orchestration loop in ``execute()``.

    G-7 unlock (2026-04-09): previously rejected ``mode="auto"`` and
    ``evaluator=...`` because ``PipelineState`` was not thread-safe. Those
    rejections are gone. ``PipelineState`` now holds a ``threading.RLock``
    that serialises all controller reads and writes, and the
    ``_post_run_controller_step`` helper (inherited from the parent
    compiler) runs inside each worker thread exactly as it does in the
    serial executor. The only surviving restriction is that
    ``mode="observe"`` is not supported in the parallel path — the serial
    path's observe mode is rarely used and has a slightly different
    telemetry contract.
    """

    def __init__(
        self,
        pipeline:    "Pipeline",
        task:        str,
        adapter:     "LLMAdapter",
        mode:        Literal["auto", "fine", "compound"],
        task_id:     str | None,
        evaluator:   "QualityEvaluator | None" = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        checkpoint:  "PipelineCheckpoint | None" = None,
    ) -> None:
        if mode not in ("auto", "fine", "compound"):
            raise ValueError(
                "_ParallelPipelineCompiler requires mode in "
                "{'auto', 'fine', 'compound'}; "
                f"got mode={mode!r}. The parallel executor does not support "
                "'observe' mode."
            )
        super().__init__(
            pipeline, task, adapter, mode, task_id,
            evaluator=evaluator, checkpoint=checkpoint,
        )
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def execute(self) -> PipelineResult:
        from .builder import _FanoutGroupSpec  # lazy — avoid cycle
        start = time.perf_counter()

        # Phase A — ToolRegistry (reused unchanged)
        registry = self._build_tool_registry()
        tool_registry_arg = registry if len(registry) > 0 else None

        # Phases B–D — compile each group to a CompoundCapsule.
        # G-6: fan-out groups are runtime-expanded once their source
        # agent's output is ready. Use None as a placeholder at the
        # compile slot; the dispatch loop expands them just before
        # submission to the ThreadPoolExecutor.
        group_compounds: list = []
        for spec in self._pipeline._groups:
            if isinstance(spec, _FanoutGroupSpec):
                group_compounds.append(None)
            else:
                group_compounds.append(self._compile_group(spec))
        spec_by_name = {spec.name: spec for spec in self._pipeline._groups}
        compound_by_name = {
            spec.name: c
            for spec, c in zip(self._pipeline._groups, group_compounds)
        }

        # Build the group-level dependency graph and topological levels
        group_deps = self._build_group_deps()
        levels     = self._topological_levels(group_deps)

        # Per-group accumulators
        all_outputs:    dict[str, str] = {}
        all_telemetry:  list           = []
        group_outputs:  dict[str, str] = {}     # group_name → final_output (for dep chaining)
        group_results:  dict[str, object] = {}  # group_name → ExecutionResult (for assembly)
        modes_used:     dict[str, str]   = {}
        recommendations: dict[str, str]  = {}
        confidences:    dict[str, float] = {}

        state        = self._pipeline._pipeline_state
        apply_switch = self._mode == "auto"

        for level in levels:
            # G-4: resolve resumed groups before submitting any futures.
            # Resumed groups pre-populate group_outputs so that any later
            # level which depends on them gets the replayed text in its
            # task augmentation. Dispatched groups are submitted below.
            dispatch_level: list[str] = []
            for group_name in level:
                resumed = self._load_group_checkpoint(group_name)
                if resumed is not None:
                    group_results[group_name] = resumed
                    group_outputs[group_name] = resumed.final_output or ""
                    all_outputs.update(resumed.outputs)
                    # Resumed groups contribute no telemetry and do not run
                    # the controller step — mode/confidence/recommendation
                    # come from the current (fresh) PipelineState.
                    composition_level = self._resolve_mode(group_name)
                    modes_used[group_name] = (
                        "compound" if composition_level == CompositionLevel.COMPOUND
                        else "fine"
                    )
                    confidences[group_name]     = state._load(group_name).confidence
                    recommendations[group_name] = state.get_recommendation(group_name)
                else:
                    dispatch_level.append(group_name)

            if not dispatch_level:
                continue

            # G-6: runtime-expand any fan-out groups in this level BEFORE
            # submitting to the thread pool. Expansion must run on the
            # orchestrator thread (not inside a worker) because the source
            # agent's output lives in the shared ``all_outputs`` dict,
            # which is only mutated by the orchestrator as each
            # as_completed future returns. Expansion is fast (extractor
            # call + compile_group); the heavy LLM dispatch still runs
            # in worker threads.
            expanded_specs: dict[str, object]      = {}
            expanded_compounds: dict[str, object]  = {}
            skipped_empty: list[str]               = []
            for group_name in dispatch_level:
                spec = spec_by_name[group_name]
                compound = compound_by_name[group_name]
                if isinstance(spec, _FanoutGroupSpec):
                    expansion = self._expand_fanout_group(spec, all_outputs)
                    if expansion is None:
                        # 0 items — skip this group entirely
                        skipped_empty.append(group_name)
                        confidences[group_name]     = state._load(group_name).confidence
                        recommendations[group_name] = state.get_recommendation(group_name)
                        modes_used[group_name]      = "fine"
                        continue
                    spec, compound = expansion
                expanded_specs[group_name] = spec
                expanded_compounds[group_name] = compound

            # Filter out empty fan-out groups from the actual dispatch list
            runnable = [
                g for g in dispatch_level if g not in skipped_empty
            ]
            if not runnable:
                continue

            # Groups in the same topological level have no dependency edges
            # between them and can run concurrently.
            level_workers = max(1, min(len(runnable), self._max_workers))
            with ThreadPoolExecutor(max_workers=level_workers) as ex:
                futures = {}
                for group_name in runnable:
                    spec     = expanded_specs[group_name]
                    compound = expanded_compounds[group_name]
                    deps     = group_deps[group_name]
                    task_input = self._build_group_task(deps, group_outputs)
                    composition_level = self._resolve_mode(group_name)
                    modes_used[group_name] = (
                        "compound" if composition_level == CompositionLevel.COMPOUND
                        else "fine"
                    )
                    future = ex.submit(
                        self._run_and_record_group,
                        spec,
                        compound,
                        composition_level,
                        task_input,
                        tool_registry_arg,
                        apply_switch,
                    )
                    futures[future] = (group_name, task_input)

                for future in as_completed(futures):
                    group_name, _task_input = futures[future]
                    group_result, updated = future.result()  # propagates exceptions
                    group_results[group_name] = group_result
                    group_outputs[group_name] = group_result.final_output or ""
                    all_outputs.update(group_result.outputs)
                    all_telemetry.extend(group_result.telemetry)
                    confidences[group_name]     = updated.confidence
                    recommendations[group_name] = state.get_recommendation(group_name)

        # G-4: whole pipeline succeeded — drop the checkpoint.
        if self._checkpoint is not None:
            self._checkpoint.clear(self._task_id)

        # Final output is the last *declared* group's output. Topological
        # order is not necessarily declaration order, so we can't use
        # "the last completed group" — that would be non-deterministic.
        last_declared = self._pipeline._groups[-1].name
        last_result   = group_results.get(last_declared)
        final_output  = (last_result.final_output if last_result else "") or ""

        latency_ms = (time.perf_counter() - start) * 1000

        # G-7: build the same controller-derived result fields the serial
        # executor produces, now that the controller is consulted in the
        # parallel path too. Per-group effective window_size so overrides
        # are honored in aggregated efficiency stats. Lock-guarded reads.
        snapshot = state.snapshot()
        def _window_for(group_name: str) -> int:
            return state._effective_policy(group_name).window_size
        scores: dict[str, float] = {
            name: gs.last_score for name, gs in snapshot.items()
        }
        efficiency: dict[str, dict] = {
            name: {
                "token_reduction_pct":       gs.token_reduction_pct(_window_for(name)),
                "mean_latency_fine_ms":      gs.mean_latency_ms("fine",     _window_for(name)),
                "mean_latency_compound_ms":  gs.mean_latency_ms("compound", _window_for(name)),
            }
            for name, gs in snapshot.items()
        }
        quality_scores: dict[str, float] = {
            name: q
            for name, gs in snapshot.items()
            if (q := gs.mean_quality(_window_for(name))) is not None
        }

        return self._assemble_result(
            outputs=all_outputs,
            final_output=final_output,
            telemetry=all_telemetry,
            modes_used=modes_used,
            recommendations=recommendations,
            confidences=confidences,
            scores=scores,
            efficiency=efficiency,
            quality=quality_scores,
            latency_ms=latency_ms,
        )

    def _run_and_record_group(
        self,
        spec,
        compound,
        level,
        task_input: str,
        tool_registry_arg,
        apply_switch: bool,
    ):
        """
        G-7: worker-thread entry point.

        Runs the group (``_run_group``), then calls the shared post-run
        controller step inside the same thread so H1/H2/H3 fire on every
        completed group in parallel mode. Returns both the raw group
        result (for output/telemetry accumulation) and the updated
        ``GroupControllerState`` (for confidence bookkeeping in the
        orchestration loop).

        Thread-safety comes from the RLock inside ``PipelineState``;
        ``_post_run_controller_step`` performs load-modify-save sequences
        on a single group, and different worker threads never touch the
        same group_name simultaneously (each group runs exactly once per
        pipeline execution), so lock-per-PipelineState is sufficient.
        """
        group_result = self._run_group(
            spec, compound, level, task_input, tool_registry_arg
        )
        # G-4: save the completed group's outputs before the controller
        # step so a failure in H2/H3 (e.g. quality-gate shadow run) does
        # not lose the work this group already produced. Safe from
        # concurrent worker threads: PipelineCheckpoint uses an internal
        # threading.Lock, and each group runs exactly once per pipeline
        # execution so save_group is never called twice for the same key.
        self._save_group_checkpoint(spec.name, group_result)
        updated = self._post_run_controller_step(
            spec, compound, level, group_result, task_input,
            tool_registry_arg, apply_switch,
        )
        return group_result, updated

    # ------------------------------------------------------------------
    # Group-level DAG construction
    # ------------------------------------------------------------------

    def _build_group_deps(self) -> dict[str, list[str]]:
        """
        Build the group-level dependency graph from ``_GroupSpec.depends_on``.

        Defaults preserve the historical implicit linear chain: a group with
        no declared ``depends_on`` depends on the group declared immediately
        before it. This means existing pipelines (which never set
        ``depends_on``) execute as a single linear chain in the parallel
        executor too — equivalent to the serial executor's behaviour.
        """
        deps: dict[str, list[str]] = {}
        prev: str | None = None
        for spec in self._pipeline._groups:
            declared = getattr(spec, "depends_on", None)
            if declared is not None:
                deps[spec.name] = list(declared)
            elif prev is not None:
                deps[spec.name] = [prev]
            else:
                deps[spec.name] = []
            prev = spec.name
        return deps

    def _topological_levels(
        self, deps: dict[str, list[str]]
    ) -> list[list[str]]:
        """
        Group group-names into topological levels (Kahn's algorithm by levels).

        Returns a list of lists. Each inner list is a set of group names with
        no remaining unsatisfied dependencies — all groups in one level can
        run concurrently. Within a level, declaration order is preserved for
        deterministic test output.

        Raises:
            ValueError: if a group references an unknown group, or if the
                        graph contains a cycle.
        """
        all_names = set(deps.keys())
        for name, ds in deps.items():
            for d in ds:
                if d not in all_names:
                    raise ValueError(
                        f"Group {name!r} declares depends_on={d!r}, "
                        f"which is not a group in this pipeline. "
                        f"Declared groups: {sorted(all_names)}"
                    )

        remaining = {name: set(ds) for name, ds in deps.items()}
        # Preserve declaration order across the whole DAG so each level is
        # emitted in declaration order — important for deterministic
        # snapshots in tests.
        declaration_order = [spec.name for spec in self._pipeline._groups]

        levels: list[list[str]] = []
        while remaining:
            ready = [
                name for name in declaration_order
                if name in remaining and not remaining[name]
            ]
            if not ready:
                cycle = sorted(remaining.keys())
                raise ValueError(
                    f"Cycle in group dependency graph: groups {cycle} "
                    "have unresolved dependencies. Check `depends_on` "
                    "declarations for circular references."
                )
            levels.append(ready)
            for name in ready:
                del remaining[name]
            ready_set = set(ready)
            for name in remaining:
                remaining[name] -= ready_set
        return levels

    def _build_group_task(
        self, deps: list[str], group_outputs: dict[str, str]
    ) -> str:
        """
        Build the task input for a group from its declared dependencies.

        Parallel-executor analogue of the serial executor's task augmentation
        (compiler.py:115–118). Differences:
          * Multiple deps: each dep's output is appended in declaration order.
          * No deps: the group sees only the original task.
          * Iteration over ``deps`` is deterministic (declaration order),
            so the task string is reproducible run-to-run.
        """
        if not deps:
            return self._task
        parts = [self._task]
        for dep in deps:
            out = group_outputs.get(dep, "")
            if out:
                parts.append(f"[{dep} output]\n{out}")
        return "\n\n".join(parts)
