"""
_PipelineCompiler — translates the Pipeline builder model to internal primitives.

This is the only file in api/ that imports from core/, runtime/, controller/,
and tools/. builder.py imports this lazily inside run() to stay import-clean.

Compile sequence (phases A–I):
  A. Build ToolRegistry from all Tool instances declared across all agents
  B. Auto-generate Schema per agent (input + output)
  C. Build AgentLeaf (wrapping AgentStepCapsule) per agent
  D. Build CompoundCapsule per group; call compute_order()
  E. Build root CompoundCapsule + CapsuleHierarchy; call compute_order_recursive()
  F. Resolve per-group CompositionLevel from PipelineState
  G. Execute CapsuleExecutor per group at its resolved level
  H. Record overhead observation per group; maybe switch mode
     Phase 12 quality gate:
       H1. Store last_fine_output on every FINE run (baseline for later comparison)
       H2. On FINE→COMPOUND switch event: shadow-run COMPOUND, evaluate quality;
           revert switch if quality < quality_floor (proactive gate, T-033)
       H3. On existing COMPOUND run: evaluate quality vs stored FINE baseline;
           record rolling quality score (reactive gate)
  I. Assemble PipelineResult from ExecutionResult + PipelineState snapshot
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Literal

from ..core.capsule  import AgentStepCapsule
from ..core.hierarchy import AgentLeaf, CompoundCapsule, CapsuleHierarchy
from ..core.types    import CompositionLevel, Schema
from ..controller.telemetry import TelemetryCollector
from ..runtime.executor  import CapsuleExecutor, ExecutionResult
from ..runtime.scheduler import compute_order
from ..tools.registry    import ToolDefinition, ToolRegistry
from .result  import PipelineResult
from .state   import CompositionSignal, _do_revert

if TYPE_CHECKING:
    from ..core.types import LLMAdapter
    from ..evaluation.base import QualityEvaluator
    from ..runtime.checkpoint import PipelineCheckpoint
    from .builder import Pipeline, _GroupSpec, _AgentSpec, _FanoutGroupSpec


class _PipelineCompiler:

    def __init__(
        self,
        pipeline:   "Pipeline",
        task:       str,
        adapter:    "LLMAdapter",
        mode:       Literal["auto", "observe", "fine", "compound"],
        task_id:    str | None,
        evaluator:  "QualityEvaluator | None" = None,
        checkpoint: "PipelineCheckpoint | None" = None,
    ) -> None:
        self._pipeline   = pipeline
        self._task       = task
        self._adapter    = adapter
        self._mode       = mode
        self._task_id    = task_id or f"{pipeline._name}-{uuid.uuid4().hex[:8]}"
        self._evaluator  = evaluator
        self._checkpoint = checkpoint  # G-4: group-level resume, optional

        # Built during compile — used for output_key → agent_name reverse map
        self._output_key_to_agent: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def execute(self) -> PipelineResult:
        from .builder import _FanoutGroupSpec, _GroupSpec  # lazy — avoid cycle
        start = time.perf_counter()

        # Phase A — ToolRegistry
        registry = self._build_tool_registry()
        tool_registry_arg = registry if len(registry) > 0 else None

        # Phases B–D — compile each group to a CompoundCapsule.
        # G-6: fan-out groups are runtime-expanded; their compound is built
        # inside the execute() loop once the source agent's output is in
        # `all_outputs`. Store None as a placeholder at the compile slot.
        group_compounds: list[CompoundCapsule | None] = []
        for spec in self._pipeline._groups:
            if isinstance(spec, _FanoutGroupSpec):
                group_compounds.append(None)
            else:
                group_compounds.append(self._compile_group(spec))

        # Phases F–H — run each group at its own resolved level
        modes_used:      dict[str, str]   = {}
        recommendations: dict[str, str]   = {}
        confidences:     dict[str, float] = {}

        state        = self._pipeline._pipeline_state
        apply_switch = self._mode == "auto"
        # Per-group effective policies are resolved via
        # ``self._pipeline.effective_policy(group_name)`` at each use site. No
        # pipeline-scoped ``policy`` variable here because per-group
        # overrides are valid and the compiler must never silently use the
        # pipeline default where a group override was declared.

        # Accumulate outputs and telemetry across all groups
        all_outputs:   dict[str, str] = {}
        all_telemetry: list           = []
        last_group_result             = None

        # Pass prior group outputs into subsequent groups via task augmentation
        current_task = self._task

        for original_spec, compound in zip(self._pipeline._groups, group_compounds):
            spec_name = original_spec.name
            level = self._resolve_mode(spec_name)
            modes_used[spec_name] = "compound" if level == CompositionLevel.COMPOUND else "fine"

            # G-4: group-level checkpoint resume. If a prior run saved this
            # group's outputs under the same task_id, skip dispatch and
            # replay the saved outputs/final_output. Resumed groups
            # contribute zero telemetry and do NOT trigger the controller
            # step (documented — see PipelineCheckpoint docstring).
            resumed = self._load_group_checkpoint(spec_name)
            if resumed is not None:
                all_outputs.update(resumed.outputs)
                last_group_result = resumed
                if resumed.final_output:
                    current_task = (
                        f"{self._task}\n\n"
                        f"[{spec_name} output]\n{resumed.final_output}"
                    )
                confidences[spec_name]     = state._load(spec_name).confidence
                recommendations[spec_name] = state.get_recommendation(spec_name)
                continue

            # G-6: runtime expansion of fan-out groups. The source agent's
            # output must already be in `all_outputs` (the user declares
            # fan-out groups after the group containing the source).
            if isinstance(original_spec, _FanoutGroupSpec):
                expansion = self._expand_fanout_group(original_spec, all_outputs)
                if expansion is None:
                    # 0 items extracted — skip the whole group. Empty final
                    # output does not augment current_task.
                    confidences[spec_name]     = state._load(spec_name).confidence
                    recommendations[spec_name] = state.get_recommendation(spec_name)
                    continue
                spec, compound = expansion
            else:
                spec = original_spec

            # Phase G — execute group at resolved level
            group_result = self._run_group(
                spec, compound, level, current_task, tool_registry_arg
            )

            all_outputs.update(group_result.outputs)
            all_telemetry.extend(group_result.telemetry)
            last_group_result = group_result

            # Feed this group's final output into the next group's task context
            if group_result.final_output:
                current_task = (
                    f"{self._task}\n\n"
                    f"[{spec.name} output]\n{group_result.final_output}"
                )

            # G-4: save the completed group's outputs before the controller
            # step so a failure in H2/H3 doesn't lose the work this group
            # already produced.
            self._save_group_checkpoint(spec.name, group_result)

            # Phases H / H1 / H2 / H3 — controller observation and quality gate
            updated = self._post_run_controller_step(
                spec, compound, level, group_result, current_task,
                tool_registry_arg, apply_switch,
            )
            confidences[spec.name]     = updated.confidence
            recommendations[spec.name] = state.get_recommendation(spec.name)

        # G-4: whole pipeline succeeded — drop the checkpoint so the next
        # run with the same task_id starts fresh.
        if self._checkpoint is not None:
            self._checkpoint.clear(self._task_id)

        latency_ms = (time.perf_counter() - start) * 1000

        # Collect per-group composition scores, efficiency and quality from snapshot.
        # Each group uses its own effective policy's window_size for aggregation.
        snapshot = state.snapshot()
        scores: dict[str, float] = {
            name: gs.last_score for name, gs in snapshot.items()
        }
        def _window_for(group_name: str) -> int:
            return self._pipeline.effective_policy(group_name).window_size
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
        quality_details: dict[str, dict] = {}
        for name, gs in snapshot.items():
            if gs.last_quality_score is not None and gs.last_signal is not None:
                # Use the evaluator's last details if available via last_signal proxy
                pass  # details are populated by record_quality path below

        # Pull last quality details from the live state (transient, not in snapshot)
        for spec in self._pipeline._groups:
            gs_live = state._load(spec.name)
            if gs_live.last_quality_score is not None:
                # We can't recover details after the fact from state alone;
                # details are populated inline during H2/H3 above via evaluator.
                pass

        # Phase I — assemble PipelineResult
        return self._assemble_result(
            outputs=all_outputs,
            final_output=last_group_result.final_output if last_group_result else "",
            telemetry=all_telemetry,
            modes_used=modes_used,
            recommendations=recommendations,
            confidences=confidences,
            scores=scores,
            efficiency=efficiency,
            quality=quality_scores,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Phase A — ToolRegistry
    # ------------------------------------------------------------------

    def _build_tool_registry(self) -> ToolRegistry:
        from .builder import _FanoutGroupSpec  # lazy — avoid cycle
        registry = ToolRegistry()
        seen: set[str] = set()
        for spec in self._pipeline._groups:
            # G-6: fan-out groups contribute their worker_tools; they have
            # no compile-time .agents attribute because N is unknown.
            if isinstance(spec, _FanoutGroupSpec):
                for tool in spec.worker_tools:
                    if tool.name not in seen:
                        registry.register(ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.input_schema,
                            callable=tool.fn,
                        ))
                        seen.add(tool.name)
                continue
            for agent in spec.agents:
                for tool in agent.tools:
                    if tool.name not in seen:
                        registry.register(ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.input_schema,
                            callable=tool.fn,
                        ))
                        seen.add(tool.name)
        return registry

    # ------------------------------------------------------------------
    # Phase B–D — group and agent compilation
    # ------------------------------------------------------------------

    def _compile_group(self, spec: "_GroupSpec") -> CompoundCapsule:
        leaves = [self._compile_agent(a) for a in spec.agents]
        # Dependency edges within group.
        # - If an agent declares depends_on explicitly (including []), use it.
        # - Otherwise fall back to implicit linear: agent N depends on agent N-1.
        #   Implicit linear is preserved agent-by-agent, so a single agent in
        #   the middle can declare explicit deps while its siblings stay linear.
        dep_edges: dict[str, list[str]] = {}
        any_explicit = False
        for i, agent_spec in enumerate(spec.agents):
            if agent_spec.depends_on is not None:
                dep_edges[agent_spec.name] = list(agent_spec.depends_on)
                any_explicit = True
            elif i > 0:
                dep_edges[agent_spec.name] = [spec.agents[i - 1].name]

        compound = CompoundCapsule(
            name=spec.name,
            children=leaves,
            dependency_edges=dep_edges,
        )
        compute_order(compound)
        # T-042: classify topology and set injection strategy.
        # Pass has_explicit_dependencies so that explicit depends_on declarations
        # (including depends_on=[] for independent roots) always route through
        # the "deps" strategy, regardless of whether the resulting graph happens
        # to be classifiable as linear. This prevents the policy-level
        # predecessor_only override from silently injecting the wrong context.
        from ..runtime.topology import classify_and_set_strategy
        classify_and_set_strategy(
            compound, has_explicit_dependencies=any_explicit,
        )
        return compound

    # ------------------------------------------------------------------
    # G-6 — runtime fan-out expansion
    # ------------------------------------------------------------------

    def _expand_fanout_group(
        self,
        fanout_spec: "_FanoutGroupSpec",
        all_outputs: dict[str, str],
    ) -> "tuple[_GroupSpec, CompoundCapsule] | None":
        """
        Expand a ``_FanoutGroupSpec`` at runtime.

        Looks up the source agent's output in ``all_outputs`` via the
        canonical ``{NAME}_OUTPUT`` key, calls ``item_extractor`` to get
        a concrete item list, truncates to ``max_items``, and builds a
        runtime ``_GroupSpec`` with one ``_AgentSpec`` per item. The
        worker's ``{item}`` placeholder is substituted via ``str.replace``
        (not ``.format``) so curly braces elsewhere in the worker goal
        are preserved. The runtime group is then compiled through the
        normal ``_compile_group`` path so topology classification,
        dependency edges, and agent compilation all reuse the existing
        machinery.

        Returns ``None`` when the extractor returns an empty list — the
        caller treats this as a no-op group (no dispatch, no outputs,
        no downstream augmentation).

        Raises:
            ValueError: if the source agent's output is not in
                ``all_outputs`` (usually means the user declared a
                fan-out group before the group containing the source
                agent, or the source agent was skipped via G-2).
        """
        from .builder import _AgentSpec, _GroupSpec  # lazy — avoid cycle

        source_key = self._agent_output_key(fanout_spec.source)
        source_output = all_outputs.get(source_key)
        if source_output is None:
            raise ValueError(
                f"fanout_group '{fanout_spec.name}': source agent "
                f"{fanout_spec.source!r} has no output in the accumulated "
                "state. Check that the source agent's group runs before "
                "this fan-out group (declaration order in the serial "
                "executor; depends_on in the parallel executor)."
            )

        try:
            items_raw = fanout_spec.item_extractor(source_output)
        except Exception as exc:
            raise ValueError(
                f"fanout_group '{fanout_spec.name}': item_extractor raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(items_raw, list):
            raise ValueError(
                f"fanout_group '{fanout_spec.name}': item_extractor must "
                f"return a list of strings, got {type(items_raw).__name__}."
            )
        items = [str(it) for it in items_raw[: fanout_spec.max_items]]
        if not items:
            return None

        # Build one _AgentSpec per item. Workers all depend_on=[] so they
        # form a flat fan-out. The compound will therefore classify as
        # "independent" topology and the executor merges them in one
        # compound LLM call under COMPOUND mode.
        worker_specs: list[_AgentSpec] = []
        for i, item in enumerate(items):
            worker_specs.append(
                _AgentSpec(
                    name=f"{fanout_spec.worker_name}_{i}",
                    goal=fanout_spec.worker_goal.replace("{item}", item),
                    tools=list(fanout_spec.worker_tools),
                    model=fanout_spec.worker_model,
                    depends_on=[],  # independent — flat fan-out
                )
            )
        runtime_group = _GroupSpec(
            name=fanout_spec.name,
            agents=worker_specs,
            adapter=fanout_spec.adapter,
            depends_on=fanout_spec.depends_on,
        )
        compound = self._compile_group(runtime_group)
        return runtime_group, compound

    @staticmethod
    def _agent_output_key(agent_name: str) -> str:
        """Return the canonical output_key for an agent name.

        Mirrors ``AgentStepCapsule.__post_init__``:
        ``{name.upper().replace(' ', '_')}_OUTPUT``. Used by the G-6
        fan-out expander to look up the source agent's output by name
        without needing a lookup table.
        """
        return f"{agent_name.upper().replace(' ', '_')}_OUTPUT"

    def _compile_agent(self, spec: "_AgentSpec") -> AgentLeaf:
        capsule = AgentStepCapsule(
            name=spec.name,
            system_prompt=spec.goal,
            input_schema=Schema(
                name=f"{spec.name}_input",
                fields={"text": "str"},
            ),
            output_schema=Schema(
                name=f"{spec.name}_output",
                fields={"result": "str"},
            ),
            tools=[t.name for t in spec.tools],
            # G-2: propagate the runtime skip predicate. Executor consults it
            # before dispatching the agent's LLM call in FINE mode.
            skip_condition=spec.condition,
        )
        # Record the output_key → agent_name reverse mapping (Phase I)
        self._output_key_to_agent[capsule.output_key] = spec.name
        return AgentLeaf(capsule=capsule)

    # ------------------------------------------------------------------
    # Phase F — mode resolution
    # ------------------------------------------------------------------

    def _resolve_mode(self, group_name: str) -> CompositionLevel:
        if self._mode == "fine":
            return CompositionLevel.FINE
        if self._mode == "compound":
            return CompositionLevel.COMPOUND
        # "auto" and "observe" both use the stored current_mode
        return self._pipeline._pipeline_state.get_mode(group_name)

    # ------------------------------------------------------------------
    # Phase G — group execution helper
    # ------------------------------------------------------------------

    # T-040: escalation order for auto mode quality gate
    _ESCALATION_LADDER: list[str] = ["standard", "two_phase", "sequential"]

    def _resolve_adapter(self, spec: "_GroupSpec") -> "LLMAdapter":
        """T-043: return the group's adapter if set, else the pipeline adapter."""
        return spec.adapter if spec.adapter is not None else self._adapter

    def _resolve_compound_execution_model(
        self,
        spec: "_GroupSpec",
        level: "CompositionLevel",
        compound: "CompoundCapsule | None" = None,
    ) -> str:
        """
        T-040 / T-042: Resolve the compound execution model for this group.

        For explicit modes (standard/two_phase/sequential): honour policy directly.
        For "auto": apply gates in order:
          0. Topology gate  — non-linear dep graph (strategy="deps") → sequential
          1. Tools gate    — any tools in group → two_phase, else standard
          2. Verbosity gate — mean FINE output ≥ 3,500 tok/agent → sequential
          3. Persisted override — a previous quality escalation found a working
             mode; use it to skip redundant shadow comparisons on future runs.
        Returns "standard" for FINE runs (irrelevant, but must be a valid value).
        """
        effective_policy = self._pipeline.effective_policy(spec.name)
        policy_model = getattr(effective_policy, "compound_execution_model", "standard")

        if policy_model != "auto":
            # E-1: escalation may have set an override even when the policy model is
            # explicit (e.g. standard). The override must win so escalation actually
            # changes execution behaviour on the next run.
            override = self._pipeline._pipeline_state.get_execution_model_override(spec.name)
            if override is not None:
                return override
            return policy_model

        if level != CompositionLevel.COMPOUND:
            return "standard"

        # Check if a prior quality escalation already found the right mode
        override = self._pipeline._pipeline_state.get_execution_model_override(spec.name)
        if override is not None:
            return override

        # Gate 0 (T-042): topology — non-linear graphs always use sequential
        # so that dependency-aware injection can eliminate irrelevant context.
        if compound is not None and compound.sequential_injection_strategy == "deps":
            return "sequential"

        # Gate 1: verbosity — determines baseline mode for all agents.
        # avg_output_tokens_fine drives mode selection using configurable thresholds
        # from ControllerPolicy so different model families and pipeline shapes can
        # be tuned without touching framework code.
        gs = self._pipeline._pipeline_state._load(spec.name)
        avg_tokens = gs.mean_avg_output_tokens_fine()
        two_phase_thresh  = effective_policy.verbosity_two_phase_threshold
        sequential_thresh = effective_policy.verbosity_sequential_threshold
        if avg_tokens is None or avg_tokens < two_phase_thresh:
            mode = "standard"
        elif avg_tokens < sequential_thresh:
            mode = "two_phase"
        else:
            mode = "sequential"

        # Gate 2: tools override — if any agent has tools and mode is still
        # standard, upgrade to two_phase so tool agents get Phase A tool gathering.
        if mode == "standard" and any(a.tools for a in spec.agents):
            mode = "two_phase"

        return mode

    def _run_group(
        self,
        spec:              "_GroupSpec",
        compound:          CompoundCapsule,
        level:             CompositionLevel,
        task_input:        str,
        tool_registry_arg,
        execution_model:   str | None = None,
    ):
        """Execute a single group at the given composition level and return its result.

        execution_model overrides the policy value when provided — used by the
        T-040 quality escalation ladder to try successive modes without mutating
        the policy.
        """
        policy = self._pipeline.effective_policy(spec.name)
        hierarchy = CapsuleHierarchy(name=spec.name, root=compound)
        telemetry = TelemetryCollector()
        # T-039: auto-calibrate compound_min_output_words from FINE observations
        # when the policy does not set it explicitly.
        compound_min_output_words = getattr(policy, "compound_min_output_words", None)
        if compound_min_output_words is None and level == CompositionLevel.COMPOUND:
            compound_min_output_words = self._pipeline._pipeline_state.get_auto_min_output_words(
                spec.name
            )
        resolved_model = execution_model if execution_model is not None else \
            self._resolve_compound_execution_model(spec, level, compound=compound)

        # Track A: compute per-agent budgets for M-1 budgeted_adaptive
        merged_output_structure = getattr(policy, "merged_output_structure", "none")
        per_agent_budgets: dict[str, int] | None = None
        if merged_output_structure == "budgeted_adaptive" and level == CompositionLevel.COMPOUND:
            gs = self._pipeline._pipeline_state._load(spec.name)
            mean_tokens = gs.mean_avg_output_tokens_fine()
            if mean_tokens is not None:
                budget = max(50, int(mean_tokens * 0.8))
                per_agent_budgets = {leaf.name: budget for leaf in compound.serialization_order}

        # T-058: plumb observations for output_guidance="auto" (selector gate).
        # mean_avg_output_tokens_fine() feeds the gate; compile_single falls
        # back to no-guidance when observations are not yet available.
        output_guidance = getattr(policy, "output_guidance", "none")
        mean_fine_tokens_by_group: dict[str, int] | None = None
        if output_guidance == "auto":
            gs = self._pipeline._pipeline_state._load(spec.name)
            mean_tokens = gs.mean_avg_output_tokens_fine()
            if mean_tokens is not None:
                mean_fine_tokens_by_group = {spec.name: int(mean_tokens)}
        verbosity_guidance_threshold = getattr(
            policy, "verbosity_guidance_threshold", None
        )

        # Track A: S-1 sequential context strategy, C-1 cache-aligned prompts
        sequential_context_strategy = getattr(policy, "sequential_context_strategy", "full")
        cache_aligned_prompts       = getattr(policy, "cache_aligned_prompts", False)

        executor  = CapsuleExecutor(
            adapter=self._resolve_adapter(spec),
            composition_level=level,
            telemetry=telemetry,
            tool_registry=tool_registry_arg,
            compound_execution_model=resolved_model,
            compound_tool_budget=getattr(policy, "compound_tool_budget", 0),
            compound_min_output_words=compound_min_output_words,
            compound_prompt_style=getattr(policy, "compound_prompt_style", "standard"),
            merged_output_structure=merged_output_structure,
            per_agent_budgets=per_agent_budgets,
            output_guidance=output_guidance,
            mean_fine_tokens_by_group=mean_fine_tokens_by_group,
            verbosity_guidance_threshold=verbosity_guidance_threshold,
            sequential_context_strategy=sequential_context_strategy,
            cache_aligned_prompts=cache_aligned_prompts,
        )
        return executor.run(hierarchy, task_input=task_input, task_id=self._task_id)

    # ------------------------------------------------------------------
    # G-4 — group-level checkpoint helpers (shared with parallel compiler)
    # ------------------------------------------------------------------

    def _load_group_checkpoint(self, group_name: str) -> "ExecutionResult | None":
        """
        Return a replayed ``ExecutionResult`` if this group already has a
        saved checkpoint for the current ``task_id``, else ``None``.

        The replayed result carries the saved outputs and final_output
        and an empty telemetry list — resumed groups do not contribute
        rolling-window signals to the composition controller.
        """
        if self._checkpoint is None:
            return None
        saved = self._checkpoint.load_group(self._task_id, group_name)
        if saved is None:
            return None
        return ExecutionResult(
            outputs=dict(saved["outputs"]),
            final_output=saved["final_output"],
            token_usage={},
            telemetry=[],
        )

    def _save_group_checkpoint(
        self, group_name: str, group_result: "ExecutionResult"
    ) -> None:
        """Persist a completed group's outputs + final_output. No-op if no checkpoint."""
        if self._checkpoint is None:
            return
        self._checkpoint.save_group(
            task_id=self._task_id,
            group_name=group_name,
            outputs={k: (v or "") for k, v in group_result.outputs.items()},
            final_output=group_result.final_output or "",
        )

    # ------------------------------------------------------------------
    # Phase H / H1 / H2 / H3 — controller observation + quality gate
    # ------------------------------------------------------------------

    def _post_run_controller_step(
        self,
        spec:              "_GroupSpec",
        compound:          CompoundCapsule,
        level:             CompositionLevel,
        group_result,
        current_task:      str,
        tool_registry_arg,
        apply_switch:      bool,
    ):
        """
        G-7: extracted post-run controller and quality-gate block.

        Runs Phase H (record overhead / build composition signal), Phase H1
        (store FINE baseline, record avg_output_tokens), Phase H2 (proactive
        shadow comparison on FINE→COMPOUND switch with escalation ladder),
        and Phase H3 (reactive quality gate for existing COMPOUND runs with
        E-1 quality-driven escalation / de-escalation).

        Both serial (`_PipelineCompiler.execute`) and parallel
        (`_ParallelPipelineCompiler.execute`) executors call this helper so
        the composition controller and quality gates stay in one place.
        Thread-safety comes from the RLock inside ``PipelineState``.

        Returns the updated ``GroupControllerState`` after the observation
        and any quality-gate mutation.
        """
        state  = self._pipeline._pipeline_state
        policy = self._pipeline.effective_policy(spec.name)

        # Phase H — record overhead for this group
        group_recs = group_result.telemetry
        if group_recs:
            total_tok   = sum(r.total_tokens        for r in group_recs)
            total_coord = sum(r.coordination_tokens for r in group_recs)
            oh = total_coord / total_tok if total_tok else 0.0
            # Phase 11: build multi-signal observation
            n_agents = len(spec.agents)
            total_reasoning = sum(
                max(0, r.total_tokens - r.coordination_tokens) for r in group_recs
            )
            avg_output_tokens = total_reasoning / len(group_recs) if group_recs else 0.0
            total_tool_calls  = sum(getattr(r, "tool_calls", 0) for r in group_recs)
            tool_calls_per_agent = total_tool_calls / n_agents if n_agents else 0.0
            # T-017: propagate error_rate and context_utilization (previously dropped)
            total_errors = sum(getattr(r, "error_count", 0) for r in group_recs)
            total_batch  = sum(getattr(r, "batch_size", 1) for r in group_recs)
            error_rate   = total_errors / total_batch if total_batch else 0.0
            ctx_util     = (
                sum(getattr(r, "context_utilization", 0.0) for r in group_recs)
                / len(group_recs)
            )
            # Phase 12: aggregate per-group latency and token usage for efficiency gates
            group_latency_ms = sum(
                getattr(r, "latency_ms", 0.0) for r in group_recs
            )
            composition_signal = CompositionSignal(
                overhead_ratio=oh,
                agent_count=n_agents,
                avg_output_tokens=avg_output_tokens,
                tool_calls_per_agent=tool_calls_per_agent,
                dependency_depth=max(0, n_agents - 1),
                error_rate=error_rate,
                context_utilization=ctx_util,
                latency_ms=group_latency_ms,
                total_tokens=total_tok,
            )
        else:
            oh = 0.0
            avg_output_tokens = 0.0
            composition_signal = None

        # Phase H1 — store FINE baseline output for quality comparison
        # T-039: also record avg_output_tokens for auto-calibration of
        # compound_min_output_words before the first COMPOUND switch.
        if level == CompositionLevel.FINE:
            s = state._load(spec.name)
            s.last_fine_output = group_result.final_output
            state._save(spec.name, s)
            if avg_output_tokens > 0:
                state.record_avg_output_tokens_fine(spec.name, avg_output_tokens)

        # Record observation and check for mode switch
        old_mode = state.get_mode(spec.name)
        updated  = state.record_and_maybe_switch(
            spec.name, overhead=oh, apply_switch=apply_switch,
            signal=composition_signal,
        )

        # Phase H2 — proactive shadow comparison on FINE→COMPOUND switch (T-033/T-040)
        if (
            apply_switch
            and self._evaluator is not None
            and policy.quality_floor is not None
            and old_mode == CompositionLevel.FINE
            and updated.current_mode == "compound"
        ):
            fine_output = group_result.final_output or ""

            # T-040: resolve the starting execution model, then walk the
            # escalation ladder (standard → two_phase → sequential) until
            # quality passes or the ladder is exhausted.
            start_model = self._resolve_compound_execution_model(
                spec, CompositionLevel.COMPOUND
            )
            policy_is_auto = getattr(policy, "compound_execution_model", "standard") == "auto"
            ladder = self._ESCALATION_LADDER if policy_is_auto else [start_model]
            # Begin at the resolved starting position
            try:
                ladder_start = ladder.index(start_model)
            except ValueError:
                ladder_start = 0
            ladder = ladder[ladder_start:]

            accepted_model: str | None = None
            for candidate_model in ladder:
                shadow_result   = self._run_group(
                    spec, compound, CompositionLevel.COMPOUND, current_task,
                    tool_registry_arg, execution_model=candidate_model,
                )
                compound_output = shadow_result.final_output or ""
                quality         = self._evaluator.evaluate(
                    self._task, fine_output, compound_output
                )
                state.record_quality(spec.name, quality)
                if quality.score >= policy.quality_floor:
                    accepted_model = candidate_model
                    break

            if accepted_model is None:
                # No mode in the ladder passed quality — revert to FINE (T-049)
                gs = state._load(spec.name)
                _do_revert(gs)  # H2 quality shadow revert
                state._save(spec.name, gs)
                updated = gs
            elif policy_is_auto and accepted_model != start_model:
                # Escalation found a better mode — persist it so future runs
                # skip the gate logic and shadow comparisons
                state.set_execution_model_override(spec.name, accepted_model)

        # Phase H3 — reactive quality gate for existing COMPOUND runs
        elif (
            apply_switch
            and self._evaluator is not None
            and level == CompositionLevel.COMPOUND
        ):
            fine_baseline = state._load(spec.name).last_fine_output
            if fine_baseline is not None:
                compound_output = group_result.final_output or ""
                quality = self._evaluator.evaluate(
                    self._task, fine_baseline, compound_output
                )
                state.record_quality(spec.name, quality)

                # E-1: quality-driven escalation ladder.
                # Uses rolling mean quality (not point estimate) so a single noisy
                # observation doesn't drive escalation.  Also de-escalates after
                # escalation_decay_window consecutive above-floor runs — allows recovery
                # if the pipeline or prompt improves without manual reconfiguration.
                escalation_enabled = getattr(policy, "escalation_enabled", False)
                if escalation_enabled and policy.quality_floor is not None:
                    gs = state._load(spec.name)
                    mean_q = gs.mean_quality(policy.window_size) or quality.score
                    if mean_q < policy.quality_floor:
                        gs.quality_failure_streak += 1
                        gs.escalation_success_streak = 0
                        state._save(spec.name, gs)
                        min_failures = getattr(policy, "escalation_min_failures", 2)
                        if gs.quality_failure_streak >= min_failures:
                            current_model = (
                                gs.execution_model_override
                                or getattr(policy, "compound_execution_model", "standard")
                            )
                            try:
                                next_idx = self._ESCALATION_LADDER.index(current_model) + 1
                            except ValueError:
                                next_idx = len(self._ESCALATION_LADDER)
                            if next_idx < len(self._ESCALATION_LADDER):
                                next_model = self._ESCALATION_LADDER[next_idx]
                                state.set_execution_model_override(spec.name, next_model)
                                gs2 = state._load(spec.name)
                                gs2.quality_failure_streak = 0
                                gs2.escalation_success_streak = 0
                                state._save(spec.name, gs2)
                            # if already at top of ladder: quality gate in
                            # get_recommendation() will return DECOMPOSE → revert next run
                    else:
                        gs.quality_failure_streak = 0
                        gs.escalation_success_streak += 1
                        state._save(spec.name, gs)
                        # De-escalate if quality has been above floor long enough
                        decay_window = getattr(policy, "escalation_decay_window", 5)
                        if (
                            gs.execution_model_override is not None
                            and gs.escalation_success_streak >= decay_window
                        ):
                            current_model = gs.execution_model_override
                            try:
                                current_idx = self._ESCALATION_LADDER.index(current_model)
                            except ValueError:
                                current_idx = 0
                            prev_model = (
                                self._ESCALATION_LADDER[current_idx - 1]
                                if current_idx > 0
                                else None
                            )
                            state.set_execution_model_override(spec.name, prev_model)
                            gs2 = state._load(spec.name)
                            gs2.escalation_success_streak = 0
                            state._save(spec.name, gs2)

        return updated

    # ------------------------------------------------------------------
    # Phase I — result assembly
    # ------------------------------------------------------------------

    def _assemble_result(
        self,
        outputs:         dict[str, str],
        final_output:    str,
        telemetry:       list,
        modes_used:      dict[str, str],
        recommendations: dict[str, str],
        confidences:     dict[str, float],
        scores:          dict[str, float],
        efficiency:      dict[str, dict],
        quality:         dict[str, float],
        latency_ms:      float,
    ) -> PipelineResult:
        # Translate internal output keys to agent names
        step_outputs: dict[str, str] = {}
        for output_key, text in outputs.items():
            agent_name = self._output_key_to_agent.get(output_key, output_key)
            step_outputs[agent_name] = text or ""

        total_tokens = sum(getattr(r, "total_tokens", 0) for r in telemetry)

        return PipelineResult(
            output=final_output or "",
            recommendation=recommendations,
            mode_used=modes_used,
            confidence=confidences,
            scores=scores,
            efficiency=efficiency,
            quality=quality,
            step_outputs=step_outputs,
            token_usage=total_tokens,
            latency_ms=round(latency_ms, 2),
        )
