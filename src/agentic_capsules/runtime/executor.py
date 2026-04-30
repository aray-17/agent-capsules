"""
Capsule Executor — work queue manager and dispatch loop.

Execution modes:
  FINE      — one LLM call per leaf agent (baseline)
  COMPOUND  — one merged LLM call for a CompoundCapsule (computation-space)
  ITERATION — one LLM call per batch of K items for the same agent (Phase 2)

Phase 5: optional GranularityController integration.
  When controller= is provided, the executor calls controller.observe(record)
  after each capsule execution, and appends the controller's recommended
  action to the ExecutionResult for the caller to act on.

Design plan ref: §3.1 (Execution Runtime), §3.2.2 (hierarchy levels),
                 §5.2 Phase 1–2, Phase 5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..core.capsule import AgentItemCapsule, AgentTagCapsule
from ..core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule, IterationCapsule, ToolLeaf
from ..core.types import (
    CapsuleExecutionError,
    CapsuleState,
    CompositionLevel,
    LLMAdapter,
    OutputKey,
    Schema,
)
from ..controller.granularity import ControllerAction, GranularityController
from ..controller.telemetry import TelemetryCollector, TelemetryRecord
from .prompt_compiler import PromptCompiler
from .scheduler import compute_iteration_schedule, compute_order
from .sync_manager import BoundarySyncManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

def _aggregate_terminal_outputs(
    compound: "CompoundCapsule",
    accumulated: dict,
    fallback: str = "",
) -> str:
    """
    Compute a group's final_output by aggregating *terminal* leaf outputs.

    A terminal leaf is one that no other leaf in the same compound declares as
    a dependency — i.e. a sink in the within-group DAG. Single-terminal groups
    (the historical case for linearly-chained pipelines like due_diligence)
    return the same string the previous "last leaf in serialization_order"
    rule produced. Multi-terminal groups (fan-out groups where several agents
    are independent siblings — e.g. the entities/claims/signals trio in
    multi_source_brief) concatenate every terminal output labeled with
    [agent_name] so downstream consumers see all parallel sibling outputs
    instead of only the last one declared.

    G-1 (AC↔LG parity, Phase 2 Batch B-2, 2026-04-10): when every terminal
    leaf's name starts with the compound's own name plus "_" (e.g. compound
    "competitive" with terminals competitive_entities / competitive_claims /
    competitive_signals), the compound-name prefix is stripped from each
    label so it renders as [entities] / [claims] / [signals]. This matches
    LangGraph's merged-arm output format — the parent-group identity is
    already carried by the downstream cross-group dep injector's
    "[competitive output]" wrapper, so repeating it on every inner label
    is pure overhead. Compounds whose terminals do not all share the
    compound-name prefix (e.g. fan-out groups with worker names w_0 / w_1
    / w_2) are unchanged — the full agent name is kept as the label.

    Args:
        compound:    The CompoundCapsule whose final_output is being computed.
        accumulated: output_key → text dict produced during execution.
        fallback:    Returned when there is no usable terminal output.
    """
    leaves = [c for c in compound.serialization_order if isinstance(c, AgentLeaf)]
    if not leaves:
        return fallback
    edges = compound.dependency_edges or {}
    has_successor: set[str] = set()
    for _name, deps in edges.items():
        for d in deps:
            has_successor.add(d)
    terminals = [l for l in leaves if l.capsule.name not in has_successor]
    if not terminals:
        terminals = [leaves[-1]]
    if len(terminals) == 1:
        out = accumulated.get(terminals[0].capsule.output_key, "")
        return out if out else fallback
    # G-1 label compaction: strip compound-name prefix if every terminal
    # shares it. Only fires on multi-terminal groups where the parent name
    # is implied by the cross-group injection wrapper downstream.
    compound_prefix = f"{compound.name}_"
    strip_prefix = all(
        leaf.capsule.name.startswith(compound_prefix)
        for leaf in terminals
    )
    parts: list[str] = []
    for leaf in terminals:
        out = accumulated.get(leaf.capsule.output_key, "")
        if out:
            label = leaf.capsule.name
            if strip_prefix:
                label = label[len(compound_prefix):]
            parts.append(f"[{label}]\n{out}")
    return "\n\n".join(parts) if parts else fallback


@dataclass
class ExecutionResult:
    """
    The output of a full pipeline run.

    `outputs` maps each output_key to extracted text (all items for ITERATION mode).
    `final_output` is the last agent/item's output.
    `token_usage` tracks total tokens consumed.
    `telemetry` holds all TelemetryRecords emitted during this run.
    """
    outputs: dict[OutputKey, str]
    final_output: str
    token_usage: dict[str, int] = field(default_factory=dict)
    telemetry: list[TelemetryRecord] = field(default_factory=list)
    recommended_action: ControllerAction = ControllerAction.MAINTAIN

    def __repr__(self) -> str:
        preview = self.final_output[:120] + "..." if len(self.final_output) > 120 else self.final_output
        return f"ExecutionResult(final={preview!r}, recommended={self.recommended_action.name})"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class CapsuleExecutor:
    """
    Dispatches a CapsuleHierarchy to an LLM adapter.

    Usage:
        executor = CapsuleExecutor(adapter=AnthropicAdapter())
        result = executor.run(hierarchy, task_input="Summarize the AI safety landscape")
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        composition_level: CompositionLevel = CompositionLevel.COMPOUND,
        sync_manager: BoundarySyncManager | None = None,
        telemetry: TelemetryCollector | None = None,
        batch_size: int = 10,
        tool_orchestrator=None,   # ToolOrchestrator | None (Phase 4)
        controller: GranularityController | None = None,  # Phase 5
        checkpoint=None,          # CheckpointStore | None (Phase 6)
        tool_registry=None,       # ToolRegistry | None (Phase 10)
        compound_execution_model: str = "standard",         # T-038
        compound_tool_budget: int = 0,                      # T-038: standard mode
        compound_min_output_words: int | None = None,       # T-038: depth hint
        compound_prompt_style: str = "standard",            # T-045: "standard"|"compact"
        merged_output_structure: str = "none",              # M-1 Track A: anti-compression
        per_agent_budgets: dict[str, int] | None = None,    # M-1 budgeted_adaptive
        output_guidance: str = "none",                      # O-1 Track A: length guidance
        mean_fine_tokens_by_group: dict[str, int] | None = None,  # O-1 auto (T-058)
        verbosity_guidance_threshold: int | None = None,    # T-058: tok/agent gate for "auto"
        sequential_context_strategy: str = "full",         # S-1 Track A: context injection
        cache_aligned_prompts: bool = False,                # C-1 Track A: Anthropic prefix caching
    ) -> None:
        self._adapter = adapter
        self._level = composition_level
        self._sync = sync_manager if sync_manager is not None else BoundarySyncManager()
        self._compiler = PromptCompiler(adapter)
        self._telemetry = telemetry if telemetry is not None else TelemetryCollector()
        self._batch_size = batch_size
        self._tool_orchestrator = tool_orchestrator
        self._controller = controller
        self._checkpoint = checkpoint
        self._tool_registry = tool_registry  # Phase 10: ToolRegistry for agent tool use
        # T-038: compound execution model configuration
        self._compound_execution_model = compound_execution_model
        self._compound_tool_budget = compound_tool_budget
        self._compound_min_output_words = compound_min_output_words
        # T-045: prompt structure style ("standard" uses == PHASE N: Name ==; "compact" uses --- Step N ---)
        self._compound_prompt_style = compound_prompt_style
        # Track A: prompt-level optimization fields
        self._merged_output_structure = merged_output_structure
        self._per_agent_budgets = per_agent_budgets or {}
        self._output_guidance = output_guidance
        self._mean_fine_tokens_by_group = mean_fine_tokens_by_group or {}
        # T-058: threshold for output_guidance="auto" observations-based gate.
        # None disables auto-gating (safe fallback when the policy does not
        # configure a threshold, e.g. test/stub setups).
        self._verbosity_guidance_threshold = verbosity_guidance_threshold
        self._sequential_context_strategy = sequential_context_strategy
        # C-1: only activate when the adapter supports prompt caching; non-Anthropic
        # adapters get a no-op so their prompt layout stays comparable to baseline.
        adapter_supports = getattr(adapter, "supports_prompt_caching", False)
        self._cache_aligned_prompts = cache_aligned_prompts and adapter_supports

    # ------------------------------------------------------------------
    # Public run interface
    # ------------------------------------------------------------------

    def run(
        self,
        hierarchy: CapsuleHierarchy,
        task_input: str,
        task_id: str = "default",
        task_inputs: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute *hierarchy* for the given *task_input*.

        *task_id* scopes this execution so multiple tasks can run against the
        same executor without cross-contaminating the sync manager.

        *task_inputs* (optional) supplies per-item content for ITERATION mode.
        When provided, each item in the batch receives its own string from
        this list rather than the broadcasted *task_input*. The list must be
        at least as long as the tag space size. Ignored in FINE/COMPOUND modes.

        Returns an ExecutionResult with all phase outputs and the final answer.
        """
        logger.info(
            "Executing hierarchy %r | level=%s | task=%r",
            hierarchy.name, self._level.name, task_id,
        )

        if self._level == CompositionLevel.COMPOUND:
            return self._run_compound(hierarchy.root, task_input, task_id)
        elif self._level == CompositionLevel.ITERATION:
            return self._run_iteration(hierarchy, task_input, task_id, task_inputs=task_inputs)
        else:
            return self._run_fine(hierarchy, task_input, task_id)

    # ------------------------------------------------------------------
    # COMPOUND mode — single merged LLM call
    # ------------------------------------------------------------------

    def _run_compound(
        self,
        compound: CompoundCapsule,
        task_input: str,
        task_id: str,
        prior_outputs: dict[OutputKey, str] | None = None,
    ) -> ExecutionResult:
        """
        Dispatch a compound capsule to the appropriate execution path.

        T-006: compounds with nested CompoundCapsule children use the mixed-compound path.
        T-038: flat compounds dispatch to two_phase or standard path based on policy.
        G-2: if every agent in the compound carries a skip_condition and they
        all evaluate to False against ``prior_outputs``, short-circuit the
        whole compound LLM call and emit SKIPPED telemetry for every agent.
        """
        g2_short_circuit = self._g2_compound_short_circuit(compound, prior_outputs)
        if g2_short_circuit is not None:
            return g2_short_circuit
        has_nested = any(
            isinstance(c, CompoundCapsule) for c in compound.children
        )
        if has_nested:
            return self._run_mixed_compound(compound, task_input, task_id, prior_outputs)

        # Ensure serialization order is computed
        if not compound.serialization_order:
            compute_order(compound)

        if self._compound_execution_model == "sequential":
            # T-039: per-agent calls with shared accumulated context.
            # No merging, no compression — each agent reasons at full depth.
            return self._run_compound_sequential(compound, task_input, task_id, prior_outputs)

        if self._compound_execution_model == "two_phase":
            # T-038: only use two_phase when at least one agent has tools.
            # Tool-free groups gain nothing from Phase A and may regress slightly
            # due to prompt structure differences — fall back to standard.
            has_tools = (
                self._tool_registry is not None
                and any(
                    leaf.capsule.tools
                    and self._tool_registry.definitions_for(leaf.capsule.tools)
                    for leaf in compound.serialization_order
                )
            )
            if has_tools:
                return self._run_compound_two_phase(compound, task_input, task_id, prior_outputs)
        return self._run_compound_standard(compound, task_input, task_id, prior_outputs)

    # ------------------------------------------------------------------
    # COMPOUND standard — single merged LLM call (original + tool budget)
    # ------------------------------------------------------------------

    def _run_compound_standard(
        self,
        compound: CompoundCapsule,
        task_input: str,
        task_id: str,
        prior_outputs: dict[OutputKey, str] | None = None,
    ) -> ExecutionResult:
        """
        Single merged LLM call for the compound.

        When compound_tool_budget != 0 and a tool_registry is present, collects
        tool definitions from all agents in the batch and passes them to the
        adapter. The adapter runs its full tool loop internally (T-038).

        All intermediate outputs stay inside the single LLM response;
        no get/put is needed between phases (boundary migration, §3.2.5).
        """
        # T-038: collect tool defs from all agents when budget != 0
        all_tool_defs = None
        if self._compound_tool_budget != 0 and self._tool_registry is not None:
            tool_names: list[str] = []
            for leaf in compound.serialization_order:
                tool_names.extend(leaf.capsule.tools)
            if tool_names:
                all_tool_defs = self._tool_registry.definitions_for(tool_names)

        # T-059 adjacent fix (2026-04-23): pass output_guidance + per-group
        # mean_fine_tokens so the single-leaf shortcut in compile_compound
        # can forward them to compile_single (due_diligence synthesis group
        # is the hot single-leaf case).
        # T-059 Phase 1 fix: also pass cache_aligned_prompts through the
        # single-leaf shortcut path.
        mean_fine_tokens = self._mean_fine_tokens_by_group.get(compound.name)
        compiled = self._compiler.compile_compound(
            compound, task_input,
            prior_outputs=prior_outputs,
            min_output_words=self._compound_min_output_words,
            compact_framing=(self._compound_prompt_style == "compact"),
            merged_output_structure=self._merged_output_structure,
            per_agent_budgets=self._per_agent_budgets if self._per_agent_budgets else None,
            output_guidance=self._output_guidance,
            mean_fine_tokens=mean_fine_tokens,
            guidance_threshold=self._verbosity_guidance_threshold,
            cache_aligned_prompts=self._cache_aligned_prompts,
        )

        logger.debug(
            "Compound %r [standard]: compiled prompt ~%d tokens, phases=%s",
            compound.name, compiled.estimated_tokens, compiled.output_keys,
        )

        for leaf in compound.serialization_order:
            leaf.capsule.state = CapsuleState.RUNNING

        start = time.perf_counter()
        try:
            if all_tool_defs:
                response = self._adapter.complete(compiled.messages, tools=all_tool_defs)
            else:
                response = self._adapter.complete(compiled.messages)
        except Exception as exc:
            for leaf in compound.serialization_order:
                leaf.capsule.state = CapsuleState.FAILED
            raise CapsuleExecutionError(
                f"LLM call failed for compound {compound.name!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        tool_call_count = getattr(self._adapter, "_last_tool_call_count", 0)
        # T-047: actual billed token counts
        input_tokens  = getattr(self._adapter, "_last_input_tokens",  0)
        output_tokens = getattr(self._adapter, "_last_output_tokens", 0)

        outputs = PromptCompiler.parse_outputs(response, compiled.output_keys)

        for leaf, key in zip(compound.serialization_order, compiled.output_keys):
            tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
            item = AgentItemCapsule(
                data=outputs.get(key, ""),
                producer_tag=tag,
                schema=leaf.capsule.output_schema,
                output_key=key,
            )
            self._sync.put_sync(tag, item)
            leaf.capsule.state = CapsuleState.COMPLETE

        # Aggregate all terminal leaves so multi-sibling fan-out groups
        # propagate every parallel output, not just the last declared.
        final_output = _aggregate_terminal_outputs(
            compound, outputs, fallback=response,
        )

        if len(compiled.output_keys) > 1:
            for leaf in compound.serialization_order[:-1]:
                self._sync.evict_tag(
                    AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                )

        record = TelemetryRecord(
            capsule_name=compound.name,
            composition_mode="COMPOUND",
            batch_size=len(compound.serialization_order),
            total_tokens=compiled.estimated_tokens,
            coordination_tokens=compiled.coordination_tokens,
            latency_ms=latency_ms,
            context_utilization=self._context_util(compiled.estimated_tokens),
            tool_calls=tool_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_call_count=1,
        )
        self._telemetry.record(record)
        self._observe(record)

        action = self._controller_action()
        token_usage = {"estimated_input": compiled.estimated_tokens}
        return ExecutionResult(
            outputs=outputs,
            final_output=final_output,
            token_usage=token_usage,
            telemetry=self._telemetry.records,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # COMPOUND two_phase — Phase A gather + Phase B reasoning (T-038)
    # ------------------------------------------------------------------

    def _run_compound_two_phase(
        self,
        compound: CompoundCapsule,
        task_input: str,
        task_id: str,
        prior_outputs: dict[OutputKey, str] | None = None,
    ) -> ExecutionResult:
        """
        Two-phase compound execution (T-038 Model B).

        Phase A: For each agent that declares tools, run a lightweight
        tool-gathering LLM call. The adapter's tool loop runs to completion;
        the response text (a concise data summary) is captured as context.
        Phase A calls emit COMPOUND_PHASE_A telemetry records.

        Phase B: Compile the standard compound prompt with Phase A tool
        contexts injected per phase. Execute as a single pure-reasoning call
        with no tools= — the adapter makes no tool round-trips. Emits one
        COMPOUND telemetry record.

        Benefits over standard mode:
        - Parallel tool execution path (future Phase 13 async engine).
        - Structural compression mitigation: injected data prevents haiku
          from collapsing phase outputs to a few sentences.
        - No budget-enforcement machinery: Phase B has no tool loop.
        - Cross-phase data sharing: all agents' tool data in shared context.

        Note: Phase A calls run sequentially today. Parallel execution is
        deferred to the Phase 13 async executor (P13-1).
        """
        from ..core.types import LLMMessage

        phase_a_contexts: dict[str, str] = {}
        phase_a_tool_calls = 0
        phase_a_total_tokens = 0

        for leaf in compound.serialization_order:
            capsule = leaf.capsule
            if not capsule.tools or self._tool_registry is None:
                continue
            tool_defs = self._tool_registry.definitions_for(capsule.tools)
            if not tool_defs:
                continue

            # Phase A gather prompt: agent system + minimal gather instruction
            phase_a_system = (
                f"{capsule.system_prompt}\n\n"
                "DATA GATHERING PRE-PASS: Use your available tools to collect the "
                "data needed for this task. After calling all relevant tools, produce "
                "a concise summary of the data you found. Do not write your full "
                "analysis — that happens in a separate step."
            )
            phase_a_msgs = [
                LLMMessage(role="system", content=phase_a_system),
                LLMMessage(role="user",   content=task_input),
            ]
            phase_a_tokens = self._adapter.count_tokens(phase_a_system + task_input)
            phase_a_total_tokens += phase_a_tokens

            start_a = time.perf_counter()
            try:
                gather_response = self._adapter.complete(phase_a_msgs, tools=tool_defs)
            except Exception as exc:
                raise CapsuleExecutionError(
                    f"Phase A gather call failed for agent {leaf.name!r} "
                    f"in compound {compound.name!r}: {exc}"
                ) from exc
            latency_a_ms = (time.perf_counter() - start_a) * 1000

            tc = getattr(self._adapter, "_last_tool_call_count",    0)
            ts = getattr(self._adapter, "_last_tool_call_sequence", [])
            phase_a_tool_calls += tc
            phase_a_contexts[leaf.name] = gather_response
            # T-047: capture per-Phase A call billing
            pa_input  = getattr(self._adapter, "_last_input_tokens",  0)
            pa_output = getattr(self._adapter, "_last_output_tokens", 0)

            # Emit telemetry for this Phase A gather call
            phase_a_record = TelemetryRecord(
                capsule_name=f"{leaf.name}[phase_a]",
                composition_mode="COMPOUND_PHASE_A",
                batch_size=1,
                total_tokens=phase_a_tokens,
                coordination_tokens=0,
                latency_ms=latency_a_ms,
                tool_calls=tc,
                tool_call_sequence=list(ts),
                input_tokens=pa_input,
                output_tokens=pa_output,
                llm_call_count=1,
            )
            self._telemetry.record(phase_a_record)

        logger.debug(
            "Compound %r [two_phase]: Phase A tool_calls=%d, agents_with_context=%d",
            compound.name, phase_a_tool_calls, len(phase_a_contexts),
        )

        # Phase B: compile with injected tool contexts, run pure reasoning.
        # T-059 adjacent fix (2026-04-23): pass output_guidance + mean_fine_tokens
        # for the single-leaf shortcut path (same as _run_compound_standard).
        # When tool_contexts is set, the shortcut is skipped in compile_compound
        # and these params are effectively ignored by the multi-phase path —
        # safe no-op. T-059 Phase 1 fix: cache_aligned_prompts too.
        mean_fine_tokens = self._mean_fine_tokens_by_group.get(compound.name)
        compiled = self._compiler.compile_compound(
            compound, task_input,
            prior_outputs=prior_outputs,
            tool_contexts=phase_a_contexts if phase_a_contexts else None,
            min_output_words=self._compound_min_output_words,
            compact_framing=(self._compound_prompt_style == "compact"),
            merged_output_structure=self._merged_output_structure,
            per_agent_budgets=self._per_agent_budgets if self._per_agent_budgets else None,
            output_guidance=self._output_guidance,
            mean_fine_tokens=mean_fine_tokens,
            guidance_threshold=self._verbosity_guidance_threshold,
            cache_aligned_prompts=self._cache_aligned_prompts,
        )

        logger.debug(
            "Compound %r [two_phase]: Phase B ~%d tokens, phases=%s",
            compound.name, compiled.estimated_tokens, compiled.output_keys,
        )

        for leaf in compound.serialization_order:
            leaf.capsule.state = CapsuleState.RUNNING

        start_b = time.perf_counter()
        try:
            response = self._adapter.complete(compiled.messages)  # no tools=
        except Exception as exc:
            for leaf in compound.serialization_order:
                leaf.capsule.state = CapsuleState.FAILED
            raise CapsuleExecutionError(
                f"Phase B reasoning call failed for compound {compound.name!r}: {exc}"
            ) from exc
        latency_b_ms = (time.perf_counter() - start_b) * 1000

        outputs = PromptCompiler.parse_outputs(response, compiled.output_keys)

        for leaf, key in zip(compound.serialization_order, compiled.output_keys):
            tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
            item = AgentItemCapsule(
                data=outputs.get(key, ""),
                producer_tag=tag,
                schema=leaf.capsule.output_schema,
                output_key=key,
            )
            self._sync.put_sync(tag, item)
            leaf.capsule.state = CapsuleState.COMPLETE

        final_output = _aggregate_terminal_outputs(
            compound, outputs, fallback=response,
        )

        if len(compiled.output_keys) > 1:
            for leaf in compound.serialization_order[:-1]:
                self._sync.evict_tag(
                    AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                )

        # T-047: Phase B billing
        pb_input  = getattr(self._adapter, "_last_input_tokens",  0)
        pb_output = getattr(self._adapter, "_last_output_tokens", 0)

        # Phase B compound record — tool_calls reflects total Phase A tool usage
        compound_record = TelemetryRecord(
            capsule_name=compound.name,
            composition_mode="COMPOUND",
            batch_size=len(compound.serialization_order),
            total_tokens=compiled.estimated_tokens,
            coordination_tokens=compiled.coordination_tokens,
            latency_ms=latency_b_ms,
            context_utilization=self._context_util(compiled.estimated_tokens),
            tool_calls=phase_a_tool_calls,
            input_tokens=pb_input,
            output_tokens=pb_output,
            llm_call_count=1,  # Phase B is one call; Phase A calls are in their own records
        )
        self._telemetry.record(compound_record)
        self._observe(compound_record)

        action = self._controller_action()
        token_usage = {"estimated_input": compiled.estimated_tokens + phase_a_total_tokens}
        return ExecutionResult(
            outputs=outputs,
            final_output=final_output,
            token_usage=token_usage,
            telemetry=self._telemetry.records,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # COMPOUND sequential — per-agent calls with shared context (T-039)
    # ------------------------------------------------------------------

    def _run_compound_sequential(
        self,
        compound: CompoundCapsule,
        task_input: str,
        task_id: str,
        prior_outputs: dict[OutputKey, str] | None = None,
    ) -> ExecutionResult:
        """
        Sequential compound execution (T-039).

        Each agent in the compound's serialization_order receives its own
        full LLM call via compile_single().  Outputs accumulate: each agent
        sees all prior agents' outputs as context, exactly as in FINE mode.

        Unlike FINE mode, sequential executes within the COMPOUND dispatcher
        so coordination overhead is attributed to the compound group (not to
        individual FINE records), and per-agent tool access is preserved.

        Benefits over standard and two_phase:
        - No merging → no compression → each agent reasons at full FINE depth
        - Shared accumulated context → downstream agents build on upstream work
        - Per-agent tool loop preserved (no tool budget, no Phase A gather)
        - Token savings (~20–30%) from eliminating repeated task-context overhead
          across multiple independent FINE calls

        Recommended for verbose models (haiku, sonnet) where standard and
        two_phase compress quality below quality_floor=0.75.

        Note: latency scales linearly with agent count (N sequential calls).
        The async executor (Phase 13) can parallelize independent agents later.
        """
        accumulated: dict[OutputKey, str] = dict(prior_outputs) if prior_outputs else {}
        # Boundary outputs from before this group — always injected regardless of strategy.
        initial_prior: dict[OutputKey, str] = dict(prior_outputs) if prior_outputs else {}
        total_tokens = 0
        total_coord_tokens = 0
        total_tool_calls = 0
        total_latency_ms = 0.0
        # T-047: COGS accumulators
        total_input_tokens = 0
        total_output_tokens = 0
        total_llm_calls = 0

        # T-042: build name → output_key map for dep-aware injection
        # S-1 (Track A): policy-level strategy overrides topology default when set to
        # "predecessor_only"; topology-detected "deps" always takes precedence.
        topology_strategy = getattr(compound, "sequential_injection_strategy", "full")
        if topology_strategy == "deps":
            strategy = "deps"   # topology classifier wins
        elif self._sequential_context_strategy == "predecessor_only":
            strategy = "predecessor_only"
        else:
            strategy = topology_strategy  # "full" (default)
        leaf_output_keys: dict[str, OutputKey] = {
            leaf.name: leaf.capsule.output_key
            for leaf in compound.serialization_order
        }
        # O-1: group-level mean fine tokens for adaptive guidance
        mean_fine_tokens = self._mean_fine_tokens_by_group.get(compound.name)

        for idx, leaf in enumerate(compound.serialization_order):
            capsule = leaf.capsule
            capsule.state = CapsuleState.RUNNING

            # Resolve tool definitions for this agent (mirrors FINE mode)
            tool_defs = None
            if self._tool_registry is not None and capsule.tools:
                tool_defs = self._tool_registry.definitions_for(capsule.tools)

            # T-042 / S-1: select context to inject based on injection strategy
            if strategy == "deps":
                dep_names = compound.dependency_edges.get(leaf.name, [])
                dep_outputs = {
                    leaf_output_keys[dep]: accumulated[leaf_output_keys[dep]]
                    for dep in dep_names
                    if dep in leaf_output_keys and leaf_output_keys[dep] in accumulated
                }
                ctx = {**initial_prior, **dep_outputs} if (initial_prior or dep_outputs) else None
            elif strategy == "predecessor_only":
                # S-1: only inject the immediate predecessor's output + boundary inputs
                if idx == 0:
                    ctx = initial_prior if initial_prior else None
                else:
                    prev_leaf = compound.serialization_order[idx - 1]
                    prev_key  = prev_leaf.capsule.output_key
                    prev_out  = accumulated.get(prev_key)
                    if prev_out:
                        ctx = {**initial_prior, prev_key: prev_out} if initial_prior else {prev_key: prev_out}
                    else:
                        ctx = initial_prior if initial_prior else None
            else:
                # "full" and "summary" stub: inject all accumulated outputs
                ctx = accumulated if accumulated else None

            compiled = self._compiler.compile_single(
                leaf, task_input,
                prior_outputs=ctx,
                output_guidance=self._output_guidance,
                mean_fine_tokens=mean_fine_tokens,
                guidance_threshold=self._verbosity_guidance_threshold,
                cache_aligned_prompts=self._cache_aligned_prompts,
            )
            total_tokens      += compiled.estimated_tokens
            total_coord_tokens += compiled.coordination_tokens

            start = time.perf_counter()
            try:
                if tool_defs:
                    response = self._adapter.complete(compiled.messages, tools=tool_defs)
                else:
                    response = self._adapter.complete(compiled.messages)
            except Exception as exc:
                capsule.state = CapsuleState.FAILED
                raise CapsuleExecutionError(
                    f"Sequential compound call failed for agent {leaf.name!r} "
                    f"in compound {compound.name!r}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000
            total_latency_ms += latency_ms

            tool_call_count = getattr(self._adapter, "_last_tool_call_count", 0)
            total_tool_calls += tool_call_count
            # T-047: accumulate billed tokens per agent call
            total_input_tokens  += getattr(self._adapter, "_last_input_tokens",  0)
            total_output_tokens += getattr(self._adapter, "_last_output_tokens", 0)
            total_llm_calls     += 1

            parsed = PromptCompiler.parse_outputs(response, compiled.output_keys)
            output_text = parsed.get(capsule.output_key, response)
            accumulated[capsule.output_key] = output_text

            tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
            item = AgentItemCapsule(
                data=output_text,
                producer_tag=tag,
                schema=capsule.output_schema,
                output_key=capsule.output_key,
            )
            self._sync.put_sync(tag, item)
            capsule.state = CapsuleState.COMPLETE

        # Evict intermediate outputs — mirrors standard COMPOUND behaviour
        if len(compound.serialization_order) > 1:
            for leaf in compound.serialization_order[:-1]:
                self._sync.evict_tag(
                    AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                )

        final_output = _aggregate_terminal_outputs(compound, accumulated)

        # Emit as a single COMPOUND record so the composition scorer attributes
        # this execution to the group (not to N separate FINE observations).
        record = TelemetryRecord(
            capsule_name=compound.name,
            composition_mode="COMPOUND",
            batch_size=len(compound.serialization_order),
            total_tokens=total_tokens,
            coordination_tokens=total_coord_tokens,
            latency_ms=total_latency_ms,
            context_utilization=self._context_util(total_tokens),
            tool_calls=total_tool_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            llm_call_count=total_llm_calls,
        )
        self._telemetry.record(record)
        self._observe(record)

        action = self._controller_action()
        return ExecutionResult(
            outputs=accumulated,
            final_output=final_output,
            token_usage={"estimated_input": total_tokens},
            telemetry=self._telemetry.records,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # Mixed-compound dispatch (T-006) — nested CompoundCapsule children
    # ------------------------------------------------------------------

    def _run_mixed_compound(
        self,
        compound: CompoundCapsule,
        task_input: str,
        task_id: str,
        prior_outputs: dict[OutputKey, str] | None = None,
    ) -> ExecutionResult:
        """
        Dispatch a compound whose children include nested CompoundCapsules.

        Children are dispatched one by one in declaration order:
          - CompoundCapsule child → recurse _run_compound()
          - AgentLeaf child       → compile_single() + one LLM call

        Outputs accumulate across children so each step sees all prior results.
        No single merged prompt is built; each child is its own LLM invocation.

        Design plan ref: §3.2.1, T-006
        """
        accumulated: dict[OutputKey, str] = dict(prior_outputs) if prior_outputs else {}
        total_tokens = 0
        last_result: ExecutionResult | None = None

        # Topological order: build from dependency_edges over child names
        child_map = {c.name: c for c in compound.children}
        ordered_children = _topo_order_children(compound.children, compound.dependency_edges)

        for child in ordered_children:
            if isinstance(child, CompoundCapsule):
                # Ensure inner compound's order is computed before recursing
                if not child.serialization_order:
                    compute_order(child)
                result = self._run_compound(
                    child, task_input, task_id,
                    prior_outputs=dict(accumulated) if accumulated else None,
                )
                accumulated.update(result.outputs)
                total_tokens += result.token_usage.get("estimated_input", 0)
                last_result = result

            elif isinstance(child, AgentLeaf):
                capsule = child.capsule
                capsule.state = CapsuleState.RUNNING

                if hasattr(self._adapter, "current_capsule"):
                    self._adapter.current_capsule = child.name  # type: ignore[union-attr]

                # T-059 adjacent fix (2026-04-23): plumb output_guidance +
                # per-group observations through compile_single so auto-concise
                # fires in mixed-compound topology too. Same pattern as _run_fine
                # and _run_compound_sequential. Group name resolves from the
                # containing compound (which owns rolling observations).
                # T-059 Phase 1 fix: also pass cache_aligned_prompts so the
                # cacheable-prefix structure applies in nested-compound
                # topology.
                mean_fine_tokens = self._mean_fine_tokens_by_group.get(compound.name)
                compiled = self._compiler.compile_single(
                    child, task_input,
                    prior_outputs=dict(accumulated) if accumulated else None,
                    output_guidance=self._output_guidance,
                    mean_fine_tokens=mean_fine_tokens,
                    guidance_threshold=self._verbosity_guidance_threshold,
                    cache_aligned_prompts=self._cache_aligned_prompts,
                )
                total_tokens += compiled.estimated_tokens

                # Resolve tool definitions for this agent (Phase 10)
                tool_defs = None
                if self._tool_registry is not None and capsule.tools:
                    tool_defs = self._tool_registry.definitions_for(capsule.tools)

                start = time.perf_counter()
                try:
                    if tool_defs:
                        response = self._adapter.complete(compiled.messages, tools=tool_defs)
                    else:
                        response = self._adapter.complete(compiled.messages)
                except Exception as exc:
                    capsule.state = CapsuleState.FAILED
                    raise CapsuleExecutionError(
                        f"LLM call failed for agent {child.name!r}: {exc}"
                    ) from exc
                latency_ms = (time.perf_counter() - start) * 1000

                tool_call_count    = getattr(self._adapter, "_last_tool_call_count",    0)
                tool_call_sequence = getattr(self._adapter, "_last_tool_call_sequence", [])

                parsed = PromptCompiler.parse_outputs(response, compiled.output_keys)
                output_text = parsed.get(capsule.output_key, response)
                accumulated[capsule.output_key] = output_text

                tag = AgentTagCapsule(agent_name=child.name, task_id=task_id)
                item = AgentItemCapsule(
                    data=output_text,
                    producer_tag=tag,
                    schema=capsule.output_schema,
                    output_key=capsule.output_key,
                )
                self._sync.put_sync(tag, item)
                capsule.state = CapsuleState.COMPLETE

                record = TelemetryRecord(
                    capsule_name=child.name,
                    composition_mode="FINE",
                    batch_size=1,
                    total_tokens=compiled.estimated_tokens,
                    coordination_tokens=compiled.coordination_tokens,
                    latency_ms=latency_ms,
                    context_utilization=self._context_util(compiled.estimated_tokens),
                    tool_calls=tool_call_count,
                    tool_call_sequence=list(tool_call_sequence),
                )
                self._telemetry.record(record)
                self._observe(record)

        action = self._controller_action()
        # Determine final output. For nested compounds we delegate to the
        # inner compound's own final_output (which has already been aggregated
        # by _aggregate_terminal_outputs). For flat AgentLeaf children we run
        # the same terminal-leaf aggregation across all leaf children of the
        # mixed compound so multi-sibling fan-out groups propagate every
        # parallel output, not just the last one declared.
        final_output = ""
        if ordered_children:
            last_child = ordered_children[-1]
            if isinstance(last_child, CompoundCapsule) and last_result is not None:
                final_output = last_result.final_output
            else:
                final_output = _aggregate_terminal_outputs(compound, accumulated)

        return ExecutionResult(
            outputs=accumulated,
            final_output=final_output,
            token_usage={"estimated_input": total_tokens},
            telemetry=self._telemetry.records,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # FINE mode — each leaf as an independent LLM call
    # ------------------------------------------------------------------

    def _run_fine(
        self,
        hierarchy: CapsuleHierarchy,
        task_input: str,
        task_id: str,
    ) -> ExecutionResult:
        """
        Execute every leaf in dependency order.

        AgentLeaf nodes make one LLM call each.
        ToolLeaf nodes are dispatched to the ToolOrchestrator (no LLM round-trip).

        Results flow through the BoundarySyncManager so each subsequent leaf
        can access prior outputs via prior_outputs context.
        """
        order = compute_order(hierarchy.root)

        # T-058 (extension): plumb per-group FINE observations so the
        # output_guidance="auto" gate in compile_single can fire during FINE
        # execution. Mirrors the pattern at _run_compound_sequential:658 where
        # the same lookup already works. Before this fix, _run_fine passed
        # neither mean_fine_tokens nor guidance_threshold, so the "auto"
        # branch in compile_single always fell through to no-guidance in
        # FINE mode regardless of observed per-agent verbosity — T-058's
        # concise-gate was effectively a no-op for FINE callers, leaving
        # verbose pipelines (due_diligence on Sonnet: ~4,000 tok/agent) with
        # no automatic output compression.
        mean_fine_tokens = self._mean_fine_tokens_by_group.get(hierarchy.root.name)

        # G-2 (parity): collect explicit intra-compound dependency declarations
        # so siblings with `depends_on=[]` do not pollute each other via the
        # running accumulated_outputs dict. `declared_deps` maps leaf.name →
        # list of dep leaf names, but only for leaves whose containing
        # CompoundCapsule uses the "deps" injection strategy (set by
        # classify_and_set_strategy when the user opted into explicit
        # depends_on). Implicit-linear groups stay on "full" strategy and are
        # not added to declared_deps — they fall through to legacy behaviour.
        # `leaf_output_keys` maps leaf.name → output_key so we can translate
        # the declared dep names into the accumulated_outputs key space.
        declared_deps, leaf_output_keys = _collect_fine_dep_info(hierarchy.root)

        # Restore from checkpoint if available (Phase 6)
        accumulated_outputs: dict[OutputKey, str] = {}
        if self._checkpoint is not None:
            saved = self._checkpoint.load(task_id)
            if saved:
                accumulated_outputs = saved
                logger.info("Restored checkpoint for task %r: %d outputs", task_id, len(saved))
        total_tokens = 0

        for leaf in order:
            if isinstance(leaf, ToolLeaf):
                output_text, latency_ms = self._dispatch_tool_leaf(
                    leaf, task_input, task_id
                )
                output_key = f"{leaf.name.upper()}_OUTPUT"
                accumulated_outputs[output_key] = output_text

                tool_record = TelemetryRecord(
                    capsule_name=leaf.name,
                    composition_mode="TOOL",
                    batch_size=len(leaf.tool_capsule),  # type: ignore[arg-type]
                    total_tokens=0,   # tool calls don't consume LLM tokens
                    coordination_tokens=0,
                    latency_ms=latency_ms,
                )
                self._telemetry.record(tool_record)
                self._observe(tool_record)

                tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                item = AgentItemCapsule(
                    data=output_text,
                    producer_tag=tag,
                    schema=Schema("tool_output", fields={"result": "str"}),
                    output_key=output_key,
                )
                self._sync.put_sync(tag, item)
                continue

            # AgentLeaf path
            capsule = leaf.capsule

            # Skip leaves already present in a restored checkpoint
            if capsule.output_key in accumulated_outputs:
                capsule.state = CapsuleState.COMPLETE
                tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                item = AgentItemCapsule(
                    data=accumulated_outputs[capsule.output_key],
                    producer_tag=tag,
                    schema=capsule.output_schema,
                    output_key=capsule.output_key,
                )
                self._sync.put_sync(tag, item)
                logger.debug("Skipping checkpointed leaf %r", leaf.name)
                continue

            # G-2: runtime skip predicate. When set, call with the current
            # accumulated-outputs snapshot; False skips the LLM call entirely
            # and propagates "" as the agent's output so downstream deps see
            # an empty string instead of a KeyError. A zero-cost SKIPPED
            # telemetry record is emitted so the controller / caller still
            # observes the node. The snapshot is deliberately shallow-copied
            # so user predicates cannot mutate executor state.
            if capsule.skip_condition is not None:
                try:
                    should_run = bool(
                        capsule.skip_condition(dict(accumulated_outputs))
                    )
                except Exception as exc:
                    raise CapsuleExecutionError(
                        f"skip_condition for agent {leaf.name!r} raised: {exc}"
                    ) from exc
                if not should_run:
                    capsule.state = CapsuleState.SKIPPED
                    accumulated_outputs[capsule.output_key] = ""

                    skipped_record = TelemetryRecord(
                        capsule_name=leaf.name,
                        composition_mode="SKIPPED",
                        batch_size=1,
                        total_tokens=0,
                        coordination_tokens=0,
                        latency_ms=0.0,
                        llm_call_count=0,
                    )
                    self._telemetry.record(skipped_record)
                    self._observe(skipped_record)

                    tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
                    item = AgentItemCapsule(
                        data="",
                        producer_tag=tag,
                        schema=capsule.output_schema,
                        output_key=capsule.output_key,
                    )
                    self._sync.put_sync(tag, item)
                    logger.debug(
                        "G-2: skipped agent %r (condition returned False)",
                        leaf.name,
                    )
                    continue

            capsule.state = CapsuleState.RUNNING

            # Inform ModelRouter (if present) which capsule is active
            if hasattr(self._adapter, "current_capsule"):
                self._adapter.current_capsule = leaf.name  # type: ignore[union-attr]

            # G-2 (parity): narrow prior_outputs to declared dependencies only
            # when the containing compound opted into the "deps" injection
            # strategy (see _collect_fine_dep_info). An explicit empty list
            # (depends_on=[]) yields no prior; an explicit non-empty list
            # filters to those leaves' output_keys; absence from declared_deps
            # preserves the legacy "pass everything" path used by implicit-
            # linear pipelines (code_review, long_chain_research, etc).
            decl = declared_deps.get(leaf.name)
            if decl is None:
                prior = dict(accumulated_outputs) if accumulated_outputs else None
            elif not decl:
                prior = None
            else:
                allowed_keys = {
                    leaf_output_keys[d] for d in decl if d in leaf_output_keys
                }
                filtered = {
                    k: v for k, v in accumulated_outputs.items() if k in allowed_keys
                }
                prior = filtered if filtered else None
            # O-1 (Track A) + T-058 extension: pass output_guidance with the
            # observations-based gate inputs. When output_guidance="auto",
            # compile_single checks mean_fine_tokens against guidance_threshold
            # and applies concise guidance only when the group's mean exceeds
            # the threshold — same behavior as _run_compound_sequential.
            # T-059 Phase 1 fix (2026-04-23): pass cache_aligned_prompts so
            # FINE mode uses the cacheable-system-prefix message structure
            # identical to _run_compound_sequential. Previously defaulted to
            # False and FINE mode billed full token rate on every call's
            # user block (task + prior_outputs). Plumbing gap identified by
            # evals/dspy_ac_internal_gap_audit.md.
            compiled = self._compiler.compile_single(
                leaf, task_input,
                prior_outputs=prior,
                output_guidance=self._output_guidance,
                mean_fine_tokens=mean_fine_tokens,
                guidance_threshold=self._verbosity_guidance_threshold,
                cache_aligned_prompts=self._cache_aligned_prompts,
            )

            # Resolve tool definitions for this agent (Phase 10)
            tool_defs = None
            if self._tool_registry is not None and capsule.tools:
                tool_defs = self._tool_registry.definitions_for(capsule.tools)

            logger.debug("Fine-grained leaf %r: ~%d tokens", leaf.name, compiled.estimated_tokens)
            total_tokens += compiled.estimated_tokens

            start = time.perf_counter()
            try:
                # Only pass tools= when there are tool defs — preserves backwards
                # compatibility with adapters that don't accept the parameter.
                if tool_defs:
                    response = self._adapter.complete(compiled.messages, tools=tool_defs)
                else:
                    response = self._adapter.complete(compiled.messages)
            except Exception as exc:
                capsule.state = CapsuleState.FAILED
                raise CapsuleExecutionError(
                    f"LLM call failed for agent {leaf.name!r}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - start) * 1000

            # Capture tool call count and sequence from adapter if available (Phase 10)
            tool_call_count    = getattr(self._adapter, "_last_tool_call_count",    0)
            tool_call_sequence = getattr(self._adapter, "_last_tool_call_sequence", [])
            # T-047: actual billed token counts
            input_tokens  = getattr(self._adapter, "_last_input_tokens",  0)
            output_tokens = getattr(self._adapter, "_last_output_tokens", 0)

            fine_record = TelemetryRecord(
                capsule_name=leaf.name,
                composition_mode="FINE",
                batch_size=1,
                total_tokens=compiled.estimated_tokens,
                coordination_tokens=compiled.coordination_tokens,  # T-005 fix
                latency_ms=latency_ms,
                context_utilization=self._context_util(compiled.estimated_tokens),
                tool_calls=tool_call_count,
                tool_call_sequence=list(tool_call_sequence),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_call_count=1,
            )
            self._telemetry.record(fine_record)
            self._observe(fine_record)

            parsed = PromptCompiler.parse_outputs(response, compiled.output_keys)
            output_text = parsed.get(capsule.output_key, response)
            accumulated_outputs[capsule.output_key] = output_text

            tag = AgentTagCapsule(agent_name=leaf.name, task_id=task_id)
            item = AgentItemCapsule(
                data=output_text,
                producer_tag=tag,
                schema=capsule.output_schema,
                output_key=capsule.output_key,
            )
            self._sync.put_sync(tag, item)
            capsule.state = CapsuleState.COMPLETE

            # Checkpoint accumulated outputs after each leaf (Phase 6)
            if self._checkpoint is not None:
                self._checkpoint.save(task_id, dict(accumulated_outputs))

        # Aggregate terminal-leaf outputs so multi-sibling fan-out groups
        # (e.g. independent extractors with depends_on=[]) propagate every
        # sibling instead of just the last one declared. Single-terminal
        # groups still return exactly one leaf's output.
        final_output = _aggregate_terminal_outputs(
            hierarchy.root, accumulated_outputs,
        )
        # Tool-leaf-only fallback: if the dep graph yielded no aggregable
        # AgentLeaf terminal (e.g. a ToolLeaf-only chain), fall back to the
        # historical "last leaf in order" rule.
        if not final_output:
            last_leaf = order[-1]
            if isinstance(last_leaf, ToolLeaf):
                final_key = f"{last_leaf.name.upper()}_OUTPUT"
            else:
                final_key = last_leaf.capsule.output_key
            final_output = accumulated_outputs.get(final_key, "")

        action = self._controller_action()
        return ExecutionResult(
            outputs=accumulated_outputs,
            final_output=final_output,
            token_usage={"estimated_input": total_tokens},
            telemetry=self._telemetry.records,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # G-2: compound short-circuit on unanimous skip
    # ------------------------------------------------------------------

    def _g2_compound_short_circuit(
        self,
        compound: CompoundCapsule,
        prior_outputs: dict[OutputKey, str] | None,
    ) -> ExecutionResult | None:
        """
        If every agent in the compound has a skip_condition and every one
        returns False against the prior-group outputs dict, short-circuit the
        compound call.

        Returns a synthesised ExecutionResult (every agent recorded as
        SKIPPED, outputs empty strings) when the short-circuit fires; returns
        None otherwise so the normal dispatch continues.

        Inside a multi-agent compound LLM call the framework cannot skip an
        individual agent — the call is a single round trip to the model. The
        only honest G-2 semantic for compound mode is the unanimous case:
        if *every* agent would be skipped, there is no reason to make the
        call. Partial skipping logs a warning once and executes normally.
        """
        leaves = [c for c in compound.serialization_order if isinstance(c, AgentLeaf)]
        if not leaves:
            return None
        # Nothing to short-circuit unless every agent declares a condition.
        if not all(l.capsule.skip_condition is not None for l in leaves):
            # Partial-skip warning: at least one agent has a condition but
            # not all. Emit once per compound run at DEBUG (not WARNING) to
            # avoid noise for the common case where only one group has a
            # condition and is running compound on its own.
            if any(l.capsule.skip_condition is not None for l in leaves):
                names = [
                    l.capsule.name for l in leaves
                    if l.capsule.skip_condition is not None
                ]
                logger.debug(
                    "G-2: skip_condition set on %s but compound call cannot "
                    "skip individual agents — proceeding with full dispatch",
                    names,
                )
            return None
        snapshot = dict(prior_outputs) if prior_outputs else {}
        for leaf in leaves:
            try:
                if bool(leaf.capsule.skip_condition(snapshot)):
                    return None
            except Exception as exc:
                raise CapsuleExecutionError(
                    f"skip_condition for agent {leaf.capsule.name!r} raised: {exc}"
                ) from exc
        # Unanimous skip — synthesise the zero-cost result.
        logger.debug(
            "G-2: short-circuiting compound %r — all %d agents skipped",
            compound.name, len(leaves),
        )
        outputs: dict[OutputKey, str] = dict(snapshot)
        for leaf in leaves:
            cap = leaf.capsule
            cap.state = CapsuleState.SKIPPED
            outputs[cap.output_key] = ""
            skipped_record = TelemetryRecord(
                capsule_name=leaf.capsule.name,
                composition_mode="SKIPPED",
                batch_size=1,
                total_tokens=0,
                coordination_tokens=0,
                latency_ms=0.0,
                llm_call_count=0,
            )
            self._telemetry.record(skipped_record)
            self._observe(skipped_record)
        final_output = _aggregate_terminal_outputs(compound, outputs, fallback="")
        return ExecutionResult(
            outputs=outputs,
            final_output=final_output,
            token_usage={"estimated_input": 0},
            telemetry=self._telemetry.records,
            recommended_action=self._controller_action(),
        )

    # ------------------------------------------------------------------
    # Controller helpers (Phase 5)
    # ------------------------------------------------------------------

    def _context_util(self, tokens: int) -> float:
        """Fraction of the adapter's context window used by *tokens*."""
        ctx = getattr(self._adapter, "context_window", 0)
        if ctx <= 0:
            return 0.0
        return min(tokens / ctx, 1.0)

    def _observe(self, record: TelemetryRecord) -> None:
        """Feed a record into the controller's window if one is configured."""
        if self._controller is not None:
            self._controller.observe(record)

    def _controller_action(self) -> ControllerAction:
        """Ask the controller for its recommended next action, if configured."""
        if self._controller is None:
            return ControllerAction.MAINTAIN
        action, _ = self._controller.decide()
        return action

    def _dispatch_tool_leaf(
        self,
        leaf: ToolLeaf,
        task_input: str,
        task_id: str,
    ) -> tuple[str, float]:
        """
        Run a ToolLeaf via the ToolOrchestrator and return (output_text, latency_ms).

        Raises CapsuleExecutionError if no tool_orchestrator is configured.
        """
        if self._tool_orchestrator is None:
            raise CapsuleExecutionError(
                f"ToolLeaf {leaf.name!r} encountered but no tool_orchestrator "
                f"was provided to CapsuleExecutor. Pass tool_orchestrator= on init."
            )
        from ..tools.orchestrator import ToolExecutionError
        # T-007 fix: build initial_input from the first step's declared input_keys.
        # If the capsule's first step expects a single key, map task_input to it.
        # For multi-key first steps, fall back to the legacy "query" key.
        tool_capsule = leaf.tool_capsule  # type: ignore[union-attr]
        first_step = tool_capsule.steps[0]
        if len(first_step.input_keys) == 1:
            initial_input = {first_step.input_keys[0]: task_input}
        else:
            initial_input = {"query": task_input}
        try:
            result = self._tool_orchestrator.run(
                tool_capsule,
                initial_input=initial_input,
            )
        except ToolExecutionError as exc:
            raise CapsuleExecutionError(
                f"Tool chain {leaf.name!r} failed at step {exc.step_index} "
                f"({exc.tool_name!r}): {exc}"
            ) from exc
        # Serialize final output dict to a string for the outputs map
        final = result.final_output
        output_text = "; ".join(f"{k}={v}" for k, v in final.items())
        return output_text, result.total_latency_ms

    # ------------------------------------------------------------------
    # ITERATION mode — one batch of K items per LLM call (Phase 2)
    # ------------------------------------------------------------------

    def _run_iteration(
        self,
        hierarchy: CapsuleHierarchy,
        task_input: str,
        task_id: str,
        task_inputs: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a single-agent hierarchy in iteration-space composition.

        Partitions the hierarchy's TagSpace into batches of `batch_size`,
        dispatches one LLM call per batch, parses K outputs per call, and
        stores each item's output in the BoundarySyncManager.

        Telemetry is recorded per batch, capturing the overhead ratio
        (coordination tokens / total tokens) for later controller use.

        Design plan ref: §5.2 Phase 2, §3.2.2
        """
        from ..core.tag import TagSpace

        if hierarchy.tag_space is None:
            raise CapsuleExecutionError(
                f"Hierarchy {hierarchy.name!r} has no tag_space. "
                f"Set tag_space= to use ITERATION mode."
            )

        tag_space: TagSpace = hierarchy.tag_space  # type: ignore[assignment]

        # The hierarchy root's first (and typically only) leaf is the agent
        leaf = next(iter(hierarchy.all_leaves()))

        batches = tag_space.partition(self._batch_size)
        all_outputs: dict[OutputKey, str] = {}
        total_tokens = 0
        last_batch_outputs: dict[OutputKey, str] = {}

        for batch_idx, batch_tags in enumerate(batches):
            iteration_capsule = IterationCapsule(
                leaf=leaf,
                tags=batch_tags,
                batch_index=batch_idx,
            )

            # Per-item content: use task_inputs slice when provided (T-002),
            # otherwise broadcast task_input to all items in the batch.
            batch_start = batch_idx * self._batch_size
            if task_inputs is not None:
                item_contents = task_inputs[batch_start: batch_start + len(batch_tags)]
            else:
                item_contents = [task_input] * len(batch_tags)

            start = time.perf_counter()
            compiled = self._compiler.compile_iteration_batch(
                iteration_capsule, item_contents
            )

            leaf.capsule.state = CapsuleState.RUNNING
            try:
                response = self._adapter.complete(compiled.messages)
            except Exception as exc:
                leaf.capsule.state = CapsuleState.FAILED
                raise CapsuleExecutionError(
                    f"LLM call failed for iteration batch "
                    f"{batch_idx} of {leaf.name!r}: {exc}"
                ) from exc

            latency_ms = (time.perf_counter() - start) * 1000

            # Parse K item outputs
            batch_outputs = PromptCompiler.parse_outputs(response, compiled.output_keys)

            # Store each item output under its individual tag
            for tag, key in zip(batch_tags, compiled.output_keys):
                output_text = batch_outputs.get(key, "")
                item = AgentItemCapsule(
                    data=output_text,
                    producer_tag=tag,
                    schema=leaf.capsule.output_schema,
                    output_key=key,
                )
                self._sync.put_sync(tag, item)
                all_outputs[key] = output_text

            last_batch_outputs = batch_outputs
            total_tokens += compiled.estimated_tokens

            # Telemetry
            record = TelemetryRecord(
                capsule_name=leaf.name,
                composition_mode="ITERATION",
                batch_size=len(batch_tags),
                total_tokens=compiled.estimated_tokens,
                coordination_tokens=compiled.coordination_tokens,
                latency_ms=latency_ms,
                context_utilization=self._context_util(compiled.estimated_tokens),
            )
            self._telemetry.record(record)
            self._observe(record)

            logger.debug(
                "Iteration batch %d/%d: k=%d, tokens=%d, overhead=%.1f%%, latency=%.0fms",
                batch_idx + 1, len(batches),
                len(batch_tags),
                compiled.estimated_tokens,
                record.overhead_ratio * 100,
                latency_ms,
            )

        leaf.capsule.state = CapsuleState.COMPLETE

        # Final output = last item of last batch
        final_key = compiled.output_keys[-1] if batches else ""
        final_output = last_batch_outputs.get(final_key, "")

        action = self._controller_action()
        return ExecutionResult(
            outputs=all_outputs,
            final_output=final_output,
            token_usage={"estimated_input": total_tokens},
            telemetry=self._telemetry.records,
            recommended_action=action,
        )


# ---------------------------------------------------------------------------
# Module-level helper: topological ordering of mixed children (T-006)
# ---------------------------------------------------------------------------

def _topo_order_children(
    children: list,
    dependency_edges: dict[str, list[str]],
) -> list:
    """
    Return *children* in topological order respecting *dependency_edges*.

    Works for any mix of AgentLeaf, ToolLeaf, and CompoundCapsule children.
    Falls back to declaration order when no edges are defined for a child.
    Uses Kahn's BFS sort; raises CapsuleExecutionError on a cycle.
    """
    from collections import deque
    name_to_child = {c.name: c for c in children}
    child_names = list(name_to_child)

    in_degree: dict[str, int] = {n: 0 for n in child_names}
    adj: dict[str, list[str]] = {n: [] for n in child_names}

    for node, deps in dependency_edges.items():
        if node not in name_to_child:
            continue
        for dep in deps:
            if dep in name_to_child:
                adj[dep].append(node)
                in_degree[node] += 1

    queue: deque[str] = deque(n for n in child_names if in_degree[n] == 0)
    order: list = []

    while queue:
        name = queue.popleft()
        order.append(name_to_child[name])
        for neighbour in adj.get(name, []):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(children):
        raise CapsuleExecutionError(
            "Cycle detected in mixed-compound children during topological sort."
        )
    return order


# ---------------------------------------------------------------------------
# Module-level helper: collect explicit dep declarations for FINE mode (G-2)
# ---------------------------------------------------------------------------

def _collect_fine_dep_info(
    node,
) -> tuple[dict[str, list[str]], dict[str, OutputKey]]:
    """
    Walk a CompoundCapsule tree and return:
      * declared_deps    — {leaf.name: [dep_name, ...]} for every leaf whose
                           containing CompoundCapsule uses the ``"deps"``
                           injection strategy. Leaves in ``"full"``-strategy
                           compounds are NOT included, so their entries fall
                           through to the legacy "pass everything" path and
                           implicit-linear pipelines (e.g. code_review's
                           three-reviewer group) remain unchanged.
      * leaf_output_keys — {leaf.name: capsule.output_key} for every
                           AgentLeaf, plus the conventional
                           ``f"{name.upper()}_OUTPUT"`` key for ToolLeaf
                           nodes, so the caller can translate dep names into
                           the accumulated_outputs key space.

    Used by ``_run_fine`` (G-2 parity fix). The ``"deps"`` strategy is set
    by ``classify_and_set_strategy`` (runtime/topology.py) when either:

      (a) any agent in the group used an explicit ``depends_on=...``
          argument (including ``depends_on=[]`` for an independent root),
          which is exactly the multi_source_brief arm pattern that G-2
          closes — or
      (b) the topology is non-linear (fan_out, diamond, parallel_converge).

    This mirrors the dep-aware context selection already implemented in
    ``_run_compound_sequential`` at the ``if strategy == "deps"`` branch,
    and preserves the legacy "full" behaviour for implicit-linear groups.
    """
    declared_deps: dict[str, list[str]] = {}
    leaf_output_keys: dict[str, OutputKey] = {}

    def walk(n) -> None:
        if isinstance(n, AgentLeaf):
            leaf_output_keys[n.name] = n.capsule.output_key
            return
        if isinstance(n, ToolLeaf):
            leaf_output_keys[n.name] = f"{n.name.upper()}_OUTPUT"
            return
        if isinstance(n, CompoundCapsule):
            strategy = getattr(n, "sequential_injection_strategy", "full")
            edges = n.dependency_edges or {}
            for child in n.children:
                if strategy == "deps" and child.name in edges:
                    declared_deps[child.name] = list(edges[child.name])
                walk(child)

    walk(node)
    return declared_deps, leaf_output_keys
