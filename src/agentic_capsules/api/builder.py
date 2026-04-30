"""
Pipeline — the developer-facing entry point for agentic-capsules v2.

The four concepts a developer needs:
  Pipeline  — named container; holds groups + agents; persists controller state
  .group()  — opens a sequential agent group
  .agent()  — adds an agent to the current group
  .run()    — compiles and executes; auto-manages composition

Internal primitives (AgentStepCapsule, CompoundCapsule, CapsuleHierarchy,
compute_order, Schema, TelemetryCollector, ToolRegistry …) are never exposed.
All wiring happens inside _PipelineCompiler (compiler.py), which is imported
lazily inside run() to keep this file import-clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from .tool   import Tool
from .result import PipelineResult
from ..controller.policy import ControllerPolicy, policy_for
from .state  import PipelineState

if TYPE_CHECKING:
    from ..core.types import LLMAdapter
    from ..runtime.sync_manager import SyncBackend
    from ..evaluation.base import QualityEvaluator
    from ..evaluation.calibration import CalibrationReport
    from ..runtime.checkpoint import PipelineCheckpoint


# ---------------------------------------------------------------------------
# Internal spec types (not public)
# ---------------------------------------------------------------------------

@dataclass
class _AgentSpec:
    name:        str
    goal:        str
    tools:       list[Tool]       = field(default_factory=list)
    model:       str | None       = None
    depends_on:  list[str] | None = None  # None = implicit linear (depend on prior agent)
    # G-2: runtime skip predicate. When set, the executor consults it before
    # dispatching the agent's LLM call and skips the call if it returns False.
    # See AgentStepCapsule.skip_condition for semantics. Opt-in per agent;
    # None (default) preserves the historical always-run behaviour.
    condition:   Callable[[dict[str, str]], bool] | None = None


@dataclass
class _GroupSpec:
    name:    str
    agents:  list[_AgentSpec] = field(default_factory=list)
    adapter: "LLMAdapter | None" = None
    # Optional inter-group dependencies (T-054 parallel executor).
    # None = historical implicit linear chain (group N depends on group N-1).
    # Used only by _ParallelPipelineCompiler when Pipeline.run(parallel=True)
    # is invoked; the serial executor ignores this field and continues to run
    # groups in declaration order with cumulative task augmentation.
    depends_on: list[str] | None = None
    # Optional per-group policy override. When None, this group uses the
    # pipeline-level ControllerPolicy. When set, all threshold, gate, and
    # execution-strategy fields come from this instance instead. Use
    # dataclasses.replace(pipeline_policy, field=value) to build a group
    # policy that differs from the pipeline default in only a few fields.
    policy: "ControllerPolicy | None" = None


@dataclass
class _FanoutGroupSpec:
    """
    G-6: a runtime-expanded group.

    Unlike ``_GroupSpec``, which has a fixed set of agents known at build
    time, a ``_FanoutGroupSpec`` spawns N copies of a single worker at
    *runtime* based on the output of a previously-declared *source*
    agent. This is AC's answer to LangGraph's ``Send("worker", state)``
    primitive — the canonical shape for retrieval-augmented,
    per-document-analysis, per-entity-research pipelines.

    Lifecycle:
      1. Builder records this spec with a source agent name, an item
         extractor callable, and a worker template (name + goal).
      2. At compile time, the ``_PipelineCompiler`` skips the normal
         ``_compile_group`` path for fanout specs — there is nothing to
         compile yet because N is unknown.
      3. At execute time, once the source agent's output is available
         in ``all_outputs``, the compiler calls ``item_extractor(source_output)``
         to get a concrete item list, caps at ``max_items``, builds a
         runtime ``_GroupSpec`` with one ``_AgentSpec`` per item
         (worker name = ``f"{worker_name}_{i}"``; worker goal = the
         template with ``{item}`` substituted), compiles that through
         the normal path, and runs it via ``_run_group``.

    Design notes:
      * Workers are all declared ``depends_on=[]`` (independent), so the
        resulting compound is a flat fan-out. In COMPOUND mode the
        executor merges all N workers into one LLM call (ideal
        batching); in FINE mode they run sequentially inside the group.
        Intra-fanout parallelism across *groups* comes for free in
        parallel mode because the fanout group is still one group.
      * ``{item}`` substitution is done via ``str.replace``, not
        ``.format``, so worker goals with curly braces for other
        purposes (JSON examples, template syntax) don't blow up.
      * Empty item list → the fanout group emits no outputs and
        propagates an empty final_output downstream. Safe no-op.
      * ``max_items`` is a hard cap to prevent runaway fanout on a
        buggy extractor. Default 20 matches the soft ceiling used for
        IterationCapsule batching; override explicitly for larger fans.

    Fields:
        name:           group name (unique in the pipeline)
        source:         name of the agent whose output seeds the fanout
        item_extractor: callable ``(producer_output: str) -> list[str]``
        worker_name:    base name for workers (actual name = ``f"{worker_name}_{i}"``)
        worker_goal:    worker system prompt template with ``{item}`` placeholder
        worker_tools:   tools every worker gets
        worker_model:   optional adapter model override per worker
        adapter:        optional group-level adapter override
        depends_on:     inter-group deps (None = implicit linear chain)
        max_items:      safety cap on fan-out width
    """
    name:           str
    source:         str
    item_extractor: Callable[[str], list[str]]
    worker_name:    str
    worker_goal:    str
    worker_tools:   list[Tool]       = field(default_factory=list)
    worker_model:   str | None       = None
    adapter:        "LLMAdapter | None" = None
    depends_on:     list[str] | None = None
    max_items:      int              = 20


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Define and run a multi-agent pipeline.

    Example::

        result = (
            Pipeline("research")
            .group("research")
                .agent("researcher", "Find key facts about {task}.", tools=[search])
                .agent("verifier",   "Cross-check the findings.")
            .group("writing")
                .agent("writer", "Draft a 200-word summary.")
            .run("AI safety challenges", adapter=AnthropicAdapter())
        )
    """

    def __init__(
        self,
        name:        str,
        sensitivity: Literal["conservative", "balanced", "aggressive"] = "balanced",
        store:       SyncBackend | None = None,
        policy:      ControllerPolicy | None = None,
    ) -> None:
        """
        Create a new pipeline.

        Args:
            name:        Unique name for this pipeline. Used as the Redis key prefix
                         when a persistent store is provided.
            sensitivity: Switching profile.  One of ``"conservative"``,
                         ``"balanced"`` (default), or ``"aggressive"``.
                         Ignored when ``policy`` is provided.
            store:       Optional :class:`~agentic_capsules.runtime.sync_manager.SyncBackend`
                         for cross-process persistence (e.g. ``RedisBackend``).
                         Defaults to in-memory state within the ``Pipeline`` instance.
            policy:      Override the full :class:`~agentic_capsules.controller.policy.ControllerPolicy`
                         when you need raw threshold control.  Takes precedence over
                         ``sensitivity``.
        """
        if not name or not name.strip():
            raise ValueError("Pipeline name cannot be empty.")
        self._name:          str                               = name.strip()
        self._policy:        ControllerPolicy                  = policy if policy is not None else policy_for(sensitivity)
        self._store:         SyncBackend | None                = store
        # G-6: groups can be either static (_GroupSpec) or runtime-expanded
        # fan-out groups (_FanoutGroupSpec). Both types are serialised in
        # declaration order; the compiler dispatches per type.
        self._groups:        list["_GroupSpec | _FanoutGroupSpec"] = []
        self._current_group: str | None                        = None
        self._pipeline_state: PipelineState                    = PipelineState(self._name, self._policy, store)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def group(
        self,
        name:       str,
        adapter:    "LLMAdapter | None" = None,
        depends_on: list[str] | None    = None,
        *,
        policy:     "ControllerPolicy | None" = None,
    ) -> "Pipeline":
        """
        Open a new sequential group of agents.

        Calling .group() again auto-closes the current group and opens a new one.
        Groups execute in declaration order under the default serial executor.

        Args:
            name:    Unique name for this group within the pipeline.
            adapter: Optional adapter override for this group. When set, all
                     agents in the group use this adapter instead of the
                     pipeline-level adapter passed to .run(). Useful for
                     routing cheap vs expensive models to different stages
                     (e.g. haiku for research, opus for synthesis).
                     Pareto sweeps and due-diligence evals pass a single
                     pipeline adapter and ignore group overrides — which is
                     the correct behaviour for controller calibration.
            depends_on: Optional inter-group dependencies (T-054 parallel
                     executor). Each name must refer to a group already
                     declared in this pipeline. Pass an empty list
                     (``depends_on=[]``) to declare a root group with no
                     predecessors. Passing ``None`` (the default) preserves
                     the historical implicit linear chain — group N depends
                     on group N-1. This argument is only consulted when the
                     pipeline is executed via ``run(parallel=True)``; the
                     default serial executor ignores it and runs groups in
                     declaration order.
            policy:  Optional per-group :class:`ControllerPolicy` override.
                     When set, this group uses the given policy instead of
                     the pipeline-level policy for all threshold, gate, and
                     execution-strategy decisions (compose_at, quality_floor,
                     compound_execution_model, output_guidance, etc.). Build
                     a group policy by copying the pipeline policy with
                     ``dataclasses.replace(pipeline._policy, field=value)``.
                     Intended use: heterogeneous quality contracts per group
                     within one pipeline (e.g., a premium synthesis group at
                     ``quality_floor=0.85`` alongside a tolerant research
                     group at ``quality_floor=0.65``), or per-group
                     compound-execution tuning on a mixed-topology pipeline.
                     Defaults to None (the group inherits the pipeline policy).
        """
        if not name or not name.strip():
            raise ValueError("Group name cannot be empty.")
        name = name.strip()
        normalized_deps: list[str] | None
        if depends_on is None:
            normalized_deps = None
        else:
            existing = {g.name for g in self._groups}
            normalized_deps = []
            for dep in depends_on:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"Group {name!r}: depends_on entries must be "
                        "non-empty group names."
                    )
                dep_clean = dep.strip()
                if dep_clean == name:
                    raise ValueError(
                        f"Group {name!r} cannot depend on itself."
                    )
                if dep_clean not in existing:
                    raise ValueError(
                        f"Group {name!r}: depends_on references {dep_clean!r}, "
                        f"which is not a group declared earlier in this pipeline. "
                        f"Declared groups: {sorted(existing) or '[]'}."
                    )
                if dep_clean in normalized_deps:
                    continue  # dedupe silently
                normalized_deps.append(dep_clean)
        self._groups.append(
            _GroupSpec(name=name, adapter=adapter,
                       depends_on=normalized_deps, policy=policy)
        )
        if policy is not None:
            self._pipeline_state.register_group_policy(name, policy)
        self._current_group = name
        return self

    def agent(
        self,
        name:       str,
        goal:       str,
        tools:      list[Tool]       = [],
        model:      str | None       = None,
        depends_on: list[str] | None = None,
        condition:  Callable[[dict[str, str]], bool] | None = None,
    ) -> Pipeline:
        """
        Add an agent to the currently open group.

        By default, agents within a group form a linear chain: each agent depends
        on the one declared immediately before it. Pass ``depends_on`` to override
        this and declare an explicit dependency set, enabling fan-out, diamond,
        and parallel-converge topologies within a group.

        Args:
            name:       Identifier for this agent (used in step_outputs keys).
            goal:       Intent / system prompt — what this agent should achieve.
            tools:      Tools the agent may invoke during its own reasoning.
            model:      Override the adapter's default model for this agent only.
            depends_on: Explicit dependency list — names of agents in the same
                        group whose output this agent needs. Each name must refer
                        to an agent already declared in this group. Pass an empty
                        list (``depends_on=[]``) to declare an independent agent
                        that runs in parallel with other independent agents (fan
                        out). Passing ``None`` (the default) preserves the
                        historical linear-by-declaration-order behaviour.
            condition:  G-2 runtime skip predicate. When provided, the executor
                        calls ``condition(accumulated_outputs)`` before
                        dispatching this agent's LLM call. If it returns False,
                        the agent is skipped: no LLM call is made, state becomes
                        SKIPPED, a zero-cost SKIPPED telemetry record is emitted,
                        and the agent's output propagates as ``""`` so downstream
                        dependents see an empty string instead of a missing key.
                        ``accumulated_outputs`` is keyed by output_key (e.g.
                        ``{"RESEARCHER_OUTPUT": "..."}``). Typical use: "if the
                        upstream extractor returned empty, skip the synthesizer."
                        The predicate is consulted in FINE mode per-leaf and at
                        compound group boundaries (if every agent in a compound
                        group is skipped, the whole compound LLM call is
                        short-circuited). Inside a multi-agent compound LLM
                        call, the framework cannot skip individual agents —
                        predicates on such agents log a one-shot warning and
                        execute normally.

        Topology examples:

        - **Linear** (default): omit ``depends_on``.
        - **Fan-out**: every agent has ``depends_on=[]``.
        - **Diamond**: N agents with ``depends_on=[]``, plus one synthesis agent
          with ``depends_on=[<all N>]``.
        """
        if not name or not name.strip():
            raise ValueError("Agent name cannot be empty.")
        if not goal or not goal.strip():
            raise ValueError(f"Agent '{name}': goal cannot be empty.")
        if self._current_group is None:
            raise ValueError(
                f"Agent '{name}' declared before any .group() call. "
                "Call .group(name) before .agent()."
            )
        if condition is not None and not callable(condition):
            raise ValueError(
                f"Agent '{name}': condition must be callable or None, "
                f"got {type(condition).__name__}."
            )
        cleaned_name = name.strip()
        normalized_deps: list[str] | None
        if depends_on is None:
            normalized_deps = None
        else:
            normalized_deps = []
            existing_names = {a.name for a in self._current_group_spec().agents}
            for dep in depends_on:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"Agent '{cleaned_name}': depends_on entries must be "
                        "non-empty agent names."
                    )
                dep_clean = dep.strip()
                if dep_clean == cleaned_name:
                    raise ValueError(
                        f"Agent '{cleaned_name}' cannot depend on itself."
                    )
                if dep_clean not in existing_names:
                    raise ValueError(
                        f"Agent '{cleaned_name}': depends_on references "
                        f"'{dep_clean}', which is not an agent declared earlier "
                        f"in group '{self._current_group}'. Declared agents: "
                        f"{sorted(existing_names) or '[]'}."
                    )
                if dep_clean in normalized_deps:
                    continue  # dedupe silently
                normalized_deps.append(dep_clean)
        self._current_group_spec().agents.append(
            _AgentSpec(
                name=cleaned_name,
                goal=goal.strip(),
                tools=list(tools),
                model=model,
                depends_on=normalized_deps,
                condition=condition,
            )
        )
        return self

    # ------------------------------------------------------------------
    # G-6: dynamic fan-out (runtime-expanded group)
    # ------------------------------------------------------------------

    def fanout_group(
        self,
        name:           str,
        source:         str,
        item_extractor: Callable[[str], list[str]],
        worker_name:    str,
        worker_goal:    str,
        worker_tools:   list[Tool]         = [],
        worker_model:   str | None         = None,
        adapter:        "LLMAdapter | None" = None,
        depends_on:     list[str] | None   = None,
        max_items:      int                = 20,
    ) -> "Pipeline":
        """
        Declare a runtime-expanded fan-out group (G-6 — LangGraph ``Send`` parity).

        After the source agent's output is available, ``item_extractor`` is
        called to produce a list of items, and one copy of the worker agent
        is dispatched per item (with ``{item}`` substituted into
        ``worker_goal``). This is the canonical primitive for
        retrieval-augmented pipelines, per-document analysis, and
        "find entities then research each one" patterns — the number of
        workers is decided at runtime by the source agent's output.

        Declaring a fan-out group auto-closes any currently open ``.group()``
        — just like calling ``.group()`` again. After the fan-out, you can
        open another normal group with ``.group("next")`` to declare a
        synthesizer that consumes all the worker outputs.

        Args:
            name:            Unique group name.
            source:          Name of an agent declared earlier in the
                             pipeline whose output seeds the fan-out.
            item_extractor:  Callable mapping the source agent's raw
                             output string to a list of item strings.
                             Each item becomes one worker dispatch.
                             Return ``[]`` to skip the fan-out entirely.
            worker_name:     Base name for workers. Actual worker names
                             are ``f"{worker_name}_{i}"`` (i = 0..N-1)
                             so they stay unique within the compound.
            worker_goal:     System prompt template for workers. Any
                             occurrence of the literal string ``{item}``
                             is replaced per-worker with the extracted
                             item. Other curly braces are preserved
                             verbatim (replace, not .format).
            worker_tools:    Tools every worker gets.
            worker_model:    Optional adapter model override per worker.
            adapter:         Optional group-level adapter override.
            depends_on:      Inter-group dependencies. None = historical
                             implicit linear chain. In the parallel
                             executor, this controls when the fan-out
                             group becomes ready to dispatch; the source
                             agent's group must be a (transitive)
                             dependency, otherwise the source output
                             won't be available when expansion runs.
            max_items:       Hard cap on fan-out width (default 20).
                             Prevents runaway dispatch on a buggy
                             extractor. The extracted list is truncated
                             to the first ``max_items`` entries.

        Example::

            result = (
                Pipeline("rag")
                .group("retrieve")
                    .agent("retriever", "Find docs about {task}", tools=[search])
                .fanout_group(
                    "analyze",
                    source="retriever",
                    item_extractor=lambda out: [
                        line.strip() for line in out.split("\\n") if line.strip()
                    ],
                    worker_name="doc_analyzer",
                    worker_goal="Summarize this document: {item}",
                )
                .group("synthesize")
                    .agent("writer", "Combine the per-document summaries.")
                .run("AI safety", adapter=AnthropicAdapter())
            )
        """
        if not name or not name.strip():
            raise ValueError("fanout_group: name cannot be empty.")
        if not source or not source.strip():
            raise ValueError(
                f"fanout_group '{name}': source must be a declared agent name."
            )
        if not callable(item_extractor):
            raise ValueError(
                f"fanout_group '{name}': item_extractor must be callable, "
                f"got {type(item_extractor).__name__}."
            )
        if not worker_name or not worker_name.strip():
            raise ValueError(
                f"fanout_group '{name}': worker_name cannot be empty."
            )
        if not worker_goal or not worker_goal.strip():
            raise ValueError(
                f"fanout_group '{name}': worker_goal cannot be empty."
            )
        if max_items < 1:
            raise ValueError(
                f"fanout_group '{name}': max_items must be >= 1, got {max_items}."
            )
        name_clean   = name.strip()
        source_clean = source.strip()
        # Source must already be declared as an agent somewhere in the pipeline.
        all_agent_names: set[str] = {
            a.name
            for g in self._groups
            if isinstance(g, _GroupSpec)
            for a in g.agents
        }
        if source_clean not in all_agent_names:
            raise ValueError(
                f"fanout_group '{name_clean}': source agent "
                f"{source_clean!r} has not been declared in any prior group. "
                f"Declared agents: {sorted(all_agent_names) or '[]'}."
            )
        if "{item}" not in worker_goal:
            # Soft guard — worker goal without {item} would just dispatch N
            # identical copies, which is almost certainly a user bug. Reject
            # with a clear message so it surfaces at build time.
            raise ValueError(
                f"fanout_group '{name_clean}': worker_goal must contain the "
                "literal substring '{item}' so each dispatched worker gets "
                "its own extracted item. Got: "
                f"{worker_goal[:80]!r}"
            )
        # depends_on: reuse the same validation the normal .group() applies,
        # but keep it here inline because .group() already cleaned and stored
        # the list; we need to validate fresh input.
        normalized_deps: list[str] | None
        if depends_on is None:
            normalized_deps = None
        else:
            existing_group_names = {g.name for g in self._groups}
            normalized_deps = []
            for dep in depends_on:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"fanout_group '{name_clean}': depends_on entries "
                        "must be non-empty group names."
                    )
                dep_clean = dep.strip()
                if dep_clean == name_clean:
                    raise ValueError(
                        f"fanout_group '{name_clean}' cannot depend on itself."
                    )
                if dep_clean not in existing_group_names:
                    raise ValueError(
                        f"fanout_group '{name_clean}': depends_on references "
                        f"{dep_clean!r}, which is not a group declared earlier "
                        f"in this pipeline. Declared groups: "
                        f"{sorted(existing_group_names) or '[]'}."
                    )
                if dep_clean in normalized_deps:
                    continue
                normalized_deps.append(dep_clean)

        self._groups.append(
            _FanoutGroupSpec(
                name=name_clean,
                source=source_clean,
                item_extractor=item_extractor,
                worker_name=worker_name.strip(),
                worker_goal=worker_goal,
                worker_tools=list(worker_tools),
                worker_model=worker_model,
                adapter=adapter,
                depends_on=normalized_deps,
                max_items=max_items,
            )
        )
        # Fan-out groups occupy the "current group" slot so that a later
        # .agent() call surfaces the fan-out-specific error message via
        # ``_current_group_spec``. The next valid builder call is
        # ``.group()`` or ``.fanout_group()``.
        self._current_group = name_clean
        return self

    # ------------------------------------------------------------------
    # G-5: subpipeline composition (public API for nested pipelines)
    # ------------------------------------------------------------------

    def subpipeline(
        self,
        other:       "Pipeline",
        name_prefix: str | None        = None,
        depends_on:  list[str] | None  = None,
    ) -> "Pipeline":
        """
        Embed another :class:`Pipeline`'s groups into this one as a
        reusable unit (G-5 — LangGraph ``add_node(graph)`` parity).

        Every group from ``other`` is appended to this pipeline in
        declaration order with namespaced names so they never collide
        with existing parent groups. The subpipeline's internal
        dependency structure is preserved (``depends_on`` entries are
        rewritten to point at the prefixed names). Both static
        ``.group()`` and runtime ``.fanout_group()`` subpipeline groups
        are supported.

        This is the canonical primitive for hierarchical composition:
        build a small, verified sub-pipeline once, then embed it
        wherever that capability is needed without re-declaring its
        agents. Each embedded group goes through the normal
        ``_compile_group`` path so FINE/COMPOUND duality, the
        composition controller, quality grounding, parallel execution,
        G-4 checkpointing, and G-6 dynamic fan-out all apply to
        embedded groups identically to top-level groups.

        Scope / namespacing:
            Group names become ``f"{name_prefix}/{group.name}"``. Agent
            names become ``f"{name_prefix}/{agent.name}"``. Fan-out
            ``source`` and ``worker_name`` fields are prefixed the same
            way. Cross-group ``depends_on`` entries inside the
            subpipeline are rewritten to the prefixed names. Policy,
            sensitivity, and pipeline state are inherited from the
            **parent** — the subpipeline's own ``_policy`` /
            ``_pipeline_state`` are discarded at embed time (they only
            have meaning when the subpipeline runs standalone).

        Hooking into the parent DAG:
            The ``depends_on`` argument specifies parent-side group
            dependencies for the subpipeline's *entry points*. An
            entry point is any group in ``other`` declared as a root
            (``depends_on=[]``) — these are the groups that have no
            upstream work inside the subpipeline. All roots inherit
            the parent's ``depends_on`` list verbatim. Groups with
            ``depends_on=None`` (implicit linear chain) keep the
            implicit-linear semantics inside the flattened parent:
            the first such entry hooks onto whatever group preceded
            the subpipeline in the parent's declaration order, exactly
            as if its agents had been declared directly.

        Args:
            other:       Another ``Pipeline`` instance. Its groups are
                         **copied** into this pipeline (no shared
                         mutable state); modifying ``other`` after the
                         embed call does not affect the parent.
            name_prefix: Namespace for the embedded groups and agents.
                         Defaults to ``other._name``. Must be a
                         non-empty string and must not collide with
                         any previously prefixed subpipeline in this
                         parent.
            depends_on:  Parent-side group dependencies for the
                         subpipeline's root groups. Same semantics as
                         ``.group(depends_on=...)`` — only consulted
                         in parallel-executor mode; serial mode runs
                         flattened groups in declaration order.

        Example::

            research = (
                Pipeline("research")
                .group("retrieve")
                    .agent("retriever", "Find papers about {task}", tools=[search])
                .group("extract")
                    .agent("extractor", "Pull claims from the papers.")
            )

            # Embed it inside a larger pipeline:
            full = (
                Pipeline("brief")
                .subpipeline(research)          # adds research/retrieve, research/extract
                .group("write")
                    .agent("writer", "Draft a brief.")
            )
            full.run("AI safety", adapter=AnthropicAdapter())
        """
        if not isinstance(other, Pipeline):
            raise ValueError(
                f"subpipeline: `other` must be a Pipeline instance, "
                f"got {type(other).__name__}."
            )
        if other is self:
            raise ValueError(
                "subpipeline: cannot embed a pipeline into itself."
            )
        if not other._groups:
            raise ValueError(
                f"subpipeline: pipeline {other._name!r} has no groups defined."
            )
        prefix = (name_prefix or other._name).strip()
        if not prefix:
            raise ValueError("subpipeline: name_prefix cannot be empty.")
        if "/" in prefix:
            raise ValueError(
                f"subpipeline: name_prefix {prefix!r} cannot contain '/' "
                "(reserved as the namespace separator)."
            )

        # Reject collisions: every prefixed group/agent name we're about
        # to emit must be fresh in the parent. We keep parent-side names
        # in a separate set from the running subpipeline-side set so
        # fan-out source validation can distinguish "agent exists in the
        # subpipeline" from "agent exists anywhere in the parent".
        parent_group_names: set[str] = {g.name for g in self._groups}
        parent_agent_names: set[str] = {
            a.name
            for g in self._groups
            if isinstance(g, _GroupSpec)
            for a in g.agents
        }
        # Subpipeline agents that have been added so far, indexed by
        # their already-prefixed name. Used for fan-out source checks.
        subpipeline_agent_names: set[str] = set()

        def _prefixed(name: str) -> str:
            return f"{prefix}/{name}"

        # Validate parent-side depends_on once, up front — reuse the same
        # semantics as ``.group(depends_on=...)``.
        if depends_on is None:
            parent_side_deps: list[str] | None = None
        else:
            parent_side_deps = []
            for dep in depends_on:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"subpipeline '{prefix}': depends_on entries must be "
                        "non-empty group names."
                    )
                dep_clean = dep.strip()
                if dep_clean not in parent_group_names:
                    raise ValueError(
                        f"subpipeline '{prefix}': depends_on references "
                        f"{dep_clean!r}, which is not a group declared earlier "
                        f"in this pipeline. Declared groups: "
                        f"{sorted(parent_group_names) or '[]'}."
                    )
                if dep_clean in parent_side_deps:
                    continue
                parent_side_deps.append(dep_clean)

        # ------------------------------------------------------------------
        # Clone each subpipeline group with prefix rewrites.
        # ------------------------------------------------------------------
        for g in other._groups:
            new_group_name = _prefixed(g.name)
            if new_group_name in parent_group_names:
                raise ValueError(
                    f"subpipeline '{prefix}': prefixed group name "
                    f"{new_group_name!r} collides with an existing group "
                    f"in this pipeline. Choose a different name_prefix."
                )
            parent_group_names.add(new_group_name)

            # Rewrite the group's inter-group deps. Roots (``depends_on=[]``)
            # inherit the parent-side deps; internal edges get prefixed;
            # implicit linear (``None``) stays None and the flattened
            # declaration order handles it correctly.
            new_group_deps: list[str] | None
            if g.depends_on is None:
                new_group_deps = None
            elif len(g.depends_on) == 0:
                new_group_deps = list(parent_side_deps) if parent_side_deps is not None else []
            else:
                new_group_deps = [_prefixed(d) for d in g.depends_on]

            if isinstance(g, _FanoutGroupSpec):
                new_source = _prefixed(g.source)
                # Source must exist in what we've already embedded from
                # THIS subpipeline — subpipelines are self-contained; a
                # fan-out group cannot pull a source agent out of the
                # parent pipeline (agent outputs are produced per-run and
                # live in the executor's accumulated_outputs dict, so
                # there's no cross-subpipeline visibility at build time).
                if new_source not in subpipeline_agent_names:
                    raise ValueError(
                        f"subpipeline '{prefix}': fan-out group "
                        f"{new_group_name!r} references source agent "
                        f"{g.source!r} (prefixed to {new_source!r}) "
                        "which was not declared earlier in the "
                        "subpipeline. Subpipelines must be "
                        "self-contained — the fan-out source must be an "
                        "agent of the embedded pipeline itself, not of "
                        "the parent."
                    )
                self._groups.append(
                    _FanoutGroupSpec(
                        name=new_group_name,
                        source=new_source,
                        item_extractor=g.item_extractor,
                        worker_name=_prefixed(g.worker_name),
                        worker_goal=g.worker_goal,
                        worker_tools=list(g.worker_tools),
                        worker_model=g.worker_model,
                        adapter=g.adapter,
                        depends_on=new_group_deps,
                        max_items=g.max_items,
                    )
                )
                continue

            # Static _GroupSpec: prefix every agent and rewrite intra-group
            # depends_on to the new agent names.
            new_agents: list[_AgentSpec] = []
            for a in g.agents:
                new_agent_name = _prefixed(a.name)
                if new_agent_name in parent_agent_names:
                    raise ValueError(
                        f"subpipeline '{prefix}': prefixed agent name "
                        f"{new_agent_name!r} collides with an existing "
                        "agent in this pipeline. Choose a different "
                        "name_prefix."
                    )
                if new_agent_name in subpipeline_agent_names:
                    raise ValueError(
                        f"subpipeline '{prefix}': duplicate agent name "
                        f"{a.name!r} inside the embedded pipeline "
                        f"{other._name!r} (two agents share this name "
                        "across subpipeline groups — prefixing cannot "
                        "disambiguate them)."
                    )
                subpipeline_agent_names.add(new_agent_name)
                parent_agent_names.add(new_agent_name)
                new_agents.append(
                    _AgentSpec(
                        name=new_agent_name,
                        goal=a.goal,
                        tools=list(a.tools),
                        model=a.model,
                        depends_on=(
                            [_prefixed(d) for d in a.depends_on]
                            if a.depends_on is not None
                            else None
                        ),
                        condition=a.condition,
                    )
                )
            self._groups.append(
                _GroupSpec(
                    name=new_group_name,
                    agents=new_agents,
                    adapter=g.adapter,
                    depends_on=new_group_deps,
                )
            )

        # After embedding, require the user to explicitly open a new group
        # (or another subpipeline / fan-out) — further ``.agent()`` calls
        # would be confusing because "the current group" is now some
        # arbitrarily-named internal group of the embedded subpipeline.
        self._current_group = None
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        task:            str,
        adapter:         "LLMAdapter",
        mode:            Literal["auto", "observe", "fine", "compound"] = "auto",
        task_id:         str | None = None,
        evaluator:       "QualityEvaluator | None" = None,
        parallel:        bool = False,
        checkpoint:      "PipelineCheckpoint | None" = None,
        checkpoint_path: str | Path | None = None,
    ) -> PipelineResult:
        """
        Compile the pipeline definition and execute it.

        Args:
            task:      Input text passed to the first agent.
            adapter:   LLM adapter (AnthropicAdapter, OpenAIAdapter, ScriptedAdapter…).
            mode:      Execution mode.
                       "auto"     — controller auto-manages composition per group.
                       "observe"  — shadow mode: observes but never switches.
                       "fine"     — locked FINE for all groups.
                       "compound" — locked COMPOUND for all groups.
            task_id:   Stable identifier for this run (auto-generated if None).
            evaluator: Phase 12 — optional quality evaluator.  When set and
                       ``policy.quality_floor`` is configured, the controller will:
                         1. Run a shadow COMPOUND comparison on FINE→COMPOUND switch
                            and block the switch if quality < quality_floor.
                         2. Record rolling quality scores in COMPOUND mode and revert
                            to FINE if the rolling mean drops below quality_floor.
                       No-op when None (default — no extra LLM calls).
            parallel:  Opt-in threaded executor (T-054). When True, dispatches
                       to ``_ParallelPipelineCompiler``, which runs independent
                       groups concurrently via ``ThreadPoolExecutor``. Default
                       False preserves the historical serial execution path
                       and behaviour. As of G-7 (2026-04-09) the parallel
                       executor supports ``mode="auto"`` and quality-gated
                       ``evaluator`` arguments — a ``threading.RLock`` in
                       ``PipelineState`` makes the controller and H2/H3
                       quality gates safe under concurrent worker threads.
                       ``mode="observe"`` is still unsupported in this path.
            checkpoint:       G-4 group-level checkpoint/restore. Pass an
                              existing ``PipelineCheckpoint`` to share state
                              across runs (e.g. retry-on-failure in the same
                              process). Mutually exclusive with
                              ``checkpoint_path``; pass at most one.
            checkpoint_path:  G-4 convenience form of ``checkpoint``. Pass a
                              directory and the pipeline creates an ephemeral
                              ``PipelineCheckpoint(path=...)`` for this run.
                              Ideal for crash-recovery across process restarts
                              — the same ``task_id`` will skip every group
                              already saved on disk. Mutually exclusive with
                              ``checkpoint``.

                              Semantics: after each group completes, the
                              pipeline saves its ``{outputs, final_output}``
                              under ``(task_id, group_name)``. On a rerun
                              with the same ``task_id``, groups that already
                              have a saved record are restored from disk and
                              their LLM calls are not re-dispatched. Resumed
                              groups contribute *no* telemetry and do *not*
                              update the composition controller for this
                              run. On successful completion, the checkpoint
                              for this ``task_id`` is cleared automatically.
                              Supported in both serial and parallel mode.
        """
        self._validate_runnable()
        # G-4: resolve the checkpoint handle.
        if checkpoint is not None and checkpoint_path is not None:
            raise ValueError(
                "Pass at most one of `checkpoint` or `checkpoint_path`; "
                "they are mutually exclusive."
            )
        resolved_checkpoint = checkpoint
        if checkpoint_path is not None:
            from ..runtime.checkpoint import PipelineCheckpoint  # lazy
            resolved_checkpoint = PipelineCheckpoint(path=checkpoint_path)
        if parallel:
            # Validation of mode/evaluator happens inside the parallel
            # compiler — keep the rejection messages in one place.
            from .parallel_compiler import _ParallelPipelineCompiler  # lazy
            return _ParallelPipelineCompiler(
                self, task, adapter, mode, task_id, evaluator,
                checkpoint=resolved_checkpoint,
            ).execute()
        if mode not in ("auto", "observe", "fine", "compound"):
            raise ValueError(
                f"mode must be one of 'auto', 'observe', 'fine', 'compound', got {mode!r}."
            )
        from .compiler import _PipelineCompiler   # lazy — keeps this file import-clean
        return _PipelineCompiler(
            self, task, adapter, mode, task_id, evaluator,
            checkpoint=resolved_checkpoint,
        ).execute()

    def calibrate(
        self,
        sample_tasks:  list[str],
        adapter:       "LLMAdapter",
        evaluator:     "QualityEvaluator",
        n_paired_runs: int = 3,
    ) -> "CalibrationReport":
        """
        Pre-deployment quality and efficiency calibration (T-034).

        Runs paired FINE and COMPOUND executions for each task and group,
        compares quality via ``evaluator``, and returns a CalibrationReport.

        This method **never** writes to GroupControllerState — it is a read-only
        dry run that does not affect the live controller.  Call it before enabling
        ``mode="auto"`` in production to verify that COMPOUND is safe.

        Args:
            sample_tasks:  Representative task inputs.  Use 3–10 tasks that
                           reflect real production inputs.
            adapter:       LLM adapter to use for both FINE and COMPOUND runs.
            evaluator:     Quality evaluator (e.g. LLMJudgeEvaluator or
                           SchemaComplianceEvaluator) to compare outputs.
            n_paired_runs: Number of times to repeat each task (for averaging
                           out sampling variance).  Default 3.

        Returns:
            CalibrationReport with quality_by_group(), latency_by_group(),
            token_reduction_by_group(), and recommend_compose_at().
        """
        self._validate_runnable()
        from .compiler import _PipelineCompiler   # lazy import
        from ..evaluation.calibration import CalibrationReport, _GroupCalibration
        import time

        report = CalibrationReport(
            pipeline_name=self._name,
            quality_floor=self._policy.quality_floor,
            _groups={g.name: _GroupCalibration(group=g.name) for g in self._groups},
        )

        for task in sample_tasks:
            for _ in range(n_paired_runs):
                # FINE run
                fine_result  = _PipelineCompiler(self, task, adapter, "fine", None).execute()
                # COMPOUND run
                comp_result  = _PipelineCompiler(self, task, adapter, "compound", None).execute()

                for g in self._groups:
                    gc = report._groups[g.name]

                    fine_out = fine_result.step_outputs.get(
                        g.agents[-1].name, fine_result.output
                    )
                    comp_out = comp_result.step_outputs.get(
                        g.agents[-1].name, comp_result.output
                    )

                    quality = evaluator.evaluate(task, fine_out, comp_out)
                    gc.quality_scores.append(quality.score)

                    if fine_result.latency_ms is not None:
                        gc.latency_fine_ms.append(fine_result.latency_ms)
                    if comp_result.latency_ms is not None:
                        gc.latency_compound_ms.append(comp_result.latency_ms)

                    fine_toks = fine_result.token_usage  // len(self._groups)
                    comp_toks = comp_result.token_usage  // len(self._groups)
                    gc.tokens_fine.append(fine_toks)
                    gc.tokens_compound.append(comp_toks)

                    # Read observation-based signals accumulated by the runtime
                    # during the FINE pass so the report can recommend per-group
                    # thresholds. mean_avg_output_tokens_fine() returns None
                    # until min_observations is reached — calibrate() defaults
                    # to 3 paired runs × len(sample_tasks), so this typically
                    # populates on the 3rd call or later.
                    gs = self._pipeline_state._load(g.name)
                    if gs.observations:
                        gc.composition_scores.append(gs.observations[-1])
                    if gs.avg_output_tokens_fine:
                        gc.avg_output_tokens_fine.append(
                            gs.avg_output_tokens_fine[-1]
                        )

        return report

    # ------------------------------------------------------------------
    # Policy resolution
    # ------------------------------------------------------------------

    def effective_policy(self, group_name: str) -> ControllerPolicy:
        """
        Return the :class:`ControllerPolicy` that applies to ``group_name``.

        If the named group was declared with a ``policy=`` override, that
        policy is returned; otherwise the pipeline-level policy is returned.
        Unknown group names fall back to the pipeline policy.

        This is the single source of truth for "what policy governs this
        group?" — compiler, executor, and controller all resolve through
        this method so per-group overrides take effect uniformly.
        """
        for spec in self._groups:
            if spec.name == group_name and getattr(spec, "policy", None) is not None:
                return spec.policy
        return self._policy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_group_spec(self) -> _GroupSpec:
        # ``_AgentSpec``s can only be appended to static groups, never to
        # fan-out groups (which expand at runtime), so this is typed to the
        # static spec. Callers that might be on a fan-out group must guard
        # for that themselves — in practice only ``.agent()`` calls this.
        current = self._groups[-1]
        if isinstance(current, _FanoutGroupSpec):
            raise ValueError(
                f"Cannot call .agent() after .fanout_group('{current.name}'); "
                "a fan-out group spawns its workers at runtime. Open a new "
                "static group with .group(...) before adding more agents."
            )
        return current

    def _validate_runnable(self) -> None:
        if not self._groups:
            raise ValueError(
                f"Pipeline '{self._name}' has no groups defined. "
                "Call .group() and .agent() before .run()."
            )
        for g in self._groups:
            if isinstance(g, _FanoutGroupSpec):
                # Fan-out groups have no compile-time agents; runtime
                # expansion owns that validation. A fan-out group with a
                # broken source agent or empty extractor will surface its
                # error at execute time.
                continue
            if not g.agents:
                raise ValueError(
                    f"Group '{g.name}' in pipeline '{self._name}' has no agents. "
                    "Call .agent() after .group()."
                )
