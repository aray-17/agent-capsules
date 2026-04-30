"""
The three fundamental capsule primitives.

Design plan ref: §2.1 (Capsules analogy table), §3.2.1

Capsules (2007)          Agentic Capsules (Phase 1)
──────────────────────   ──────────────────────────────────────
StepCapsule              AgentStepCapsule
ItemCapsule              AgentItemCapsule
TagCapsule               AgentTagCapsule
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import (
    AgentName,
    CapsuleState,
    CompositionAxis,
    JsonDict,
    OutputKey,
    Schema,
    TagKey,
)


# ---------------------------------------------------------------------------
# AgentTagCapsule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentTagCapsule:
    """
    Unique identity of a capsule instance — the agentic analog of TagCapsule.

    Phase 1: a simple (agent_name, task_id) pair.
    Phase 2 extends this to a full cross-product iteration coordinate.

    Design plan ref: §2.1, §3.2.1 (Tag Dimension Registry)
    """
    agent_name: AgentName
    task_id: str

    @property
    def key(self) -> TagKey:
        return f"{self.agent_name}::{self.task_id}"

    def __str__(self) -> str:
        return self.key


# ---------------------------------------------------------------------------
# AgentItemCapsule
# ---------------------------------------------------------------------------

@dataclass
class AgentItemCapsule:
    """
    A context payload that crosses a capsule boundary — the analog of ItemCapsule.

    Produced by a completed AgentStepCapsule and consumed by its downstream
    neighbour via the BoundarySyncManager (get/put semantics, §3.2.5).

    When agents are composed, intermediate ItemCapsules become local memory
    within the compound capsule's context and never cross a real boundary.

    Design plan ref: §2.1, §3.2.5
    """
    data: Any                          # The payload — string, dict, or structured object
    producer_tag: AgentTagCapsule
    schema: Schema
    output_key: OutputKey = ""         # Phase-marker heading used in compiled prompt

    def __repr__(self) -> str:
        preview = str(self.data)[:80] + "..." if len(str(self.data)) > 80 else str(self.data)
        return f"AgentItemCapsule(key={self.output_key!r}, producer={self.producer_tag})"


# ---------------------------------------------------------------------------
# AgentStepCapsule
# ---------------------------------------------------------------------------

@dataclass
class AgentStepCapsule:
    """
    A single agent reasoning unit — the analog of StepCapsule.

    Holds everything needed to execute one agent: its prompt, its expected
    input/output shape, and its position in the composition hierarchy.

    Composition: two AgentStepCapsules can be merged into a CompoundCapsule
    (hierarchy.py) when their composition axis and dependency edges allow it.

    Design plan ref: §2.1, §3.2.1
    """
    name: AgentName
    system_prompt: str

    # Schemas describe what this capsule consumes and produces.
    # Used by rules.py (Rule 5 — dimensional consistency) and
    # prompt_compiler.py (output heading generation).
    input_schema: Schema
    output_schema: Schema

    # The axis along which this capsule participates in composition.
    # Phase 1 uses COMPUTATION only.
    composition_axis: CompositionAxis = CompositionAxis.COMPUTATION

    # Runtime state — mutable during execution.
    state: CapsuleState = field(default=CapsuleState.PENDING, compare=False)

    # The output_key is the phase-marker heading the PromptCompiler will use
    # when this capsule is part of a compound prompt.
    # Defaults to "<NAME>_OUTPUT" if not set.
    output_key: OutputKey = ""

    # Tool names this agent can invoke during its own reasoning (Phase 10).
    # Names are resolved against ToolRegistry at execution time.
    # Empty list means no tool access (default behaviour, fully backwards compatible).
    tools: list[str] = field(default_factory=list)

    # G-2: optional runtime skip predicate. When set, the executor calls
    # `skip_condition(accumulated_outputs)` before dispatching this agent's
    # LLM call; if it returns False the agent is skipped, its state becomes
    # SKIPPED, a zero-cost SKIPPED telemetry record is emitted, and its
    # output is propagated as "" so downstream agents see an empty value
    # instead of a KeyError. `accumulated_outputs` is keyed by output_key
    # (e.g. {"RESEARCHER_OUTPUT": "..."}), matching the prior_outputs dict
    # passed to PromptCompiler.compile_single. Only consulted in FINE mode
    # per-leaf dispatch and at compound group boundaries (all-agents-skip
    # short-circuit); COMPOUND-internal per-agent skipping is out of scope
    # because a compound call is a single LLM dispatch.
    skip_condition: Callable[[dict[str, str]], bool] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.output_key:
            self.output_key = f"{self.name.upper().replace(' ', '_')}_OUTPUT"

    def __repr__(self) -> str:
        tools_str = f", tools={self.tools!r}" if self.tools else ""
        return f"AgentStepCapsule(name={self.name!r}, state={self.state.name}{tools_str})"
