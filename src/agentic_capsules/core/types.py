"""
Shared type definitions used across all modules.

Design plan ref: §2.1 (capsule analogs), §3.2.1 (capsule definition layer), §5.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CapsuleState(Enum):
    """Lifecycle state of a capsule instance."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETE = auto()
    FAILED = auto()
    # G-2: agent was skipped at runtime because its skip_condition returned
    # False. Distinct from COMPLETE because no LLM call was made and from
    # FAILED because nothing went wrong — the agent's inputs just didn't
    # warrant running it.
    SKIPPED = auto()


class CompositionAxis(Enum):
    """
    The axis along which capsules are composed.

    ITERATION  — same agent × multiple data items (Phase 2)
    COMPUTATION — different agents merged into one compound agent (Phase 3)
    TOOL       — tool chain / batch composition (Phase 4)

    Phase 1 uses COMPUTATION only (sequential agent merging).
    """
    ITERATION = auto()
    COMPUTATION = auto()
    TOOL = auto()


class CompositionLevel(Enum):
    """
    Granularity level descriptor passed to the executor.

    FINE      — every leaf executes as an independent capsule (one call per item)
    COMPOUND  — compound capsules execute as merged units (computation-space)
    ITERATION — same agent batched across K items per call (iteration-space, Phase 2)
    """
    FINE = auto()
    COMPOUND = auto()
    ITERATION = auto()


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    """A single message in an LLM conversation."""
    role: str          # "system" | "user" | "assistant"
    content: str
    # C-1 (Track A): Anthropic prompt caching marker.
    # When set on a system message, the Anthropic adapter emits a multi-block
    # system array and marks this block with the given cache_control dict.
    # Example: {"type": "ephemeral"} — caches the block for 5 minutes (90% discount).
    # Ignored by non-Anthropic adapters.
    cache_control: dict | None = None


@dataclass
class Schema:
    """
    Minimal schema descriptor for an AgentItemCapsule's data.

    For Phase 1 this is just a name and a description; stricter
    JSON Schema validation can be added in later phases.
    """
    name: str
    description: str = ""
    fields: dict[str, str] = field(default_factory=dict)  # field_name -> type hint str


# ---------------------------------------------------------------------------
# LLM Adapter Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMAdapter(Protocol):
    """
    Protocol that all LLM endpoint adapters must satisfy.

    Resolves design decision D-3 (adapter-delegated token counting).
    Design plan ref: §5.1 (Anthropic/OpenAI/generic adapter rows)
    """

    @property
    def context_window(self) -> int:
        """Maximum token capacity of the model."""
        ...

    def complete(self, messages: list[LLMMessage], tools: list | None = None) -> str:
        """
        Send messages to the LLM and return the assistant's reply as a string.

        When *tools* is non-empty (list of ToolDefinition), the adapter passes
        the tool schemas to the model API and runs the multi-turn tool invocation
        loop until the model returns a final text response.

        Synchronous for Phase 1; async variant added in Phase 13.
        Phase 10: tools parameter added (default None — fully backwards compatible).
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Count the tokens in *text* as the model would see them.
        Used by PromptCompiler to enforce Rule 6 (context budget feasibility).
        """
        ...

    @property
    def supports_prompt_caching(self) -> bool:
        """
        Whether this adapter supports C-1 cache-aligned prompt restructuring.

        When False, compile_single() treats cache_aligned_prompts=True as a
        no-op — messages keep their standard layout so non-Anthropic provider
        quality comparisons stay on an equal footing.  Adapters that do NOT
        override this property get False by default.
        """
        return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CompositionError(Exception):
    """
    Raised when a composition violates one of the 6 composition rules.
    Design plan ref: §2.3
    """
    def __init__(self, rule: int, message: str) -> None:
        self.rule = rule
        super().__init__(f"Rule {rule} violation: {message}")


class CapsuleExecutionError(Exception):
    """Raised when a capsule fails during execution."""


class ToolCompositionError(Exception):
    """
    Raised when a ToolCapsule violates a tool composition rule (T-Rule 1–4).
    Design plan ref: §4.3
    """
    def __init__(self, rule: int, message: str) -> None:
        self.rule = rule
        super().__init__(f"T-Rule {rule} violation: {message}")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TagKey = str                    # unique identifier for a capsule instance
OutputKey = str                 # heading name used in phase-marker prompts (e.g. "RESEARCH_OUTPUT")
AgentName = str
JsonDict = dict[str, Any]
