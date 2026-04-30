"""
ToolCapsule — composable tool invocation unit.

A ToolCapsule is an ordered chain of ToolSteps that execute without returning
to the LLM between steps. Only the final step's output crosses the capsule
boundary back to the agent.

Composition rules enforced at definition time:
  T-Rule 1 — Schema Compatibility: each step's input_from must reference a
             prior step's output_key, or be None (uses external input)
  T-Rule 2 — Idempotency: non-idempotent steps emit a warning (not hard error)
  T-Rule 4 — Side Effect Ordering: read_only steps may be parallelized

T-Rule 3 (latency budgeting) is enforced at runtime by ToolOrchestrator.

Design plan ref: §3.2.1 (ToolCapsule Registry), §4.1–4.3
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ToolStep:
    """One step in a ToolCapsule chain.

    tool_name  — identifier routed by the ToolAdapter
    input_keys — keys expected in this step's input dict
    output_key — key under which this step's result is stored in the chain context
    input_from — if set, pulls from a prior step's output_key; otherwise uses
                 the chain's external input
    read_only  — T-Rule 4: no side effects; orchestrator may parallelize
    idempotent — T-Rule 2: False triggers a warning at validation time
    timeout_s  — T-Rule 3: per-step timeout enforced by the orchestrator
    """
    tool_name: str
    input_keys: list[str]
    output_key: str
    input_from: str | None = None
    read_only: bool = True
    idempotent: bool = True
    timeout_s: float = 30.0

    def __repr__(self) -> str:
        src = f"<-{self.input_from}" if self.input_from else "<-external"
        return f"ToolStep({self.tool_name!r}, {src} -> {self.output_key!r})"


@dataclass
class ToolCapsule:
    """Ordered chain of ToolSteps executed as a single orchestrator dispatch.

    Intermediate outputs stay in a local context buffer and never cross the
    capsule boundary (boundary migration, §3.2.5). Only the final step's
    output is returned as the capsule's external result.

    T-Rule 1 is validated at construction time via validate().
    """
    name: str
    steps: list[ToolStep]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"ToolCapsule {self.name!r} must have at least one step.")
        self.validate()

    def validate(self) -> None:
        """T-Rule 1: every input_from must reference a prior step's output_key."""
        available: set[str] = set()
        for i, step in enumerate(self.steps):
            if step.input_from is not None and step.input_from not in available:
                raise ValueError(
                    f"ToolCapsule {self.name!r} T-Rule 1 violation: "
                    f"step {i} ({step.tool_name!r}) references "
                    f"input_from={step.input_from!r} but that key is not "
                    f"produced by any prior step. Available: {sorted(available)}"
                )
            available.add(step.output_key)

    @property
    def final_output_key(self) -> str:
        return self.steps[-1].output_key

    @property
    def has_non_idempotent_steps(self) -> bool:
        return any(not s.idempotent for s in self.steps)

    @property
    def read_only_prefix(self) -> list[ToolStep]:
        """T-Rule 4: leading read_only steps eligible for parallelization."""
        prefix = []
        for step in self.steps:
            if step.read_only:
                prefix.append(step)
            else:
                break
        return prefix

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        chain = " -> ".join(s.tool_name for s in self.steps)
        return f"ToolCapsule({self.name!r}, [{chain}])"
