"""
TagCapsule tree — iteration space definition and enumeration.

Defines the space over which agents are batched in iteration-space
composition (Phase 2). A TagSpace with one dimension maps to a flat
list of items. Multiple dimensions produce a cross-product.

Example (§3.2.1 Tag Dimension Registry):
    space = TagSpace(
        agent_name="analyst",
        dimensions=[
            TagDimension("document_id", list(range(1, 101))),
            TagDimension("analysis_type", ["sentiment", "summary", "entities"]),
        ]
    )
    tags = space.enumerate()   # 300 AgentTagCapsules
    batches = space.partition(k=10)  # 30 batches of 10

Design plan ref: §2.1 (TagCapsule), §3.2.1 (Tag Dimension Registry),
                 §5.2 Phase 2
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .capsule import AgentTagCapsule
from .types import AgentName


# ---------------------------------------------------------------------------
# TagDimension
# ---------------------------------------------------------------------------

@dataclass
class TagDimension:
    """
    One axis of the iteration space.

    name   — dimension label (e.g. "document_id", "analysis_type")
    values — ordered sequence of values for this axis

    Design plan ref: §3.2.1 (Tag Dimension Registry)
    """
    name: str
    values: list[Any]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"TagDimension {self.name!r} must have at least one value.")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"TagDimension({self.name!r}, n={len(self.values)})"


# ---------------------------------------------------------------------------
# TagSpace
# ---------------------------------------------------------------------------

@dataclass
class TagSpace:
    """
    Defines the full iteration space for one agent across one or more dimensions.

    Supports:
      enumerate()   — yields all AgentTagCapsules in cross-product order
      partition(k)  — splits into batches of size k for iteration composition
      size          — total number of items (product of all dimension sizes)

    Design plan ref: §3.2.1, §5.2 Phase 2
    """
    agent_name: AgentName
    dimensions: list[TagDimension]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("TagSpace must have at least one dimension.")

    @property
    def size(self) -> int:
        """Total number of items in the iteration space."""
        result = 1
        for dim in self.dimensions:
            result *= len(dim)
        return result

    def enumerate(self) -> list[AgentTagCapsule]:
        """
        Return all AgentTagCapsules in this space, in cross-product order.

        The task_id for each tag is built from its dimension coordinates:
        e.g. "document_id=3__analysis_type=sentiment"
        """
        dim_value_lists = [dim.values for dim in self.dimensions]
        dim_names = [dim.name for dim in self.dimensions]

        tags: list[AgentTagCapsule] = []
        for combo in itertools.product(*dim_value_lists):
            task_id = "__".join(
                f"{name}={value}" for name, value in zip(dim_names, combo)
            )
            tags.append(AgentTagCapsule(agent_name=self.agent_name, task_id=task_id))
        return tags

    def partition(self, k: int) -> list[list[AgentTagCapsule]]:
        """
        Split the enumerated tag space into batches of at most *k* tags each.

        Each batch will be compiled into one IterationCapsule and dispatched
        as a single LLM call by the executor.

        k=1  → one capsule per item (fine-grained baseline)
        k=N  → one capsule for the entire space (maximally coarse)
        """
        if k < 1:
            raise ValueError(f"Batch size k must be >= 1, got {k}.")
        all_tags = self.enumerate()
        return [all_tags[i:i + k] for i in range(0, len(all_tags), k)]

    def __repr__(self) -> str:
        dims = ", ".join(repr(d) for d in self.dimensions)
        return f"TagSpace(agent={self.agent_name!r}, size={self.size}, dims=[{dims}])"
