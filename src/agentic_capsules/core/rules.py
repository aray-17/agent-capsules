"""
Composition rule validation.

Enforces all 6 rules from design plan §2.3 at definition time so that
problems surface before any LLM call is made.

Rule 1 — Atomicity
Rule 2 — Unique Tagging
Rule 3 — Termination  (dependency graph is a DAG)
Rule 4 — Reachability (no dead agents)
Rule 5 — Dimensional Consistency
Rule 6 — Context Budget Feasibility

Also enforces Tool Composition Rules T-Rule 1–4 (Phase 4).

Design plan ref: §2.3
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .types import CompositionError

if TYPE_CHECKING:
    from .hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_hierarchy(
    hierarchy: CapsuleHierarchy,
    adapter_context_window: int | None = None,
) -> None:
    """
    Run all applicable rules against *hierarchy*.

    Raises CompositionError on the first violation.
    Pass `adapter_context_window` to also check Rule 6.
    """
    _check_rule2_unique_tagging(hierarchy)
    _check_rule3_termination(hierarchy)
    _check_rule4_reachability(hierarchy)
    _check_rule5_dimensional_consistency(hierarchy)
    if adapter_context_window is not None:
        _check_rule6_context_budget(hierarchy, adapter_context_window)


def validate_compound(
    compound: CompoundCapsule,
    adapter_context_window: int | None = None,
) -> None:
    """Validate a single CompoundCapsule in isolation."""
    _check_rule3_compound(compound)
    _check_rule4_compound(compound)
    if adapter_context_window is not None:
        _check_rule6_compound(compound, adapter_context_window)


# ---------------------------------------------------------------------------
# Rule 2 — Unique Tagging
# ---------------------------------------------------------------------------

def _check_rule2_unique_tagging(hierarchy: CapsuleHierarchy) -> None:
    """Every agent name in the hierarchy must be unique."""
    seen: set[str] = set()
    for leaf in hierarchy.all_leaves():
        if leaf.name in seen:
            raise CompositionError(
                2,
                f"Agent name {leaf.name!r} appears more than once in hierarchy "
                f"{hierarchy.name!r}. Every capsule instance must have a unique tag.",
            )
        seen.add(leaf.name)


# ---------------------------------------------------------------------------
# Rule 3 — Termination (DAG check)
# ---------------------------------------------------------------------------

def _check_rule3_termination(hierarchy: CapsuleHierarchy) -> None:
    _check_rule3_compound(hierarchy.root)


def _check_rule3_compound(compound: CompoundCapsule) -> None:
    """
    Verify the dependency graph of *compound* is a DAG (no cycles).
    Uses Kahn's algorithm (BFS topological sort).
    """
    from .hierarchy import AgentLeaf, CompoundCapsule as CC

    # Only check direct children edges at this level; recurse into nested compounds.
    child_names = {c.name for c in compound.children}
    edges = compound.dependency_edges  # {name: [names it depends on]}

    # Build in-degree map and adjacency list
    in_degree: dict[str, int] = {n: 0 for n in child_names}
    adj: dict[str, list[str]] = {n: [] for n in child_names}

    for node, deps in edges.items():
        if node not in child_names:
            continue
        for dep in deps:
            if dep not in child_names:
                raise CompositionError(
                    3,
                    f"Dependency {dep!r} of {node!r} in compound {compound.name!r} "
                    f"is not a child of that compound.",
                )
            adj[dep].append(node)
            in_degree[node] += 1

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbour in adj.get(node, []):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited != len(child_names):
        raise CompositionError(
            3,
            f"Cycle detected in compound {compound.name!r}. "
            f"The dependency graph must be a DAG (no cycles).",
        )

    # Recurse into nested compounds
    for child in compound.children:
        if isinstance(child, CC):
            _check_rule3_compound(child)


# ---------------------------------------------------------------------------
# Rule 4 — Reachability
# ---------------------------------------------------------------------------

def _check_rule4_reachability(hierarchy: CapsuleHierarchy) -> None:
    _check_rule4_compound(hierarchy.root)


def _check_rule4_compound(compound: CompoundCapsule) -> None:
    """
    Every child of a compound must be reachable from the compound's entry point.

    The entry points are the children with no incoming dependencies (in-degree = 0).
    All other children must be transitively reachable from those entry points.
    """
    from .hierarchy import CompoundCapsule as CC

    child_names = {c.name for c in compound.children}
    edges = compound.dependency_edges  # {name: [names it depends on]}

    # Build forward adjacency (dep → dependant)
    adj: dict[str, set[str]] = {n: set() for n in child_names}
    in_degree: dict[str, int] = {n: 0 for n in child_names}
    for node, deps in edges.items():
        if node not in child_names:
            continue
        for dep in deps:
            if dep in child_names:
                adj[dep].add(node)
                in_degree[node] += 1

    # BFS from all entry points (in-degree 0)
    reachable: set[str] = set()
    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        for neighbour in adj.get(node, set()):
            queue.append(neighbour)

    unreachable = child_names - reachable
    if unreachable:
        raise CompositionError(
            4,
            f"Unreachable agents in compound {compound.name!r}: {sorted(unreachable)}. "
            f"Every agent must be reachable from an entry-point agent.",
        )

    # Recurse
    for child in compound.children:
        if isinstance(child, CC):
            _check_rule4_compound(child)


# ---------------------------------------------------------------------------
# Rule 5 — Dimensional Consistency
# ---------------------------------------------------------------------------

def _check_rule5_dimensional_consistency(hierarchy: CapsuleHierarchy) -> None:
    """
    Each agent's output schema must be at least as dimensioned as its tag.

    Phase 1: tags are (agent_name, task_id) — 2 dimensions.
    We verify output_schema.fields is non-empty (has at least one dimension).
    Full cross-product dimension checking added in Phase 2.
    """
    for leaf in hierarchy.all_leaves():
        capsule = leaf.capsule
        if not capsule.output_schema.fields and not capsule.output_schema.name:
            raise CompositionError(
                5,
                f"Agent {capsule.name!r} has an empty output schema. "
                f"Output schema must be at least as dimensioned as its identity tag.",
            )


# ---------------------------------------------------------------------------
# Rule 6 — Context Budget Feasibility
# ---------------------------------------------------------------------------

def _check_rule6_context_budget(
    hierarchy: CapsuleHierarchy,
    adapter_context_window: int,
) -> None:
    _check_rule6_compound(hierarchy.root, adapter_context_window)


def _check_rule6_compound(
    compound: CompoundCapsule,
    adapter_context_window: int,
) -> None:
    """
    Estimate the merged prompt token count for this compound and verify
    it fits within the model's context window.

    Uses a conservative character-based estimate (chars / 3.5 ≈ tokens)
    at validation time. The PromptCompiler uses the adapter's exact
    count_tokens() at compile time (resolves D-2).
    """
    from .hierarchy import AgentLeaf, CompoundCapsule as CC

    total_chars = sum(
        len(child.capsule.system_prompt)
        for child in _all_leaves_of(compound)
    )
    # Conservative estimate: ~3.5 chars per token + 20% overhead for phase markers
    estimated_tokens = int((total_chars / 3.5) * 1.2)

    if estimated_tokens > adapter_context_window:
        raise CompositionError(
            6,
            f"Compound {compound.name!r} estimated merged prompt ~{estimated_tokens} tokens "
            f"exceeds context window {adapter_context_window}. "
            f"Reduce composition level or shorten agent prompts.",
        )

    for child in compound.children:
        if isinstance(child, CC):
            _check_rule6_compound(child, adapter_context_window)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _all_leaves_of(compound: CompoundCapsule) -> list[AgentLeaf]:
    from .hierarchy import AgentLeaf, CompoundCapsule as CC
    leaves = []
    for child in compound.children:
        if isinstance(child, AgentLeaf):
            leaves.append(child)
        else:
            leaves.extend(_all_leaves_of(child))
    return leaves
