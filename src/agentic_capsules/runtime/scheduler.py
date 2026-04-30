"""
Serialization Schedule Generator.

Takes a CompoundCapsule's dependency edges and produces a topological
ordering of its AgentLeaf children. This ordering becomes the execution
sequence inside the compound: agent[0] runs first, its output is passed
to agent[1], and so on.

The schedule is also what the PromptCompiler uses to assign phase numbers
and output headings in the merged prompt (§3.2.3).

Rule 3 (Termination) is enforced by rules.py at definition time; the
Scheduler trusts the graph is a DAG by the time it is called.

Design plan ref: §3.2.1 (serialization order), §3.2.3 (PromptCompiler input)
"""

from __future__ import annotations

from collections import deque

from ..core.hierarchy import AgentLeaf, CompoundCapsule, IterationCapsule, ToolLeaf
from ..core.capsule import AgentTagCapsule
from ..core.types import AgentName, CompositionError


def compute_order(compound: CompoundCapsule) -> list[AgentLeaf | ToolLeaf]:
    """
    Return the AgentLeafs of *compound* in valid execution order.

    Uses Kahn's topological sort (BFS). Raises CompositionError (Rule 3)
    if a cycle is detected — this should not happen after validate_hierarchy()
    but is checked defensively here too.

    Only processes the *direct* AgentLeaf children of *compound*; nested
    CompoundCapsules are treated as atomic units and scheduled as a block.

    The result is also stored in `compound.serialization_order` for caching.
    """
    # Collect AgentLeaf and ToolLeaf direct children (both are schedulable units)
    leaves: list[AgentLeaf | ToolLeaf] = [
        c for c in compound.children if isinstance(c, (AgentLeaf, ToolLeaf))
    ]
    # Use ordered structures so that agents with no edges preserve children order
    leaf_map: dict[AgentName, AgentLeaf] = {leaf.name: leaf for leaf in leaves}
    leaf_names: set[AgentName] = set(leaf_map)

    edges = compound.dependency_edges  # {name: [names it depends on]}

    # Build in-degree and forward adjacency, keyed in children order
    in_degree: dict[AgentName, int] = {leaf.name: 0 for leaf in leaves}
    adj: dict[AgentName, list[AgentName]] = {leaf.name: [] for leaf in leaves}

    for node, deps in edges.items():
        if node not in leaf_names:
            continue
        for dep in deps:
            if dep in leaf_names:
                adj[dep].append(node)
                in_degree[node] += 1

    queue: deque[AgentName] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    order: list[AgentLeaf] = []

    while queue:
        name = queue.popleft()
        order.append(leaf_map[name])
        for neighbour in adj.get(name, []):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(leaves):
        raise CompositionError(
            3,
            f"Cycle detected in compound {compound.name!r} during schedule "
            f"computation. Executed {len(order)} of {len(leaves)} agents.",
        )

    # Cache on the compound so callers don't recompute
    compound.serialization_order = order
    return order


def compute_order_recursive(compound: CompoundCapsule) -> None:
    """
    Recursively compute and cache serialization order for *compound* and
    all nested CompoundCapsules.
    """
    compute_order(compound)
    for child in compound.children:
        if isinstance(child, CompoundCapsule):
            compute_order_recursive(child)


# ---------------------------------------------------------------------------
# Iteration-space schedule (Phase 2)
# ---------------------------------------------------------------------------

def compute_iteration_schedule(
    iteration_capsule: IterationCapsule,
) -> list[tuple[int, AgentTagCapsule]]:
    """
    Return the ordered execution schedule for one IterationCapsule batch.

    Produces a list of (item_index, AgentTagCapsule) pairs in the order
    the PromptCompiler will assign ITEM_N headings. Item indices are
    1-based for human-readable prompt output.

    The PromptCompiler uses this to build:
        == ITEM 1 == ... ITEM_1_OUTPUT
        == ITEM 2 == ... ITEM_2_OUTPUT
        ...

    Design plan ref: §5.2 Phase 2, §3.2.1
    """
    return [
        (i + 1, tag)
        for i, tag in enumerate(iteration_capsule.tags)
    ]
