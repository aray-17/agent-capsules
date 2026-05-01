"""
CapsuleHierarchy — the composition tree.

Defines what compositions are statically valid. Built at design time from
the application graph. The Granularity Controller uses get_level() at
runtime to decide which tree depth to execute at.

Design plan ref: §3.1 (Capsule Definition Layer), §3.2.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .capsule import AgentStepCapsule, AgentTagCapsule
from .types import AgentName, CompositionAxis


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------

@dataclass
class AgentLeaf:
    """
    A leaf node in the hierarchy — wraps one AgentStepCapsule.

    `dependencies` lists the names of other AgentLeafs (in the same
    CompoundCapsule) that must complete before this leaf can run.

    Design plan ref: §3.2.1 (AgentLeaf pseudocode)
    """
    capsule: AgentStepCapsule
    dependencies: list[AgentName] = field(default_factory=list)

    @property
    def name(self) -> AgentName:
        return self.capsule.name

    def __repr__(self) -> str:
        return f"AgentLeaf({self.name!r})"


@dataclass
class CompoundCapsule:
    """
    An internal node — a valid merge of two or more children.

    `children` may be AgentLeafs or nested CompoundCapsules, allowing
    arbitrary composition depth.

    `serialization_order` is computed by the Scheduler from `dependency_edges`
    and cached here after the hierarchy is validated. The PromptCompiler reads
    it to build the phase-marker prompt.

    Design plan ref: §3.2.1 (CompoundCapsule pseudocode), §3.2.3
    """
    name: str
    children: list[AgentLeaf | ToolLeaf | CompoundCapsule]
    composition_axis: CompositionAxis = CompositionAxis.COMPUTATION

    # Edges among direct children: {child_name: [names it depends on]}
    # Derived from the data-flow definition by the caller.
    dependency_edges: dict[AgentName, list[AgentName]] = field(default_factory=dict)

    # Filled in by Scheduler.compute_order() after rule validation.
    serialization_order: list[AgentLeaf | ToolLeaf] = field(default_factory=list, compare=False)

    # Filled in by topology.classify_and_set_strategy() after compute_order().
    # Controls which context injection strategy the sequential executor uses:
    #   "full"    — inject all prior outputs (current default behaviour)
    #   "deps"    — inject only declared-dependency outputs (non-linear topologies)
    #   "summary" — inject summarised prior outputs (linear + verbose, T-042.2 stub)
    sequential_injection_strategy: str = field(default="full", compare=False)

    def leaf_names(self) -> list[AgentName]:
        """All leaf agent names reachable from this node."""
        names: list[AgentName] = []
        for child in self.children:
            if isinstance(child, AgentLeaf):
                names.append(child.name)
            else:
                names.extend(child.leaf_names())
        return names

    def get_leaf(self, name: AgentName) -> AgentLeaf | None:
        for child in self.children:
            if isinstance(child, AgentLeaf) and child.name == name:
                return child
            if isinstance(child, CompoundCapsule):
                found = child.get_leaf(name)
                if found:
                    return found
        return None

    def __repr__(self) -> str:
        return f"CompoundCapsule({self.name!r}, children={[c.name for c in self.children]})"


# ---------------------------------------------------------------------------
# ToolLeaf — leaf node wrapping a ToolCapsule instead of an agent (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class ToolLeaf:
    """
    A leaf node in the hierarchy that wraps a ToolCapsule.

    The executor dispatches ToolLeaf nodes to the ToolOrchestrator instead
    of the LLM adapter, enabling mixed agent+tool pipelines within a single
    CapsuleHierarchy.

    `name` is used as the output_key prefix and for sync_manager tagging.

    Design plan ref: §3.2.1, §4.2 (Agent-Tool Co-Composition), §5.2 Phase 4
    """
    tool_capsule: object  # ToolCapsule (imported lazily to avoid circular deps)
    dependencies: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tool_capsule.name  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"ToolLeaf({self.name!r})"


# ---------------------------------------------------------------------------
# IterationCapsule — one batch of same-agent × multiple items (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class IterationCapsule:
    """
    A batch execution unit for iteration-space composition.

    Wraps one AgentLeaf with a list of AgentTagCapsules (one per data item
    in the batch). The executor dispatches this as a single LLM call that
    processes all items sequentially in one prompt.

    Analogous to `ResearchAgent<doc=1..10>` in the design plan pseudocode.

    Design plan ref: §2.1 (TagCapsule), §3.2.1, §5.2 Phase 2
    """
    leaf: AgentLeaf
    tags: list[AgentTagCapsule]       # one tag per item in this batch
    batch_index: int = 0              # which partition this batch is (for logging)

    @property
    def name(self) -> AgentName:
        return self.leaf.name

    @property
    def batch_size(self) -> int:
        return len(self.tags)

    def __repr__(self) -> str:
        return (
            f"IterationCapsule({self.name!r}, "
            f"batch={self.batch_index}, size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# Root: CapsuleHierarchy
# ---------------------------------------------------------------------------

@dataclass
class CapsuleHierarchy:
    """
    The root of the composition tree.

    Stores the full hierarchy and provides:

    - ``get_level(n)`` — retrieve all capsule nodes at depth n (used by the
      Granularity Controller to navigate between fine and coarse execution)
    - ``all_leaves()`` — iterate every AgentLeaf in definition order
    - ``validate()`` — run rule checks (delegates to core/rules.py)

    Depth 0 = root compound (most coarse); depth = max_depth = all leaves
    (most fine-grained, one capsule per agent).

    Design plan ref: §3.1 (Capsule Definition Layer), §3.2.2 (controller uses
    hierarchy levels to compose/decompose)
    """
    name: str
    root: CompoundCapsule
    # Optional iteration space; when set, the executor can use ITERATION mode.
    tag_space: object | None = field(default=None)  # TagSpace | None (imported lazily)
    iteration_dims: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def all_leaves(self) -> Iterator[AgentLeaf]:
        """Yield every AgentLeaf in the tree, left-to-right depth-first."""
        yield from _iter_leaves(self.root)

    def get_level(self, depth: int) -> list[AgentLeaf | CompoundCapsule]:
        """
        Return all nodes at the given depth.

        depth=0  → [root CompoundCapsule]  (single compound, fully merged)
        depth=1  → direct children of root
        ...
        depth=N  → all leaves (fully fine-grained)

        The Granularity Controller uses this to pick which level to execute.
        """
        if depth == 0:
            return [self.root]
        results: list[AgentLeaf | CompoundCapsule] = []
        _collect_at_depth(self.root, target_depth=depth, current_depth=0, out=results)
        return results

    def max_depth(self) -> int:
        """The depth at which all nodes are leaves (fully fine-grained level)."""
        return _tree_depth(self.root)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, adapter_context_window: int | None = None) -> None:
        """
        Run all composition rule checks.

        Raises CompositionError on the first violation found.
        Imports rules lazily to avoid circular dependency during construction.
        """
        from .rules import validate_hierarchy
        validate_hierarchy(self, adapter_context_window=adapter_context_window)


# ---------------------------------------------------------------------------
# Internal traversal utilities
# ---------------------------------------------------------------------------

def _iter_leaves(node: AgentLeaf | ToolLeaf | CompoundCapsule) -> Iterator[AgentLeaf | ToolLeaf]:
    if isinstance(node, (AgentLeaf, ToolLeaf)):
        yield node
    else:
        for child in node.children:
            yield from _iter_leaves(child)


def _collect_at_depth(
    node: AgentLeaf | CompoundCapsule,
    target_depth: int,
    current_depth: int,
    out: list,
) -> None:
    if current_depth == target_depth or isinstance(node, AgentLeaf):
        out.append(node)
        return
    for child in node.children:
        _collect_at_depth(child, target_depth, current_depth + 1, out)





def _tree_depth(node: AgentLeaf | CompoundCapsule, current: int = 0) -> int:
    if isinstance(node, AgentLeaf):
        return current
    return max(_tree_depth(child, current + 1) for child in node.children)
