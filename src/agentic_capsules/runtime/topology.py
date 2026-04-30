"""
Topology Classifier — T-042 Sequential COMPOUND Context Optimization.

Classifies the dependency graph of a CompoundCapsule into one of four
topology classes, then selects the appropriate sequential injection strategy
based on that topology and, optionally, observed output verbosity.

Topology classes:
  linear            — A→B→C→D — all nodes have in_degree ≤ 1 and out_degree ≤ 1
  fan_out           — A→B, A→C, A→D — some node has out_degree ≥ 2
  diamond           — A→B→D, A→C→D — some node has in_degree ≥ 2
  parallel_converge — root→(B‖C)→synth — root out_degree ≥ 2, max_depth = 2

Injection strategies (stored on CompoundCapsule.sequential_injection_strategy):
  "full"    — all accumulated prior outputs (current default)
  "deps"    — only declared-dependency outputs (for non-linear topologies)
  "summary" — summarised prior outputs (stub — T-042.2; falls back to "full")

Design plan ref: T-042, Option 1b (Graph Topology Heuristics)
"""

from __future__ import annotations

from ..core.hierarchy import CompoundCapsule


def _classify_topology(compound: CompoundCapsule) -> str:
    """
    Return the topology class of *compound*'s dependency graph.

    Uses `compound.dependency_edges` which maps each node to its list of
    direct dependencies: {node_name: [dep1, dep2, ...]}.

    Returns one of: "linear", "fan_out", "diamond", "parallel_converge"
    """
    edges = compound.dependency_edges  # {name: [deps]}
    leaf_names = {
        c.name for c in compound.children
        if hasattr(c, "name")
    }

    if not edges or len(leaf_names) <= 1:
        return "linear"

    # in_degree[n] = number of declared dependencies for n
    in_degree: dict[str, int] = {name: 0 for name in leaf_names}
    # out_degree[n] = number of nodes that depend on n
    out_degree: dict[str, int] = {name: 0 for name in leaf_names}

    for node, deps in edges.items():
        if node not in leaf_names:
            continue
        for dep in deps:
            if dep in leaf_names:
                in_degree[node] += 1
                out_degree[dep] = out_degree.get(dep, 0) + 1

    # Diamond: any node has 2+ declared dependencies
    if any(v >= 2 for v in in_degree.values()):
        return "diamond"

    # Fan-out: any node is depended on by 2+ others
    if any(v >= 2 for v in out_degree.values()):
        return "fan_out"

    # Parallel converge: root out_degree ≥ 2 AND max path-depth = 2
    # (root → independent branches → single synthesis node)
    # When fan_out was not detected above, a node with out_degree=0 and
    # in_degree=0 is the root.  Check whether the root's direct children
    # all converge at depth 2 via a shared sink.
    roots = [n for n in leaf_names if in_degree.get(n, 0) == 0]
    if len(roots) == 1:
        root = roots[0]
        root_out = out_degree.get(root, 0)
        # Sinks = nodes with no dependents (out_degree = 0)
        sinks = [n for n in leaf_names if out_degree.get(n, 0) == 0 and n != root]
        if root_out >= 2 and len(sinks) == 1:
            return "parallel_converge"

    return "linear"


def classify_and_set_strategy(
    compound: CompoundCapsule,
    avg_output_tokens: float | None = None,
    has_explicit_dependencies: bool = False,
) -> None:
    """
    Classify the topology of *compound* and store the recommended
    sequential injection strategy in ``compound.sequential_injection_strategy``.

    ``avg_output_tokens`` is the observed mean FINE-mode output size for this
    group (tokens).  When provided and topology is linear, a high value
    (≥ 3,500 tok/agent) triggers the "summary" stub.

    ``has_explicit_dependencies`` is True iff any agent in the group used the
    public ``depends_on=...`` argument (including ``depends_on=[]`` for an
    independent root). When True, strategy is forced to ``"deps"`` regardless
    of topology class — explicit dependency declarations are an opt-in to
    graph-aware injection, and the executor must honor them precisely. This
    closes the foot-gun where the policy-level ``predecessor_only`` strategy
    could silently inject the wrong context for graphs whose serialization
    order does not match the declared dependency edges (e.g. a group with an
    ``depends_on=[]`` independent root that happens to be classified as
    "linear" because no node has in_degree ≥ 2 and no node has out_degree ≥ 2).

    Called once per group at pipeline compile time (``_compile_group``).
    The executor reads the cached field — no recomputation per run.
    """
    if has_explicit_dependencies:
        # User opted into the dependency graph; honor it strictly.
        compound.sequential_injection_strategy = "deps"
        return

    topo = _classify_topology(compound)

    if topo in ("diamond", "fan_out", "parallel_converge"):
        compound.sequential_injection_strategy = "deps"
    elif topo == "linear":
        if avg_output_tokens is not None and avg_output_tokens >= 3_500:
            # T-042.2 stub: summary injection for verbose linear chains.
            # Falls back to "full" until T-042.2 is implemented.
            compound.sequential_injection_strategy = "full"  # stub
        else:
            compound.sequential_injection_strategy = "full"
    else:
        compound.sequential_injection_strategy = "full"
