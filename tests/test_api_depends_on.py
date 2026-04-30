"""
T-019 — DSL depends_on → compiler dependency_edges → topology classifier.

Verifies that the public `.agent(depends_on=...)` builder argument flows all
the way through `_PipelineCompiler._compile_group()` into the
`CompoundCapsule.dependency_edges` dict, and that the resulting edge graph is
what the topology classifier sees.

Before T-019, the compiler hard-coded a linear chain regardless of user intent,
so `diamond`, `fan_out`, and `parallel_converge` topologies were unreachable
from the public DSL.
"""
import pytest
from unittest.mock import MagicMock

from agentic_capsules.api.builder import Pipeline
from agentic_capsules.api.compiler import _PipelineCompiler
from agentic_capsules.controller.policy import ControllerPolicy
from agentic_capsules.runtime.topology import _classify_topology


def _compiler(pipeline):
    return _PipelineCompiler(pipeline, "task", MagicMock(), "auto", None)


def _compile_group(pipeline, group_index=0):
    spec = pipeline._groups[group_index]
    return _compiler(pipeline)._compile_group(spec)


# ---------------------------------------------------------------------------
# Implicit linear (historical default, no depends_on passed)
# ---------------------------------------------------------------------------

def test_implicit_linear_chain_three_agents():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "second")
            .agent("c", "third")
    )
    compound = _compile_group(p)
    assert compound.dependency_edges == {"b": ["a"], "c": ["b"]}
    assert _classify_topology(compound) == "linear"


def test_implicit_linear_single_agent_has_no_edges():
    p = Pipeline("t").group("g").agent("only", "do it")
    compound = _compile_group(p)
    assert compound.dependency_edges == {}
    assert _classify_topology(compound) == "linear"


# ---------------------------------------------------------------------------
# Fan-out: every agent has depends_on=[]
# ---------------------------------------------------------------------------

def test_fan_out_topology_from_dsl():
    p = (
        Pipeline("t")
        .group("reviewers")
            .agent("root", "seed")
            .agent("r1", "review from angle 1", depends_on=["root"])
            .agent("r2", "review from angle 2", depends_on=["root"])
            .agent("r3", "review from angle 3", depends_on=["root"])
    )
    compound = _compile_group(p)
    assert compound.dependency_edges == {
        "r1": ["root"],
        "r2": ["root"],
        "r3": ["root"],
    }
    assert _classify_topology(compound) == "fan_out"


# ---------------------------------------------------------------------------
# Diamond: two independent agents converging into one synthesizer
# ---------------------------------------------------------------------------

def test_diamond_topology_from_dsl():
    p = (
        Pipeline("t")
        .group("g")
            .agent("root",      "seed")
            .agent("branch_a",  "branch A", depends_on=["root"])
            .agent("branch_b",  "branch B", depends_on=["root"])
            .agent("synth",     "merge",    depends_on=["branch_a", "branch_b"])
    )
    compound = _compile_group(p)
    assert compound.dependency_edges == {
        "branch_a": ["root"],
        "branch_b": ["root"],
        "synth":    ["branch_a", "branch_b"],
    }
    # Diamond wins over fan_out when any node has in_degree >= 2.
    assert _classify_topology(compound) == "diamond"


def test_diamond_sets_deps_injection_strategy():
    p = (
        Pipeline("t")
        .group("g")
            .agent("root",      "seed")
            .agent("branch_a",  "branch A", depends_on=["root"])
            .agent("branch_b",  "branch B", depends_on=["root"])
            .agent("synth",     "merge",    depends_on=["branch_a", "branch_b"])
    )
    compound = _compile_group(p)
    # classify_and_set_strategy is called from _compile_group itself.
    assert compound.sequential_injection_strategy == "deps"


def test_independent_root_via_explicit_empty_deps_routes_to_deps_strategy():
    """
    Edge case: a group with two agents where one declares depends_on=[].
    The resulting graph has no in_degree ≥ 2 and no out_degree ≥ 2, so the
    topology classifier would label it "linear" and (under the policy default)
    fall through to the "full" or "predecessor_only" strategy. With
    predecessor_only enabled, the second agent would receive the first
    agent's output even though it declared independence — the foot-gun.

    Fix: any explicit depends_on (including []) routes through "deps" so the
    declared graph is honored exactly. The independent root receives no
    upstream context, as declared.
    """
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "independent", depends_on=[])
    )
    compound = _compile_group(p)
    # Edges: only the implicit/explicit declarations land here.
    # b declared depends_on=[], so it has no edges; a has no entry either.
    assert compound.dependency_edges == {"b": []}
    # Strategy must be "deps" because the user opted in via explicit deps.
    assert compound.sequential_injection_strategy == "deps"


def test_explicit_linear_chain_routes_to_deps_strategy():
    """
    Even when explicit deps form a strict linear chain that's identical to
    what implicit-linear would produce, the explicit declaration is an opt-in
    to graph-aware injection. The strategy must be "deps", not "full" — this
    forecloses the predecessor_only override on graphs the user wrote out.
    """
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "second", depends_on=["a"])
            .agent("c", "third",  depends_on=["b"])
    )
    compound = _compile_group(p)
    assert compound.sequential_injection_strategy == "deps"


def test_implicit_linear_chain_still_routes_to_full_strategy():
    """
    Sanity check: implicit-linear chains (no explicit depends_on anywhere)
    keep the historical "full" strategy. The policy-level predecessor_only
    override is still allowed to fire on these — that's the S-1 experiment's
    target topology and must be unaffected by this fix.
    """
    p = (
        Pipeline("t")
        .group("g")
            .agent("a", "first")
            .agent("b", "second")
            .agent("c", "third")
    )
    compound = _compile_group(p)
    assert compound.sequential_injection_strategy == "full"


# ---------------------------------------------------------------------------
# Mixed: one agent overrides, siblings fall back to implicit linear
# ---------------------------------------------------------------------------

def test_mixed_implicit_and_explicit_deps():
    """If only one agent declares deps, siblings still get implicit linear."""
    p = (
        Pipeline("t")
        .group("g")
            .agent("a1", "first")
            .agent("a2", "second")                            # implicit → ["a1"]
            .agent("a3", "third", depends_on=["a1"])          # explicit → ["a1"]
            .agent("a4", "fourth")                            # implicit → ["a3"]
    )
    compound = _compile_group(p)
    assert compound.dependency_edges == {
        "a2": ["a1"],
        "a3": ["a1"],
        "a4": ["a3"],
    }


# ---------------------------------------------------------------------------
# Serialization order still valid after topological sort
# ---------------------------------------------------------------------------

def test_diamond_serialization_order_respects_deps():
    p = (
        Pipeline("t")
        .group("g")
            .agent("root",      "seed")
            .agent("branch_a",  "branch A", depends_on=["root"])
            .agent("branch_b",  "branch B", depends_on=["root"])
            .agent("synth",     "merge",    depends_on=["branch_a", "branch_b"])
    )
    compound = _compile_group(p)
    order = [leaf.name for leaf in compound.serialization_order]
    # root must precede branches; branches must precede synth.
    assert order.index("root") < order.index("branch_a")
    assert order.index("root") < order.index("branch_b")
    assert order.index("branch_a") < order.index("synth")
    assert order.index("branch_b") < order.index("synth")
