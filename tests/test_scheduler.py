"""Tests for runtime/scheduler.py"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CompoundCapsule
from agentic_capsules.core.types import CompositionError, Schema
from agentic_capsules.runtime.scheduler import compute_order


def _leaf(name: str) -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=f"{name} prompt",
            input_schema=Schema("in", fields={"q": "str"}),
            output_schema=Schema("out", fields={"r": "str"}),
        )
    )


def test_linear_order():
    """a → b → c should produce [a, b, c]."""
    a, b, c = _leaf("a"), _leaf("b"), _leaf("c")
    compound = CompoundCapsule(
        name="pipeline",
        children=[a, b, c],
        dependency_edges={"b": ["a"], "c": ["b"]},
    )
    order = compute_order(compound)
    assert [l.name for l in order] == ["a", "b", "c"]


def test_parallel_roots():
    """a and b both independent; c depends on both."""
    a, b, c = _leaf("a"), _leaf("b"), _leaf("c")
    compound = CompoundCapsule(
        name="pipeline",
        children=[a, b, c],
        dependency_edges={"c": ["a", "b"]},
    )
    order = compute_order(compound)
    names = [l.name for l in order]
    # a and b before c in any valid topological order
    assert names.index("c") > names.index("a")
    assert names.index("c") > names.index("b")


def test_no_edges_any_order():
    """No dependencies: all orderings are valid."""
    a, b = _leaf("a"), _leaf("b")
    compound = CompoundCapsule(name="p", children=[a, b], dependency_edges={})
    order = compute_order(compound)
    assert len(order) == 2
    assert {l.name for l in order} == {"a", "b"}


def test_caches_on_compound():
    a, b = _leaf("a"), _leaf("b")
    compound = CompoundCapsule(
        name="p", children=[a, b], dependency_edges={"b": ["a"]}
    )
    order = compute_order(compound)
    assert compound.serialization_order is order


def test_cycle_raises():
    a, b = _leaf("a"), _leaf("b")
    compound = CompoundCapsule(
        name="p",
        children=[a, b],
        dependency_edges={"a": ["b"], "b": ["a"]},
    )
    with pytest.raises(CompositionError) as exc_info:
        compute_order(compound)
    assert exc_info.value.rule == 3
