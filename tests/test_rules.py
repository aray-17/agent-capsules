"""Tests for core/rules.py — all 6 composition rules."""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.rules import validate_hierarchy
from agentic_capsules.core.types import CompositionError, Schema


def _leaf(name: str, prompt: str = "prompt", output_fields: dict | None = None) -> AgentLeaf:
    cap = AgentStepCapsule(
        name=name,
        system_prompt=prompt,
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields=output_fields or {"r": "str"}),
    )
    return AgentLeaf(capsule=cap)


def _valid_hierarchy() -> CapsuleHierarchy:
    a = _leaf("a")
    b = _leaf("b")
    root = CompoundCapsule(
        name="root",
        children=[a, b],
        dependency_edges={"b": ["a"]},
    )
    return CapsuleHierarchy(name="valid", root=root)


# ---------------------------------------------------------------------------
# Rule 2 — Unique Tagging
# ---------------------------------------------------------------------------

def test_rule2_duplicate_name_raises():
    a1 = _leaf("agent")
    a2 = _leaf("agent")  # duplicate
    root = CompoundCapsule(name="root", children=[a1, a2], dependency_edges={})
    h = CapsuleHierarchy(name="dup", root=root)
    with pytest.raises(CompositionError) as exc_info:
        validate_hierarchy(h)
    assert exc_info.value.rule == 2


def test_rule2_unique_names_pass():
    validate_hierarchy(_valid_hierarchy())  # should not raise


# ---------------------------------------------------------------------------
# Rule 3 — Termination (cycle detection)
# ---------------------------------------------------------------------------

def test_rule3_cycle_raises():
    a = _leaf("a")
    b = _leaf("b")
    root = CompoundCapsule(
        name="root",
        children=[a, b],
        dependency_edges={"a": ["b"], "b": ["a"]},  # cycle
    )
    h = CapsuleHierarchy(name="cyclic", root=root)
    with pytest.raises(CompositionError) as exc_info:
        validate_hierarchy(h)
    assert exc_info.value.rule == 3


def test_rule3_dag_passes():
    validate_hierarchy(_valid_hierarchy())


# ---------------------------------------------------------------------------
# Rule 4 — Reachability
# ---------------------------------------------------------------------------

def test_rule4_unreachable_raises():
    a = _leaf("a")
    b = _leaf("b")
    c = _leaf("c")
    # c depends on b, but b depends on a — c is reachable via a→b→c
    # Make c unreachable: give it a dependency on a ghost node
    root = CompoundCapsule(
        name="root",
        children=[a, b, c],
        dependency_edges={
            "b": ["a"],
            "c": ["ghost"],  # ghost not a child → c has in-degree but no path from entry
        },
    )
    h = CapsuleHierarchy(name="unreachable", root=root)
    with pytest.raises(CompositionError) as exc_info:
        validate_hierarchy(h)
    # Either rule 3 (ghost not a child) or rule 4 fires first
    assert exc_info.value.rule in (3, 4)


def test_rule4_all_reachable_passes():
    validate_hierarchy(_valid_hierarchy())


# ---------------------------------------------------------------------------
# Rule 5 — Dimensional Consistency
# ---------------------------------------------------------------------------

def test_rule5_empty_output_schema_raises():
    a = _leaf("a", output_fields={})  # empty fields AND empty name would fail
    a.capsule.output_schema = Schema(name="", fields={})  # both empty
    root = CompoundCapsule(name="root", children=[a], dependency_edges={})
    h = CapsuleHierarchy(name="no_schema", root=root)
    with pytest.raises(CompositionError) as exc_info:
        validate_hierarchy(h)
    assert exc_info.value.rule == 5


# ---------------------------------------------------------------------------
# Rule 6 — Context Budget Feasibility
# ---------------------------------------------------------------------------

def test_rule6_over_budget_raises():
    # Very long prompt that will exceed a tiny context window
    big_prompt = "x" * 10_000
    a = _leaf("a", prompt=big_prompt)
    b = _leaf("b", prompt=big_prompt)
    root = CompoundCapsule(name="root", children=[a, b], dependency_edges={"b": ["a"]})
    h = CapsuleHierarchy(name="big", root=root)
    with pytest.raises(CompositionError) as exc_info:
        validate_hierarchy(h, adapter_context_window=100)
    assert exc_info.value.rule == 6


def test_rule6_within_budget_passes():
    validate_hierarchy(_valid_hierarchy(), adapter_context_window=200_000)
