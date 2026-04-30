"""Tests for core/hierarchy.py"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import Schema


def _leaf(name: str) -> AgentLeaf:
    cap = AgentStepCapsule(
        name=name,
        system_prompt=f"You are {name}.",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"r": "str"}),
    )
    return AgentLeaf(capsule=cap)


def _simple_hierarchy():
    """researcher → summarizer, no dependencies (sequential by position)."""
    researcher = _leaf("researcher")
    summarizer = _leaf("summarizer")
    summarizer.dependencies = ["researcher"]

    root = CompoundCapsule(
        name="pipeline",
        children=[researcher, summarizer],
        dependency_edges={"summarizer": ["researcher"]},
    )
    return CapsuleHierarchy(name="test_pipeline", root=root)


# ---------------------------------------------------------------------------
# all_leaves
# ---------------------------------------------------------------------------

def test_all_leaves_flat():
    h = _simple_hierarchy()
    names = [l.name for l in h.all_leaves()]
    assert names == ["researcher", "summarizer"]


def test_all_leaves_nested():
    inner = CompoundCapsule(
        name="inner",
        children=[_leaf("a"), _leaf("b")],
        dependency_edges={"b": ["a"]},
    )
    outer = CompoundCapsule(
        name="outer",
        children=[inner, _leaf("c")],
        dependency_edges={"c": ["b"]},
    )
    h = CapsuleHierarchy(name="nested", root=outer)
    names = [l.name for l in h.all_leaves()]
    assert set(names) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# get_level
# ---------------------------------------------------------------------------

def test_get_level_0_returns_root():
    h = _simple_hierarchy()
    level = h.get_level(0)
    assert len(level) == 1
    assert level[0] is h.root


def test_get_level_1_returns_leaves():
    h = _simple_hierarchy()
    level = h.get_level(1)
    assert len(level) == 2
    names = {n.name for n in level}
    assert names == {"researcher", "summarizer"}


# ---------------------------------------------------------------------------
# max_depth
# ---------------------------------------------------------------------------

def test_max_depth_simple():
    h = _simple_hierarchy()
    assert h.max_depth() == 1


def test_max_depth_nested():
    inner = CompoundCapsule(
        name="inner",
        children=[_leaf("a"), _leaf("b")],
        dependency_edges={"b": ["a"]},
    )
    outer = CompoundCapsule(
        name="outer",
        children=[inner, _leaf("c")],
        dependency_edges={},
    )
    h = CapsuleHierarchy(name="nested", root=outer)
    assert h.max_depth() == 2
