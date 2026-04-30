"""Tests for core/capsule.py"""

import pytest

from agentic_capsules.core.capsule import (
    AgentItemCapsule,
    AgentStepCapsule,
    AgentTagCapsule,
)
from agentic_capsules.core.types import CompositionAxis, Schema


def _make_capsule(name="researcher"):
    return AgentStepCapsule(
        name=name,
        system_prompt=f"You are a {name}.",
        input_schema=Schema(name="input", fields={"query": "str"}),
        output_schema=Schema(name="output", fields={"result": "str"}),
    )


# ---------------------------------------------------------------------------
# AgentTagCapsule
# ---------------------------------------------------------------------------

def test_tag_key_format():
    tag = AgentTagCapsule(agent_name="researcher", task_id="t1")
    assert tag.key == "researcher::t1"


def test_tag_is_hashable():
    tag = AgentTagCapsule(agent_name="a", task_id="1")
    s = {tag}
    assert tag in s


def test_tag_equality():
    t1 = AgentTagCapsule("a", "1")
    t2 = AgentTagCapsule("a", "1")
    t3 = AgentTagCapsule("b", "1")
    assert t1 == t2
    assert t1 != t3


# ---------------------------------------------------------------------------
# AgentStepCapsule
# ---------------------------------------------------------------------------

def test_output_key_default():
    cap = _make_capsule("fact checker")
    assert cap.output_key == "FACT_CHECKER_OUTPUT"


def test_output_key_custom():
    cap = AgentStepCapsule(
        name="x",
        system_prompt="prompt",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"r": "str"}),
        output_key="MY_CUSTOM_OUTPUT",
    )
    assert cap.output_key == "MY_CUSTOM_OUTPUT"


def test_default_composition_axis():
    cap = _make_capsule()
    assert cap.composition_axis == CompositionAxis.COMPUTATION


# ---------------------------------------------------------------------------
# AgentItemCapsule
# ---------------------------------------------------------------------------

def test_item_capsule_stores_data():
    tag = AgentTagCapsule("researcher", "t1")
    item = AgentItemCapsule(
        data={"result": "some findings"},
        producer_tag=tag,
        schema=Schema("output", fields={"result": "str"}),
        output_key="RESEARCHER_OUTPUT",
    )
    assert item.data == {"result": "some findings"}
    assert item.producer_tag == tag
