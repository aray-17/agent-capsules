"""Tests for core/types.py"""

import pytest

from agentic_capsules.core.types import (
    CapsuleState,
    CompositionAxis,
    CompositionError,
    LLMMessage,
    Schema,
)


def test_capsule_state_values():
    assert CapsuleState.PENDING != CapsuleState.RUNNING
    assert CapsuleState.COMPLETE != CapsuleState.FAILED


def test_composition_axis_values():
    assert CompositionAxis.ITERATION != CompositionAxis.COMPUTATION
    assert CompositionAxis.TOOL != CompositionAxis.COMPUTATION


def test_llm_message():
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_schema_defaults():
    s = Schema(name="result")
    assert s.description == ""
    assert s.fields == {}


def test_composition_error_carries_rule():
    err = CompositionError(rule=3, message="cycle detected")
    assert err.rule == 3
    assert "Rule 3" in str(err)
    assert "cycle detected" in str(err)
