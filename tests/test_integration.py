"""
Integration tests — real LLM adapter smoke tests.

These tests make actual API calls and are skipped by default.
To run them, set the relevant environment variable and use the marker:

    ANTHROPIC_API_KEY=sk-... pytest -m integration
    OPENAI_API_KEY=sk-...    pytest -m integration

All tests follow the same pattern:
  1. Build a minimal single-agent FINE pipeline
  2. Run it with the real adapter
  3. Assert the output is non-empty and structurally sound (contains the
     expected heading — confirming the parser works against real LLM output)

Design plan ref: §5.2 Phase 8, T-004
"""

from __future__ import annotations

import os

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order


# ---------------------------------------------------------------------------
# Shared pipeline builder
# ---------------------------------------------------------------------------

def _single_agent_hierarchy(name: str = "summarizer") -> CapsuleHierarchy:
    """One-leaf pipeline for a simple summarization task."""
    leaf = AgentLeaf(capsule=AgentStepCapsule(
        name=name,
        system_prompt="You are a helpful summarizer. Summarize the given text in one sentence.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("output", fields={"summary": "str"}),
    ))
    root = CompoundCapsule(name="pipeline", children=[leaf], dependency_edges={})
    compute_order(root)
    return CapsuleHierarchy(name="integration_test", root=root)


TASK_INPUT = (
    "Agentic capsules is a Python framework for dynamic granularity composition "
    "across agents, data, and tools. It provides iteration-space, computation-space, "
    "and tool-space composition primitives that let developers trade off LLM call "
    "count against coordination overhead at runtime."
)


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Set ANTHROPIC_API_KEY to run Anthropic integration tests",
)
def test_anthropic_single_agent_round_trip():
    """Single FINE-mode call via AnthropicAdapter returns non-empty structured output."""
    from agentic_capsules.adapters.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(
        _single_agent_hierarchy("summarizer"),
        task_input=TASK_INPUT,
        task_id="integration-anthropic-1",
    )

    assert result.final_output.strip(), "Expected non-empty output from AnthropicAdapter"
    assert len(result.outputs) == 1
    key = list(result.outputs.keys())[0]
    assert result.outputs[key].strip(), f"Output for key {key!r} is empty"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Set ANTHROPIC_API_KEY to run Anthropic integration tests",
)
def test_anthropic_compound_mode_round_trip():
    """Two-agent COMPOUND call via AnthropicAdapter — both output keys parsed correctly."""
    from agentic_capsules.adapters.anthropic import AnthropicAdapter

    analyzer = AgentLeaf(capsule=AgentStepCapsule(
        name="analyzer",
        system_prompt="Identify the main topic of the text in one phrase.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("topic", fields={"topic": "str"}),
    ))
    summarizer = AgentLeaf(capsule=AgentStepCapsule(
        name="summarizer",
        system_prompt="Summarize the text in one sentence.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("output", fields={"summary": "str"}),
    ))
    root = CompoundCapsule(
        name="pipeline", children=[analyzer, summarizer], dependency_edges={}
    )
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="compound_integration", root=root)

    adapter = AnthropicAdapter()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(hierarchy, task_input=TASK_INPUT, task_id="integration-anthropic-2")

    assert "ANALYZER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs
    assert result.outputs["ANALYZER_OUTPUT"].strip()
    assert result.outputs["SUMMARIZER_OUTPUT"].strip()


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI integration tests",
)
def test_openai_single_agent_round_trip():
    """Single FINE-mode call via OpenAIAdapter returns non-empty structured output."""
    from agentic_capsules.adapters.openai import OpenAIAdapter

    adapter = OpenAIAdapter(model="gpt-4.1")
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(
        _single_agent_hierarchy("summarizer"),
        task_input=TASK_INPUT,
        task_id="integration-openai-1",
    )

    assert result.final_output.strip(), "Expected non-empty output from OpenAIAdapter"
    assert len(result.outputs) == 1
    key = list(result.outputs.keys())[0]
    assert result.outputs[key].strip(), f"Output for key {key!r} is empty"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI integration tests",
)
def test_openai_compound_mode_round_trip():
    """Two-agent COMPOUND call via OpenAIAdapter — both output keys parsed correctly."""
    from agentic_capsules.adapters.openai import OpenAIAdapter

    analyzer = AgentLeaf(capsule=AgentStepCapsule(
        name="analyzer",
        system_prompt="Identify the main topic of the text in one phrase.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("topic", fields={"topic": "str"}),
    ))
    summarizer = AgentLeaf(capsule=AgentStepCapsule(
        name="summarizer",
        system_prompt="Summarize the text in one sentence.",
        input_schema=Schema("input", fields={"text": "str"}),
        output_schema=Schema("output", fields={"summary": "str"}),
    ))
    root = CompoundCapsule(
        name="pipeline", children=[analyzer, summarizer], dependency_edges={}
    )
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="compound_integration", root=root)

    adapter = OpenAIAdapter(model="gpt-4.1")
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(hierarchy, task_input=TASK_INPUT, task_id="integration-openai-2")

    assert "ANALYZER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs
    assert result.outputs["ANALYZER_OUTPUT"].strip()
    assert result.outputs["SUMMARIZER_OUTPUT"].strip()
