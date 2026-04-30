"""Tests for runtime/prompt_compiler.py"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CompoundCapsule
from agentic_capsules.core.types import CompositionError, LLMMessage, Schema
from agentic_capsules.runtime.prompt_compiler import PromptCompiler
from agentic_capsules.runtime.scheduler import compute_order


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class StubAdapter:
    context_window = 200_000

    def complete(self, messages, tools=None):
        return "stub response"

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(name: str, prompt: str) -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=prompt,
            input_schema=Schema("in", fields={"q": "str"}),
            output_schema=Schema("out", fields={"r": "str"}),
        )
    )


def _two_agent_compound():
    researcher = _leaf("researcher", "You are a researcher. Find key facts.")
    summarizer = _leaf("summarizer", "You are a summarizer. Write a concise summary.")
    compound = CompoundCapsule(
        name="pipeline",
        children=[researcher, summarizer],
        dependency_edges={"summarizer": ["researcher"]},
    )
    compute_order(compound)
    return compound


# ---------------------------------------------------------------------------
# compile_compound
# ---------------------------------------------------------------------------

def test_compile_compound_returns_two_messages():
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="Explain quantum computing")
    assert len(result.messages) == 2
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"


def test_compile_compound_output_keys_in_order():
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    assert result.output_keys == ["RESEARCHER_OUTPUT", "SUMMARIZER_OUTPUT"]


def test_compile_compound_phase_markers_in_system_prompt():
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    system = result.messages[0].content
    assert "== PHASE 1:" in system
    assert "== PHASE 2:" in system


def test_compile_compound_inter_phase_reference():
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    system = result.messages[0].content
    # Phase 2 should reference RESEARCHER_OUTPUT from Phase 1
    assert "RESEARCHER_OUTPUT" in system


def test_compile_compound_rule6_raises_over_budget():
    class TinyAdapter:
        context_window = 10
        def complete(self, messages, tools=None): return ""
        def count_tokens(self, text): return len(text)  # 1 token per char

    compiler = PromptCompiler(TinyAdapter())
    compound = _two_agent_compound()
    with pytest.raises(CompositionError) as exc_info:
        compiler.compile_compound(compound, task_input="task")
    assert exc_info.value.rule == 6


# ---------------------------------------------------------------------------
# G-3 (AC↔LG parity, Phase 2 Batch B) — single-leaf compound routing
#
# A CompoundCapsule with exactly one AgentLeaf child ("single-leaf compound")
# is structurally equivalent to a single call. The legacy compound path
# still wraps it in the "You are executing a compound task with 1 sequential
# phases" preamble plus the "== PHASE 1: Name ==" header and the per-phase
# M-1 budget hint — all pure overhead on a single call. In multi_source_brief
# both `scoping` and `synthesis` are single-leaf groups; this framing adds
# +430 and +396 chars respectively on the offline probe, measured against
# LG's equivalent calls which have none of it.
#
# The fix: compile_compound detects len(order) == 1 and delegates to
# compile_single. Output contract is unchanged (same output_keys, same
# messages structure); only the preamble/header/M-1-hint framing is dropped.
# ---------------------------------------------------------------------------

def _single_leaf_compound():
    """A one-agent compound — mirrors multi_source_brief's scoping / briefer."""
    leaf = _leaf("briefer", "You are writing a one-page brief.")
    compound = CompoundCapsule(
        name="synthesis",
        children=[leaf],
        dependency_edges={},
    )
    compute_order(compound)
    return compound, leaf


def test_single_leaf_compound_has_no_compound_preamble():
    compiler = PromptCompiler(StubAdapter())
    compound, _ = _single_leaf_compound()
    result = compiler.compile_compound(compound, task_input="Write the brief.")
    system_text = result.messages[0].content
    # G-3: no "compound task with N sequential phases" preamble
    assert "compound task" not in system_text
    assert "sequential phases" not in system_text
    # G-3: no phase header
    assert "== PHASE" not in system_text


def test_single_leaf_compound_has_no_m1_hint():
    """M-1 budget hints are meaningless on a single leaf — there's nothing
    to balance against. The fix drops them by routing through compile_single."""
    compiler = PromptCompiler(StubAdapter())
    compound, _ = _single_leaf_compound()
    result = compiler.compile_compound(
        compound,
        task_input="task",
        merged_output_structure="budgeted",
    )
    system_text = result.messages[0].content
    assert "Target approximately 800 words" not in system_text
    assert "All 1 sections" not in system_text


def test_single_leaf_compound_preserves_output_key():
    compiler = PromptCompiler(StubAdapter())
    compound, _ = _single_leaf_compound()
    result = compiler.compile_compound(compound, task_input="task")
    # The single leaf's output_key must still appear so the executor
    # can parse the LLM response the same way fine-mode does.
    assert result.output_keys == ["BRIEFER_OUTPUT"]
    user_text = result.messages[1].content
    assert "BRIEFER_OUTPUT" in user_text


def test_single_leaf_compound_matches_compile_single_framing():
    """Single-leaf compound should be byte-for-byte equivalent to
    compile_single on the same leaf and task_input. Guards against future
    drift that could re-introduce compound-mode overhead."""
    compiler = PromptCompiler(StubAdapter())
    compound, leaf = _single_leaf_compound()
    compound_result = compiler.compile_compound(compound, task_input="Write the brief.")
    single_result = compiler.compile_single(leaf, task_input="Write the brief.")
    # Normalise message list into (role, content) tuples — they must match.
    compound_msgs = [(m.role, m.content) for m in compound_result.messages]
    single_msgs = [(m.role, m.content) for m in single_result.messages]
    assert compound_msgs == single_msgs
    assert compound_result.output_keys == single_result.output_keys


def test_single_leaf_compound_carries_prior_outputs():
    """Single-leaf compounds still need prior_outputs routed through."""
    compiler = PromptCompiler(StubAdapter())
    compound, _ = _single_leaf_compound()
    result = compiler.compile_compound(
        compound,
        task_input="task",
        prior_outputs={"SCOPING_OUTPUT": "Stripe, fintech, SF, 2010."},
    )
    user_text = result.messages[1].content
    assert "SCOPING_OUTPUT" in user_text
    assert "Stripe, fintech" in user_text


def test_two_leaf_compound_still_uses_phase_framing():
    """Regression guard: multi-leaf compounds must keep phase framing."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    system_text = result.messages[0].content
    assert "== PHASE 1:" in system_text
    assert "== PHASE 2:" in system_text


# ---------------------------------------------------------------------------
# compile_single
# ---------------------------------------------------------------------------

def test_compile_single_has_output_instruction():
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("analyst", "You are an analyst.")
    result = compiler.compile_single(leaf, task_input="Analyze this")
    user_text = result.messages[1].content
    assert "ANALYST_OUTPUT" in user_text


# ---------------------------------------------------------------------------
# parse_outputs
# ---------------------------------------------------------------------------

def test_parse_outputs_extracts_sections():
    response = (
        "RESEARCHER_OUTPUT:\nHere are the facts.\n\n"
        "SUMMARIZER_OUTPUT:\nHere is the summary."
    )
    parsed = PromptCompiler.parse_outputs(
        response, ["RESEARCHER_OUTPUT", "SUMMARIZER_OUTPUT"]
    )
    assert "facts" in parsed["RESEARCHER_OUTPUT"]
    assert "summary" in parsed["SUMMARIZER_OUTPUT"]


def test_parse_outputs_fallback_when_heading_missing():
    response = "Some unstructured response without headings."
    parsed = PromptCompiler.parse_outputs(response, ["MISSING_OUTPUT"])
    assert parsed["MISSING_OUTPUT"] == response.strip()


def test_parse_outputs_single_key():
    response = "ANALYST_OUTPUT:\nThe analysis is complete."
    parsed = PromptCompiler.parse_outputs(response, ["ANALYST_OUTPUT"])
    assert "analysis is complete" in parsed["ANALYST_OUTPUT"]


# ---------------------------------------------------------------------------
# T-038: tool_contexts injection and min_output_words depth hint
# ---------------------------------------------------------------------------

def test_tool_context_injected_into_matching_phase():
    """Pre-gathered tool data for this phase: appears in the system prompt for the matching agent."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    tool_contexts = {"researcher": "Search returned: AAPL Q4 revenue $124.3B."}
    result = compiler.compile_compound(
        compound, task_input="Analyse AAPL",
        tool_contexts=tool_contexts,
    )
    system_text = result.messages[0].content
    assert "Pre-gathered tool data for this phase:" in system_text
    assert "AAPL Q4 revenue" in system_text


def test_tool_context_only_for_matching_leaf():
    """Tool context for 'researcher' does NOT appear under the summarizer phase."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    tool_contexts = {"researcher": "Researcher tool data."}
    result = compiler.compile_compound(
        compound, task_input="task",
        tool_contexts=tool_contexts,
    )
    system_text = result.messages[0].content
    # Context appears exactly once (researcher phase only)
    assert system_text.count("Pre-gathered tool data for this phase:") == 1


def test_no_tool_context_when_not_provided():
    """compile_compound with no tool_contexts produces no injection block."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    assert "Pre-gathered tool data for this phase:" not in result.messages[0].content


def test_min_output_words_hint_appears_per_phase():
    """Depth hint appears once per phase when min_output_words is set."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(
        compound, task_input="task",
        min_output_words=200,
    )
    system_text = result.messages[0].content
    assert system_text.count("at least 200 words") == 2  # once per phase


def test_min_output_words_none_produces_no_hint():
    """No depth hint when min_output_words is None (default)."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(compound, task_input="task")
    assert "words" not in result.messages[0].content


def test_tool_context_and_depth_hint_together():
    """Both tool_contexts and min_output_words can be used simultaneously."""
    compiler = PromptCompiler(StubAdapter())
    compound = _two_agent_compound()
    result = compiler.compile_compound(
        compound, task_input="task",
        tool_contexts={"researcher": "some data"},
        min_output_words=150,
    )
    system_text = result.messages[0].content
    assert "Pre-gathered tool data for this phase:" in system_text
    assert "at least 150 words" in system_text


# ---------------------------------------------------------------------------
# T-058 — output_guidance="auto" (observations-based gate)
#
# Binary selector in compile_single:
#   - mean_fine_tokens >= guidance_threshold → apply concise
#   - otherwise → no guidance
#   - mean_fine_tokens is None or guidance_threshold is None → no guidance
#
# Replaces the failed "adaptive" variant (continuous 0.8× budget) which
# regressed haiku quality by −0.079 in §15.2 by reinforcing verbosity.
# ---------------------------------------------------------------------------

_CONCISE_MARKER = "Be concise"


def test_auto_above_threshold_applies_concise():
    """Verbose group (mean_fine_tokens > threshold) gets the concise hint."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="auto",
        mean_fine_tokens=3_600,      # haiku-like verbosity
        guidance_threshold=1_500,
    )
    user_text = result.messages[-1].content
    assert _CONCISE_MARKER in user_text


def test_auto_below_threshold_no_guidance():
    """Already-concise group (mean_fine_tokens < threshold) gets no hint.

    This is the load-bearing case: under forced concise, gemini-flash
    regresses −0.160 quality. Auto must skip the hint here.
    """
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="auto",
        mean_fine_tokens=960,        # gemini-flash-like terseness
        guidance_threshold=1_500,
    )
    user_text = result.messages[-1].content
    assert _CONCISE_MARKER not in user_text


def test_auto_at_boundary_applies_concise():
    """At exact threshold: >= comparison applies concise (inclusive boundary)."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="auto",
        mean_fine_tokens=1_500,
        guidance_threshold=1_500,
    )
    assert _CONCISE_MARKER in result.messages[-1].content


def test_auto_missing_observations_no_guidance():
    """No per-group observations yet (FINE warmup phase): auto no-ops."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="auto",
        mean_fine_tokens=None,
        guidance_threshold=1_500,
    )
    assert _CONCISE_MARKER not in result.messages[-1].content


def test_auto_missing_threshold_no_guidance():
    """Threshold not plumbed (test/stub setups): auto falls back safely."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="auto",
        mean_fine_tokens=10_000,
        guidance_threshold=None,
    )
    assert _CONCISE_MARKER not in result.messages[-1].content


def test_explicit_concise_ignores_threshold():
    """Explicit 'concise' override always applies regardless of observations."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="concise",
        mean_fine_tokens=100,        # would be below any sensible threshold
        guidance_threshold=1_500,
    )
    assert _CONCISE_MARKER in result.messages[-1].content


def test_none_guidance_no_hint_regardless():
    """'none' never emits a hint; threshold/observations irrelevant."""
    compiler = PromptCompiler(StubAdapter())
    leaf = _leaf("writer", "Write a section.")
    result = compiler.compile_single(
        leaf, task_input="topic",
        output_guidance="none",
        mean_fine_tokens=10_000,
        guidance_threshold=1_500,
    )
    assert _CONCISE_MARKER not in result.messages[-1].content
