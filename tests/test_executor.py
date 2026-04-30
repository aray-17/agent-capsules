"""
Tests for runtime/executor.py

Uses a stub LLM adapter that returns pre-scripted responses so tests
run without any API calls.
"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CapsuleState, CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    """Returns a preset response for each successive complete() call."""

    context_window = 200_000

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def complete(self, messages, tools=None):
        if self._call_count >= len(self._responses):
            return "DEFAULT RESPONSE"
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @property
    def call_count(self):
        return self._call_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(name: str) -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=f"You are {name}.",
            input_schema=Schema("in", fields={"q": "str"}),
            output_schema=Schema("out", fields={"r": "str"}),
        )
    )


def _two_agent_hierarchy():
    researcher = _leaf("researcher")
    summarizer = _leaf("summarizer")
    root = CompoundCapsule(
        name="pipeline",
        children=[researcher, summarizer],
        dependency_edges={"summarizer": ["researcher"]},
    )
    compute_order(root)
    return CapsuleHierarchy(name="test_pipeline", root=root)


# ---------------------------------------------------------------------------
# COMPOUND mode tests
# ---------------------------------------------------------------------------

def test_compound_mode_single_llm_call():
    compound_response = (
        "RESEARCHER_OUTPUT:\nKey facts found.\n\n"
        "SUMMARIZER_OUTPUT:\nConcise summary here."
    )
    adapter = ScriptedAdapter([compound_response])
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)

    result = executor.run(_two_agent_hierarchy(), task_input="Research AI safety")

    assert adapter.call_count == 1  # only ONE LLM call in compound mode
    assert "summary" in result.final_output.lower()


def test_compound_mode_extracts_final_output():
    response = (
        "RESEARCHER_OUTPUT:\nResearch done.\n\n"
        "SUMMARIZER_OUTPUT:\nFinal answer."
    )
    adapter = ScriptedAdapter([response])
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    result = executor.run(_two_agent_hierarchy(), task_input="task")

    assert result.final_output == "Final answer."


def test_compound_mode_capsule_states_complete():
    response = "RESEARCHER_OUTPUT:\nR.\n\nSUMMARIZER_OUTPUT:\nS."
    adapter = ScriptedAdapter([response])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.COMPOUND)
    executor.run(hierarchy, task_input="task")

    for leaf in hierarchy.all_leaves():
        assert leaf.capsule.state == CapsuleState.COMPLETE


# ---------------------------------------------------------------------------
# FINE mode tests
# ---------------------------------------------------------------------------

def test_fine_mode_two_llm_calls():
    adapter = ScriptedAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(_two_agent_hierarchy(), task_input="Research AI safety")

    assert adapter.call_count == 2  # one call per agent in fine mode


def test_fine_mode_correct_final_output():
    adapter = ScriptedAdapter([
        "RESEARCHER_OUTPUT:\nDetailed findings.",
        "SUMMARIZER_OUTPUT:\nBrief summary.",
    ])
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(_two_agent_hierarchy(), task_input="task")
    assert result.final_output == "Brief summary."


def test_fine_mode_accumulates_all_outputs():
    adapter = ScriptedAdapter([
        "RESEARCHER_OUTPUT:\nResearch text.",
        "SUMMARIZER_OUTPUT:\nSummary text.",
    ])
    executor = CapsuleExecutor(adapter, composition_level=CompositionLevel.FINE)
    result = executor.run(_two_agent_hierarchy(), task_input="task")
    assert "RESEARCHER_OUTPUT" in result.outputs
    assert "SUMMARIZER_OUTPUT" in result.outputs


# ---------------------------------------------------------------------------
# T-058 extension to FINE mode: _run_fine plumbs mean_fine_tokens +
# guidance_threshold to compile_single so output_guidance="auto" actually
# fires during FINE execution (previously a silent no-op).
# ---------------------------------------------------------------------------

class _CapturingAdapter:
    """Adapter that records every message sent to complete() for assertion."""

    context_window = 200_000

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.captured_messages: list[list] = []

    def complete(self, messages, tools=None):
        self.captured_messages.append(list(messages))
        if self._call_count >= len(self._responses):
            reply = "DEFAULT RESPONSE"
        else:
            reply = self._responses[self._call_count]
        self._call_count += 1
        return reply

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


_CONCISE_MARKER = "Be concise"


def _all_message_text(captured: list[list]) -> str:
    return "\n".join(
        msg.content for call in captured for msg in call
    )


def test_run_fine_auto_above_threshold_injects_concise():
    """FINE mode with mean_fine_tokens above threshold applies concise.

    The load-bearing claim: the T-058 auto gate works in FINE mode the
    same way it works in sequential compound (_run_compound_sequential:694).
    """
    adapter = _CapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.FINE,
        output_guidance="auto",
        mean_fine_tokens_by_group={hierarchy.root.name: 3_600},
        verbosity_guidance_threshold=1_500,
    )
    executor.run(hierarchy, task_input="task")
    all_text = _all_message_text(adapter.captured_messages)
    assert _CONCISE_MARKER in all_text, (
        "FINE-mode auto-concise should fire when mean_fine_tokens >= threshold"
    )


def test_run_fine_auto_below_threshold_no_guidance():
    """FINE mode with mean_fine_tokens below threshold gets no guidance.

    Prevents regression on already-terse models (gemini-flash −0.160 Δq
    under forced concise; T-058 auto-gate design).
    """
    adapter = _CapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.FINE,
        output_guidance="auto",
        mean_fine_tokens_by_group={hierarchy.root.name: 960},
        verbosity_guidance_threshold=1_500,
    )
    executor.run(hierarchy, task_input="task")
    all_text = _all_message_text(adapter.captured_messages)
    assert _CONCISE_MARKER not in all_text


def test_run_fine_auto_no_observations_no_guidance():
    """FINE warmup (no observations yet) falls through to no guidance.

    Preserves legacy behaviour: before observations accumulate, auto is a
    safe no-op. Also the invariant test for stub/test setups that do not
    plumb mean_fine_tokens_by_group.
    """
    adapter = _CapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.FINE,
        output_guidance="auto",
        # mean_fine_tokens_by_group intentionally empty
        verbosity_guidance_threshold=1_500,
    )
    executor.run(hierarchy, task_input="task")
    all_text = _all_message_text(adapter.captured_messages)
    assert _CONCISE_MARKER not in all_text


# ---------------------------------------------------------------------------
# T-059 adjacent fixes (2026-04-23): forward output_guidance + observations
# through the two other call sites of compile_single that previously dropped
# them — compile_compound single-leaf shortcut and _run_mixed_compound.
# ---------------------------------------------------------------------------

def _single_leaf_hierarchy():
    """Single-agent compound — triggers compile_compound's single-leaf shortcut
    (prompt_compiler.py:315). due_diligence's synthesis group has this shape.
    """
    writer = _leaf("writer")
    root = CompoundCapsule(
        name="synthesis",
        children=[writer],
        dependency_edges={},
    )
    compute_order(root)
    return CapsuleHierarchy(name="synthesis_pipeline", root=root)


def test_compound_single_leaf_shortcut_forwards_auto_concise():
    """Single-leaf compound routes through compile_compound's shortcut to
    compile_single — it MUST forward output_guidance + mean_fine_tokens or
    the T-058 auto-concise gate is silently lost on that call site.
    """
    adapter = _CapturingAdapter(["WRITER_OUTPUT:\nResult."])
    hierarchy = _single_leaf_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="standard",
        output_guidance="auto",
        mean_fine_tokens_by_group={hierarchy.root.name: 3_600},
        verbosity_guidance_threshold=1_500,
    )
    executor.run(hierarchy, task_input="task")
    all_text = _all_message_text(adapter.captured_messages)
    assert _CONCISE_MARKER in all_text, (
        "compile_compound single-leaf shortcut must forward output_guidance + "
        "observations; otherwise T-058 auto-concise silently drops on "
        "single-agent compound groups (e.g. due_diligence synthesis)"
    )


def _mixed_compound_hierarchy():
    """Mixed compound: nested CompoundCapsule + AgentLeaf siblings — exercises
    _run_mixed_compound (executor.py:798). Verifies the AgentLeaf branch's
    compile_single call receives the auto-concise gate inputs.
    """
    researcher = _leaf("researcher")
    inner_a = _leaf("inner_a")
    inner_b = _leaf("inner_b")
    inner_compound = CompoundCapsule(
        name="inner",
        children=[inner_a, inner_b],
        dependency_edges={"inner_b": ["inner_a"]},
    )
    compute_order(inner_compound)
    root = CompoundCapsule(
        name="outer",
        children=[researcher, inner_compound],
        dependency_edges={"inner": ["researcher"]},
    )
    compute_order(root)
    return CapsuleHierarchy(name="mixed_pipeline", root=root)


def test_run_mixed_compound_forwards_auto_concise():
    """_run_mixed_compound's AgentLeaf branch must plumb output_guidance +
    mean_fine_tokens to compile_single so nested-compound topologies don't
    silently lose the auto-concise gate.
    """
    # 3 LM responses: researcher (agent-leaf branch), inner_a + inner_b
    # (inner compound runs as its own compound call, which may be a single
    # merged call depending on the inner compound dispatch path).
    adapter = _CapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "INNER_A_OUTPUT:\nA.\n\nINNER_B_OUTPUT:\nB.",
        "EXTRA_OK:\nbackup",
    ])
    hierarchy = _mixed_compound_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="standard",
        output_guidance="auto",
        mean_fine_tokens_by_group={hierarchy.root.name: 3_600},
        verbosity_guidance_threshold=1_500,
    )
    executor.run(hierarchy, task_input="task")
    # The researcher (AgentLeaf direct child) goes through the mixed-compound
    # agent-leaf branch at executor.py:845. Its compiled prompt must contain
    # the concise marker.
    researcher_messages = adapter.captured_messages[0]
    researcher_text = "\n".join(m.content for m in researcher_messages)
    assert _CONCISE_MARKER in researcher_text, (
        "_run_mixed_compound's AgentLeaf branch must forward output_guidance "
        "+ mean_fine_tokens to compile_single"
    )


# ---------------------------------------------------------------------------
# T-059 Phase 1 fix (2026-04-23): plumb cache_aligned_prompts to _run_fine,
# _run_mixed_compound, and compile_compound so FINE mode (and
# nested-compound AgentLeaf calls) get the cacheable-system-prefix message
# structure identical to _run_compound_sequential. Before this fix,
# _run_fine / _run_mixed_compound / compile_compound single-leaf shortcut
# dropped cache_aligned_prompts entirely, so FINE mode callers paid full
# token rate on task+prior_outputs in the user block on every call.
# Identified in evals/dspy_ac_internal_gap_audit.md.
# ---------------------------------------------------------------------------

class _CacheCapableCapturingAdapter(_CapturingAdapter):
    """CapturingAdapter with supports_prompt_caching=True so the executor's
    adapter-capability check lets cache_aligned_prompts activate.
    """
    supports_prompt_caching = True


def _system_block_count(captured_messages_per_call: list) -> int:
    """Count system messages in a single call's message list."""
    return sum(1 for m in captured_messages_per_call if m.role == "system")


def test_run_fine_forwards_cache_aligned_prompts():
    """_run_fine with cache_aligned_prompts=True must pass it through to
    compile_single so the message structure has TWO system blocks
    (task prefix + agent prompt) instead of one. Before this fix, FINE
    mode always produced a single-system/single-user structure regardless
    of the cache_aligned_prompts policy setting.
    """
    adapter = _CacheCapableCapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.FINE,
        cache_aligned_prompts=True,
    )
    executor.run(hierarchy, task_input="task")
    # Under cache_aligned_prompts, each call has: [system=task, system=agent_prompt, user=...]
    # Without it: [system=agent_prompt, user=task+priors+...].
    for i, call_msgs in enumerate(adapter.captured_messages):
        assert _system_block_count(call_msgs) == 2, (
            f"Call {i}: expected 2 system blocks under cache_aligned_prompts, "
            f"got {_system_block_count(call_msgs)}. Messages: "
            f"{[(m.role, len(m.content)) for m in call_msgs]}"
        )


def test_run_fine_no_cache_aligned_prompts_legacy_structure():
    """Invariant: when cache_aligned_prompts defaults False, FINE keeps the
    legacy single-system structure. Preserves behavior for non-Anthropic
    adapters and pre-C-1 callers.
    """
    adapter = _CapturingAdapter([  # no supports_prompt_caching attribute
        "RESEARCHER_OUTPUT:\nFacts.",
        "SUMMARIZER_OUTPUT:\nSummary.",
    ])
    hierarchy = _two_agent_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.FINE,
        cache_aligned_prompts=True,  # policy asks for it, but adapter doesn't support
    )
    executor.run(hierarchy, task_input="task")
    for i, call_msgs in enumerate(adapter.captured_messages):
        assert _system_block_count(call_msgs) == 1, (
            f"Call {i}: expected 1 system block when adapter lacks "
            f"supports_prompt_caching, got {_system_block_count(call_msgs)}"
        )


def test_run_mixed_compound_forwards_cache_aligned_prompts():
    """_run_mixed_compound's AgentLeaf branch must also plumb
    cache_aligned_prompts. Before this fix, nested-compound AgentLeaf
    calls silently used the legacy non-cache-aligned structure even when
    policy requested cache_aligned.
    """
    adapter = _CacheCapableCapturingAdapter([
        "RESEARCHER_OUTPUT:\nFacts.",
        "INNER_A_OUTPUT:\nA.\n\nINNER_B_OUTPUT:\nB.",
        "EXTRA_OK:\nbackup",
    ])
    hierarchy = _mixed_compound_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="standard",
        cache_aligned_prompts=True,
    )
    executor.run(hierarchy, task_input="task")
    # First call is the researcher (AgentLeaf direct child via the
    # mixed-compound branch). It should have 2 system blocks under
    # cache_aligned_prompts.
    researcher_messages = adapter.captured_messages[0]
    assert _system_block_count(researcher_messages) == 2, (
        f"Mixed-compound AgentLeaf branch must forward cache_aligned_prompts. "
        f"Got {_system_block_count(researcher_messages)} system blocks; "
        f"expected 2."
    )


def test_compound_single_leaf_shortcut_forwards_cache_aligned_prompts():
    """compile_compound single-leaf shortcut must forward cache_aligned_prompts
    so single-agent compound groups (due_diligence synthesis,
    multi_source_brief scoping/briefer) get the cacheable-prefix structure
    when policy requests it.
    """
    adapter = _CacheCapableCapturingAdapter(["WRITER_OUTPUT:\nResult."])
    hierarchy = _single_leaf_hierarchy()
    executor = CapsuleExecutor(
        adapter,
        composition_level=CompositionLevel.COMPOUND,
        compound_execution_model="standard",
        cache_aligned_prompts=True,
    )
    executor.run(hierarchy, task_input="task")
    # Single call via the shortcut must have 2 system blocks.
    single_call_messages = adapter.captured_messages[0]
    assert _system_block_count(single_call_messages) == 2, (
        f"compile_compound single-leaf shortcut must forward "
        f"cache_aligned_prompts. Got {_system_block_count(single_call_messages)} "
        f"system blocks; expected 2."
    )
