"""Tests for compile_compound() in runtime/prompt_compiler.py"""

import pytest

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CompoundCapsule
from agentic_capsules.core.types import CompositionError, Schema
from agentic_capsules.runtime.prompt_compiler import PromptCompiler
from agentic_capsules.runtime.scheduler import compute_order


class StubAdapter:
    context_window = 200_000
    def complete(self, messages, tools=None): return ""
    def count_tokens(self, text): return max(1, len(text) // 4)


def _leaf(name: str, system_prompt: str) -> AgentLeaf:
    return AgentLeaf(
        capsule=AgentStepCapsule(
            name=name,
            system_prompt=system_prompt,
            input_schema=Schema("in", fields={"text": "str"}),
            output_schema=Schema("out", fields={"result": "str"}),
        )
    )


def _compound(*names_and_prompts: tuple[str, str]) -> CompoundCapsule:
    leaves = [_leaf(name, prompt) for name, prompt in names_and_prompts]
    compound = CompoundCapsule(name="pipeline", children=leaves, dependency_edges={})
    compute_order(compound)
    return compound


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_phase_headers_present():
    compound = _compound(("researcher", "Research the topic."), ("summarizer", "Summarize the findings."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="AI safety")
    system = result.messages[0].content
    assert "== PHASE 1:" in system
    assert "== PHASE 2:" in system


def test_phase_header_uses_agent_name():
    compound = _compound(("researcher", "Research."), ("summarizer", "Summarize."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    assert "Researcher" in system
    assert "Summarizer" in system


def test_output_keys_match_leaf_output_keys():
    compound = _compound(("researcher", "R."), ("fact_checker", "FC."), ("summarizer", "S."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    assert result.output_keys == ["RESEARCHER_OUTPUT", "FACT_CHECKER_OUTPUT", "SUMMARIZER_OUTPUT"]


def test_input_instruction_absent_for_phase1():
    compound = _compound(("researcher", "Research."), ("summarizer", "Summarize."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # Phase 1 section must NOT reference a prior key
    phase1_section = system.split("== PHASE 2:")[0]
    assert "Use" not in phase1_section or "PHASE 1" not in phase1_section.split("Use")[0]


def test_input_instruction_present_from_phase2():
    compound = _compound(("researcher", "Research."), ("fact_checker", "Fact check."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    assert "RESEARCHER_OUTPUT" in system
    assert "Phase 1" in system


def test_three_phase_dataflow_chain():
    compound = _compound(
        ("researcher", "Research."),
        ("fact_checker", "Fact check."),
        ("summarizer", "Summarize."),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # Phase 2 references Phase 1 output; Phase 3 references Phase 2 output
    assert "RESEARCHER_OUTPUT" in system
    assert "FACT_CHECKER_OUTPUT" in system


def test_task_input_in_user_message():
    compound = _compound(("researcher", "Research."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="quantum computing")
    assert "quantum computing" in result.messages[1].content


def test_prior_outputs_injected_into_user_message():
    compound = _compound(("summarizer", "Summarize."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound,
        task_input="topic",
        prior_outputs={"EXTERNAL_OUTPUT": "some prior result"},
    )
    user = result.messages[1].content
    assert "EXTERNAL_OUTPUT" in user
    assert "some prior result" in user


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

def test_coordination_tokens_nonzero():
    compound = _compound(("researcher", "Research the topic in depth."), ("summarizer", "Summarize findings."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    assert result.coordination_tokens > 0


def test_coordination_tokens_less_than_total():
    compound = _compound(("researcher", "Research."), ("summarizer", "Summarize."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    assert result.coordination_tokens < result.estimated_tokens


def test_single_phase_coordination_tokens_nonzero():
    compound = _compound(("analyst", "Analyse the document carefully."))
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="text")
    assert result.coordination_tokens > 0


# ---------------------------------------------------------------------------
# Rule 6
# ---------------------------------------------------------------------------

def test_rule6_fires_when_too_large():
    class TinyAdapter:
        context_window = 5
        def complete(self, m): return ""
        def count_tokens(self, t): return len(t)

    compound = _compound(("researcher", "Research deeply."), ("summarizer", "Summarize concisely."))
    compiler = PromptCompiler(TinyAdapter())
    with pytest.raises(CompositionError) as exc_info:
        compiler.compile_compound(compound, task_input="topic")
    assert exc_info.value.rule == 6


# ---------------------------------------------------------------------------
# PE (prompt-economy, 2026-04-09) — parallel-independent compile path
# ---------------------------------------------------------------------------
#
# These tests cover the new parallel-independent compile path added by the
# 2026-04-09 prompt-economy fix. Trigger condition: every leaf in the
# compound has an explicit depends_on=[] in ``dependency_edges``. The
# existing `_compound()` helper above uses ``dependency_edges={}`` which
# leaves every leaf NOT in the dict → treated as implicit linear → legacy
# path. That is what keeps the older tests in this file green after the
# fix; the helper below is what activates the new path.


def _parallel_compound(*names_and_prompts: tuple[str, str]) -> CompoundCapsule:
    """Build a compound where every leaf has explicit depends_on=[]."""
    leaves = [_leaf(name, prompt) for name, prompt in names_and_prompts]
    edges = {name: [] for name, _ in names_and_prompts}
    compound = CompoundCapsule(name="pipeline", children=leaves, dependency_edges=edges)
    compute_order(compound)
    return compound


# Shared source material used to simulate the multi_source_brief shape where
# every extractor's system_prompt begins with the same bundle. Must be
# >200 chars (the default min_chars for shared-prefix hoisting) and must
# contain at least one "\n\n" paragraph boundary *within* the prefix.
_SHARED_BUNDLE = (
    "SOURCE MATERIAL:\n"
    "Stripe is a payments company founded in 2010 by Patrick and John Collison. "
    "The company is headquartered in San Francisco and Dublin, processes hundreds "
    "of billions of dollars in payments annually, and operates in over 40 countries. "
    "Its primary products include payment processing, billing, and financial infrastructure.\n"
)


def _prompt(leaf_specific: str) -> str:
    """Build a leaf prompt: shared bundle + leaf-specific instruction."""
    return _SHARED_BUNDLE + "\n" + leaf_specific


# ---- _is_parallel_independent helper ----

def test_is_parallel_independent_all_empty_deps():
    compound = _parallel_compound(
        ("entities", "Extract entities."),
        ("claims", "Extract claims."),
        ("signals", "Extract signals."),
    )
    assert PromptCompiler._is_parallel_independent(compound) is True


def test_is_parallel_independent_any_dep_blocks():
    # entities depends on nothing, but claims depends on entities → not parallel
    leaves = [
        _leaf("entities", "Extract entities."),
        _leaf("claims", "Extract claims."),
    ]
    compound = CompoundCapsule(
        name="mixed",
        children=leaves,
        dependency_edges={"entities": [], "claims": ["entities"]},
    )
    compute_order(compound)
    assert PromptCompiler._is_parallel_independent(compound) is False


def test_is_parallel_independent_single_leaf_is_false():
    # A 1-leaf compound has nothing to parallelize; keep legacy path.
    compound = _parallel_compound(("only_one", "Do the thing."))
    assert PromptCompiler._is_parallel_independent(compound) is False


def test_is_parallel_independent_implicit_linear_is_false():
    # Legacy _compound() uses dependency_edges={} — NOT parallel.
    compound = _compound(
        ("researcher", "Research."),
        ("summarizer", "Summarize."),
    )
    assert PromptCompiler._is_parallel_independent(compound) is False


# ---- _extract_shared_prefix helper ----

def test_extract_shared_prefix_hoists_shared_bundle():
    leaves = [
        _leaf("entities", _prompt("Extract entities as a numbered list.")),
        _leaf("claims",   _prompt("Extract claims as a numbered list.")),
        _leaf("signals",  _prompt("Extract signals as a numbered list.")),
    ]
    shared, tails = PromptCompiler._extract_shared_prefix(leaves, min_chars=200)
    # Shared must include the full SOURCE MATERIAL block up to the \n\n
    assert shared.startswith("SOURCE MATERIAL:")
    assert "Stripe" in shared
    # Each tail is the leaf-specific instruction (no source material)
    assert len(tails) == 3
    for tail in tails:
        assert "SOURCE MATERIAL" not in tail
        assert "Extract" in tail and "numbered list" in tail


def test_extract_shared_prefix_below_threshold_returns_empty():
    # "Research." and "Summarize." share only a leading capital letter → no hoist
    leaves = [
        _leaf("researcher", "Research."),
        _leaf("summarizer", "Summarize."),
    ]
    shared, tails = PromptCompiler._extract_shared_prefix(leaves, min_chars=200)
    assert shared == ""
    assert tails == ["Research.", "Summarize."]


def test_extract_shared_prefix_snaps_to_paragraph_boundary():
    # Shared prefix is big but the divergence point is mid-sentence → snap
    # back to the previous \n\n boundary, not mid-word.
    common = "A" * 250 + "\n\n" + "B" * 50  # 250 A's, blank line, 50 B's
    leaves = [
        _leaf("a", common + "EXTRACTOR_ONE: do one thing."),
        _leaf("b", common + "EXTRACTOR_TWO: do another thing."),
    ]
    shared, tails = PromptCompiler._extract_shared_prefix(leaves, min_chars=200)
    # Must end at the paragraph boundary after the A's, NOT leak the B block
    # into shared (both leaves share the B block too, so the raw LCP would
    # extend past it, but we accept that snap). Critically, the shared block
    # must not mid-slice anything.
    assert shared  # non-empty
    assert not shared.endswith("\n")  # trimmed
    # Each tail starts where shared left off (after a \n\n separator)
    assert tails[0].startswith("EXTRACTOR_ONE") or tails[0].startswith("B")
    assert tails[1].startswith("EXTRACTOR_TWO") or tails[1].startswith("B")


# ---- compile_compound parallel-independent path: preamble + structure ----

def test_parallel_path_emits_parallel_preamble():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
        ("signals",  _prompt("Extract signals.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="brief Stripe")
    system = result.messages[0].content
    # Parallel preamble markers
    assert "independent extractions" in system
    assert "from the source material directly" in system
    assert "Do not write a preamble" in system
    # Legacy sequential framing MUST be absent
    assert "compound task with" not in system
    assert "sequential phases" not in system


def test_parallel_path_emits_section_markers_not_phase_headers():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    assert "--- Section 1: Entities ---" in system
    assert "--- Section 2: Claims ---" in system
    # Legacy markers absent
    assert "== PHASE 1:" not in system
    assert "== PHASE 2:" not in system


def test_parallel_path_no_phase_chaining_lines():
    # The "Use X_OUTPUT from Phase N" directive must NOT appear when leaves
    # are parallel-independent — these were causing haiku to chain outputs.
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
        ("signals",  _prompt("Extract signals.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    assert "from Phase" not in system
    assert "ENTITIES_OUTPUT" in system  # section 1 output key still declared
    # But NOT as an input to claims
    claims_block = system.split("--- Section 2: Claims ---")[1].split("--- Section 3")[0]
    assert "ENTITIES_OUTPUT" not in claims_block
    assert "Use" not in claims_block.split("Produce your output")[0]


def test_parallel_path_shared_prefix_hoisted_once():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities as numbered list.")),
        ("claims",   _prompt("Extract claims as numbered list.")),
        ("signals",  _prompt("Extract signals as numbered list.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # SHARED CONTEXT block present ONCE
    assert system.count("SHARED CONTEXT:") == 1
    # SOURCE MATERIAL bundle appears ONCE, not N=3 times
    assert system.count("SOURCE MATERIAL:") == 1
    # Leaf-specific tails all still present
    assert "Extract entities" in system
    assert "Extract claims" in system
    assert "Extract signals" in system


def test_parallel_path_output_keys_preserved():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
        ("signals",  _prompt("Extract signals.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    assert result.output_keys == ["ENTITIES_OUTPUT", "CLAIMS_OUTPUT", "SIGNALS_OUTPUT"]
    system = result.messages[0].content
    for key in result.output_keys:
        assert key in system


# ---- Global M-1 hint (parallel mode) ----

def test_parallel_path_m1_budgeted_global_single_hint():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
        ("signals",  _prompt("Extract signals.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="budgeted"
    )
    system = result.messages[0].content
    # Parallel-mode budget (250 words), NOT the linear 800
    assert "Target approximately 250 words" in system
    assert "Target approximately 800 words" not in system
    # Hint appears ONCE in the preamble, not per-section
    assert system.count("Target approximately 250 words") == 1


def test_parallel_path_m1_budgeted_adaptive_clamps_to_parallel_ceiling():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    # Adaptive targets derived from FINE — 800 tokens/leaf → ~600 words →
    # but must clamp to parallel ceiling (250).
    result = compiler.compile_compound(
        compound,
        task_input="topic",
        merged_output_structure="budgeted_adaptive",
        per_agent_budgets={"entities": 800, "claims": 800},
    )
    system = result.messages[0].content
    # Clamped — NOT 600 words
    assert "Target approximately 250 words" in system
    assert "600 words" not in system


def test_parallel_path_m1_reinforced_single_hint():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="reinforced"
    )
    system = result.messages[0].content
    # Reinforced attention-redirect present, but only ONCE globally
    assert "Fully address all requirements" in system
    assert system.count("Fully address all requirements") == 1


def test_parallel_path_m1_none_emits_no_budget_hint():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="none"
    )
    system = result.messages[0].content
    assert "Target approximately" not in system
    assert "Fully address all requirements" not in system


# ---- Regression guard: linear path unchanged ----

def test_sequential_compound_preserves_legacy_framing():
    # Implicit linear deps → must hit the legacy path (== PHASE N == markers,
    # "compound task with N sequential phases" preamble, per-phase output
    # instructions). Protects Track A M-1 wins on due_diligence/code_review.
    compound = _compound(
        ("researcher", "Research the topic."),
        ("fact_checker", "Verify the facts."),
        ("summarizer", "Summarize findings."),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="budgeted",
    )
    system = result.messages[0].content
    # Legacy markers present
    assert "compound task with 3 sequential phases" in system
    assert "== PHASE 1:" in system
    assert "== PHASE 2:" in system
    assert "== PHASE 3:" in system
    # Legacy per-phase M-1 hint (800 words) unchanged
    assert "Target approximately 800 words" in system
    # Parallel markers absent
    assert "--- Section 1:" not in system
    assert "independent extractions" not in system


def test_sequential_compound_m1_still_per_phase():
    # Per-phase firing is the Track A win — must not regress.
    compound = _compound(
        ("researcher", "Research."),
        ("fact_checker", "Verify."),
        ("summarizer", "Summarize."),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="budgeted",
    )
    system = result.messages[0].content
    # Budgeted hint fires 3x — once per phase.
    assert system.count("Target approximately 800 words") == 3


# ---- Token accounting: parallel path strictly lower on shared-source shape ----

def test_shared_source_deduped_on_both_parallel_and_sequential_paths():
    # Same 3 leaves, same prompts — one as parallel-independent, one as
    # implicit-linear. After the P0-linear extension (2026-04-09), BOTH
    # paths dedup the shared bundle to a single SOURCE MATERIAL block.
    # Parallel pays a slightly larger per-prompt coordination cost for
    # its parallel preamble + no-preamble directive, which is acceptable
    # — the goal of P0 is dedup, not absolute token minimization.
    names_and_prompts = [
        ("entities", _prompt("Extract entities as a numbered list.")),
        ("claims",   _prompt("Extract claims as a numbered list.")),
        ("signals",  _prompt("Extract signals as a numbered list.")),
    ]
    parallel = _parallel_compound(*names_and_prompts)
    sequential = _compound(*names_and_prompts)
    compiler = PromptCompiler(StubAdapter())
    p_result = compiler.compile_compound(parallel,   task_input="brief Stripe")
    s_result = compiler.compile_compound(sequential, task_input="brief Stripe")
    # Core invariant: shared bundle appears exactly once in BOTH paths.
    assert s_result.messages[0].content.count("SOURCE MATERIAL:") == 1
    assert p_result.messages[0].content.count("SOURCE MATERIAL:") == 1
    # Both paths must emit a SHARED CONTEXT: block (hoisting triggered).
    assert "SHARED CONTEXT:" in s_result.messages[0].content
    assert "SHARED CONTEXT:" in p_result.messages[0].content
    # Both paths must still produce all three extractor instructions.
    for out in (s_result, p_result):
        sys = out.messages[0].content
        assert "Extract entities" in sys
        assert "Extract claims" in sys
        assert "Extract signals" in sys


def test_parallel_path_coordination_tokens_nonzero_and_below_total():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
        ("signals",  _prompt("Extract signals.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    assert result.coordination_tokens > 0
    assert result.coordination_tokens < result.estimated_tokens


# ---- task_input / prior_outputs behaviour unchanged ----

def test_parallel_path_task_input_in_user_message():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="brief Stripe")
    assert "brief Stripe" in result.messages[1].content


# ---------------------------------------------------------------------------
# PE P0 (linear-path extension, 2026-04-09) — shared-prefix hoisting on the
# legacy sequential-framing path. These tests verify that the P0 extension
# catches shared-bundle linear compounds while leaving every other aspect
# of the legacy framing (phase markers, phase chaining, per-phase M-1)
# completely unchanged.
# ---------------------------------------------------------------------------

def test_linear_shared_prefix_hoisted_sequential_path():
    # 3-phase implicit-linear compound with the same bundle baked into
    # every phase's system_prompt. After P0-linear, the bundle must be
    # hoisted to a single SHARED CONTEXT: block.
    compound = _compound(
        ("researcher",  _prompt("Research the subject.")),
        ("fact_checker", _prompt("Verify the researcher's claims.")),
        ("summarizer",  _prompt("Summarize the verified findings.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # Dedup: bundle appears exactly once.
    assert system.count("SOURCE MATERIAL:") == 1
    assert system.count("SHARED CONTEXT:") == 1
    # Per-phase instructions still all present.
    assert "Research the subject" in system
    assert "Verify the researcher's claims" in system
    assert "Summarize the verified findings" in system


def test_linear_shared_prefix_preserves_legacy_framing():
    # Hoisting must NOT change any other aspect of the legacy framing:
    # preamble, phase headers, phase-chaining directives, per-phase M-1.
    compound = _compound(
        ("researcher",  _prompt("Research.")),
        ("fact_checker", _prompt("Verify.")),
        ("summarizer",  _prompt("Summarize.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound, task_input="topic", merged_output_structure="budgeted",
    )
    system = result.messages[0].content
    # Legacy preamble
    assert "compound task with 3 sequential phases" in system
    # Phase markers
    assert "== PHASE 1:" in system
    assert "== PHASE 2:" in system
    assert "== PHASE 3:" in system
    # Phase chaining (these compounds have implicit linear deps — chaining
    # directives are correct and must remain).
    assert "RESEARCHER_OUTPUT" in system
    assert "Phase 1" in system or "from Phase 1" in system
    # Per-phase M-1 budgeted hint — still fires 3x (Track A quality win).
    assert system.count("Target approximately 800 words") == 3
    # Parallel markers must NOT appear.
    assert "--- Section 1:" not in system
    assert "independent extractions" not in system


def test_linear_without_shared_prefix_unchanged():
    # Legacy compounds with genuinely different per-phase prompts (the
    # shipping case: due_diligence, code_review, long_chain_research)
    # must see NO hoisting — the SHARED CONTEXT: block must be absent
    # and the system prompts must appear in full inside their phase
    # blocks. This is the regression guard that protects every existing
    # Track A result.
    compound = _compound(
        ("researcher",   "You are a researcher. Gather facts about the topic."),
        ("fact_checker", "You are a fact-checker. Verify claims against sources."),
        ("summarizer",   "You are a summarizer. Condense findings into 3 bullets."),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # No hoist
    assert "SHARED CONTEXT:" not in system
    # Full per-phase prompts present
    assert "You are a researcher." in system
    assert "You are a fact-checker." in system
    assert "You are a summarizer." in system
    # Legacy framing preserved
    assert "compound task with 3 sequential phases" in system
    assert "== PHASE 1:" in system


def test_linear_hoist_skipped_below_min_chars():
    # Two short prompts with a 50-char shared prefix — below the 200-char
    # threshold, so no hoist.
    shared = "Context: topic X. "  # 17 chars
    compound = _compound(
        ("a", shared + "Do task A extensively with detailed reasoning."),
        ("b", shared + "Do task B extensively with detailed reasoning."),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    assert "SHARED CONTEXT:" not in system
    # Both full prompts (with the short prefix inline) are emitted.
    assert system.count("Context: topic X.") == 2


def test_linear_shared_prefix_token_savings_when_bundle_is_large():
    # With a bundle, the hoisted version must be strictly cheaper than
    # a synthetic "no hoist" baseline where every phase duplicates it.
    # We approximate the baseline by counting "SOURCE MATERIAL:" and
    # asserting the dedup happened.
    compound = _compound(
        ("researcher",   _prompt("Research. Write a 2-paragraph summary.")),
        ("fact_checker", _prompt("Fact-check the research.")),
        ("summarizer",   _prompt("Summarize in 3 bullets.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(compound, task_input="topic")
    system = result.messages[0].content
    # Dedup verification: bundle appears 1x, not 3x.
    assert system.count("SOURCE MATERIAL:") == 1
    # Bundle content (the Stripe description) must still be present.
    assert "Stripe" in system


def test_parallel_path_prior_outputs_injected_into_user_message():
    compound = _parallel_compound(
        ("entities", _prompt("Extract entities.")),
        ("claims",   _prompt("Extract claims.")),
    )
    compiler = PromptCompiler(StubAdapter())
    result = compiler.compile_compound(
        compound,
        task_input="topic",
        prior_outputs={"EXTERNAL_OUTPUT": "some upstream data"},
    )
    user = result.messages[1].content
    assert "EXTERNAL_OUTPUT" in user
    assert "some upstream data" in user
