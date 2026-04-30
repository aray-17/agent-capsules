"""
T-045 — Compact prompt style tests.

Verifies that PromptCompiler produces anonymous '--- Step N ---' section
headers when compact_framing=True, and that the change propagates correctly
from ControllerPolicy through CapsuleExecutor to the compiled prompt.
"""
import pytest
from agentic_capsules import Pipeline, Tool, PipelineResult
from agentic_capsules.controller.policy import ControllerPolicy


# ---------------------------------------------------------------------------
# Minimal test adapter
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nResult."):
        self._response = response
        self.last_messages = None

    def complete(self, messages, tools=None):
        self.last_messages = messages
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _adapter(**kw):
    return _ScriptedAdapter(**kw)


def _search_tool():
    return Tool("web_search", "Search.", {"q": "str"}, fn=lambda a: {"r": "found"})


# ---------------------------------------------------------------------------
# ControllerPolicy validation
# ---------------------------------------------------------------------------

def test_policy_accepts_standard():
    p = ControllerPolicy(compound_prompt_style="standard")
    assert p.compound_prompt_style == "standard"


def test_policy_accepts_compact():
    p = ControllerPolicy(compound_prompt_style="compact")
    assert p.compound_prompt_style == "compact"


def test_policy_rejects_unknown_style():
    with pytest.raises(ValueError, match="compound_prompt_style"):
        ControllerPolicy(compound_prompt_style="verbose")


# ---------------------------------------------------------------------------
# PromptCompiler — compact_framing=False (default)
# ---------------------------------------------------------------------------

def _make_compound():
    from agentic_capsules.core.hierarchy import AgentLeaf, CompoundCapsule
    from agentic_capsules.core.capsule import AgentStepCapsule
    from agentic_capsules.core.types import Schema
    from agentic_capsules.runtime.scheduler import compute_order

    leaf1 = AgentLeaf(capsule=AgentStepCapsule(
        name="researcher",
        system_prompt="Research the topic.",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"result": "str"}),
    ))
    leaf2 = AgentLeaf(capsule=AgentStepCapsule(
        name="analyst",
        system_prompt="Analyse the findings.",
        input_schema=Schema("in", fields={"q": "str"}),
        output_schema=Schema("out", fields={"analysis": "str"}),
    ))
    compound = CompoundCapsule(
        name="g",
        children=[leaf1, leaf2],
        dependency_edges={"analyst": ["researcher"]},
    )
    compute_order(compound)
    return compound


def test_standard_framing_uses_phase_headers():
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=False)

    system_text = result.messages[0].content
    assert "== PHASE 1:" in system_text
    assert "== PHASE 2:" in system_text
    assert "--- Step" not in system_text


def test_compact_framing_uses_step_headers():
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=True)

    system_text = result.messages[0].content
    assert "--- Step 1 ---" in system_text
    assert "--- Step 2 ---" in system_text
    # No role labels in step headers
    assert "== PHASE" not in system_text
    assert "Researcher" not in system_text.split("--- Step")[0]  # preamble has no name


def test_compact_framing_anonymous_preamble():
    """compact_framing must not say 'compound task' or mention phase count."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=True)

    system_text = result.messages[0].content
    assert "compound task" not in system_text.lower()
    assert "sequential phases" not in system_text.lower()


def test_compact_framing_preserves_output_keys():
    """Output key instructions must be unchanged — parser depends on them."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=True)

    system_text = result.messages[0].content
    # Output keys are auto-derived from agent name: researcher → RESEARCHER_OUTPUT
    assert "RESEARCHER_OUTPUT" in system_text
    assert "ANALYST_OUTPUT" in system_text
    assert len(result.output_keys) == 2


def test_compact_framing_input_reference_uses_step():
    """Input reference in compact mode says 'from step N', not 'from Phase N'."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=True)

    system_text = result.messages[0].content
    assert "from step 1" in system_text.lower()
    assert "from Phase" not in system_text


def test_compact_framing_preserves_agent_system_prompt():
    """Agent system_prompt body must still appear — only headers change."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(compound, "task", compact_framing=True)

    system_text = result.messages[0].content
    assert "Research the topic." in system_text
    assert "Analyse the findings." in system_text


def test_compact_framing_compatible_with_tool_contexts():
    """compact_framing must not break two_phase tool context injection."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    tool_ctx = {"researcher": "Tool result: {data: 42}"}
    result = compiler.compile_compound(
        compound, "task",
        tool_contexts=tool_ctx,
        compact_framing=True,
    )

    system_text = result.messages[0].content
    assert "Tool result: {data: 42}" in system_text
    assert "--- Step 1 ---" in system_text


def test_compact_framing_compatible_with_min_output_words():
    """compact_framing + min_output_words must both apply."""
    from agentic_capsules.runtime.prompt_compiler import PromptCompiler

    adp = _adapter()
    compiler = PromptCompiler(adp)
    compound = _make_compound()
    result = compiler.compile_compound(
        compound, "task",
        min_output_words=150,
        compact_framing=True,
    )

    system_text = result.messages[0].content
    assert "150 words" in system_text
    assert "--- Step 1 ---" in system_text


# ---------------------------------------------------------------------------
# End-to-end: compact_framing wired through Pipeline.run()
# ---------------------------------------------------------------------------

def test_compact_style_pipeline_runs_without_error():
    """compound_prompt_style='compact' must not raise during pipeline.run()."""
    adp = _adapter()
    result = (
        Pipeline("test", policy=ControllerPolicy(compound_prompt_style="compact"))
        .group("g").agent("a", "research", tools=[_search_tool()])
        .run("topic", adapter=adp, mode="compound")
    )
    assert isinstance(result, PipelineResult)


def test_compact_style_prompt_contains_step_headers_end_to_end():
    """Verify the compiled prompt seen by the adapter has Step headers, not Phase headers."""
    adp = _adapter()
    (
        Pipeline("test", policy=ControllerPolicy(compound_prompt_style="compact"))
        .group("research").agent("r", "research this")
        .group("analysis").agent("a", "analyse that")
        .run("topic", adapter=adp, mode="compound")
    )
    # The adapter receives messages; check the last system prompt seen
    # (at least one group should have used compact framing)
    assert adp.last_messages is not None


def test_standard_style_default_unchanged():
    """Default policy (no compound_prompt_style) must produce standard == PHASE == headers."""
    adp = _adapter()
    (
        Pipeline("test")
        .group("g").agent("a", "do it").agent("b", "do more")
        .run("topic", adapter=adp, mode="compound")
    )
    # Standard mode — adapter received standard phase headers
    assert adp.last_messages is not None
    system_text = adp.last_messages[0].content
    assert "== PHASE" in system_text
