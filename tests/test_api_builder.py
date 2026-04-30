"""Tests for api/builder.py — builder model and validation."""
import pytest
from agentic_capsules.api.builder import Pipeline
from agentic_capsules.api.tool import Tool
from agentic_capsules.controller.policy import ControllerPolicy


def _tool():
    return Tool("search", "Search.", {"q": "str"}, fn=lambda a: {})


# ---------------------------------------------------------------------------
# Method chaining returns Pipeline
# ---------------------------------------------------------------------------

def test_group_returns_pipeline():
    p = Pipeline("test")
    assert p.group("g") is p


def test_agent_returns_pipeline():
    p = Pipeline("test").group("g")
    assert p.agent("a", "do stuff") is p


def test_full_chain_returns_pipeline():
    p = (
        Pipeline("test")
        .group("research")
            .agent("researcher", "Find facts.")
            .agent("verifier",   "Verify.")
        .group("writing")
            .agent("writer", "Write.")
    )
    assert isinstance(p, Pipeline)


# ---------------------------------------------------------------------------
# Group and agent ordering
# ---------------------------------------------------------------------------

def test_groups_in_declaration_order():
    p = Pipeline("test").group("a").agent("x","g").group("b").agent("y","g")
    assert [g.name for g in p._groups] == ["a", "b"]


def test_agents_in_declaration_order():
    p = (Pipeline("test").group("g")
         .agent("a1", "first").agent("a2", "second").agent("a3", "third"))
    assert [a.name for a in p._groups[0].agents] == ["a1", "a2", "a3"]


def test_second_group_auto_closes_first():
    p = (Pipeline("test")
         .group("research").agent("r", "research")
         .group("writing").agent("w", "write"))
    assert len(p._groups) == 2
    assert p._groups[0].name == "research"
    assert p._groups[1].name == "writing"


def test_agents_attached_to_correct_group():
    p = (Pipeline("test")
         .group("g1").agent("a", "g1 agent")
         .group("g2").agent("b", "g2 agent"))
    assert p._groups[0].agents[0].name == "a"
    assert p._groups[1].agents[0].name == "b"


# ---------------------------------------------------------------------------
# Tools and model per agent
# ---------------------------------------------------------------------------

def test_tools_attached_to_agent_spec():
    t = _tool()
    p = Pipeline("test").group("g").agent("a", "goal", tools=[t])
    assert p._groups[0].agents[0].tools == [t]


def test_model_attached_to_agent_spec():
    p = Pipeline("test").group("g").agent("a", "goal", model="claude-opus-4-6")
    assert p._groups[0].agents[0].model == "claude-opus-4-6"


def test_model_defaults_to_none():
    p = Pipeline("test").group("g").agent("a", "goal")
    assert p._groups[0].agents[0].model is None


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

def test_pipeline_name_stored():
    assert Pipeline("research")._name == "research"


def test_pipeline_name_stripped():
    assert Pipeline("  test  ")._name == "test"


def test_sensitivity_balanced_by_default():
    p = Pipeline("test")
    assert p._policy.compose_at == 0.23  # T-021/T-032: lowered from 0.36 after removing w3 bias


def test_sensitivity_aggressive():
    p = Pipeline("test", sensitivity="aggressive")
    assert p._policy.compose_at == 0.18  # T-021/T-032: lowered from 0.25 after removing w3 bias


def test_explicit_policy_overrides_sensitivity():
    custom = ControllerPolicy(compose_at=0.35)
    p = Pipeline("test", sensitivity="aggressive", policy=custom)
    assert p._policy.compose_at == 0.35


# ---------------------------------------------------------------------------
# Validation — error cases
# ---------------------------------------------------------------------------

def test_empty_pipeline_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        Pipeline("")


def test_empty_group_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        Pipeline("test").group("")


def test_empty_agent_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        Pipeline("test").group("g").agent("", "goal")


def test_empty_agent_goal_raises():
    with pytest.raises(ValueError, match="goal cannot be empty"):
        Pipeline("test").group("g").agent("a", "")


def test_agent_without_group_raises():
    with pytest.raises(ValueError, match="Call .group()"):
        Pipeline("test").agent("a", "goal")


def test_run_with_no_groups_raises():
    from unittest.mock import MagicMock
    p = Pipeline("test")
    with pytest.raises(ValueError, match="no groups defined"):
        p.run("task", adapter=MagicMock())


def test_run_with_empty_group_raises():
    from unittest.mock import MagicMock
    p = Pipeline("test").group("g")   # no agents added
    with pytest.raises(ValueError, match="has no agents"):
        p.run("task", adapter=MagicMock())


def test_run_with_invalid_mode_raises():
    from unittest.mock import MagicMock
    p = Pipeline("test").group("g").agent("a", "goal")
    with pytest.raises(ValueError, match="mode must be one of"):
        p.run("task", adapter=MagicMock(), mode="invalid")


# ---------------------------------------------------------------------------
# depends_on — explicit dependency declaration
# ---------------------------------------------------------------------------

def test_depends_on_defaults_to_none():
    p = Pipeline("t").group("g").agent("a", "goal")
    assert p._groups[0].agents[0].depends_on is None


def test_depends_on_empty_list_stored():
    p = Pipeline("t").group("g").agent("a", "goal", depends_on=[])
    assert p._groups[0].agents[0].depends_on == []


def test_depends_on_single_dep_stored():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a1", "first")
            .agent("a2", "second", depends_on=["a1"])
    )
    assert p._groups[0].agents[1].depends_on == ["a1"]


def test_depends_on_multiple_deps_preserves_order():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a1", "first")
            .agent("a2", "second")
            .agent("a3", "third", depends_on=["a1", "a2"])
    )
    assert p._groups[0].agents[2].depends_on == ["a1", "a2"]


def test_depends_on_dedupes_silently():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a1", "first")
            .agent("a2", "second", depends_on=["a1", "a1"])
    )
    assert p._groups[0].agents[1].depends_on == ["a1"]


def test_depends_on_strips_whitespace():
    p = (
        Pipeline("t")
        .group("g")
            .agent("a1", "first")
            .agent("a2", "second", depends_on=["  a1  "])
    )
    assert p._groups[0].agents[1].depends_on == ["a1"]


def test_depends_on_unknown_name_raises():
    with pytest.raises(ValueError, match="not an agent declared earlier"):
        (
            Pipeline("t")
            .group("g")
                .agent("a1", "first", depends_on=["ghost"])
        )


def test_depends_on_self_reference_raises():
    with pytest.raises(ValueError, match="cannot depend on itself"):
        (
            Pipeline("t")
            .group("g")
                .agent("a1", "first", depends_on=["a1"])
        )


def test_depends_on_forward_reference_raises():
    # a1 cannot depend on a2 because a2 hasn't been declared yet
    with pytest.raises(ValueError, match="not an agent declared earlier"):
        (
            Pipeline("t")
            .group("g")
                .agent("a1", "first", depends_on=["a2"])
                .agent("a2", "second")
        )


def test_depends_on_cross_group_reference_raises():
    # b1 is in a different group; should not be visible
    with pytest.raises(ValueError, match="not an agent declared earlier"):
        (
            Pipeline("t")
            .group("g1").agent("a1", "first")
            .group("g2").agent("b1", "second", depends_on=["a1"])
        )


def test_depends_on_empty_string_raises():
    with pytest.raises(ValueError, match="non-empty agent names"):
        (
            Pipeline("t")
            .group("g")
                .agent("a1", "first")
                .agent("a2", "second", depends_on=[""])
        )


# ---------------------------------------------------------------------------
# Per-group policy override
# ---------------------------------------------------------------------------

def test_group_without_override_inherits_pipeline_policy():
    base = ControllerPolicy(quality_floor=0.80)
    p = Pipeline("t", policy=base).group("g1").agent("a", "do")
    assert p.effective_policy("g1") is base


def test_group_with_override_uses_override():
    base   = ControllerPolicy(quality_floor=0.80)
    strict = ControllerPolicy(quality_floor=0.90)
    p = (
        Pipeline("t", policy=base)
        .group("g1", policy=strict).agent("a", "do")
    )
    assert p.effective_policy("g1") is strict


def test_override_does_not_leak_to_other_groups():
    base   = ControllerPolicy(quality_floor=0.80)
    strict = ControllerPolicy(quality_floor=0.90)
    p = (
        Pipeline("t", policy=base)
        .group("g1", policy=strict).agent("a", "do")
        .group("g2").agent("b", "do")
    )
    assert p.effective_policy("g1") is strict
    assert p.effective_policy("g2") is base


def test_unknown_group_name_falls_back_to_pipeline_policy():
    base = ControllerPolicy(quality_floor=0.80)
    p = Pipeline("t", policy=base).group("g1").agent("a", "do")
    assert p.effective_policy("does_not_exist") is base


def test_override_registers_with_pipeline_state():
    """Per-group policy is also visible via state._effective_policy for
    state-internal call sites (record_and_maybe_switch, etc.)."""
    base   = ControllerPolicy(quality_floor=0.80)
    strict = ControllerPolicy(quality_floor=0.90)
    p = (
        Pipeline("t", policy=base)
        .group("g1", policy=strict).agent("a", "do")
        .group("g2").agent("b", "do")
    )
    assert p._pipeline_state._effective_policy("g1") is strict
    assert p._pipeline_state._effective_policy("g2") is base


def test_override_is_keyword_only():
    """policy= must be passed by keyword — prevents accidental positional
    misuse where adapter/depends_on positions shift."""
    with pytest.raises(TypeError):
        # Attempt positional: group(name, adapter, depends_on, policy) — 4 positional
        Pipeline("t").group("g", None, None, ControllerPolicy())  # noqa: E501


def test_dataclass_replace_pattern_for_partial_override():
    """Documented operator pattern: dataclasses.replace the pipeline policy
    for fields that differ, pass the result as the group override. The
    override is a full ControllerPolicy — every field resolves from the
    override, not merged with pipeline defaults."""
    from dataclasses import replace
    base  = ControllerPolicy(quality_floor=0.80, compose_at=0.23)
    synth = replace(base, quality_floor=0.65)   # override one field only
    p = (
        Pipeline("t", policy=base)
        .group("research").agent("a", "do")
        .group("synthesis", policy=synth).agent("b", "do")
    )
    # synthesis group uses the override (lower floor); research uses base
    assert p.effective_policy("synthesis").quality_floor == 0.65
    assert p.effective_policy("research").quality_floor  == 0.80
    # Other fields in synth's policy come from base (via replace)
    assert p.effective_policy("synthesis").compose_at == 0.23
