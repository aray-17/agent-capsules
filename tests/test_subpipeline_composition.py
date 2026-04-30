"""
Tests for G-5 — subpipeline composition (``Pipeline.subpipeline(other, …)``).

G-5 adds a public builder API for embedding one :class:`Pipeline` into
another as a reusable unit. At build time, the embedded pipeline's
groups are cloned into the parent with namespaced names
(``"{prefix}/{group}"``, ``"{prefix}/{agent}"``); intra-subpipeline
``depends_on`` entries are rewritten to the prefixed names; roots
(``depends_on=[]``) inherit the parent-side ``.subpipeline(depends_on=...)``
argument. The flattened groups then go through the normal
``_compile_group`` path so FINE/COMPOUND duality, the composition
controller, G-4 checkpointing, G-6 dynamic fan-out, and the parallel
executor all apply identically to embedded groups.

Coverage:
  * Builder validation: non-Pipeline rejection, self-embed rejection,
    empty subpipeline rejection, empty/invalid prefix, slash in prefix,
    group name collisions, agent name collisions, bad depends_on
  * Flattening: prefixed group names, prefixed agent names, intra-group
    agent depends_on rewrites, intra-subpipeline group depends_on
    rewrites, None (implicit-linear) preservation, [] (root) inheriting
    parent-side depends_on
  * Runtime execution (serial): full run end-to-end, parent downstream
    group sees the subpipeline's final output
  * Runtime execution (parallel): same semantics under parallel=True
  * Fan-out inside subpipeline: embedded ``fanout_group`` expands with
    its prefixed source agent
  * Nested subpipelines: re-embedding a pipeline that already has an
    embedded subpipeline — prefixes stack cleanly

All tests use stub adapters; no live APIs are touched.
"""
from __future__ import annotations

import threading

import pytest

from agentic_capsules import Pipeline
from agentic_capsules.api.builder import _FanoutGroupSpec, _GroupSpec


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    """Returns successive scripted responses in call order, then loops."""
    context_window = 200_000

    def __init__(self, responses: list[str] | None = None,
                 default: str = "OUTPUT:\ndone.") -> None:
        self._responses = list(responses) if responses else []
        self._default   = default
        self._lock      = threading.Lock()
        self.call_count = 0
        self.captured_system: list[str] = []

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
            idx = self.call_count - 1
            for m in messages:
                role    = getattr(m, "role",    None)
                content = getattr(m, "content", None) or ""
                if role == "system":
                    self.captured_system.append(content)
                    break
        if idx < len(self._responses):
            return self._responses[idx]
        return self._default

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_research_subpipeline() -> Pipeline:
    """Two-group subpipeline used as the standard embed target in tests."""
    return (
        Pipeline("research")
        .group("retrieve")
            .agent("retriever", "Find papers about the topic.")
        .group("extract")
            .agent("extractor", "Pull claims from the retrieved papers.")
    )


# ---------------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------------

class TestBuilderValidation:

    def test_rejects_non_pipeline(self):
        with pytest.raises(ValueError, match="must be a Pipeline instance"):
            Pipeline("parent").subpipeline("not a pipeline")  # type: ignore[arg-type]

    def test_rejects_self_embed(self):
        p = Pipeline("parent").group("g").agent("a", "do")
        with pytest.raises(ValueError, match="cannot embed a pipeline into itself"):
            p.subpipeline(p)

    def test_rejects_empty_subpipeline(self):
        other = Pipeline("empty")  # no groups
        with pytest.raises(ValueError, match="has no groups defined"):
            Pipeline("parent").subpipeline(other)

    def test_rejects_empty_prefix(self):
        other = _make_research_subpipeline()
        with pytest.raises(ValueError, match="name_prefix cannot be empty"):
            Pipeline("parent").subpipeline(other, name_prefix="   ")

    def test_rejects_slash_in_prefix(self):
        other = _make_research_subpipeline()
        with pytest.raises(ValueError, match="cannot contain '/'"):
            Pipeline("parent").subpipeline(other, name_prefix="a/b")

    def test_rejects_group_name_collision(self):
        # Parent has a group "research/retrieve" (unusual but legal)
        parent = (
            Pipeline("parent")
            .group("research/retrieve")
                .agent("x", "do")
        )
        with pytest.raises(ValueError, match="collides with an existing group"):
            parent.subpipeline(_make_research_subpipeline())

    def test_rejects_agent_name_collision(self):
        # Parent has an agent already named "research/retriever"
        parent = (
            Pipeline("parent")
            .group("g")
                .agent("research/retriever", "do")
        )
        with pytest.raises(ValueError, match="collides with an existing"):
            parent.subpipeline(_make_research_subpipeline())

    def test_rejects_bad_depends_on_reference(self):
        other = _make_research_subpipeline()
        with pytest.raises(ValueError, match="is not a group declared earlier"):
            Pipeline("parent").subpipeline(other, depends_on=["no_such_group"])

    def test_rejects_non_string_depends_on_entry(self):
        other = _make_research_subpipeline()
        with pytest.raises(ValueError, match="must be non-empty group names"):
            Pipeline("parent").subpipeline(other, depends_on=[""])

    def test_multiple_subpipelines_require_distinct_prefixes(self):
        other = _make_research_subpipeline()
        with pytest.raises(ValueError, match="collides with an existing group"):
            (
                Pipeline("parent")
                .subpipeline(other)
                .subpipeline(other)  # same prefix (default = "research") → collides
            )

    def test_multiple_subpipelines_with_distinct_prefixes_ok(self):
        other = _make_research_subpipeline()
        parent = (
            Pipeline("parent")
            .subpipeline(other, name_prefix="r1")
            .subpipeline(other, name_prefix="r2")
        )
        assert [g.name for g in parent._groups] == [
            "r1/retrieve", "r1/extract",
            "r2/retrieve", "r2/extract",
        ]

    def test_custom_prefix_is_respected(self):
        parent = Pipeline("parent").subpipeline(
            _make_research_subpipeline(), name_prefix="phase1"
        )
        assert [g.name for g in parent._groups] == ["phase1/retrieve", "phase1/extract"]

    def test_embedding_is_deep_copied(self):
        """Mutating the source pipeline after embedding must not affect the parent."""
        other = _make_research_subpipeline()
        parent = Pipeline("parent").subpipeline(other, name_prefix="sub")
        # Mutate the original afterwards:
        other.group("new_group").agent("new_agent", "do")
        # Parent should still have exactly 2 groups from the embed.
        assert len(parent._groups) == 2


# ---------------------------------------------------------------------------
# Flattening correctness
# ---------------------------------------------------------------------------

class TestFlattening:

    def test_prefixed_group_names(self):
        parent = Pipeline("parent").subpipeline(
            _make_research_subpipeline(), name_prefix="ns"
        )
        names = [g.name for g in parent._groups]
        assert names == ["ns/retrieve", "ns/extract"]

    def test_prefixed_agent_names(self):
        parent = Pipeline("parent").subpipeline(
            _make_research_subpipeline(), name_prefix="ns"
        )
        retrieve = [g for g in parent._groups if g.name == "ns/retrieve"][0]
        extract  = [g for g in parent._groups if g.name == "ns/extract"][0]
        assert [a.name for a in retrieve.agents] == ["ns/retriever"]
        assert [a.name for a in extract.agents]  == ["ns/extractor"]

    def test_agent_goal_is_preserved_verbatim(self):
        """Agent goals must not be mutated by embedding — they're plain strings."""
        parent = Pipeline("parent").subpipeline(
            _make_research_subpipeline(), name_prefix="ns"
        )
        extract = [g for g in parent._groups if g.name == "ns/extract"][0]
        assert extract.agents[0].goal == "Pull claims from the retrieved papers."

    def test_intra_agent_depends_on_rewritten(self):
        other = (
            Pipeline("sub")
            .group("g")
                .agent("a", "root", depends_on=[])
                .agent("b", "uses a", depends_on=["a"])
        )
        parent = Pipeline("parent").subpipeline(other, name_prefix="ns")
        g = parent._groups[0]
        assert g.agents[0].name == "ns/a"
        assert g.agents[0].depends_on == []
        assert g.agents[1].name == "ns/b"
        assert g.agents[1].depends_on == ["ns/a"]

    def test_intra_group_depends_on_rewritten(self):
        # Subpipeline has a DAG of groups: A, B, C where C depends on both A and B
        other = (
            Pipeline("sub")
            .group("A", depends_on=[])
                .agent("a", "do")
            .group("B", depends_on=[])
                .agent("b", "do")
            .group("C", depends_on=["A", "B"])
                .agent("c", "do")
        )
        parent = (
            Pipeline("parent")
            .group("seed", depends_on=[])
                .agent("s", "do")
            .subpipeline(other, name_prefix="ns", depends_on=["seed"])
        )
        names_deps = [(g.name, g.depends_on) for g in parent._groups]
        assert names_deps == [
            ("seed",    []),
            # Roots inherit parent-side depends_on = ["seed"]
            ("ns/A",    ["seed"]),
            ("ns/B",    ["seed"]),
            # Internal edges get prefixed
            ("ns/C",    ["ns/A", "ns/B"]),
        ]

    def test_implicit_linear_depends_on_preserved(self):
        """Groups with depends_on=None stay None after embedding; the
        flattened declaration order handles the implicit linear chain
        naturally in the parent."""
        other = _make_research_subpipeline()  # both groups have depends_on=None
        parent = Pipeline("parent").subpipeline(other, name_prefix="ns")
        for g in parent._groups:
            assert g.depends_on is None

    def test_root_depends_on_without_parent_deps_stays_empty(self):
        other = (
            Pipeline("sub")
            .group("A", depends_on=[])
                .agent("a", "do")
        )
        parent = Pipeline("parent").subpipeline(other)  # no depends_on arg
        # Root with no parent-side deps stays as a root []
        assert parent._groups[0].depends_on == []

    def test_current_group_is_none_after_subpipeline(self):
        """After .subpipeline(), a subsequent .agent() call must error
        — the user has to explicitly open a new group (or another
        subpipeline / fan-out). This keeps subpipelines sealed."""
        parent = Pipeline("parent").subpipeline(_make_research_subpipeline())
        with pytest.raises(ValueError, match="before any .group"):
            parent.agent("stray", "do")

    def test_subpipeline_can_be_followed_by_group(self):
        parent = (
            Pipeline("parent")
            .subpipeline(_make_research_subpipeline(), name_prefix="ns")
            .group("synth")
                .agent("writer", "Combine the claims.")
        )
        assert [g.name for g in parent._groups] == [
            "ns/retrieve", "ns/extract", "synth",
        ]

    def test_policy_and_state_are_parent_side(self):
        """The subpipeline's own pipeline state must not leak into the
        parent — the parent's policy governs every flattened group."""
        from agentic_capsules.controller.policy import policy_for
        other = Pipeline("sub", sensitivity="aggressive").group("g").agent("a", "do")
        parent = Pipeline("parent", sensitivity="conservative").subpipeline(
            other, name_prefix="ns"
        )
        # Parent policy wins; sub's "aggressive" settings are discarded.
        assert parent._policy.compose_at == policy_for("conservative").compose_at


# ---------------------------------------------------------------------------
# Runtime execution — serial
# ---------------------------------------------------------------------------

class TestSerialExecution:

    def test_embedded_pipeline_runs_end_to_end(self):
        other = _make_research_subpipeline()
        parent = Pipeline("parent").subpipeline(other, name_prefix="ns")
        adapter = _ScriptedAdapter([
            "NS/RETRIEVER_OUTPUT:\npaper A, paper B",
            "NS/EXTRACTOR_OUTPUT:\nclaim X, claim Y",
        ])
        result = parent.run("AI safety", adapter=adapter, mode="fine")
        # 2 subpipeline agents = 2 LLM calls
        assert adapter.call_count == 2
        # The final output should reflect the last-group (ns/extract)
        # output — verified via step_outputs.
        assert "ns/retriever" in result.step_outputs
        assert "ns/extractor" in result.step_outputs

    def test_parent_downstream_group_sees_subpipeline_output(self):
        other = _make_research_subpipeline()
        captured: list[str] = []

        class _CaptureAdapter(_ScriptedAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                    idx = self.call_count - 1
                # Capture the user prompt of call #3 (the synthesizer)
                # where the subpipeline's final output should appear as
                # task augmentation.
                if idx == 2:
                    for m in messages:
                        if getattr(m, "role", None) == "user":
                            captured.append(getattr(m, "content", "") or "")
                            break
                if idx < len(self._responses):
                    return self._responses[idx]
                return self._default

        adapter = _CaptureAdapter([
            "NS/RETRIEVER_OUTPUT:\npaper-A",
            "NS/EXTRACTOR_OUTPUT:\nclaim-from-paper-A",
            "WRITER_OUTPUT:\nfinal brief",
        ])
        parent = (
            Pipeline("parent")
            .subpipeline(other, name_prefix="ns")
            .group("write")
                .agent("writer", "Draft a brief.")
        )
        parent.run("AI safety", adapter=adapter, mode="fine")
        assert adapter.call_count == 3
        # The synthesizer should have seen the subpipeline's last group
        # output in its task augmentation.
        synth_prompt = "\n".join(captured)
        assert "claim-from-paper-A" in synth_prompt

    def test_fanout_inside_subpipeline(self):
        """A subpipeline containing a fan-out group must have the
        fan-out's source reference prefix-rewritten and expand correctly
        at runtime."""
        sub = (
            Pipeline("rag")
            .group("retrieve")
                .agent("retriever", "Find docs.")
            .fanout_group(
                "analyze", source="retriever",
                item_extractor=lambda out: out.strip().split(","),
                worker_name="analyzer",
                worker_goal="Analyze: {item}",
            )
        )
        parent = Pipeline("parent").subpipeline(sub, name_prefix="rag1")
        # Verify the fan-out spec was prefixed correctly at build time
        fanout = [
            g for g in parent._groups if isinstance(g, _FanoutGroupSpec)
        ][0]
        assert fanout.name         == "rag1/analyze"
        assert fanout.source       == "rag1/retriever"
        assert fanout.worker_name  == "rag1/analyzer"

        # Runtime: source emits 3 comma-separated items → 3 worker dispatches
        adapter = _ScriptedAdapter([
            "RAG1/RETRIEVER_OUTPUT:\ndoc1,doc2,doc3",
        ], default="WORKER_OUTPUT:\nanalyzed.")
        parent.run("task", adapter=adapter, mode="fine")
        # 1 source + 3 workers
        assert adapter.call_count == 4

    def test_independent_subpipelines_produce_independent_outputs(self):
        """Two embeds of the same subpipeline with different prefixes
        must produce independent outputs (no shared state)."""
        other = Pipeline("sub").group("g").agent("a", "do")
        parent = (
            Pipeline("parent")
            .subpipeline(other, name_prefix="first")
            .subpipeline(other, name_prefix="second")
        )
        adapter = _ScriptedAdapter([
            "FIRST/A_OUTPUT:\nresult-one",
            "SECOND/A_OUTPUT:\nresult-two",
        ])
        result = parent.run("task", adapter=adapter, mode="fine")
        assert adapter.call_count == 2
        assert "first/a" in result.step_outputs
        assert "second/a" in result.step_outputs


# ---------------------------------------------------------------------------
# Runtime execution — parallel
# ---------------------------------------------------------------------------

class TestParallelExecution:

    def test_parallel_mode_runs_subpipeline(self):
        """Subpipelines are flattened into normal groups so the parallel
        executor should handle them with zero special-case code."""
        other = (
            Pipeline("sub")
            .group("A", depends_on=[])
                .agent("a", "do")
            .group("B", depends_on=[])
                .agent("b", "do")
        )
        parent = (
            Pipeline("parent")
            .group("seed", depends_on=[])
                .agent("s", "do")
            .subpipeline(other, name_prefix="ns", depends_on=["seed"])
        )
        adapter = _ScriptedAdapter([
            "S_OUTPUT:\nseed-result",
            "NS/A_OUTPUT:\na-result",
            "NS/B_OUTPUT:\nb-result",
        ])
        result = parent.run("task", adapter=adapter, mode="fine", parallel=True)
        assert adapter.call_count == 3
        assert "s"    in result.step_outputs
        assert "ns/a" in result.step_outputs
        assert "ns/b" in result.step_outputs


# ---------------------------------------------------------------------------
# Nested subpipelines
# ---------------------------------------------------------------------------

class TestNestedSubpipelines:

    def test_subpipeline_of_subpipeline(self):
        """A pipeline that already has an embedded subpipeline can itself
        be embedded — prefixes stack."""
        inner = Pipeline("inner").group("g").agent("a", "do")
        middle = Pipeline("middle").subpipeline(inner, name_prefix="in")
        outer = Pipeline("outer").subpipeline(middle, name_prefix="mid")

        names = [g.name for g in outer._groups]
        # The middle pipeline's groups were already named "in/g" after the
        # first embed; re-embedding into outer prefixes again to "mid/in/g".
        assert names == ["mid/in/g"]

        # And the agent name stacks the same way:
        assert [a.name for a in outer._groups[0].agents] == ["mid/in/a"]
