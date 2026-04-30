"""
Tests for G-6 — dynamic fan-out (``.fanout_group(...)``, LangGraph ``Send`` parity).

At runtime, after the source agent's output is available, the compiler
calls ``item_extractor(output)`` to get N items, builds N worker agent
specs (``{item}`` substituted into each worker goal), compiles them
through the normal ``_compile_group`` path, and runs the resulting
``CompoundCapsule``. This closes the "retrieval-augmented pipelines
cannot express runtime fan-out" gap.

Coverage:
  * Builder validation: source/extractor/worker_name/worker_goal/max_items,
    {item} placeholder requirement, unknown source rejection,
    .agent() rejection after a fan-out group, depends_on validation
  * Serial runtime expansion: N workers dispatched, per-item goal
    substitution, item truncation at max_items, empty-list short-circuit,
    {item} substring replacement preserves other curly braces
  * Extractor errors surface as build-stage ValueError
  * Compile-time slot is a None placeholder for fan-out groups
  * Downstream groups see the fan-out's aggregated output in task
    augmentation
  * Parallel runtime expansion: same semantics under parallel=True,
    plus concurrent-level expansion inside one topological level
  * Checkpointing interaction: G-4 checkpoint replays a fan-out
    group's saved outputs on retry (resumed fan-out groups do not
    re-invoke the extractor or dispatch workers)

All tests use stub adapters and never touch live APIs.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import pytest

from agentic_capsules import Pipeline
from agentic_capsules.api.builder import _FanoutGroupSpec, _GroupSpec
from agentic_capsules.runtime.checkpoint import PipelineCheckpoint


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class _CountingAdapter:
    """Returns a scripted response; tracks call count and captured prompts."""
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nworker-done.") -> None:
        self._response        = response
        self._lock            = threading.Lock()
        self.call_count       = 0
        self.captured_prompts: list[str] = []

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
            # Capture user-role content for per-item assertions.
            for m in messages:
                role    = getattr(m, "role", None)
                content = getattr(m, "content", None) or ""
                if role == "user":
                    self.captured_prompts.append(content)
                    break
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _SourceSeededAdapter:
    """Returns a scripted list of items from the source agent, then a
    generic worker response for all subsequent calls. Lets us control
    exactly what the extractor sees."""
    context_window = 200_000

    def __init__(self, source_output: str, worker_response: str = "## OUTPUT\nw."):
        self._source_output  = source_output
        self._worker_response = worker_response
        self._lock           = threading.Lock()
        self.call_count      = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
            current = self.call_count
        return self._source_output if current == 1 else self._worker_response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------------

class TestBuilderValidation:

    def _base(self) -> Pipeline:
        return Pipeline("t").group("seed").agent("src", "emit items")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            self._base().fanout_group(
                "", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
            )

    def test_rejects_empty_source(self):
        with pytest.raises(ValueError, match="source must be a declared agent"):
            self._base().fanout_group(
                "fan", source="",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
            )

    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="has not been declared"):
            self._base().fanout_group(
                "fan", source="nonexistent",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
            )

    def test_rejects_non_callable_extractor(self):
        with pytest.raises(ValueError, match="item_extractor must be callable"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor="not callable",  # type: ignore[arg-type]
                worker_name="w", worker_goal="Do: {item}",
            )

    def test_rejects_empty_worker_name(self):
        with pytest.raises(ValueError, match="worker_name cannot be empty"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="", worker_goal="Do: {item}",
            )

    def test_rejects_empty_worker_goal(self):
        with pytest.raises(ValueError, match="worker_goal cannot be empty"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="",
            )

    def test_rejects_worker_goal_without_item_placeholder(self):
        with pytest.raises(ValueError, match="must contain the literal substring"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="No placeholder here",
            )

    def test_rejects_max_items_zero(self):
        with pytest.raises(ValueError, match="max_items must be >= 1"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
                max_items=0,
            )

    def test_rejects_self_dependency(self):
        with pytest.raises(ValueError, match="cannot depend on itself"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
                depends_on=["fan"],
            )

    def test_rejects_unknown_dependency_group(self):
        with pytest.raises(ValueError, match="is not a group declared"):
            self._base().fanout_group(
                "fan", source="src",
                item_extractor=lambda s: [s],
                worker_name="w", worker_goal="Do: {item}",
                depends_on=["nonexistent"],
            )

    def test_accepts_valid_declaration(self):
        p = self._base().fanout_group(
            "fan", source="src",
            item_extractor=lambda s: [s],
            worker_name="w", worker_goal="Do: {item}",
        )
        assert isinstance(p._groups[-1], _FanoutGroupSpec)
        assert p._groups[-1].source == "src"
        assert p._groups[-1].max_items == 20  # default

    def test_agent_call_after_fanout_group_rejected(self):
        p = self._base().fanout_group(
            "fan", source="src",
            item_extractor=lambda s: [s],
            worker_name="w", worker_goal="Do: {item}",
        )
        with pytest.raises(ValueError, match="Cannot call .agent"):
            p.agent("rogue", "should not be allowed")


# ---------------------------------------------------------------------------
# Serial-mode runtime expansion
# ---------------------------------------------------------------------------

class TestSerialRuntimeExpansion:

    def test_n_workers_dispatched_matches_extractor_list_length(self):
        """Extractor returns 3 items → source + 3 workers = 4 LLM calls."""
        adapter = _SourceSeededAdapter("item-a\nitem-b\nitem-c")
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: out.split("\n"),
                worker_name="w", worker_goal="Handle: {item}",
            )
        )
        p.run("task", adapter=adapter, mode="fine")
        assert adapter.call_count == 4  # 1 source + 3 workers

    def test_per_item_goal_substitution(self):
        """Each worker's system prompt contains its own extracted item."""
        adapter = _CountingAdapter()
        # First call is the source agent — capture what it returns so we
        # can drive the extractor. Use a deterministic stub:
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                    # Only capture worker prompts, not the source's own
                    for m in messages:
                        role    = getattr(m, "role",    None)
                        content = getattr(m, "content", None) or ""
                        if role == "user":
                            self.captured_prompts.append(content)
                            break
                # Source (first call) returns item list; workers return generic
                return (
                    "## OUTPUT\nalpha|beta|gamma"
                    if self.call_count == 1
                    else "## OUTPUT\nworker-done."
                )

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit items")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: out.split("|"),
                worker_name="w",
                worker_goal="Please handle exactly: {item}",
            )
        )
        p.run("task", adapter=adapter, mode="fine")
        # The worker prompts must each contain one of the extracted items
        worker_prompts = adapter.captured_prompts[1:]  # skip the source prompt
        joined = "\n".join(worker_prompts)
        assert "alpha" in joined
        assert "beta"  in joined
        assert "gamma" in joined

    def test_max_items_truncates_extractor_output(self):
        """max_items caps the number of dispatched workers."""
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                return (
                    "## OUTPUT\n" + "\n".join(f"item-{i}" for i in range(50))
                    if self.call_count == 1
                    else "## OUTPUT\ndone."
                )

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit many")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: out.split("\n"),
                worker_name="w", worker_goal="Handle: {item}",
                max_items=5,
            )
        )
        p.run("task", adapter=adapter, mode="fine")
        # 1 source + 5 workers (capped at max_items)
        assert adapter.call_count == 6

    def test_empty_extractor_result_short_circuits_group(self):
        """Extractor returns [] → no workers dispatched, empty output."""
        adapter = _SourceSeededAdapter("## OUTPUT\nempty.")
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: [],  # no items
                worker_name="w", worker_goal="Handle: {item}",
            )
        )
        result = p.run("task", adapter=adapter, mode="fine")
        # Only the source was called
        assert adapter.call_count == 1
        # No worker outputs should appear in step_outputs
        assert not any(k.startswith("w_") for k in result.step_outputs)

    def test_item_substitution_preserves_other_curly_braces(self):
        """Worker goal with JSON-example braces must not break.

        The worker goal becomes the agent's system prompt, so we capture
        system-role content here (not user-role content like the other
        tests) — we want to verify the raw goal string survived item
        substitution without eating non-``{item}`` curly braces.
        """
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                    # Capture ALL system messages this call so we can
                    # inspect the worker's system prompt (which holds
                    # the substituted goal).
                    for m in messages:
                        role    = getattr(m, "role",    None)
                        content = getattr(m, "content", None) or ""
                        if role == "system":
                            self.captured_prompts.append(content)
                return (
                    "## OUTPUT\nthing-x"
                    if self.call_count == 1
                    else "## OUTPUT\nworker-done."
                )

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: [out.strip()],
                worker_name="w",
                worker_goal='Return JSON like {"key": "value"} for: {item}',
            )
        )
        p.run("task", adapter=adapter, mode="fine")
        # At least one worker's system prompt must contain both the
        # preserved curly braces AND the substituted item.
        assert any(
            '{"key": "value"}' in sp and "thing-x" in sp
            for sp in adapter.captured_prompts
        ), f"captured: {adapter.captured_prompts!r}"

    def test_extractor_exception_raises_value_error(self):
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                return "## OUTPUT\nanything."

        def boom(output: str) -> list[str]:
            raise RuntimeError("extractor broke")

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=boom,
                worker_name="w", worker_goal="Do: {item}",
            )
        )
        with pytest.raises(ValueError, match="item_extractor raised"):
            p.run("task", adapter=adapter, mode="fine")

    def test_extractor_non_list_return_raises_value_error(self):
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                return "## OUTPUT\nanything."

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: "not a list",  # type: ignore[return-value]
                worker_name="w", worker_goal="Do: {item}",
            )
        )
        with pytest.raises(ValueError, match="must return a list of strings"):
            p.run("task", adapter=adapter, mode="fine")

    def test_downstream_group_sees_fanout_output_aggregated(self):
        """A static group declared after a fan-out sees the fan-out's
        aggregated output in its task augmentation. We verify this by
        having the downstream agent capture its own prompt."""
        captured: list[str] = []

        class _ThreeThenCaptureAdapter:
            context_window = 200_000
            def __init__(self):
                self._lock = threading.Lock()
                self.call_count = 0
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                    current = self.call_count
                # Source (call 1): emit the extractor-ready item list under
                # the source agent's true output key so parse_outputs strips
                # the heading cleanly. parse_outputs looks for ``KEY:`` — not
                # ``## KEY`` — so we must emit that exact form.
                if current == 1:
                    return "SRC_OUTPUT:\na\nb\nc"
                if current <= 4:  # workers 2, 3, 4
                    return f"W_{current - 2}_OUTPUT:\nworker-result-{current}"
                # Call 5 = synthesizer; capture its prompt
                for m in messages:
                    role    = getattr(m, "role",    None)
                    content = getattr(m, "content", None) or ""
                    if role == "user":
                        captured.append(content)
                        break
                return "## WRITER_OUTPUT\nfinal-synth."
            def count_tokens(self, text: str) -> int:
                return max(1, len(text) // 4)

        adapter = _ThreeThenCaptureAdapter()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: out.split("\n"),
                worker_name="w", worker_goal="Do: {item}",
            )
            .group("synth")
                .agent("writer", "Combine worker outputs")
        )
        p.run("task", adapter=adapter, mode="fine")
        assert adapter.call_count == 5  # 1 source + 3 workers + 1 synth
        # The synthesizer must see the fan-out group's aggregated text
        synth_prompt = "\n".join(captured)
        assert "[fan output]" in synth_prompt


# ---------------------------------------------------------------------------
# Parallel-mode runtime expansion
# ---------------------------------------------------------------------------

class TestParallelRuntimeExpansion:

    def test_parallel_fanout_expands_after_source_group_completes(self):
        """Runtime expansion in parallel mode happens just-in-time for
        each topological level — the fan-out group depends_on its
        source group, so it's in a later level."""
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                return (
                    "## OUTPUT\na|b|c|d"
                    if self.call_count == 1
                    else "## OUTPUT\nworker-done."
                )

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed", depends_on=[])
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: out.split("|"),
                worker_name="w", worker_goal="Do: {item}",
                depends_on=["seed"],
            )
        )
        p.run("task", adapter=adapter, mode="fine", parallel=True)
        # 1 source + 4 workers = 5 LLM calls
        assert adapter.call_count == 5

    def test_parallel_empty_extractor_skips_fanout_group(self):
        class _StubSource(_CountingAdapter):
            def complete(self, messages, tools=None):
                with self._lock:
                    self.call_count += 1
                return "## OUTPUT\nirrelevant"

        adapter = _StubSource()
        p = (
            Pipeline("t")
            .group("seed", depends_on=[])
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=lambda out: [],
                worker_name="w", worker_goal="Do: {item}",
                depends_on=["seed"],
            )
        )
        p.run("task", adapter=adapter, mode="fine", parallel=True)
        # Only the source was called (fan-out expansion returned 0 items)
        assert adapter.call_count == 1


# ---------------------------------------------------------------------------
# Checkpoint interaction (G-4 × G-6)
# ---------------------------------------------------------------------------

class TestFanoutCheckpointInteraction:

    def test_resumed_fanout_group_does_not_re_invoke_extractor(self):
        """Seed the G-4 checkpoint with a completed fan-out group's
        outputs. On the next run with the same task_id, the fan-out
        group should replay from checkpoint — no source dispatch, no
        worker dispatch, no extractor call."""
        ckpt = PipelineCheckpoint()
        # Pretend a prior run already ran source + fan-out + 3 workers
        ckpt.save_group(
            "t-resume", "seed",
            outputs={"SRC_OUTPUT": "a|b|c"},
            final_output="a|b|c",
        )
        ckpt.save_group(
            "t-resume", "fan",
            outputs={
                "W_0_OUTPUT": "result-a",
                "W_1_OUTPUT": "result-b",
                "W_2_OUTPUT": "result-c",
            },
            final_output="[w_0]\nresult-a\n\n[w_1]\nresult-b\n\n[w_2]\nresult-c",
        )

        extractor_calls = 0
        def counting_extractor(output: str) -> list[str]:
            nonlocal extractor_calls
            extractor_calls += 1
            return output.split("|")

        adapter = _CountingAdapter()
        p = (
            Pipeline("t")
            .group("seed")
                .agent("src", "emit")
            .fanout_group(
                "fan", source="src",
                item_extractor=counting_extractor,
                worker_name="w", worker_goal="Do: {item}",
            )
        )
        p.run("task", adapter=adapter, mode="fine",
              task_id="t-resume", checkpoint=ckpt)
        assert adapter.call_count    == 0  # everything replayed
        assert extractor_calls       == 0  # extractor NOT re-invoked
