"""
Tests for the multi-source competitive intelligence brief pipeline (T-054).

Verifies:
  * Pipeline shape matches the design (14 agents, 6 groups, correct topology)
  * Inter-group dependency edges encode the 4-way fan-out (G2-G5 depend on G1)
  * Intra-group: each arm has three independent extractors with depends_on=[]
  * Synthesizer depends on all four lens groups
  * Bundles are wired correctly: each arm's extractors share the same bundle
    in their system prompts (the cache-alignment precondition for C-1)
  * Smoke test through serial executor with a stub adapter (FINE mode)
  * Smoke test through parallel executor with a stub adapter (FINE + COMPOUND)
  * The four arms actually run concurrently in parallel mode
  * Unknown target raises a clear error

These tests use stub adapters and never touch live APIs.
"""
from __future__ import annotations

import threading
import time

import pytest

from agentic_capsules import PipelineResult

from evals.data.multi_source_bundles import LENSES, TARGETS
from evals.shared.multi_source_brief import (
    PIPELINE_NAME,
    build_pipeline,
)


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

class _ScriptedAdapter:
    """Thread-safe scripted adapter."""
    context_window = 200_000

    def __init__(self, response: str = "## OUTPUT\nstub output."):
        self._response  = response
        self._lock      = threading.Lock()
        self.call_count = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self.call_count += 1
        return self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _SleepingAdapter:
    """Sleeps for `delay` seconds per call; tracks peak concurrent in-flight."""
    context_window = 200_000

    def __init__(self, delay: float = 0.2):
        self._delay      = delay
        self._lock       = threading.Lock()
        self._in_flight  = 0
        self.peak_in_flight = 0
        self.call_count  = 0

    def complete(self, messages, tools=None):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.call_count += 1
        try:
            time.sleep(self._delay)
            return "## OUTPUT\nslept response."
        finally:
            with self._lock:
                self._in_flight -= 1

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Topology — group count, agent count, dependency structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", sorted(TARGETS))
def test_pipeline_has_six_groups(target):
    p = build_pipeline(target)
    assert len(p._groups) == 6, (
        f"expected 6 groups (scoping + 4 arms + synthesis), got {len(p._groups)}"
    )


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_pipeline_has_fourteen_agents(target):
    p = build_pipeline(target)
    total_agents = sum(len(g.agents) for g in p._groups)
    assert total_agents == 14, (
        f"expected 14 agents (1 + 4×3 + 1), got {total_agents}"
    )


def test_group_names_in_canonical_order():
    p = build_pipeline("Stripe")
    actual = [g.name for g in p._groups]
    assert actual == ["scoping", *LENSES, "synthesis"]


def test_scoping_is_root_with_no_deps():
    p = build_pipeline("Stripe")
    scoping = p._groups[0]
    assert scoping.name == "scoping"
    assert scoping.depends_on == []


@pytest.mark.parametrize("lens", LENSES)
def test_arm_groups_depend_only_on_scoping(lens):
    p = build_pipeline("Stripe")
    arm = next(g for g in p._groups if g.name == lens)
    assert arm.depends_on == ["scoping"], (
        f"arm {lens!r} should depend only on scoping, got {arm.depends_on}"
    )


def test_synthesis_depends_on_all_four_arms():
    p = build_pipeline("Stripe")
    synth = p._groups[-1]
    assert synth.name == "synthesis"
    assert synth.depends_on == list(LENSES)


@pytest.mark.parametrize("lens", LENSES)
def test_arm_has_three_independent_extractors(lens):
    p = build_pipeline("Stripe")
    arm = next(g for g in p._groups if g.name == lens)
    assert len(arm.agents) == 3
    extractor_names = {a.name for a in arm.agents}
    assert extractor_names == {
        f"{lens}_entities",
        f"{lens}_claims",
        f"{lens}_signals",
    }
    # All three are fan-out (depends_on=[]) — required for compound merging
    for agent in arm.agents:
        assert agent.depends_on == [], (
            f"{agent.name} should have depends_on=[] (fan-out within arm) "
            f"but got {agent.depends_on}"
        )


def test_scoping_has_one_agent_named_target_scoper():
    p = build_pipeline("Stripe")
    scoping = p._groups[0]
    assert len(scoping.agents) == 1
    assert scoping.agents[0].name == "target_scoper"


def test_synthesis_has_one_agent_named_briefer():
    p = build_pipeline("Stripe")
    synth = p._groups[-1]
    assert len(synth.agents) == 1
    assert synth.agents[0].name == "briefer"


# ---------------------------------------------------------------------------
# Bundle wiring — each arm's extractors share the same source bundle prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("lens", LENSES)
def test_arm_extractors_share_identical_bundle_prefix(target, lens):
    """All three extractors in one arm must share an identical
    'SOURCE MATERIAL: ...' prefix — this is what makes C-1 prompt caching
    effective for this pipeline."""
    p = build_pipeline(target)
    arm = next(g for g in p._groups if g.name == lens)
    prefixes = {a.goal.split("\n\nTASK:\n")[0] for a in arm.agents}
    assert len(prefixes) == 1, (
        f"arm {lens!r} extractors do not share an identical source-material prefix "
        f"(got {len(prefixes)} distinct prefixes); C-1 cache alignment will not "
        "work for this arm"
    )


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("lens", LENSES)
def test_arm_extractors_have_distinct_instructions(target, lens):
    """The three extractors in an arm must differ in their TASK instruction —
    otherwise they would just be three copies of the same call."""
    p = build_pipeline(target)
    arm = next(g for g in p._groups if g.name == lens)
    instructions = {a.goal.split("\n\nTASK:\n")[1] for a in arm.agents}
    assert len(instructions) == 3, (
        f"arm {lens!r}: extractors share an instruction (only "
        f"{len(instructions)} unique tails); they should be entities/claims/signals"
    )


@pytest.mark.parametrize("lens", LENSES)
def test_different_targets_get_different_bundles(lens):
    """Different targets must produce different bundle text — sanity check
    that the factory actually reads from TARGETS instead of a hard-coded
    string."""
    p_stripe   = build_pipeline("Stripe")
    p_anthropic = build_pipeline("Anthropic")
    arm_stripe   = next(g for g in p_stripe._groups   if g.name == lens)
    arm_anthropic = next(g for g in p_anthropic._groups if g.name == lens)
    assert arm_stripe.agents[0].goal != arm_anthropic.agents[0].goal


def test_unknown_target_raises():
    with pytest.raises(KeyError, match="Unknown target"):
        build_pipeline("NotARealCompany")


def test_pipeline_name_matches_constant():
    p = build_pipeline("Stripe")
    assert p._name == PIPELINE_NAME


# ---------------------------------------------------------------------------
# Smoke tests — serial and parallel executors with stub adapters
# ---------------------------------------------------------------------------

def test_smoke_serial_fine_mode():
    """End-to-end smoke through the serial executor in FINE mode."""
    adapter = _ScriptedAdapter()
    p = build_pipeline("Stripe")
    result = p.run("Stripe", adapter=adapter, mode="fine")

    assert isinstance(result, PipelineResult)
    # 14 agents in FINE mode = 14 LLM calls
    assert adapter.call_count == 14
    assert len(result.output) > 0


def test_smoke_parallel_fine_mode():
    """End-to-end smoke through the parallel executor in FINE mode."""
    adapter = _ScriptedAdapter()
    p = build_pipeline("Stripe")
    result = p.run("Stripe", adapter=adapter, mode="fine", parallel=True)

    assert isinstance(result, PipelineResult)
    assert adapter.call_count == 14
    assert len(result.output) > 0


def test_smoke_parallel_compound_mode():
    """End-to-end smoke through the parallel executor in COMPOUND mode.

    In COMPOUND mode, each arm's three extractors collapse into one merged
    LLM call. Single-agent groups (scoping, synthesis) make one call each.
    Total: 1 + (4 × 1) + 1 = 6 LLM calls — significantly fewer than FINE.
    This is the compound-merging-on-top-of-parallelism win in microcosm.
    """
    adapter = _ScriptedAdapter()
    p = build_pipeline("Stripe")
    result = p.run("Stripe", adapter=adapter, mode="compound", parallel=True)

    assert isinstance(result, PipelineResult)
    # Each arm = 1 merged call; scoping + synthesis = 1 call each
    assert adapter.call_count == 6, (
        f"COMPOUND parallel should produce exactly 6 LLM calls "
        f"(1 scoping + 4 merged arms + 1 synthesis), got {adapter.call_count}"
    )


def test_arms_run_concurrently_in_parallel_mode():
    """The four arms must actually run in parallel, not sequentially.

    Three topological levels: scoping → 4 arms → synthesis. With per-call
    delay=0.15s and four arms in the middle level, peak in-flight must
    reach 4 (not 1) and total wall time must be well under 14×0.15s=2.1s.
    """
    delay = 0.15
    adapter = _SleepingAdapter(delay=delay)
    p = build_pipeline("Stripe")

    start = time.perf_counter()
    p.run("Stripe", adapter=adapter, mode="fine", parallel=True)
    elapsed = time.perf_counter() - start

    # 14 calls happened
    assert adapter.call_count == 14
    # Peak in-flight must reach at least 4 — proves the four arms ran concurrently.
    # With 12 fan-out agents in the arms level (3 per arm × 4 arms), peak could
    # be as high as 12 if intra-group parallelism were also implemented; with
    # group-only parallelism it's bounded at 4 (one in-flight call per arm at
    # any given moment, since within-arm execution stays serial).
    assert adapter.peak_in_flight >= 4, (
        f"expected peak_in_flight >= 4 (one per parallel arm), "
        f"got {adapter.peak_in_flight}"
    )
    # Wall-clock upper bound: serial would take 14 × delay = 2.1s.
    # Parallel with group-level concurrency: roughly (1 + 3 + 1) × delay = 0.75s
    # (scoping serial, then 3 serial agents per arm in parallel across arms,
    # then synthesis serial). Use a generous 8× delay ceiling (1.2s) to avoid
    # CI flake.
    assert elapsed < delay * 8, (
        f"parallel run took {elapsed:.2f}s; expected < {delay * 8:.2f}s"
    )


def test_serial_mode_does_not_run_arms_concurrently():
    """Sanity check: parallel=False keeps execution single-threaded."""
    adapter = _SleepingAdapter(delay=0.05)
    p = build_pipeline("Stripe")
    p.run("Stripe", adapter=adapter, mode="fine")  # parallel=False default
    assert adapter.peak_in_flight == 1
