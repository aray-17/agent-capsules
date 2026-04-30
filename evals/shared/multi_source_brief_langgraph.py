"""
LangGraph baseline for the multi_source_brief pipeline (T-055).

A *tuned* (not naive) port of the 14-agent multi_source_brief pipeline, used
as the comparison point in the head-to-head against Agentic Capsules. The
goal is fairness — the LangGraph version uses the same prompts, the same
Anthropic SDK calls, and the same adapter wrapper, so the only variable
between the two systems is *orchestration*.

Two graph builders:

  build_fine_graph()
      14 nodes — one per extractor + scoping + briefer. Same call count as
      the Agentic Capsules fine path. Uses LangGraph's built-in concurrent
      branch execution: scoping fans out to 12 extractor nodes (4 lenses ×
      3 extractors per arm), all of which run concurrently when the graph
      is awaited; briefer joins them all.

  build_compound_graph()
      6 nodes — one scoping + 4 *merged* arm nodes (each asks for entities,
      claims, and signals in a single prompt and parses the combined reply)
      + 1 briefer. Same call count as Agentic Capsules compound mode. This
      is the manual analog of compound merging — a careful LangGraph user
      could write this by hand without any framework support.

Both graphs are *async*. Serial vs parallel execution is selected at run
time:

  run_graph(graph, target, adapter, parallel=True)
      Calls graph.ainvoke() directly. Sibling branches execute concurrently
      via asyncio scheduling — LangGraph 1.x dispatches them in the same
      superstep when nodes are async.

  run_graph(graph, target, adapter, parallel=False)
      Same async invocation, but a global asyncio.Lock is acquired around
      every LLM call so the underlying API requests serialize. Wall-clock
      latency in this mode is the sum of per-call latencies (matching the
      Agentic Capsules ``parallel=False`` path).

The adapter is the *same* AnthropicAdapter used by Agentic Capsules, so the
SDK behavior, retry logic, and token accounting are identical between the
two systems. Call counts are tracked by a wrapper that increments on each
``complete()`` invocation.

Scoping-injection fairness (prompt-economy branch, 2026-04-09)
---------------------------------------------------------------
Every extractor and merged-arm node injects the scoping output into its
user prompt under a ``[scoping output]`` header. This matches Agentic
Capsules' ``_build_group_task`` behavior (parallel_compiler.py:426): in
AC, a group that declares ``depends_on=["scoping"]`` receives scoping's
output appended to its task input, so every arm call pays the scoping
token cost. The initial LG port skipped this injection, which made the
LG side artificially cheap — extractors saw the bundle but no scoping
context. The fix aligns per-call *content* between the two systems so
the only variable in the head-to-head is orchestration.

Tracking ref: T-055.3
"""
from __future__ import annotations

import asyncio
import re
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, START, END

from agentic_capsules.core.types import LLMMessage
from evals.data.multi_source_bundles import LENSES, TARGETS

# Reuse the exact prompt strings from the Agentic Capsules pipeline so the
# only variable between systems is orchestration, not prompts.
from evals.shared.multi_source_brief import (
    _ENTITIES_INSTRUCTION,
    _CLAIMS_INSTRUCTION,
    _SIGNALS_INSTRUCTION,
    _SCOPING_INSTRUCTION,
    _BRIEFER_INSTRUCTION,
    _make_arm_prompt,
    _make_scoping_prompt,
)


# ---------------------------------------------------------------------------
# Adapter wrapper — counts complete() calls
# ---------------------------------------------------------------------------

class CountingAdapter:
    """Pass-through wrapper that counts complete() invocations.

    Identical interface to the wrapper used by the Agentic Capsules eval
    runner so call counts are directly comparable.
    """
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def complete(self, messages, tools=None):
        self.calls += 1
        return self._inner.complete(messages, tools=tools)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

def _dict_merge(a: dict, b: dict) -> dict:
    """Reducer for concurrent dict updates from parallel branches."""
    merged = dict(a)
    merged.update(b)
    return merged


class _BriefState(TypedDict, total=False):
    target:             str
    scoping_output:     str
    extractor_outputs:  Annotated[dict[str, str], _dict_merge]
    merged_outputs:     Annotated[dict[str, str], _dict_merge]
    final_brief:        str


# ---------------------------------------------------------------------------
# LLM call helper — async wrapper around the sync adapter
# ---------------------------------------------------------------------------

async def _llm(adapter, system_prompt: str, user_prompt: str,
               lock: asyncio.Lock | None) -> str:
    """Run one LLM call. If *lock* is provided, serialize all calls."""
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user",   content=user_prompt),
    ]
    if lock is not None:
        async with lock:
            return await asyncio.to_thread(adapter.complete, messages, None)
    return await asyncio.to_thread(adapter.complete, messages, None)


# ---------------------------------------------------------------------------
# Fine graph — 14 nodes
# ---------------------------------------------------------------------------

def build_fine_graph(target: str, adapter, lock: asyncio.Lock | None) -> Any:
    """
    Build the fine-mode LangGraph: 14 nodes, same call count as Agentic
    Capsules fine mode.

    Topology:
        scoping -> {12 extractor nodes in parallel} -> briefer

    The 12 extractor nodes all declare ``scoping`` as their only predecessor,
    so LangGraph schedules them in a single superstep — they run concurrently
    under .ainvoke() when *lock* is None.
    """
    if target not in TARGETS:
        raise KeyError(f"Unknown target {target!r}")
    bundles = TARGETS[target]

    # ------ scoping ------
    async def n_scoping(state: _BriefState) -> dict:
        out = await _llm(
            adapter,
            _make_scoping_prompt(bundles["overview"]),
            f"Produce the scope for {target}.",
            lock,
        )
        return {"scoping_output": out}

    # ------ extractor factory ------
    #
    # Matches Agentic Capsules' ``_build_group_task``: every downstream group
    # that declares ``depends_on=[...]`` has the dependency's output appended
    # to its user prompt. In this pipeline that means each extractor's user
    # prompt carries the scoping output under a ``[scoping output]`` header,
    # so both systems see the same content per LLM call.
    def make_extractor(lens: str, kind: str, instruction: str):
        key = f"{lens}_{kind}"
        async def node(state: _BriefState) -> dict:
            scope = state.get("scoping_output", "")
            user_prompt = (
                f"Run the {kind} extractor for the {lens} lens of {target}."
                f"\n\n[scoping output]\n{scope}"
            )
            out = await _llm(
                adapter,
                _make_arm_prompt(bundles[lens], instruction),
                user_prompt,
                lock,
            )
            return {"extractor_outputs": {key: out}}
        node.__name__ = f"n_{key}"
        return node

    # ------ briefer ------
    async def n_briefer(state: _BriefState) -> dict:
        extractor_outputs = state.get("extractor_outputs", {})
        scope = state.get("scoping_output", "")
        # Format the 12 extractor outputs into 4 lens blocks for the briefer.
        blocks: list[str] = [f"SCOPING:\n{scope}\n"]
        for lens in LENSES:
            blocks.append(f"\n=== {lens.upper()} ===")
            for kind in ("entities", "claims", "signals"):
                key = f"{lens}_{kind}"
                blocks.append(f"\n[{kind}]\n{extractor_outputs.get(key, '')}")
        user_prompt = "\n".join(blocks)
        out = await _llm(adapter, _BRIEFER_INSTRUCTION, user_prompt, lock)
        return {"final_brief": out}

    # ------ assemble ------
    g = StateGraph(_BriefState)
    g.add_node("scoping", n_scoping)
    extractor_node_names: list[str] = []
    for lens in LENSES:
        for kind, instruction in (
            ("entities", _ENTITIES_INSTRUCTION),
            ("claims",   _CLAIMS_INSTRUCTION),
            ("signals",  _SIGNALS_INSTRUCTION),
        ):
            name = f"{lens}_{kind}"
            g.add_node(name, make_extractor(lens, kind, instruction))
            extractor_node_names.append(name)
    g.add_node("briefer", n_briefer)

    g.add_edge(START, "scoping")
    for name in extractor_node_names:
        g.add_edge("scoping", name)
        g.add_edge(name, "briefer")
    g.add_edge("briefer", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Compound graph — 6 nodes
# ---------------------------------------------------------------------------

# Each merged arm asks for all three extractions in one prompt with delimited
# sections. The parser splits the reply on the section markers.

_MERGED_ARM_INSTRUCTION = (
    "Run THREE extractors on the source above and return them in clearly "
    "delimited sections. Use the EXACT section markers shown.\n\n"
    "=== ENTITIES ===\n"
    + _ENTITIES_INSTRUCTION + "\n\n"
    "=== CLAIMS ===\n"
    + _CLAIMS_INSTRUCTION + "\n\n"
    "=== SIGNALS ===\n"
    + _SIGNALS_INSTRUCTION + "\n\n"
    "Begin your reply directly with '=== ENTITIES ===' and use exactly those "
    "three section headers in that order. Do not add any preamble."
)


def _split_merged(reply: str) -> dict[str, str]:
    """Parse a merged arm reply into the three sub-sections."""
    parts = re.split(r"===\s*(ENTITIES|CLAIMS|SIGNALS)\s*===", reply)
    # parts: ["preamble", "ENTITIES", "...body...", "CLAIMS", "...", "SIGNALS", "..."]
    out: dict[str, str] = {"entities": "", "claims": "", "signals": ""}
    for i in range(1, len(parts) - 1, 2):
        marker = parts[i].strip().lower()
        body   = parts[i + 1].strip()
        if marker in out:
            out[marker] = body
    return out


def build_compound_graph(target: str, adapter, lock: asyncio.Lock | None) -> Any:
    """
    Build the compound-mode LangGraph: 6 nodes, same call count as Agentic
    Capsules compound mode.

    Topology:
        scoping -> {4 merged arm nodes in parallel} -> briefer

    Each merged arm node asks for entities + claims + signals in one prompt
    with delimited sections, then parses the reply. This is the manual analog
    of compound merging — a careful LangGraph user can write it by hand.
    """
    if target not in TARGETS:
        raise KeyError(f"Unknown target {target!r}")
    bundles = TARGETS[target]

    async def n_scoping(state: _BriefState) -> dict:
        out = await _llm(
            adapter,
            _make_scoping_prompt(bundles["overview"]),
            f"Produce the scope for {target}.",
            lock,
        )
        return {"scoping_output": out}

    def make_arm(lens: str):
        async def node(state: _BriefState) -> dict:
            scope = state.get("scoping_output", "")
            user_prompt = (
                f"Run the merged extractor set for the {lens} lens of {target}."
                f"\n\n[scoping output]\n{scope}"
            )
            out = await _llm(
                adapter,
                _make_arm_prompt(bundles[lens], _MERGED_ARM_INSTRUCTION),
                user_prompt,
                lock,
            )
            return {"merged_outputs": {lens: out}}
        node.__name__ = f"arm_{lens}"
        return node

    async def n_briefer(state: _BriefState) -> dict:
        merged = state.get("merged_outputs", {})
        scope  = state.get("scoping_output", "")
        blocks: list[str] = [f"SCOPING:\n{scope}\n"]
        for lens in LENSES:
            reply  = merged.get(lens, "")
            parsed = _split_merged(reply)
            blocks.append(f"\n=== {lens.upper()} ===")
            for kind in ("entities", "claims", "signals"):
                blocks.append(f"\n[{kind}]\n{parsed[kind]}")
        user_prompt = "\n".join(blocks)
        out = await _llm(adapter, _BRIEFER_INSTRUCTION, user_prompt, lock)
        return {"final_brief": out}

    g = StateGraph(_BriefState)
    g.add_node("scoping", n_scoping)
    arm_nodes: list[str] = []
    for lens in LENSES:
        name = f"arm_{lens}"
        g.add_node(name, make_arm(lens))
        arm_nodes.append(name)
    g.add_node("briefer", n_briefer)

    g.add_edge(START, "scoping")
    for name in arm_nodes:
        g.add_edge("scoping", name)
        g.add_edge(name, "briefer")
    g.add_edge("briefer", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_pipeline(target: str, adapter, *, mode: str, parallel: bool) -> dict:
    """
    Run the LangGraph pipeline for *target* in the requested *mode*.

    Args:
        target:   Target company name.
        adapter:  AnthropicAdapter (or any adapter with .complete()). Wrap
                  with CountingAdapter beforehand if you want call counts.
        mode:     "fine" or "compound".
        parallel: When True, sibling branches run concurrently via asyncio.
                  When False, a global lock serializes LLM calls.

    Returns:
        The final state dict from .ainvoke() — includes ``final_brief``.
    """
    lock: asyncio.Lock | None = None if parallel else asyncio.Lock()
    if mode == "fine":
        graph = build_fine_graph(target, adapter, lock)
    elif mode == "compound":
        graph = build_compound_graph(target, adapter, lock)
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return asyncio.run(graph.ainvoke({"target": target}))
