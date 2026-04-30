"""
[Advanced / Internal API] Multi-Source Research Pipeline with Checkpoint/Resume

Uses internal primitives directly (AgentStepCapsule, CheckpointStore,
CapsuleExecutor). Intended for contributors and advanced users.

For most use cases, prefer the v2 public SDK:
    See: examples/research_pipeline.py

Demonstrates:
  - Tool-space composition: web_search + database → synthesizer
  - Checkpoint/restore: pipeline can resume mid-run after a failure
  - Adaptive granularity controller: switches composition level if overhead is high

Pipeline:
  web_search (AgentLeaf) → db_lookup (AgentLeaf) → synthesizer (AgentLeaf)

The checkpoint is written after each leaf, so a failure at the synthesizer
step can resume without re-running the web_search and db_lookup.

This example runs offline with a scripted adapter; replace with real adapters
and API keys for a live run.

Usage:
    python -m examples.multi_source_research
    python -m examples.multi_source_research --resume

Design plan ref: §5.2 Phase 4, Phase 6 (checkpoint/resume), §5.3 examples/
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.checkpoint import CheckpointStore
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.scheduler import compute_order


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, tools=None):
        self.call_count += 1
        combined = messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        parts = []
        for key in keys:
            if "WEB" in key:
                parts.append(f"{key}:\nWeb results: found 5 articles on the topic.")
            elif "DB" in key:
                parts.append(f"{key}:\nDB results: 12 related records retrieved.")
            elif "SYNTH" in key:
                parts.append(f"{key}:\nSynthesis: topic is well-covered by multiple sources.")
            else:
                parts.append(f"{key}:\nResult found.")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def _make_hierarchy() -> tuple[CapsuleHierarchy, AgentLeaf, AgentLeaf, AgentLeaf]:
    web_search = AgentLeaf(capsule=AgentStepCapsule(
        name="web_search",
        system_prompt="Search the web for recent articles on the given topic.",
        input_schema=Schema("query", fields={"query": "str"}),
        output_schema=Schema("web_results", fields={"results": "str"}),
    ))
    db_lookup = AgentLeaf(capsule=AgentStepCapsule(
        name="db_lookup",
        system_prompt="Query the knowledge database for structured records on the topic.",
        input_schema=Schema("query", fields={"query": "str"}),
        output_schema=Schema("db_results", fields={"records": "str"}),
    ))
    synthesizer = AgentLeaf(capsule=AgentStepCapsule(
        name="synthesizer",
        system_prompt="Synthesize web and database results into a comprehensive report.",
        input_schema=Schema("all_results", fields={"web": "str", "db": "str"}),
        output_schema=Schema("report", fields={"report": "str"}),
    ))
    root = CompoundCapsule(
        name="research",
        children=[web_search, db_lookup, synthesizer],
        dependency_edges={"synthesizer": ["web_search", "db_lookup"]},
    )
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="multi_source_research", root=root)
    return hierarchy, web_search, db_lookup, synthesizer


def run_fresh(task_id: str = "research-1") -> None:
    """Run the full pipeline from scratch, with checkpoint enabled."""
    hierarchy, web_leaf, db_leaf, synth_leaf = _make_hierarchy()
    store = CheckpointStore()
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.FINE, checkpoint=store
    )

    print("\nMulti-Source Research Pipeline (fresh run with checkpointing)")
    result = executor.run(hierarchy, task_input="AI safety research 2025", task_id=task_id)
    print(f"LLM calls:    {adapter.call_count}")
    print(f"Checkpointed: {len(store.load(task_id) or {})} outputs")
    print(f"Final report: {result.final_output[:80]!r}")


def run_with_resume() -> None:
    """
    Simulate a failure after the first two leaves, then resume.
    The resumed run should skip web_search and db_lookup (already checkpointed)
    and only call the synthesizer.
    """
    _, web_leaf, db_leaf, _ = _make_hierarchy()
    task_id = "research-resume-demo"

    # Simulate partial checkpoint (first two leaves already done)
    store = CheckpointStore()
    store.save(task_id, {
        web_leaf.capsule.output_key: "Web results: found 5 articles on the topic.",
        db_leaf.capsule.output_key: "DB results: 12 related records retrieved.",
    })
    print(f"\nCheckpoint pre-seeded: {len(store.load(task_id))} outputs (simulating partial run)")

    # Resume: only synthesizer should run
    hierarchy, _, _, _ = _make_hierarchy()
    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.FINE, checkpoint=store
    )

    print("Resuming pipeline (web_search + db_lookup already checkpointed)...")
    result = executor.run(hierarchy, task_input="AI safety research 2025", task_id=task_id)
    print(f"LLM calls this run: {adapter.call_count} (expected 1 — only synthesizer)")
    print(f"Final report: {result.final_output[:80]!r}")
    assert adapter.call_count == 1, f"Expected 1 call, got {adapter.call_count}"
    print("\nResume verified — only 1 LLM call needed to complete the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Demonstrate checkpoint resume (skip first 2 leaves)")
    args = parser.parse_args()

    if args.resume:
        run_with_resume()
    else:
        run_fresh()
        run_with_resume()
