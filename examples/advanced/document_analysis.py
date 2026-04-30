"""
[Advanced / Internal API] Document Analysis Pipeline

Uses internal primitives directly (AgentStepCapsule, TagSpace, CapsuleExecutor).
Intended for contributors and advanced users needing direct iteration-space control.

For most use cases, prefer the v2 public SDK:
    See: examples/document_analysis_pipeline.py

Demonstrates iteration-space composition: one LLM call processes a batch of
documents simultaneously, reducing round-trips compared to per-document calls.

Pipeline:
  Summarizer → processes each document in batches of K

This example runs offline using a scripted adapter; swap ScriptedAdapter for
AnthropicAdapter() or OpenAIAdapter() for a live run with real API keys.

Usage:
    python -m examples.document_analysis
    python -m examples.document_analysis --batch-size 3 --docs 9

Design plan ref: §5.2 Phase 2, §5.3 examples/
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.tag import TagDimension, TagSpace
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor


# ---------------------------------------------------------------------------
# Scripted adapter (no API key required — swap with real adapter for live use)
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, tools=None):
        self.call_count += 1
        combined = messages[-1].content
        keys = re.findall(r"(SUMMARIZER_OUTPUT_ITEM_\d+)", combined)
        parts = [f"{k}:\nSummary: this document covers the topic concisely." for k in keys]
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def run(doc_count: int = 6, batch_size: int = 2) -> None:
    summarizer = AgentLeaf(capsule=AgentStepCapsule(
        name="summarizer",
        system_prompt="You are a document summarization agent. For each document, produce a 1-sentence summary.",
        input_schema=Schema("document", fields={"text": "str"}),
        output_schema=Schema("summary", fields={"summary": "str"}),
    ))
    root = CompoundCapsule(name="doc_pipeline", children=[summarizer], dependency_edges={})
    tag_space = TagSpace(
        agent_name="summarizer",
        dimensions=[TagDimension("doc_id", list(range(doc_count)))],
    )
    hierarchy = CapsuleHierarchy(
        name="document_analysis", root=root, tag_space=tag_space
    )

    adapter = ScriptedAdapter()
    executor = CapsuleExecutor(
        adapter, composition_level=CompositionLevel.ITERATION, batch_size=batch_size
    )

    print(f"\nDocument Analysis — iteration-space composition")
    print(f"Documents: {doc_count}  |  Batch size: {batch_size}\n")

    result = executor.run(hierarchy, task_input="AI safety report", task_id="doc-run-1")

    expected_batches = (doc_count + batch_size - 1) // batch_size
    print(f"LLM calls: {adapter.call_count} ({expected_batches} batches × 1 call)")
    print(f"Outputs:   {len(result.outputs)} summaries")
    print(f"Throughput: {doc_count}/{adapter.call_count} = {doc_count/adapter.call_count:.1f}x docs per call")
    summary = executor._telemetry.summary()
    print(f"Avg overhead: {summary.get('avg_overhead_ratio', 0):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run(doc_count=args.docs, batch_size=args.batch_size)
