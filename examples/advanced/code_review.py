"""
[Advanced / Internal API] Code Review Pipeline with Multi-Model Routing

Uses internal primitives directly (AgentStepCapsule, CompoundCapsule,
CapsuleHierarchy, CapsuleExecutor). Intended for contributors and advanced
users who need fine-grained control over the execution layer.

For most use cases, prefer the v2 public SDK:
    See: examples/code_review_pipeline.py

Demonstrates computation-space composition with ModelRouter:
  static_analyzer → security_scanner → review_synthesizer

Each agent is routed to a different model via ModelRouter.

Usage:
    python -m examples.advanced.code_review
    python -m examples.advanced.code_review --snippet "def foo(): pass"
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import CompositionLevel, Schema
from agentic_capsules.runtime.executor import CapsuleExecutor
from agentic_capsules.runtime.model_router import ModelRouter
from agentic_capsules.runtime.scheduler import compute_order


# ---------------------------------------------------------------------------
# Scripted adapters (named so we can verify routing)
# ---------------------------------------------------------------------------

class NamedAdapter:
    context_window = 200_000

    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []
        self.current_capsule: str = ""

    def complete(self, messages, tools=None):
        self.calls.append(self.current_capsule)
        content = messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", content)
        parts = []
        for key in keys:
            if "STATIC" in key:
                parts.append(f"{key}:\nStatic analysis: no syntax errors. [via {self.name}]")
            elif "SECURITY" in key:
                parts.append(f"{key}:\nSecurity: no obvious vulnerabilities found. [via {self.name}]")
            elif "REVIEW" in key:
                parts.append(f"{key}:\nReview: code is clean and well-structured. [via {self.name}]")
            else:
                parts.append(f"{key}:\nAnalysis complete. [via {self.name}]")
        return "\n\n".join(parts) if parts else f"OUTPUT:\nDone. [via {self.name}]"

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def run(snippet: str = "def greet(name): return f'Hello, {name}!'") -> None:
    # Build pipeline
    static_analyzer = AgentLeaf(capsule=AgentStepCapsule(
        name="static_analyzer",
        system_prompt="Perform static analysis: check syntax, imports, and code style.",
        input_schema=Schema("code", fields={"code": "str"}),
        output_schema=Schema("static_report", fields={"issues": "str"}),
    ))
    security_scanner = AgentLeaf(capsule=AgentStepCapsule(
        name="security_scanner",
        system_prompt="Perform security scanning: check for injection, secrets, unsafe patterns.",
        input_schema=Schema("code", fields={"code": "str"}),
        output_schema=Schema("security_report", fields={"findings": "str"}),
    ))
    review_synthesizer = AgentLeaf(capsule=AgentStepCapsule(
        name="review_synthesizer",
        system_prompt="Synthesize the static and security reports into a final code review.",
        input_schema=Schema("reports", fields={"static": "str", "security": "str"}),
        output_schema=Schema("review", fields={"review": "str"}),
    ))
    root = CompoundCapsule(
        name="code_review",
        children=[static_analyzer, security_scanner, review_synthesizer],
        dependency_edges={"review_synthesizer": ["static_analyzer", "security_scanner"]},
    )
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="code_review_pipeline", root=root)

    # ModelRouter: static_analyzer → fast model, security_scanner → specialized model
    fast = NamedAdapter("fast-model")
    security = NamedAdapter("security-model")
    default = NamedAdapter("default-model")
    router = ModelRouter(default=default, routes={
        "static_analyzer": fast,
        "security_scanner": security,
    })

    executor = CapsuleExecutor(router, composition_level=CompositionLevel.FINE)

    print("\nCode Review Pipeline (computation-space + ModelRouter)")
    print(f"Snippet: {snippet[:60]!r}\n")

    result = executor.run(hierarchy, task_input=snippet, task_id="code-review-1")

    print(f"Final review:\n{result.final_output}\n")
    print("Routing summary:")
    print(f"  static_analyzer  → fast-model     (calls: {len(fast.calls)})")
    print(f"  security_scanner → security-model (calls: {len(security.calls)})")
    print(f"  review_synthesizer → default-model (calls: {len(default.calls)})")
    print(f"\nTotal LLM calls: {len(fast.calls) + len(security.calls) + len(default.calls)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snippet", type=str, default="def greet(name): return f'Hello, {name}!'")
    args = parser.parse_args()
    run(snippet=args.snippet)
