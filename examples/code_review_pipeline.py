"""
Code Review Pipeline — v2 SDK

v2 SDK equivalent of examples/advanced/code_review.py.
Shows how the same code review workflow maps to the public Pipeline API.

Pipeline:
  analysis group — static_analyzer → security_scanner
  review group   — review_synthesizer

For multi-model routing (routing different agents to different LLM models),
see examples/advanced/code_review.py which uses the internal ModelRouter.
The v2 SDK routes all agents through the single adapter passed to run().

Run offline:
    python -m examples.code_review_pipeline

Run live:
    ANTHROPIC_API_KEY=... python -m examples.code_review_pipeline --live
    ANTHROPIC_API_KEY=... python -m examples.code_review_pipeline --live --snippet "def foo(): pass"
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules import Pipeline, PipelineResult


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    def complete(self, messages, tools=None) -> str:
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        responses = {
            "STATIC_ANALYZER": "Static analysis: no syntax errors, no unused imports, PEP 8 compliant.",
            "SECURITY_SCANNER": "Security scan: no hardcoded secrets, no SQL injection risk, no unsafe deserialization.",
            "REVIEW_SYNTHESIZER": "Code review: the snippet is clean and production-ready. No issues found.",
        }
        parts = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{responses.get(agent, 'Review complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    return (
        Pipeline("code_review", sensitivity="balanced")
        .group("analysis")
            .agent(
                "static_analyzer",
                "Perform static analysis: check syntax, imports, naming conventions, and code style.",
            )
            .agent(
                "security_scanner",
                "Perform security scanning: check for injection vulnerabilities, "
                "hardcoded secrets, unsafe patterns, and dependency risks.",
            )
        .group("review")
            .agent(
                "review_synthesizer",
                "Synthesise the static analysis and security scan into a final code review. "
                "Be specific about what is good and what needs fixing.",
            )
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_offline(snippet: str) -> None:
    pipeline = _build_pipeline()
    result: PipelineResult = pipeline.run(
        f"Review this code:\n\n{snippet}",
        adapter=ScriptedAdapter(),
    )
    print(f"\nCode Review Pipeline — offline")
    print(f"Snippet: {snippet[:70]!r}\n")
    print(f"Review:\n{result.output}")
    print(f"\nIndividual agent outputs:")
    for agent, output in result.step_outputs.items():
        print(f"  [{agent}] {output[:80]}")
    print(f"\nmode_used: {result.mode_used}")


def run_live(snippet: str) -> None:
    from agentic_capsules.adapters.anthropic import AnthropicAdapter
    pipeline = _build_pipeline()
    result: PipelineResult = pipeline.run(
        f"Review this code:\n\n{snippet}",
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
    )
    print(result.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Review Pipeline — v2 SDK")
    parser.add_argument(
        "--snippet", type=str,
        default="def divide(a, b):\n    return a / b",
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.live:
        run_live(args.snippet)
    else:
        run_offline(args.snippet)
