"""
LLMJudgeEvaluator — LLM-as-judge quality evaluation (P12-3).

Uses any LLMAdapter as a judge to compare FINE and COMPOUND outputs on a
structured rubric (completeness, accuracy, structure, conciseness).

Score = mean of four 0–10 rubric dimensions, normalised to [0, 1].

Cost: one extra LLM call per evaluation.  Use a cheap judge model
(e.g. claude-haiku, gpt-4o-mini) even when the pipeline runs on a more
capable model — judge cost is ~$0.001/eval.

Parse guard: malformed JSON from the judge returns QualityScore(score=0.0,
confidence=0.0) without raising.  Partial scores (some keys missing) are
handled with default values.

Phase 12 ref: P12-3, T-027.
"""
from __future__ import annotations

import json

from .base import QualityScore

JUDGE_PROMPT = """\
You are evaluating two AI pipeline outputs for the same task.

Task: {task_input}

Output A (reference — fine-grained agents):
{fine_output}

Output B (to evaluate — composed agents):
{compound_output}

Rate Output B relative to Output A on each dimension (0–10):
  completeness: does B cover all key points and facts in A?
  accuracy:     does B introduce errors or omit facts present in A?
  structure:    does B follow the same structure and format as A?
  conciseness:  is B appropriately concise (not padded, not truncated)?

Respond with JSON only, no commentary:
{{"completeness": N, "accuracy": N, "structure": N, "conciseness": N}}
"""

_DIMENSIONS = ("completeness", "accuracy", "structure", "conciseness")


class LLMJudgeEvaluator:
    """
    LLM-as-judge quality evaluator.

    Sends a structured rubric prompt to a judge LLM adapter and parses the
    response as a JSON object with four 0–10 scores.

    Args:
        judge_adapter: Any LLMAdapter instance.  The adapter's default model
                       is used; pass a cheap model (haiku, gpt-4o-mini) to
                       minimise cost.
        prompt:        Rubric prompt template.  Must contain ``{task_input}``,
                       ``{fine_output}``, and ``{compound_output}`` placeholders.
    """

    def __init__(self, judge_adapter, prompt: str = JUDGE_PROMPT) -> None:
        self._adapter = judge_adapter
        self._prompt  = prompt

    def evaluate(
        self,
        task_input:      str,
        fine_output:     str,
        compound_output: str,
    ) -> QualityScore:
        """
        Compare COMPOUND output against FINE baseline using an LLM judge.

        Returns:
            QualityScore with score = mean rubric score (0–1), confidence 0.85.
            Returns score=0.0, confidence=0.0 on any parse or adapter error.
        """
        prompt = self._prompt.format(
            task_input=task_input,
            fine_output=fine_output,
            compound_output=compound_output,
        )
        try:
            raw = self._adapter.complete([_UserMessage(prompt)])
            return self._parse(raw)
        except Exception as exc:
            return QualityScore(
                score=0.0,
                confidence=0.0,
                evaluator="LLMJudgeEvaluator",
                details={"error": str(exc)},
            )

    def _parse(self, raw: str) -> QualityScore:
        """Parse judge JSON response; return zero-confidence score on failure."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return QualityScore(
                score=0.0,
                confidence=0.0,
                evaluator="LLMJudgeEvaluator",
                details={"parse_error": str(exc), "raw": raw[:200]},
            )

        # Extract and normalise each dimension (default 5/10 if missing)
        dim_scores: dict[str, float] = {}
        for dim in _DIMENSIONS:
            raw_val = data.get(dim, 5)
            try:
                val = float(raw_val)
                dim_scores[dim] = max(0.0, min(10.0, val)) / 10.0
            except (TypeError, ValueError):
                dim_scores[dim] = 0.5  # default mid-score on unparse-able value

        score = sum(dim_scores.values()) / len(_DIMENSIONS)

        return QualityScore(
            score=max(0.0, min(1.0, score)),
            confidence=0.85,
            evaluator="LLMJudgeEvaluator",
            details=dim_scores,
        )


# ---------------------------------------------------------------------------
# Minimal message wrapper — avoids importing from core.types to stay decoupled
# ---------------------------------------------------------------------------

class _UserMessage:
    """Minimal wrapper to pass a user message to any LLMAdapter.complete()."""
    role    = "user"
    content: str

    def __init__(self, content: str) -> None:
        self.content = content
