"""
QualityScore and QualityEvaluator — the shared evaluation protocol (P12-1).

All evaluators in this package implement QualityEvaluator and return a
QualityScore so that different evaluation strategies (structural, LLM-based,
consistency) can be swapped in and compared without changing the controller.

Score contract:
  1.0  — COMPOUND output is fully equivalent to FINE baseline.
  0.0  — Complete quality degradation.
  Scores outside [0, 1] are clamped by compute helpers in each evaluator.

Confidence semantics (approximate reliability estimates):
  SchemaComplianceEvaluator  0.70  — structural heuristic, not semantic
  LLMJudgeEvaluator          0.85  — strong but prompt-sensitive
  ConsistencyEvaluator       0.60  — measures variance, not absolute quality
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class QualityScore:
    """
    Result of comparing a COMPOUND output against a FINE baseline.

    Attributes:
        score:      0.0 (no quality preserved) – 1.0 (identical quality).
        confidence: How reliable this estimate is (0.0–1.0).
                    SchemaCompliance≈0.70, LLMJudge≈0.85, Consistency≈0.60.
        details:    Evaluator-specific breakdown dict (for debugging/logging).
        evaluator:  Class name of the evaluator that produced this score.
    """
    score:      float
    confidence: float
    details:    dict = field(default_factory=dict)
    evaluator:  str  = ""

    def __post_init__(self) -> None:
        self.score      = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@runtime_checkable
class QualityEvaluator(Protocol):
    """
    Protocol for all output quality evaluators.

    Compares a COMPOUND-mode output against a FINE-mode baseline for the
    same task and group.  Returns a QualityScore where 1.0 = COMPOUND output
    is fully equivalent to FINE; 0.0 = completely degraded.

    Implementations must not raise exceptions — on parse/runtime errors they
    should return ``QualityScore(score=0.0, confidence=0.0)`` so the quality
    gate can proceed safely.
    """
    def evaluate(
        self,
        task_input:      str,
        fine_output:     str,
        compound_output: str,
    ) -> QualityScore: ...
