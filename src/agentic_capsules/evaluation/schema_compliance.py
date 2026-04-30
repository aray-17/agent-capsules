"""
SchemaComplianceEvaluator — free structural quality check (P12-2).

No LLM call required.  Scores COMPOUND output against FINE baseline using
three fast structural signals:

  Field coverage   (weight 0.50) — fraction of FINE "key terms" present in COMPOUND
  Completeness     (weight 0.30) — COMPOUND length >= completeness_floor * FINE length
  Format match     (weight 0.20) — JSON-parseable if FINE was JSON-parseable

Overall score = 0.50 * field_coverage + 0.30 * completeness + 0.20 * format_match

Confidence: 0.70 (structural heuristic — captures obvious degradation but cannot
detect semantic errors or hallucinations).

Phase 12 ref: P12-2, T-027.
"""
from __future__ import annotations

import json
import re

from .base import QualityScore

# Stop-words excluded from field-coverage extraction (common words add noise)
_STOP = frozenset({
    "the", "and", "for", "are", "was", "has", "have", "been", "with",
    "that", "this", "from", "will", "they", "their", "which", "when",
    "also", "into", "its", "our", "can", "but", "not", "all", "one",
    "two", "any", "more", "each", "per", "both",
})


def _extract_key_terms(text: str) -> set[str]:
    """
    Extract significant tokens from text for field-coverage comparison.

    Returns lowercase words that are:
      - >= 5 characters (avoids short noise tokens)
      - not stop-words
      - contain only letters (avoids fragmented numbers/punctuation)

    This targets domain vocabulary (metrics, company names, concepts) that
    a high-quality COMPOUND output should preserve from the FINE baseline.
    """
    tokens = re.findall(r"[A-Za-z]{5,}", text)
    return {t.lower() for t in tokens if t.lower() not in _STOP}


def _field_coverage(fine_output: str, compound_output: str) -> float:
    """Fraction of FINE key terms present in COMPOUND output."""
    fine_terms = _extract_key_terms(fine_output)
    if not fine_terms:
        return 1.0  # no extractable terms — no coverage requirement
    compound_terms = _extract_key_terms(compound_output)
    return len(fine_terms & compound_terms) / len(fine_terms)


def _completeness(fine_output: str, compound_output: str, floor: float) -> float:
    """1.0 if COMPOUND length >= floor * FINE length, else linear partial credit."""
    if not fine_output:
        return 1.0
    ratio = len(compound_output) / len(fine_output)
    if ratio >= floor:
        return 1.0
    # Partial credit proportional to how close to floor we are
    return max(0.0, ratio / floor)


def _format_match(fine_output: str, compound_output: str) -> float:
    """
    1.0  if FINE was not JSON (no format constraint).
    1.0  if both are valid JSON.
    0.0  if FINE was valid JSON but COMPOUND is not.
    """
    fine_is_json = _is_json(fine_output)
    if not fine_is_json:
        return 1.0
    return 1.0 if _is_json(compound_output) else 0.0


def _is_json(text: str) -> bool:
    try:
        stripped = text.strip()
        if not stripped:
            return False
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


class SchemaComplianceEvaluator:
    """
    Structural quality check — no LLM call required.

    Scores based on:
      - Field coverage   (0.50): fraction of FINE key terms present in COMPOUND
      - Completeness     (0.30): COMPOUND length >= completeness_floor * FINE length
      - Format match     (0.20): COMPOUND is JSON-parseable if FINE was

    Overall score = 0.50 * field_coverage + 0.30 * completeness + 0.20 * format_match

    Args:
        completeness_floor: Minimum length ratio (COMPOUND / FINE) to score
                            completeness as 1.0.  Default 0.60 (40% shortening
                            is acceptable; more triggers a partial-credit penalty).
    """

    def __init__(self, completeness_floor: float = 0.60) -> None:
        if not 0 < completeness_floor <= 1.0:
            raise ValueError(
                f"completeness_floor must be in (0, 1], got {completeness_floor}"
            )
        self._completeness_floor = completeness_floor

    def evaluate(
        self,
        task_input:      str,
        fine_output:     str,
        compound_output: str,
    ) -> QualityScore:
        """
        Compare COMPOUND output against FINE baseline structurally.

        Args:
            task_input:      The task prompt (unused by structural check,
                             kept for protocol compatibility).
            fine_output:     Reference output produced in FINE mode.
            compound_output: Output produced in COMPOUND mode to evaluate.

        Returns:
            QualityScore with score in [0, 1] and confidence 0.70.
        """
        field_score   = _field_coverage(fine_output, compound_output)
        compl_score   = _completeness(fine_output, compound_output, self._completeness_floor)
        format_score  = _format_match(fine_output, compound_output)
        score = 0.50 * field_score + 0.30 * compl_score + 0.20 * format_score

        return QualityScore(
            score=max(0.0, min(1.0, score)),
            confidence=0.70,
            evaluator="SchemaComplianceEvaluator",
            details={
                "field_coverage":    round(field_score,  4),
                "completeness":      round(compl_score,  4),
                "format_match":      round(format_score, 4),
                "fine_len":          len(fine_output),
                "compound_len":      len(compound_output),
                "completeness_floor": self._completeness_floor,
            },
        )
