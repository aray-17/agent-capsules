"""
ConsistencyEvaluator — output variance measurement across N COMPOUND runs (P12-4).

High pairwise similarity between repeated COMPOUND runs means the composed
prompt is well-specified and stable.  Low similarity means sampling variance
is high and COMPOUND results are less predictable.

Two interfaces:

  evaluate(task_input, fine_output, compound_output) -> QualityScore
      Protocol-compliant comparison.  Returns Jaccard similarity between
      fine_output and compound_output (token-level).  Use this for inline
      quality gate integration.

  run_consistency_check(pipeline, adapter, task, n_runs) -> QualityScore
      Standalone consistency sweep: runs the pipeline N times in COMPOUND
      mode, measures pairwise Jaccard similarity across all run pairs.
      Mean pairwise similarity is the consistency score.

Similarity options:
  "jaccard"  (default) — token-level Jaccard coefficient, zero dependencies.
  "cosine"   — cosine similarity via sentence-transformers (skipped gracefully
               if the package is not installed).

Phase 12 ref: P12-4, T-027.
"""
from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from .base import QualityScore

if TYPE_CHECKING:
    pass


def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens for Jaccard similarity."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity in [0, 1]."""
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _cosine_similarity(a: str, b: str) -> float | None:
    """
    Cosine similarity using sentence-transformers.

    Returns None when sentence-transformers is not installed so callers can
    fall back to Jaccard.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = model.encode([a, b])
        num = float(np.dot(vecs[0], vecs[1]))
        den = float(np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        return num / den if den > 0 else 0.0
    except ImportError:
        return None


class ConsistencyEvaluator:
    """
    Output consistency evaluator for COMPOUND mode.

    Args:
        pipeline:   Pipeline instance to run for consistency checks.
        adapter:    LLMAdapter to use for the consistency runs.
        n_runs:     Number of COMPOUND runs for pairwise comparison (default 5).
        similarity: "jaccard" (default) or "cosine" (requires sentence-transformers).
    """

    def __init__(
        self,
        pipeline   = None,
        adapter    = None,
        n_runs:     int = 5,
        similarity: str = "jaccard",
    ) -> None:
        self._pipeline   = pipeline
        self._adapter    = adapter
        self._n_runs     = max(2, n_runs)
        self._similarity = similarity

    def evaluate(
        self,
        task_input:      str,
        fine_output:     str,
        compound_output: str,
    ) -> QualityScore:
        """
        Protocol-compliant evaluation: similarity between fine and compound output.

        Returns the token-level Jaccard similarity between fine_output and
        compound_output.  This provides a cheap inline consistency signal
        without requiring extra LLM calls.
        """
        if self._similarity == "cosine":
            score = _cosine_similarity(fine_output, compound_output)
            if score is None:
                score = _jaccard(fine_output, compound_output)
                method = "jaccard_fallback"
            else:
                method = "cosine"
        else:
            score = _jaccard(fine_output, compound_output)
            method = "jaccard"

        return QualityScore(
            score=max(0.0, min(1.0, score)),
            confidence=0.60,
            evaluator="ConsistencyEvaluator",
            details={
                "method":          method,
                "fine_len":        len(fine_output),
                "compound_len":    len(compound_output),
                "jaccard_score":   score,
            },
        )

    def run_consistency_check(
        self,
        task:   str,
        n_runs: int | None = None,
    ) -> QualityScore:
        """
        Standalone consistency sweep: run pipeline N times in COMPOUND mode.

        Measures mean pairwise similarity across all runs.  High score means
        COMPOUND outputs are stable; low score means high variance.

        Requires pipeline and adapter to be set in the constructor.

        Args:
            task:   Task input for each pipeline run.
            n_runs: Override constructor n_runs for this call.

        Returns:
            QualityScore where score = mean pairwise similarity.
        """
        if self._pipeline is None or self._adapter is None:
            raise ValueError(
                "pipeline and adapter must be set to use run_consistency_check(). "
                "Pass them to ConsistencyEvaluator(pipeline=..., adapter=...)."
            )

        k = n_runs or self._n_runs
        outputs: list[str] = []
        for _ in range(k):
            result = self._pipeline.run(task, adapter=self._adapter, mode="compound")
            outputs.append(result.output)

        # Pairwise similarity across all run pairs
        pairs = list(itertools.combinations(outputs, 2))
        if not pairs:
            return QualityScore(
                score=1.0, confidence=0.60,
                evaluator="ConsistencyEvaluator",
                details={"n_runs": k, "pairs": 0},
            )

        similarities = [
            _jaccard(a, b) if self._similarity != "cosine"
            else (_cosine_similarity(a, b) or _jaccard(a, b))
            for a, b in pairs
        ]
        mean_sim = sum(similarities) / len(similarities)

        return QualityScore(
            score=max(0.0, min(1.0, mean_sim)),
            confidence=0.60,
            evaluator="ConsistencyEvaluator",
            details={
                "n_runs":          k,
                "pairs":           len(pairs),
                "mean_similarity": round(mean_sim, 4),
                "min_similarity":  round(min(similarities), 4),
                "max_similarity":  round(max(similarities), 4),
            },
        )
