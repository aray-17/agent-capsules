"""
PipelineResult — developer-facing execution result.

All fields are keyed by the names the developer used when declaring agents
and groups — never by internal output-key format (e.g. "researcher" not
"RESEARCHER_OUTPUT").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    """
    The output of a pipeline.run() call.

    Attributes:
        output:         Final agent's output text.
        recommendation: Per-group controller recommendation.
                        Keys are group names; values are "COMPOSE" |
                        "DECOMPOSE" | "MAINTAIN".
        mode_used:      Per-group composition level actually used this run.
                        Keys are group names; values are "fine" | "compound".
        confidence:     Per-group confidence score at end of this run (0.0–1.0).
        scores:         Per-group multi-signal composition score (0.0–1.0) computed
                        this run.  Values above ``compose_at`` count as COMPOSE
                        evidence.  0.0 when score_weights is not configured.
        step_outputs:   Per-agent output text, keyed by agent name.
        token_usage:    Total tokens consumed this run.
        latency_ms:     Wall-clock time for the full pipeline run (ms).
        efficiency:     Phase 12 — per-group efficiency summary, populated from
                        the rolling GroupControllerState window after each run.
                        Keys are group names; each value is a dict with:
                          token_reduction_pct  — % tokens saved vs FINE (positive = savings)
                          mean_latency_fine_ms — rolling mean latency in FINE mode
                          mean_latency_compound_ms — rolling mean latency in COMPOUND mode
                        Fields are None when not enough data for that mode yet.
        quality:        Phase 12 — per-group rolling mean quality score (0.0–1.0).
                        Empty dict when no evaluator was passed to pipeline.run().
        quality_details: Phase 12 — per-group last QualityScore.details dict for
                        debugging (field_coverage, accuracy, etc. depending on
                        evaluator type).  Empty dict when no evaluator active.
    """
    output:          str
    recommendation:  dict[str, str]   = field(default_factory=dict)
    mode_used:       dict[str, str]   = field(default_factory=dict)
    confidence:      dict[str, float] = field(default_factory=dict)
    scores:          dict[str, float] = field(default_factory=dict)
    step_outputs:    dict[str, str]   = field(default_factory=dict)
    token_usage:     int              = 0
    latency_ms:      float | None     = None
    efficiency:      dict[str, dict]  = field(default_factory=dict)
    quality:         dict[str, float] = field(default_factory=dict)
    quality_details: dict[str, dict]  = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.output[:80] + "..." if len(self.output) > 80 else self.output
        quality_str = f", quality={self.quality}" if self.quality else ""
        return (
            f"PipelineResult("
            f"output={preview!r}, "
            f"mode_used={self.mode_used}, "
            f"recommendation={self.recommendation}"
            f"{quality_str})"
        )
