"""
Eval result persistence — appends run summaries to evals/last_eval.md.

Every live eval run (OpenAI, Anthropic, suite, tuning) appends a structured
entry to last_eval.md so that:
  - You can see at a glance how the controller behaved across providers / models
  - Calibration history is preserved as models evolve
  - The file can be committed to track tuning decisions over time

Format: chronological, most-recent at the top.  Each run appends one section.
"""
from __future__ import annotations

import os
from datetime import datetime

from .results import EvalResult

_EVAL_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "last_eval.md")

_HEADER = """\
# Eval Log — agentic_capsules controller calibration

Results are appended chronologically (most recent at top).
Each entry covers one eval run: provider, model, sensitivity, and per-group outcomes.

---
"""


def _entry(result: EvalResult) -> str:
    """Format one EvalResult as a markdown section."""
    lines: list[str] = []
    n = len(result.runs)
    compose_at   = getattr(result.policy, "compose_at",   0.40) if result.policy else 0.40
    decompose_at = getattr(result.policy, "decompose_at", 0.15) if result.policy else 0.15
    confidence   = getattr(result.policy, "confidence",   0.80) if result.policy else 0.80

    lines.append(f"## {result.timestamp}  ·  {result.provider} / {result.model}  ·  {result.sensitivity}")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Company | {result.company} |")
    lines.append(f"| Runs | {n} |")
    lines.append(f"| compose\\_at | {compose_at} |")
    lines.append(f"| decompose\\_at | {decompose_at} |")
    lines.append(f"| confidence | {confidence} |")

    weights = getattr(result.policy, "score_weights", None) if result.policy else None
    if weights:
        lines.append(f"| score\\_weights | {tuple(round(w, 2) for w in weights)} |")

    lines.append("")

    # Per-group summary table
    lines.append("| Group | Switches | Mean score | ±std | First switch | Mean conf |")
    lines.append("|---|---|---|---|---|---|")
    for group in result.groups():
        sw  = result.switch_count(group)
        ms  = result.mean_score(group)
        std = result.score_std(group)
        mc  = result.mean_confidence(group)
        fs  = result.first_switch_run(group)
        fs_str = f"Run {fs}" if fs else "—"
        lines.append(f"| {group} | {sw}/{n} | {ms:.3f} | {std:.3f} | {fs_str} | {mc:.0%} |")

    lines.append("")

    # Per-run detail
    lines.append("<details>")
    lines.append("<summary>Per-run detail</summary>")
    lines.append("")
    lines.append("| Run | Group | Mode | Conf | Score | Rec |")
    lines.append("|---|---|---|---|---|---|")
    for run in result.runs:
        for rec in run.records:
            lines.append(
                f"| {run.run_index + 1} | {rec.group} | {rec.mode_used} "
                f"| {rec.confidence:.0%} | {rec.score:.3f} | {rec.recommendation} |"
            )
    lines.append("")
    lines.append("</details>")

    # Signal breakdown for the last run
    last_run = result.runs[-1] if result.runs else None
    if last_run:
        has_signals = any(rec.signal is not None for rec in last_run.records)
        if has_signals:
            lines.append("")
            lines.append("**Signal breakdown (last run):**")
            lines.append("")
            for rec in last_run.records:
                sig = rec.signal
                if sig is None:
                    continue
                norm_agents = min(sig.agent_count / 4.0, 1.0)
                norm_tokens = min(sig.avg_output_tokens / 300.0, 1.0)
                norm_tools  = min(sig.tool_calls_per_agent / 3.0, 1.0)
                max_depth   = max(sig.agent_count - 1, 1)
                norm_depth  = min(sig.dependency_depth / max_depth, 1.0)
                lines.append(
                    f"- **{rec.group}**: "
                    f"overhead={sig.overhead_ratio:.3f}, "
                    f"agents={sig.agent_count}(→{norm_agents:.2f}), "
                    f"avg_tokens={sig.avg_output_tokens:.0f}(→{norm_tokens:.2f}), "
                    f"tool_calls={sig.tool_calls_per_agent:.1f}(→{norm_tools:.2f}), "
                    f"depth={sig.dependency_depth}(→{norm_depth:.2f}), "
                    f"score={rec.score:.3f}"
                )

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def append_result(result: EvalResult) -> None:
    """
    Append *result* to evals/last_eval.md.

    Creates the file with a header if it does not exist.
    Inserts the new entry after the header (most-recent first).
    """
    entry = _entry(result)

    if not os.path.exists(_EVAL_LOG):
        with open(_EVAL_LOG, "w") as f:
            f.write(_HEADER)
            f.write(entry)
        return

    with open(_EVAL_LOG, "r") as f:
        content = f.read()

    # Insert after the header separator "---\n"
    sep = "---\n"
    idx = content.find(sep)
    if idx == -1:
        with open(_EVAL_LOG, "a") as f:
            f.write(entry)
    else:
        insert_at = idx + len(sep) + 1  # after the separator + newline
        with open(_EVAL_LOG, "w") as f:
            f.write(content[:insert_at])
            f.write(entry)
            f.write(content[insert_at:])
