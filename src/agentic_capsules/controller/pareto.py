"""
Pareto threshold finder — find the compose_at that maximises quality × token savings.

Sweeps a grid of compose_at values, running the pipeline at each threshold and
evaluating quality.  Returns the threshold that maximises:

  pareto_score = tokens_saved_pct × quality_score

where tokens_saved_pct = (tokens_fine - tokens_compound) / tokens_fine.

Phase 12 ref: P12-8, T-027.

Usage::

    from agentic_capsules.controller.pareto import find_pareto_threshold

    result = find_pareto_threshold(
        pipeline=pipeline,
        adapter=adapter,
        evaluator=SchemaComplianceEvaluator(),
        task_inputs=["Analyse Acme Corp", "Analyse Widget Inc."],
    )
    # result → {"research": 0.30, "analysis": 0.36, "synthesis": 0.40}
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.builder import Pipeline
    from ..core.types import LLMAdapter
    from ..evaluation.base import QualityEvaluator


_DEFAULT_GRID = [0.20, 0.25, 0.30, 0.36, 0.40, 0.50]


@dataclass
class ParetoPoint:
    """Evaluation result for one compose_at value and one group."""
    compose_at:      float
    group:           str
    tokens_fine:     float
    tokens_compound: float
    quality_score:   float
    tokens_saved_pct: float = 0.0
    pareto_score:    float = 0.0

    def __post_init__(self) -> None:
        if self.tokens_fine > 0:
            self.tokens_saved_pct = max(
                0.0, (self.tokens_fine - self.tokens_compound) / self.tokens_fine
            )
        self.pareto_score = self.tokens_saved_pct * self.quality_score


@dataclass
class ParetoResult:
    """
    Output of find_pareto_threshold().

    Attributes:
        recommended:        group_name → recommended compose_at value.
        points:             All sweep data points (group_name → list[ParetoPoint]).
        execution_model:    compound_execution_model used for the COMPOUND runs.
    """
    recommended:     dict[str, float]             = field(default_factory=dict)
    points:          dict[str, list[ParetoPoint]] = field(default_factory=dict)
    execution_model: str                          = "standard"

    def print_table(self) -> None:
        """Print a formatted sweep table to stdout."""
        _BOLD  = "\033[1m"
        _GREEN = "\033[32m"
        _RESET = "\033[0m"

        print(f"\n{_BOLD}--- Pareto threshold sweep (compound_execution_model={self.execution_model}) ---{_RESET}")
        print(f"  {'Group':<14} {'compose_at':>10} {'tokens_saved':>13} {'quality':>8} {'pareto':>8}  note")
        print(f"  {'-'*14} {'-'*10} {'-'*13} {'-'*8} {'-'*8}  {'-'*14}")

        for group, pts in sorted(self.points.items()):
            best = self.recommended.get(group)
            for p in sorted(pts, key=lambda x: x.compose_at):
                marker = f"{_GREEN}← recommended{_RESET}" if p.compose_at == best else ""
                print(
                    f"  {group:<14} {p.compose_at:>10.2f} "
                    f"{p.tokens_saved_pct:>12.1%} "
                    f"{p.quality_score:>8.3f} "
                    f"{p.pareto_score:>8.3f}  {marker}"
                )


def find_pareto_threshold(
    pipeline:                    "Pipeline",
    adapter:                     "LLMAdapter",
    evaluator:                   "QualityEvaluator",
    task_inputs:                 list[str],
    compose_grid:                list[float] | None = None,
    n_runs:                      int = 3,
    compound_execution_model:    str = "standard",
    compound_min_output_words:   int | None = None,
    compound_prompt_style:       str = "standard",
    progress:                    bool = True,
    merged_output_structure:     str = "none",
    output_guidance:             str = "none",
    sequential_context_strategy: str = "full",
) -> ParetoResult:
    """
    Sweep compose_at values and find the Pareto-optimal threshold per group.

    For each compose_at value in the grid:
      1. Run the pipeline N times at that compose_at (balanced preset).
      2. Record token usage in FINE and COMPOUND mode per group.
      3. Evaluate quality of COMPOUND vs FINE output.
      4. Compute pareto_score = tokens_saved_pct × quality_score.

    Returns the compose_at that maximises pareto_score per group.

    Args:
        pipeline:                  Pipeline to sweep.  Its controller state is reset between runs.
        adapter:                   LLM adapter for pipeline execution.
        evaluator:                 Quality evaluator for FINE vs COMPOUND comparison.
        task_inputs:               Task inputs to evaluate (use representative production tasks).
        compose_grid:              compose_at values to sweep.  Default: [0.20, 0.25, 0.30, 0.36, 0.40, 0.50].
        n_runs:                    Runs per compose_at value (averaged for noise reduction).
        compound_execution_model:  T-038: 'standard' or 'two_phase'.  Applied to all COMPOUND runs.
        compound_min_output_words: T-038: depth hint passed to two_phase Phase B (words).
        compound_prompt_style:       T-045: 'standard' or 'compact'.  'compact' removes role-label
                                     structural markers from compound prompts.
        merged_output_structure:     M-1: anti-compression hint for standard compound
                                     ('none'|'budgeted'|'budgeted_adaptive'|'reinforced').
        output_guidance:             O-1: output length guidance for sequential/FINE
                                     ('none'|'adaptive'|'concise'|'moderate'|'brief').
        sequential_context_strategy: S-1: context injection strategy for sequential mode
                                     ('full'|'predecessor_only').

    Returns:
        ParetoResult with recommended thresholds and all data points.
    """
    from ..api.builder import Pipeline
    from ..api.compiler import _PipelineCompiler
    from ..controller.policy import ControllerPolicy, SENSITIVITY_PRESETS

    grid = compose_grid or _DEFAULT_GRID

    # points[group][compose_at] → aggregated sums for averaging
    _sums: dict[str, dict[float, dict]] = {}
    for g in pipeline._groups:
        _sums[g.name] = {
            ca: {"tokens_fine": 0.0, "tokens_compound": 0.0, "quality": 0.0, "n": 0}
            for ca in grid
        }

    base = SENSITIVITY_PRESETS["balanced"]

    total_steps    = len(grid) * len(task_inputs)
    completed      = 0
    step_times:    list[float] = []   # seconds per (ca, task) step
    sweep_start    = time.monotonic()

    for ca_idx, ca in enumerate(grid):
        # Fresh pipeline state for each compose_at sweep
        from ..api.state import PipelineState
        from ..controller.policy import ControllerPolicy

        sweep_policy = ControllerPolicy(
            compose_at=ca,
            decompose_at=min(ca * 0.35, ca - 0.01),   # keep valid gap
            confidence=base.confidence,
            min_observations=base.min_observations,
            window_size=base.window_size,
            score_weights=base.score_weights,
            compound_execution_model=compound_execution_model,
            compound_min_output_words=compound_min_output_words,
            compound_prompt_style=compound_prompt_style,
            merged_output_structure=merged_output_structure,
            output_guidance=output_guidance,
            sequential_context_strategy=sequential_context_strategy,
        )

        for task_idx, task in enumerate(task_inputs):
            step_start = time.monotonic()

            # FINE run (always standard — two_phase/sequential only affects COMPOUND)
            sweep_pipeline_fine = _clone_pipeline(pipeline, sweep_policy)
            fine_result = _PipelineCompiler(
                sweep_pipeline_fine, task, adapter, "fine", None
            ).execute()

            # COMPOUND run — uses compound_execution_model from policy
            sweep_pipeline_comp = _clone_pipeline(pipeline, sweep_policy)
            comp_result = _PipelineCompiler(
                sweep_pipeline_comp, task, adapter, "compound", None
            ).execute()

            step_qualities: dict[str, float] = {}
            for g in pipeline._groups:
                fine_out = fine_result.step_outputs.get(g.agents[-1].name, fine_result.output)
                comp_out = comp_result.step_outputs.get(g.agents[-1].name, comp_result.output)
                quality  = evaluator.evaluate(task, fine_out, comp_out)

                n_groups = len(pipeline._groups)
                tok_fine = fine_result.token_usage / n_groups
                tok_comp = comp_result.token_usage / n_groups

                bucket = _sums[g.name][ca]
                bucket["tokens_fine"]     += tok_fine
                bucket["tokens_compound"] += tok_comp
                bucket["quality"]         += quality.score
                bucket["n"]               += 1

                step_qualities[g.name] = quality.score

            # Progress telemetry
            completed += 1
            step_elapsed = time.monotonic() - step_start
            step_times.append(step_elapsed)

            if progress:
                elapsed_total = time.monotonic() - sweep_start
                avg_step      = sum(step_times) / len(step_times)
                remaining_s   = avg_step * (total_steps - completed)

                quality_str = "  ".join(
                    f"{g}={q:.3f}" for g, q in sorted(step_qualities.items())
                )
                print(
                    f"  [{completed:>{len(str(total_steps))}}/{total_steps}]"
                    f"  ca={ca:.2f}  task {task_idx + 1}/{len(task_inputs)}"
                    f"  {quality_str}"
                    f"  +{_fmt_duration(step_elapsed)}"
                    f"  elapsed {_fmt_duration(elapsed_total)}"
                    f"  ~{_fmt_duration(remaining_s)} left",
                    flush=True,
                )

    # Build ParetoPoints and find best per group
    result = ParetoResult(execution_model=compound_execution_model)
    for group, ca_map in _sums.items():
        pts: list[ParetoPoint] = []
        for ca, bucket in ca_map.items():
            n = max(bucket["n"], 1)
            pt = ParetoPoint(
                compose_at=ca,
                group=group,
                tokens_fine=bucket["tokens_fine"] / n,
                tokens_compound=bucket["tokens_compound"] / n,
                quality_score=bucket["quality"] / n,
            )
            pts.append(pt)
        result.points[group] = pts
        best = max(pts, key=lambda p: p.pareto_score)
        result.recommended[group] = best.compose_at

    return result


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as Xm Ys or Xs."""
    seconds = max(0.0, seconds)
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.0f}s"


def _clone_pipeline(source: "Pipeline", policy: "ControllerPolicy") -> "Pipeline":
    """Create a fresh Pipeline with the same groups/agents but a new policy and clean state."""
    from ..api.builder import Pipeline
    from ..api.state import PipelineState

    clone              = Pipeline.__new__(Pipeline)
    clone._name        = source._name
    clone._policy      = policy
    clone._store       = None
    clone._groups      = source._groups   # shared — read-only during sweep
    clone._current_group = None
    clone._pipeline_state = PipelineState(source._name, policy, store=None)
    return clone
