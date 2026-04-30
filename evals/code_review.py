"""
Code-review pipeline eval runner — T-028 (eval breadth).

Runs the code review pipeline (fan-out, 3-agent review group) against any
supported provider and records controller adaptation over N runs.

Exercises a different topology than the due-diligence pipeline:
  - 3 agents in the review group (vs 2 in due-diligence)
  - Tool-heavy fan-out pattern (all 3 reviewers call the same 2 tools)
  - Different score profile: review group expected to compose by run 3

Usage:
    # Offline — scripted adapter, no API key needed:
    python -m evals.code_review

    # Live against Anthropic:
    python -m evals.code_review --live --provider anthropic --runs 5
    python -m evals.code_review --live --provider anthropic --model claude-sonnet-4-6

    # Live against OpenAI:
    python -m evals.code_review --live --provider openai --runs 5
    python -m evals.code_review --live --provider openai --model gpt-4o

    # Offline with variable tool response sizes:
    python -m evals.code_review --variable-tools

    # Offline with error injection (verifies error_rate signal):
    python -m evals.code_review --error-rate 0.25

Results are printed to stdout and appended to evals/last_eval.md when --live.
"""
from __future__ import annotations

import argparse

from evals.shared.code_review  import (
    CodeReviewScriptedAdapter, TASK_TEMPLATE, PIPELINE_DESCRIPTION,
    build_pipeline, build_pipeline_with_judge,
)
from evals.shared.pipeline     import VariableToolAdapter, ErrorInjectionAdapter
from evals.shared.results      import (
    EvalResult, collect_run, make_eval_result,
    print_run_header, print_run, print_summary,
    print_signal_breakdown, print_calibration_notes, print_quality_report,
)
from evals.shared.persist      import append_result

_DEFAULT_OAI_MODEL   = "gpt-4o-mini"
_DEFAULT_ANTH_MODEL  = "claude-haiku-4-5-20251001"
_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
_DEFAULT_JUDGE_PROV  = "openai"


def run_eval(
    pr_id:                   str   = "pr-42",
    n_runs:                  int   = 3,
    provider:                str   = "anthropic",
    model:                   str | None = None,
    sensitivity:             str   = "balanced",
    live:                    bool  = False,
    variable_tools:          bool  = False,
    error_rate:              float = 0.0,
    quality:                 bool  = False,
    judge_provider:          str   = _DEFAULT_JUDGE_PROV,
    judge_model:             str   = _DEFAULT_JUDGE_MODEL,
    quality_floor:           float = 0.75,
    compound_execution_model: str  = "standard",
    merged_output_structure: str   = "none",
    output_guidance:         str   = "none",
    sequential_context_strategy: str = "full",
    cache_aligned_prompts:       bool = False,
    escalation_enabled:          bool = False,
) -> EvalResult:
    """
    Run the code review pipeline N times and return structured results.

    Args:
        pr_id:                   Pull request ID to use in the task prompt.
        n_runs:                  Number of pipeline.run() calls.
        provider:                "anthropic" or "openai".
        model:                   LLM model ID. Defaults to provider's cost-efficient model.
        sensitivity:             Controller preset (conservative / balanced / aggressive).
        live:                    Use real LLM API. Requires matching API key env var.
        variable_tools:          Use VariableToolAdapter for realistic response-size variance.
        error_rate:              Probability of injecting agent errors (0.0 = off).
        quality:                 Attach LLMJudgeEvaluator and print quality scores.
        judge_provider:          Provider for the LLM judge (openai / anthropic).
        judge_model:             Judge model ID.
        quality_floor:           Minimum quality score before COMPOUND is permitted.
        compound_execution_model: "standard" | "sequential" | "two_phase".
        merged_output_structure: M-1 variant: "none"|"budgeted"|"budgeted_adaptive"|"reinforced".
        output_guidance:         O-1 variant: "none"|"auto"|"concise"|"moderate"|"brief".
        sequential_context_strategy: S-1 variant: "full"|"predecessor_only".
        cache_aligned_prompts:   C-1 — Anthropic prefix caching restructure (Anthropic only).
        escalation_enabled:      E-1 — quality-driven execution model escalation ladder.
    """
    evaluator = None
    if quality and live:
        if judge_provider == "openai":
            from agentic_capsules.adapters.openai import OpenAIAdapter
            judge_adapter = OpenAIAdapter(model=judge_model)
        elif judge_provider == "anthropic":
            from agentic_capsules.adapters.anthropic import AnthropicAdapter
            judge_adapter = AnthropicAdapter(model=judge_model)
        else:
            raise ValueError(f"Unknown judge provider: {judge_provider!r}")
        pipeline, evaluator = build_pipeline_with_judge(
            sensitivity=sensitivity,
            judge_adapter=judge_adapter,
            quality_floor=quality_floor,
            compound_execution_model=compound_execution_model,
            merged_output_structure=merged_output_structure,
            output_guidance=output_guidance,
            sequential_context_strategy=sequential_context_strategy,
            cache_aligned_prompts=cache_aligned_prompts,
            escalation_enabled=escalation_enabled,
        )
    else:
        pipeline = build_pipeline(sensitivity=sensitivity)

    if live:
        if provider == "anthropic":
            from agentic_capsules.adapters.anthropic import AnthropicAdapter
            adapter = AnthropicAdapter(model=model or _DEFAULT_ANTH_MODEL)
            effective_model = model or _DEFAULT_ANTH_MODEL
        elif provider == "openai":
            from agentic_capsules.adapters.openai import OpenAIAdapter
            adapter = OpenAIAdapter(model=model or _DEFAULT_OAI_MODEL)
            effective_model = model or _DEFAULT_OAI_MODEL
        else:
            raise ValueError(f"Unknown provider {provider!r}. Use 'anthropic' or 'openai'.")
        effective_provider = provider
    else:
        if variable_tools:
            adapter = VariableToolAdapter(cache_hit_rate=0.40, seed=42)
        else:
            adapter = CodeReviewScriptedAdapter()
        effective_model    = "scripted" + ("-variable" if variable_tools else "")
        effective_provider = "scripted"

    if error_rate > 0.0:
        adapter = ErrorInjectionAdapter(adapter, error_rate=error_rate, seed=0)

    policy = pipeline._policy

    result = make_eval_result(
        provider=effective_provider,
        model=effective_model,
        sensitivity=sensitivity,
        company=f"PR {pr_id}",
        policy=policy,
    )

    print_run_header(effective_provider, effective_model, sensitivity, f"PR {pr_id}", n_runs)
    print(f"  Pipeline   : {PIPELINE_DESCRIPTION}")
    if error_rate > 0.0:
        print(f"  Error rate : {error_rate:.0%} (injected)")
    print()

    task = TASK_TEMPLATE.format(pr_id=pr_id)
    for i in range(n_runs):
        pr  = pipeline.run(task, adapter=adapter, evaluator=evaluator)
        run = collect_run(i, pr, pipeline)
        result.runs.append(run)
        print_run(run, policy)

    print_summary(result)
    print_signal_breakdown(result)
    print_calibration_notes(result)
    if quality:
        print_quality_report(result, quality_floor=quality_floor)

    if live:
        append_result(result)
        print(f"\n  Results appended to evals/last_eval.md")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code review eval — T-028 eval breadth")
    parser.add_argument("--pr-id",          default="pr-42",
                        help="Pull request ID (default: pr-42)")
    parser.add_argument("--runs",           type=int, default=3,
                        help="Number of pipeline runs")
    parser.add_argument("--provider",       default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider for live runs")
    parser.add_argument("--model",          default=None,
                        help="Model ID (default: provider's cost-efficient model)")
    parser.add_argument("--sensitivity",    default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--live",           action="store_true",
                        help="Call real LLM API")
    parser.add_argument("--variable-tools", action="store_true",
                        dest="variable_tools",
                        help="Use VariableToolAdapter for response-size variance")
    parser.add_argument("--error-rate",     type=float, default=0.0,
                        dest="error_rate",
                        help="Probability of agent error injection (0.0–1.0, default 0)")
    parser.add_argument("--quality",        action="store_true",
                        help="Attach LLM judge and print quality scores")
    parser.add_argument("--judge-provider", default=_DEFAULT_JUDGE_PROV,
                        dest="judge_provider",
                        choices=["openai", "anthropic"],
                        help="Judge provider (default: openai)")
    parser.add_argument("--quality-model",  default=_DEFAULT_JUDGE_MODEL,
                        dest="judge_model",
                        help="Judge model ID (default: gpt-4o-mini)")
    parser.add_argument("--compound-execution-model", default="standard",
                        dest="compound_execution_model",
                        choices=["standard", "sequential", "two_phase"],
                        help="Compound execution model (default: standard)")
    parser.add_argument("--merged-output-structure", default="none",
                        dest="merged_output_structure",
                        choices=["none", "budgeted", "budgeted_adaptive", "reinforced"],
                        help="M-1 output structure hint (default: none)")
    parser.add_argument("--output-guidance", default="none",
                        dest="output_guidance",
                        choices=["none", "auto", "concise", "moderate", "brief"],
                        help="O-1 output length guidance (default: none)")
    parser.add_argument("--sequential-context-strategy", default="full",
                        dest="sequential_context_strategy",
                        choices=["full", "predecessor_only"],
                        help="S-1 context injection strategy (default: full)")
    parser.add_argument("--cache-aligned-prompts", action="store_true",
                        dest="cache_aligned_prompts",
                        help="C-1 — restructure prompts for Anthropic prefix caching (Anthropic only)")
    parser.add_argument("--escalation-enabled", action="store_true",
                        dest="escalation_enabled",
                        help="E-1 — quality-driven execution model escalation ladder")
    args = parser.parse_args()

    run_eval(
        pr_id=args.pr_id,
        n_runs=args.runs,
        provider=args.provider,
        model=args.model,
        sensitivity=args.sensitivity,
        live=args.live,
        variable_tools=args.variable_tools,
        error_rate=args.error_rate,
        quality=args.quality,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        compound_execution_model=args.compound_execution_model,
        merged_output_structure=args.merged_output_structure,
        output_guidance=args.output_guidance,
        sequential_context_strategy=args.sequential_context_strategy,
        cache_aligned_prompts=args.cache_aligned_prompts,
        escalation_enabled=args.escalation_enabled,
    )
