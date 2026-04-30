"""
Long-chain research pipeline eval runner — P-3.

Runs the long-chain research pipeline (gather 4 agents + analyze 3 agents +
report 1 agent) against any supported provider and records controller
adaptation over N runs.

Exercises topology gaps left by P-1 and P-2:
  - 4-agent sequential tool chain (primary S-1 signal source)
  - 3-agent sequential reasoning chain (M-1/O-1 at depth > 2)
  - Long context accumulation in gather group

Model recommendations:
  haiku       — primary S-1 signal (verbose; context accumulates fastest)
  sonnet      — quality-credible result (passes 0.75 floor, T-052)
  gpt-4o-mini — cross-provider check

Usage:
    # Offline — scripted adapter, no API key needed:
    python -m evals.long_chain_research

    # Live against Anthropic (haiku — primary S-1 model):
    python -m evals.long_chain_research --live --provider anthropic --runs 5

    # Live against Anthropic (sonnet — quality-credible):
    python -m evals.long_chain_research --live --provider anthropic --runs 5 \\
        --model claude-sonnet-4-6

    # Live against OpenAI (cross-provider check):
    python -m evals.long_chain_research --live --provider openai --runs 5

    # S-1: test predecessor-only context strategy:
    python -m evals.long_chain_research --live --provider anthropic --runs 5 \\
        --sequential-context-strategy predecessor_only

    # Track A variant flags (M-1, O-1) work here too:
    python -m evals.long_chain_research --live --provider anthropic --runs 5 \\
        --compound-execution-model sequential --output-guidance concise

Results are printed to stdout and appended to evals/last_eval.md when --live.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys

from evals.shared.long_chain_research import (
    LongChainScriptedAdapter, TASK_TEMPLATE, PIPELINE_DESCRIPTION,
    build_pipeline, build_pipeline_with_judge,
)
from evals.shared.results import (
    EvalResult, collect_run, make_eval_result,
    print_run_header, print_run, print_summary,
    print_signal_breakdown, print_calibration_notes, print_quality_report,
)
from evals.shared.persist import append_result

_DEFAULT_OAI_MODEL  = "gpt-4o-mini"
_DEFAULT_ANTH_MODEL = "claude-haiku-4-5-20251001"

_PROVIDER_MAP = {
    "anthropic": ("agentic_capsules.adapters.anthropic", "AnthropicAdapter", _DEFAULT_ANTH_MODEL),
    "openai":    ("agentic_capsules.adapters.openai",    "OpenAIAdapter",    _DEFAULT_OAI_MODEL),
}


# ---------------------------------------------------------------------------
# Checkpoint helpers (hung-socket recovery)
# ---------------------------------------------------------------------------
#
# This eval can stall for hours on a single hung Anthropic/OpenAI socket
# (observed 2026-04-06: 5 runs completed, run 6 wedged in recv() for 9h
# before manual kill). Checkpoint lets us persist completed runs after each
# iteration so run_resilient.sh retries pick up where the previous attempt
# died instead of losing 45+ minutes of work.
#
# Format: pickle of (config_fingerprint_str, list[EvalRun]).
# The fingerprint guards against silently reusing a checkpoint from a
# different variant — if it doesn't match current args, we abort loudly.
# On successful completion of all runs, the checkpoint file is deleted.

# n_runs is deliberately NOT in the fingerprint. Resuming needs to work when
# a prior invocation completed 5/7 runs and the retry is still "--runs 7",
# and it should also be OK to resume 5/7 into a --runs 10 invocation (adds
# 5 more runs of the same config). Only per-run configuration matters here.
_CHECKPOINT_FIELDS = (
    "company", "provider", "model", "sensitivity",
    "compound_execution_model", "compound_min_output_words", "mode",
    "merged_output_structure", "output_guidance",
    "sequential_context_strategy", "cache_aligned_prompts",
    "escalation_enabled", "quality", "judge_provider", "judge_model",
    "quality_floor",
)


def _checkpoint_fingerprint(locals_dict: dict) -> str:
    """Stable hash of the eval configuration. Different args → different hash."""
    payload = "|".join(f"{k}={locals_dict[k]!r}" for k in _CHECKPOINT_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_load(path: str, fingerprint: str):
    """
    Load a checkpoint if present and its fingerprint matches.

    Returns list[EvalRun] on hit, [] on miss. Raises SystemExit if the
    checkpoint is corrupt or belongs to a different config — silent reuse
    would contaminate the next variant's data.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as fh:
            saved_fp, saved_runs = pickle.load(fh)
    except Exception as exc:
        print(f"  [checkpoint] ERROR: corrupt file {path}: {exc}", file=sys.stderr)
        print(f"  [checkpoint] Delete the file and retry: rm {path}", file=sys.stderr)
        raise SystemExit(2)
    if saved_fp != fingerprint:
        print(f"  [checkpoint] ERROR: fingerprint mismatch at {path}", file=sys.stderr)
        print(f"  [checkpoint]   saved={saved_fp}  current={fingerprint}", file=sys.stderr)
        print(f"  [checkpoint] This checkpoint belongs to a different eval config.", file=sys.stderr)
        print(f"  [checkpoint] Delete the file if you really want to overwrite: rm {path}", file=sys.stderr)
        raise SystemExit(2)
    print(f"  [checkpoint] Resuming from {path} — {len(saved_runs)} run(s) already completed")
    return saved_runs


def _checkpoint_save(path: str, fingerprint: str, runs: list) -> None:
    """Atomic write: tmp file + os.replace so a crash mid-write can't corrupt."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump((fingerprint, runs), fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _checkpoint_clear(path: str) -> None:
    """Remove checkpoint after successful completion. Silent on missing."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _build_judge_adapter(judge_provider: str, judge_model: str):
    if judge_provider == "openai":
        from agentic_capsules.adapters.openai import OpenAIAdapter
        return OpenAIAdapter(model=judge_model)
    elif judge_provider == "anthropic":
        from agentic_capsules.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=judge_model)
    else:
        raise ValueError(f"Unknown judge provider: {judge_provider!r}")


def run_eval(
    company:                     str   = "Acme Logistics Inc.",
    n_runs:                      int   = 3,
    provider:                    str   = "anthropic",
    model:                       str | None = None,
    sensitivity:                 str   = "balanced",
    live:                        bool  = False,
    quality:                     bool  = False,
    judge_provider:              str   = "openai",
    judge_model:                 str   = "gpt-4o",
    quality_floor:               float = 0.75,
    compound_execution_model:    str   = "standard",
    compound_min_output_words:   int | None = None,
    mode:                        str   = "auto",
    merged_output_structure:     str   = "none",
    output_guidance:             str   = "none",
    sequential_context_strategy: str   = "full",
    cache_aligned_prompts:       bool  = False,
    escalation_enabled:          bool  = False,
    checkpoint_path:             str | None = None,
) -> EvalResult:
    """
    Run the long-chain research pipeline N times and return structured results.

    Args:
        company:       Company name to use in the task prompt.
        n_runs:        Number of pipeline.run() calls.
        provider:      "anthropic" or "openai".
        model:         LLM model ID.  Defaults to provider's cost-efficient model.
        sensitivity:   Controller preset: conservative / balanced / aggressive.
        live:          Use real LLM API.  Requires matching API key env var.
        quality:       Attach LLM judge evaluator (requires --live).
        judge_provider: Provider for the judge ("anthropic" or "openai").
        judge_model:   Judge model ID.
        quality_floor: Minimum quality score for COMPOUND mode.
        compound_execution_model: standard / two_phase / sequential.
        mode:          auto / observe / fine / compound.
        merged_output_structure: M-1 variant (none/budgeted/budgeted_adaptive/reinforced).
        output_guidance:         O-1 variant (none/auto/concise/moderate/brief).
        sequential_context_strategy: S-1 variant (full/predecessor_only).
        cache_aligned_prompts:   C-1 Anthropic prefix caching restructure.
        escalation_enabled:      E-1 quality-driven execution model escalation ladder.
        checkpoint_path:         When set, persist completed runs to this
                                 path after each iteration so a crash or
                                 retry resumes instead of restarting from
                                 run 0. File is deleted on successful
                                 completion of all n_runs iterations.
    """
    evaluator = None

    if live:
        if provider not in _PROVIDER_MAP:
            raise ValueError(f"Unknown provider {provider!r}. Use 'anthropic' or 'openai'.")

        module_path, cls_name, default_model = _PROVIDER_MAP[provider]
        effective_model    = model or default_model
        effective_provider = provider

        import importlib
        adapter_cls = getattr(importlib.import_module(module_path), cls_name)

        judge_adapter_obj = _build_judge_adapter(judge_provider, judge_model) if quality else None
        pipeline, evaluator = build_pipeline_with_judge(
            sensitivity=sensitivity,
            judge_adapter=judge_adapter_obj,
            quality_floor=quality_floor,
            compound_execution_model=compound_execution_model,
            compound_min_output_words=compound_min_output_words,
            merged_output_structure=merged_output_structure,
            output_guidance=output_guidance,
            sequential_context_strategy=sequential_context_strategy,
            cache_aligned_prompts=cache_aligned_prompts,
            escalation_enabled=escalation_enabled,
        )
        adapter = adapter_cls(model=effective_model)
    else:
        pipeline, evaluator = build_pipeline_with_judge(
            sensitivity=sensitivity,
            compound_execution_model=compound_execution_model,
            compound_min_output_words=compound_min_output_words,
            merged_output_structure=merged_output_structure,
            output_guidance=output_guidance,
            sequential_context_strategy=sequential_context_strategy,
            cache_aligned_prompts=cache_aligned_prompts,
            escalation_enabled=escalation_enabled,
        )
        adapter            = LongChainScriptedAdapter()
        effective_model    = "scripted"
        effective_provider = "scripted"

    policy = pipeline._policy

    result = make_eval_result(
        provider=effective_provider,
        model=effective_model,
        sensitivity=sensitivity,
        company=company,
        policy=policy,
    )

    print_run_header(effective_provider, effective_model, sensitivity, company, n_runs)
    print(f"  Pipeline   : {PIPELINE_DESCRIPTION}")
    print()

    # Checkpoint resume — must happen after result is built so we can attach
    # recovered runs, but before the loop so start_idx skips them.
    fingerprint = ""
    if checkpoint_path:
        fingerprint = _checkpoint_fingerprint({
            "company": company, "provider": effective_provider,
            "model": effective_model, "sensitivity": sensitivity,
            "compound_execution_model": compound_execution_model,
            "compound_min_output_words": compound_min_output_words, "mode": mode,
            "merged_output_structure": merged_output_structure,
            "output_guidance": output_guidance,
            "sequential_context_strategy": sequential_context_strategy,
            "cache_aligned_prompts": cache_aligned_prompts,
            "escalation_enabled": escalation_enabled, "quality": quality,
            "judge_provider": judge_provider, "judge_model": judge_model,
            "quality_floor": quality_floor,
        })
        recovered = _checkpoint_load(checkpoint_path, fingerprint)
        if recovered:
            result.runs.extend(recovered)
            for run in recovered:
                print_run(run, policy)

    start_idx = len(result.runs)
    task = TASK_TEMPLATE.format(company=company)
    for i in range(start_idx, n_runs):
        pr  = pipeline.run(task, adapter=adapter, evaluator=evaluator, mode=mode)
        run = collect_run(i, pr, pipeline)
        result.runs.append(run)
        print_run(run, policy)
        if checkpoint_path:
            _checkpoint_save(checkpoint_path, fingerprint, result.runs)

    print_summary(result)
    print_signal_breakdown(result)
    print_calibration_notes(result)
    if quality:
        print_quality_report(result, quality_floor=quality_floor)

    if live:
        append_result(result)
        print(f"\n  Results appended to evals/last_eval.md")

    # Only clear the checkpoint after *everything* downstream of the run loop
    # succeeds (summary, quality report, persist). If the process dies in one
    # of those stages the checkpoint is still useful on retry.
    if checkpoint_path:
        _checkpoint_clear(checkpoint_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-chain research eval — P-3")
    parser.add_argument("--company",        default="Acme Logistics Inc.",
                        help="Company to analyse (default: Acme Logistics Inc.)")
    parser.add_argument("--runs",           type=int, default=3,
                        help="Number of pipeline runs (default: 3)")
    parser.add_argument("--provider",       default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider for live runs (default: anthropic)")
    parser.add_argument("--model",          default=None,
                        help="Model ID (default: provider's cost-efficient model)")
    parser.add_argument("--sensitivity",    default="balanced",
                        choices=["conservative", "balanced", "aggressive"],
                        help="Controller sensitivity preset (default: balanced)")
    parser.add_argument("--live",           action="store_true",
                        help="Call real LLM API (requires API key env var)")
    parser.add_argument("--quality",        action="store_true",
                        help="Enable LLM judge quality evaluation (requires --live)")
    parser.add_argument("--judge-provider", default="openai",
                        choices=["anthropic", "openai"],
                        dest="judge_provider",
                        help="Provider for judge model (default: openai)")
    parser.add_argument("--judge-model",    default="gpt-4o",
                        dest="judge_model",
                        help="Judge model ID (default: gpt-4o)")
    parser.add_argument("--quality-floor",  type=float, default=0.75,
                        dest="quality_floor",
                        help="Minimum quality score for COMPOUND mode (default: 0.75)")
    parser.add_argument("--compound-execution-model", default="standard",
                        choices=["standard", "two_phase", "sequential"],
                        dest="compound_execution_model",
                        help="Compound execution model (default: standard)")
    parser.add_argument("--compound-min-output-words", type=int, default=None,
                        dest="compound_min_output_words",
                        help="Depth hint per phase in words (default: None)")
    parser.add_argument("--mode",           default="auto",
                        choices=["auto", "observe", "fine", "compound"],
                        help="Execution mode (default: auto)")
    parser.add_argument("--merged-output-structure", default="none",
                        choices=["none", "budgeted", "budgeted_adaptive", "reinforced"],
                        dest="merged_output_structure",
                        help="M-1: anti-compression hint for standard compound (default: none)")
    parser.add_argument("--output-guidance", default="none",
                        choices=["none", "auto", "concise", "moderate", "brief"],
                        dest="output_guidance",
                        help="O-1: output length guidance for sequential/FINE (default: none)")
    parser.add_argument("--sequential-context-strategy", default="full",
                        choices=["full", "predecessor_only"],
                        dest="sequential_context_strategy",
                        help="S-1: context injection strategy (default: full)")
    parser.add_argument("--cache-aligned-prompts", action="store_true",
                        dest="cache_aligned_prompts",
                        help="C-1: restructure prompts for Anthropic prefix caching")
    parser.add_argument("--escalation-enabled", action="store_true",
                        dest="escalation_enabled",
                        help="E-1: quality-driven execution model escalation ladder")
    parser.add_argument("--checkpoint", default=None, dest="checkpoint_path",
                        help="Path to persist completed runs for hung-socket "
                             "recovery. If the file exists and its config "
                             "fingerprint matches, the eval resumes from the "
                             "next un-run index. Deleted on successful completion.")
    args = parser.parse_args()

    run_eval(
        company=args.company,
        n_runs=args.runs,
        provider=args.provider,
        model=args.model,
        sensitivity=args.sensitivity,
        live=args.live,
        quality=args.quality,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        quality_floor=args.quality_floor,
        compound_execution_model=args.compound_execution_model,
        compound_min_output_words=args.compound_min_output_words,
        mode=args.mode,
        merged_output_structure=args.merged_output_structure,
        output_guidance=args.output_guidance,
        sequential_context_strategy=args.sequential_context_strategy,
        cache_aligned_prompts=args.cache_aligned_prompts,
        escalation_enabled=args.escalation_enabled,
        checkpoint_path=args.checkpoint_path,
    )
