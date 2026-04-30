"""
Multi-source brief eval runner — P-4 (T-055).

Runs the 14-agent multi_source_brief pipeline against one target company in
one execution mode and records tokens, latency, LLM call counts, and
(optionally) post-hoc judge quality on the briefer's synthesis output.

The three execution modes that matter for the LangGraph head-to-head:

  fine_serial      — parallel=False, mode="fine"     — baseline (one call at a time)
  fine_parallel    — parallel=True,  mode="fine"     — parallelism win on the 4 arms
  compound_parallel— parallel=True,  mode="compound" — parallelism + compound merging

Quality measurement note:
    The parallel executor rejects evaluators (the H2/H3 quality gates write to
    ControllerState, which is not thread-safe). To get quality scores on
    parallel runs, this runner does **post-hoc judging**: after pipeline.run()
    completes, it sends ``result.output`` (the briefer's synthesis) to a judge
    model with a fixed rubric and stores the score on the EvalRun. This is
    decoupled from the pipeline executor and works for all three exec modes.

Usage:
    # Offline preflight (no API calls):
    python3 -m tools.verify_p4_strategy

    # Live single config:
    python3 -m evals.multi_source_brief --live --target Stripe \\
        --exec fine_parallel --runs 3

    # With quality judge (post-hoc):
    python3 -m evals.multi_source_brief --live --target Stripe \\
        --exec compound_parallel --runs 3 --quality \\
        --judge-provider openai --judge-model gpt-4o

Tracking ref: T-055 (LangGraph head-to-head, P-4 14-agent pipeline)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time

from agentic_capsules.api.compiler import _PipelineCompiler
from evals.shared.multi_source_brief import build_pipeline
from evals.data.multi_source_bundles import TARGETS
from evals.shared.results import (
    EvalResult, EvalRun, RunRecord, make_eval_result, collect_run,
    print_run_header, print_summary,
)
from evals.shared.persist import append_result


_DEFAULT_OAI_MODEL  = "gpt-4o-mini"
_DEFAULT_ANTH_MODEL = "claude-haiku-4-5-20251001"

_PROVIDER_MAP = {
    "anthropic": ("agentic_capsules.adapters.anthropic", "AnthropicAdapter", _DEFAULT_ANTH_MODEL),
    "openai":    ("agentic_capsules.adapters.openai",    "OpenAIAdapter",    _DEFAULT_OAI_MODEL),
}

# (parallel, mode) for each exec label
_EXEC_MAP = {
    "fine_serial":      (False, "fine"),
    "fine_parallel":    (True,  "fine"),
    "compound_parallel":(True,  "compound"),
}


# ---------------------------------------------------------------------------
# Adapter wrapper — count complete() calls
# ---------------------------------------------------------------------------

class _CountingAdapter:
    """Pass-through wrapper that counts complete() invocations."""
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def complete(self, messages, tools=None):
        self.calls += 1
        return self._inner.complete(messages, tools=tools)


# ---------------------------------------------------------------------------
# Strategy assertion (mirrors tools/verify_p4_strategy.py)
# ---------------------------------------------------------------------------

def _assert_strategies(target: str) -> None:
    pipeline = build_pipeline(target)
    compiler = _PipelineCompiler(
        pipeline=pipeline, task="verify", adapter=None,  # type: ignore[arg-type]
        mode="fine", task_id="verify-p4", evaluator=None,
    )
    failures: list[tuple[str, str]] = []
    for spec in pipeline._groups:
        compound = compiler._compile_group(spec)
        n = len(spec.agents)
        if n >= 2 and compound.sequential_injection_strategy != "deps":
            failures.append((spec.name, compound.sequential_injection_strategy))
    if failures:
        print("STRATEGY ASSERTION FAILED:")
        for name, strat in failures:
            print(f"  {name}: expected 'deps', got {strat!r}")
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Post-hoc judge — score the briefer's synthesis
# ---------------------------------------------------------------------------

_JUDGE_RUBRIC = """\
You are evaluating a competitive intelligence brief on the target company \
{target}. Score it on three dimensions, each in [0.0, 1.0]:

1. coverage    — does the brief surface the most important entities, claims, \
                 and forward-looking signals you would expect from public \
                 materials about {target}?
2. grounding   — are claims plausible and consistent with what is widely \
                 known about {target}? Penalize fabrications and \
                 unsupported assertions.
3. structure   — is it well-organized, with an executive summary, per-lens \
                 paragraphs, and key takeaways, written at one-page length?

Reply with ONLY a JSON object on a single line:
{{"coverage": <float>, "grounding": <float>, "structure": <float>}}

Brief to evaluate:
---
{brief}
---
"""


def _build_judge_adapter(judge_provider: str, judge_model: str):
    if judge_provider == "openai":
        from agentic_capsules.adapters.openai import OpenAIAdapter
        return OpenAIAdapter(model=judge_model)
    elif judge_provider == "anthropic":
        from agentic_capsules.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=judge_model)
    raise ValueError(f"Unknown judge provider: {judge_provider!r}")


def _judge(judge_adapter, target: str, brief: str) -> tuple[float, dict]:
    """Send the brief to the judge and return (mean_score, details_dict)."""
    from agentic_capsules.core.types import LLMMessage
    prompt = _JUDGE_RUBRIC.format(target=target, brief=brief)
    reply = judge_adapter.complete(
        [LLMMessage(role="user", content=prompt)],
        tools=None,
    )
    # Robust JSON extraction — model may wrap in code fence or add commentary.
    m = re.search(r"\{[^{}]*\}", reply)
    if not m:
        return 0.0, {"raw": reply, "error": "no JSON in reply"}
    try:
        details = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return 0.0, {"raw": reply, "error": str(exc)}
    keys = ("coverage", "grounding", "structure")
    vals = [float(details.get(k, 0.0)) for k in keys]
    mean = sum(vals) / len(vals)
    return mean, details


# ---------------------------------------------------------------------------
# Checkpoint helpers (parallel of long_chain_research.py)
# ---------------------------------------------------------------------------

_CHECKPOINT_FIELDS = (
    "target", "provider", "model", "exec_mode", "quality",
    "judge_provider", "judge_model",
)


def _checkpoint_fingerprint(d: dict) -> str:
    payload = "|".join(f"{k}={d[k]!r}" for k in _CHECKPOINT_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_load(path: str, fingerprint: str):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        saved_fp, saved_runs = pickle.load(fh)
    if saved_fp != fingerprint:
        print(f"  [checkpoint] fingerprint mismatch — saved={saved_fp} current={fingerprint}",
              file=sys.stderr)
        print(f"  [checkpoint] delete to overwrite: rm {path}", file=sys.stderr)
        raise SystemExit(2)
    print(f"  [checkpoint] resuming from {path} — {len(saved_runs)} run(s) recovered")
    return saved_runs


def _checkpoint_save(path: str, fingerprint: str, runs: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump((fingerprint, runs), fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _checkpoint_clear(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Main eval driver
# ---------------------------------------------------------------------------

def run_eval(
    target:          str  = "Stripe",
    n_runs:          int  = 3,
    provider:        str  = "anthropic",
    model:           str | None = None,
    exec_mode:       str  = "fine_parallel",
    live:            bool = False,
    quality:         bool = False,
    judge_provider:  str  = "openai",
    judge_model:     str  = "gpt-4o",
    checkpoint_path: str | None = None,
) -> EvalResult:
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}. Available: {sorted(TARGETS)}")
    if exec_mode not in _EXEC_MAP:
        raise ValueError(f"Unknown exec mode {exec_mode!r}. Available: {sorted(_EXEC_MAP)}")
    parallel, mode = _EXEC_MAP[exec_mode]

    _assert_strategies(target)

    if live:
        if provider not in _PROVIDER_MAP:
            raise ValueError(f"Unknown provider {provider!r}")
        module_path, cls_name, default_model = _PROVIDER_MAP[provider]
        effective_model = model or default_model
        effective_provider = provider
        import importlib
        adapter_cls = getattr(importlib.import_module(module_path), cls_name)
        inner_adapter = adapter_cls(model=effective_model)
    else:
        # Offline path is intentionally not implemented — this pipeline has
        # 14 unique system prompts and a scripted adapter would not exercise
        # the parallel/compound code paths meaningfully. Use --live.
        raise SystemExit("This eval requires --live (no scripted adapter for P-4).")

    judge_adapter = _build_judge_adapter(judge_provider, judge_model) if quality else None

    pipeline = build_pipeline(target)
    policy = pipeline._policy

    result = make_eval_result(
        provider=effective_provider,
        model=effective_model,
        sensitivity="balanced",
        company=f"{target} ({exec_mode})",
        policy=policy,
    )

    print_run_header(effective_provider, effective_model, "balanced", target, n_runs)
    print(f"  Pipeline   : multi_source_brief — 14 agents, 6 groups")
    print(f"  Exec mode  : {exec_mode}  (parallel={parallel}, mode={mode})")
    if quality:
        print(f"  Judge      : {judge_provider} / {judge_model} (post-hoc on briefer output)")
    print()

    fingerprint = ""
    if checkpoint_path:
        fingerprint = _checkpoint_fingerprint({
            "target": target, "provider": effective_provider,
            "model": effective_model, "exec_mode": exec_mode,
            "quality": quality, "judge_provider": judge_provider,
            "judge_model": judge_model,
        })
        recovered = _checkpoint_load(checkpoint_path, fingerprint)
        if recovered:
            result.runs.extend(recovered)

    start_idx = len(result.runs)
    task = f"Produce a competitive intelligence brief on {target}."

    for i in range(start_idx, n_runs):
        # Fresh adapter wrapper per run so call counts are per-run.
        counting_adapter = _CountingAdapter(inner_adapter)
        # Fresh pipeline per run so internal state doesn't accumulate across runs.
        pipeline = build_pipeline(target)

        run_start = time.perf_counter()
        pr = pipeline.run(
            task=task, adapter=counting_adapter,
            mode=mode, parallel=parallel,
        )
        run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0

        # Override latency with the wall-clock we measured (parallel executor's
        # latency_ms aggregation may sum group latencies rather than measure
        # wall-clock; we want wall-clock for the head-to-head).
        pr.latency_ms = run_elapsed_ms

        run = collect_run(i, pr, pipeline)

        # Post-hoc judge: score the briefer's synthesis output.
        if quality and judge_adapter is not None:
            mean, details = _judge(judge_adapter, target, pr.output)
            # Attach to the synthesis group's RunRecord so existing reporting
            # picks it up. The synthesis group is always the last one.
            for rec in run.records:
                if rec.group == "synthesis":
                    rec.quality_score = mean
                    rec.quality_details = details
                    break

        result.runs.append(run)

        # Inline summary for each run (the standard print_run is too verbose
        # here because we have 6 groups × forced mode → no useful per-group
        # signal). Print a single line per run instead.
        q_str = ""
        if quality:
            for rec in run.records:
                if rec.group == "synthesis" and rec.quality_score is not None:
                    q_str = f"  quality={rec.quality_score:.3f}"
                    break
        print(f"  run {i+1:>2}/{n_runs}  calls={counting_adapter.calls:>3}  "
              f"tokens={pr.token_usage:>6}  latency={run_elapsed_ms/1000:6.1f}s{q_str}")

        if checkpoint_path:
            _checkpoint_save(checkpoint_path, fingerprint, result.runs)

    print()
    print_summary(result)

    # Aggregate quality across runs (post-hoc).
    if quality:
        scores = [
            rec.quality_score
            for run in result.runs
            for rec in run.records
            if rec.group == "synthesis" and rec.quality_score is not None
        ]
        if scores:
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            print(f"  quality (briefer): mean={mean:.3f} ±{std:.3f}  n={len(scores)}")

    if live:
        append_result(result)
        print(f"\n  Results appended to evals/last_eval.md")

    if checkpoint_path:
        _checkpoint_clear(checkpoint_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source brief eval — P-4")
    parser.add_argument("--target", default="Stripe", choices=sorted(TARGETS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--exec", dest="exec_mode", default="fine_parallel",
                        choices=sorted(_EXEC_MAP),
                        help="Execution mode (default: fine_parallel)")
    parser.add_argument("--live", action="store_true",
                        help="Call real LLM API (required — no scripted adapter)")
    parser.add_argument("--quality", action="store_true",
                        help="Post-hoc judge quality on briefer output")
    parser.add_argument("--judge-provider", default="openai",
                        choices=["anthropic", "openai"], dest="judge_provider")
    parser.add_argument("--judge-model", default="gpt-4o", dest="judge_model")
    parser.add_argument("--checkpoint", default=None, dest="checkpoint_path")
    args = parser.parse_args()

    run_eval(
        target=args.target,
        n_runs=args.runs,
        provider=args.provider,
        model=args.model,
        exec_mode=args.exec_mode,
        live=args.live,
        quality=args.quality,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        checkpoint_path=args.checkpoint_path,
    )
