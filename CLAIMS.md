# Paper claims and supporting evidence

This file lists the load-bearing claims of
*Agent Capsules: Quality-Gated Granularity Control for Multi-Agent
LLM Pipelines* and points each claim at the artifact in this
repository that backs it. The paper PDF is at
[`paper/paper.pdf`](paper/paper.pdf); the exact code state cited
in the paper is tagged
[`v1.0-arxiv`](https://github.com/aray-17/agent-capsules/releases/tag/v1.0-arxiv).

## Headline numbers

| Comparison | Pipeline | Result | Evidence file |
|---|---|---|---|
| Agent Capsules vs.\ hand-tuned LangGraph (FINE mode) | multi_source_brief, 14 agents, Haiku, 15 runs/cell | −51% input tokens at +0.020 quality | [`evals/headtohead_langgraph.json`](evals/headtohead_langgraph.json) |
| Agent Capsules vs.\ hand-tuned LangGraph (compound mode) | multi_source_brief, 14 agents, Haiku, 15 runs/cell | −42% input tokens at +0.017 quality | [`evals/headtohead_langgraph.json`](evals/headtohead_langgraph.json) |
| Agent Capsules vs.\ uncompiled DSPy | due_diligence, 5 agents, Sonnet, 7 tasks | −19% total tokens at quality parity | [`evals/headtohead_dspy.md`](evals/headtohead_dspy.md) |
| Agent Capsules vs.\ DSPy + MIPROv2 | due_diligence, 5 agents, Sonnet, 7 tasks | −68% total tokens at +0.052 quality | [`evals/headtohead_dspy.md`](evals/headtohead_dspy.md) and [`evals/dspy/compiled/due_diligence.json`](evals/dspy/compiled/due_diligence.json) |

The Agent Capsules cell behind both DSPy comparisons is
``ac_compound_sequential`` (the mode the controller escalates to
after the quality gate fires on standard compound). See
[`evals/headtohead_dspy.md`](evals/headtohead_dspy.md) for the
five-row aggregate (`ac_fine`, `ac_auto_with_evaluator`,
`ac_compound_sequential`, `dspy_uncompiled`, `dspy_mipro`).

## Mechanism claims

**C1. The composition score partitions models by tool-call
behavior.** OpenAI and Google models score below the balanced
threshold (tools/agent ≈ 1.0); Anthropic models score above
(tools/agent ≈ 2.0–2.5). The controller fires compound execution
on Anthropic models and stays in FINE on OpenAI / Google,
without per-model configuration. Paper §6; data in
[`evals/paper_tables.md`](evals/paper_tables.md) (Table 4).

**C2. The escalation ladder recovers quality on standard-compound
failures.** When standard compound fails the rolling-mean quality
floor, the controller escalates to two-phase, then to sequential.
Sonnet research-group quality recovers from 0.313 (standard) to
0.724 (sequential) on the code-review pipeline, ~14× the judge's
minimum detectable difference. Paper §9.3; data in
[`evals/paper_tables.md`](evals/paper_tables.md) (Table 12).

**C3. The quality gate matches the LLM-judge oracle on every
measured cell.** Across all (model, group, mode) cells in the
paper's evaluation, the controller's routing decision agrees with
an oracle that knows every cell's outcome in advance. The paper
treats this as the load-bearing operational claim and is explicit
about scope: the agreement holds within the LLM-judge regime,
once the rolling-mean window has saturated. Paper §9.4–§9.5;
data in [`evals/paper_tables.md`](evals/paper_tables.md) (Table 13)
and the structured form at
[`evals/oracle_routing.json`](evals/oracle_routing.json).

**C4. Compound execution preserves structure on synthesis-style
groups and degrades it on tool-using groups.** A schema-compliance
side-channel measures structural drop independently of the LLM
judge and agrees directionally: synthesis groups preserve 100% of
required structure under compound; research and analysis groups
drop 27%–42%. Paper §9.5(b); data in
[`evals/paper_tables.md`](evals/paper_tables.md) (Table 14).

## Negative result

**N1. Forced reasoning Phase A does not rescue compound quality.**
We tested extending Phase A to tool-free agents with a structured
reasoning pre-pass; on Sonnet it improved analysis quality
(+0.084 peak) but regressed research quality (−0.067 to −0.175
across the `compose_at` sweep). Injecting more context into a
merged call worsens compression rather than relieving it. The
framework's escalation ladder therefore recovers quality by
moving *toward* per-agent dispatch (sequential), not by rewriting
merged prompts. Paper §7.4; data in
[`evals/paper_tables.md`](evals/paper_tables.md) (§7.4 section).

## Reproducibility notes

- The mechanism-claim and tunable-default measurements cited
  throughout the paper are reproduced in
  [`evals/paper_tables.md`](evals/paper_tables.md) — Tables 4, 6,
  7, 8, 9, 10, 11, 12, 13, 14, 17, 18 plus the §7.4 negative
  result. Table 13 also has a structured form at
  [`evals/oracle_routing.json`](evals/oracle_routing.json) for
  programmatic verification of the oracle-equivalence claim.
- The pipeline definitions are at
  [`evals/shared/pipeline.py`](evals/shared/pipeline.py) (due
  diligence, P-1) and the per-pipeline files
  [`evals/code_review.py`](evals/code_review.py),
  [`evals/long_chain_research.py`](evals/long_chain_research.py),
  [`evals/multi_source_brief.py`](evals/multi_source_brief.py).
  Same agent prompts, same tool stubs, same topology as cited in
  the paper.
- The DSPy head-to-head uses the compiled artifact at
  [`evals/dspy/compiled/due_diligence.json`](evals/dspy/compiled/due_diligence.json).
  Recompiling MIPROv2 from scratch costs ~$15–20 in real API
  calls; the compiled artifact lets reviewers verify the
  head-to-head without paying for a recompile.
- The LangGraph baseline is implemented as a published-tutorial
  topology with the same tool stubs as the Agent Capsules
  pipeline. The implementation is *ours*; reviewers should
  consider that source attribution when interpreting the
  comparison numbers.
- Quality is measured by LLM judge: `claude-opus-4-6` on Anthropic
  evaluation runs, `gpt-4o` on OpenAI / Google runs. Cross-provider
  absolute scores are not directly comparable; the paper
  explicitly disclaims this in §9.

## Operational data not in this repository

A larger body of operational evaluation work — per-call probes,
overnight resilience harnesses, gap audits, multi-week eval logs
— is maintained outside this public repository. If your work
depends on understanding *how* the paper's numbers were produced
(rather than verifying *that* they reproduce), email
**research@anindaray.com**.

## Citing

```bibtex
@article{ray2026agentcapsules,
  title  = {Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines},
  author = {Ray, Aninda},
  year   = {2026},
  note   = {arXiv preprint, forthcoming.},
  url    = {https://github.com/aray-17/agent-capsules}
}
```
