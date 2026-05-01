# Paper tables and figures — measurement data

This file reproduces the load-bearing tables, figures, and the §7.4
negative-result data points cited in *Agent Capsules: Quality-Gated
Granularity Control for Multi-Agent LLM Pipelines*. It is the
public, sliced form of the underlying measurements; raw per-cell
probe traces and overnight run logs that produced the values stay
in private operational logs.

Quality is measured by an LLM judge: `claude-opus-4-6` on Anthropic
provider runs, `gpt-4o` on OpenAI / Google runs (paper §9.1).
Within-provider relative comparisons are valid; cross-provider
absolute scores are **not** directly comparable.

The companion structured form of Table 13 is at
[`evals/oracle_routing.json`](oracle_routing.json).

---

## §6 — Composition Score

### Table 4. Composition scores in FINE mode (research group, 5-run mean ± std)

Measurement basis for the cross-provider partition claim (C1 in
[CLAIMS.md](../CLAIMS.md)). Models above the balanced threshold
(`compose_at = 0.23`) are candidates for compound execution.

| Model         | Score          | Std    | Fires?         | Driver                  |
|---------------|----------------|--------|----------------|-------------------------|
| GPT-4o-mini   | 0.177          | 0.002  | No             | tools/agent = 1.0       |
| GPT-4o        | 0.181          | 0.003  | No             | tools/agent = 1.0       |
| Gemini-flash  | 0.208          | 0.001  | No             | tools/agent = 1.0       |
| Sonnet        | 0.245          | 0.019  | Aggressive     | tools/agent ≈ 2.0       |
| Haiku         | 0.264 – 0.299  | 0.034  | Yes            | tools/agent = 2.0 – 2.5 |

Score variance is ≤ ±0.034 across runs for any (model, pipeline)
pair. Gemini-2.5-pro scores 0.177 – 0.180 — statistically identical
to GPT-4o-mini despite being a more capable model — confirming the
score is a behavioral (not capability) signal.

---

## §7.4 — Negative Result: Reasoning Phase A Does Not Rescue Compound (N1)

Backs **N1** in [CLAIMS.md](../CLAIMS.md). The framework's
escalation ladder recovers quality by moving *toward* per-agent
dispatch, not by rewriting merged prompts.

We tested extending Phase A to tool-free agents with a structured
reasoning pre-pass, so each agent would produce its full analysis
before Phase B merged the results. Measurements on Sonnet, code-review
pipeline:

| Mode                                  | Research quality      | Analysis quality      |
|---------------------------------------|-----------------------|-----------------------|
| Standard compound (baseline)          | 0.742                 | 0.683                 |
| Two-phase + reasoning Phase A         | 0.675 (−0.067)        | 0.767 (+0.084)        |
| Sequential compound                   | 0.758 – 0.833         | 0.775 – 0.783         |

Across the `compose_at` sweep, the reasoning Phase A regressed
research quality by **−0.067 to −0.175** while improving analysis by
**+0.084** at the `compose_at=0.36` peak. Sequential compound, by
contrast, clears the floor on both groups.

Conclusion: injecting more context into a merged Phase B gives the
model more material to compress, not less reason to compress.
Merged calls are the fundamental compression bottleneck; sequential
execution (no merging) is required for multi-agent reasoning groups.

---

## §8 — Tunables (defaults justification)

### Table 6. Forced concise output guidance (sequential compound, research group, 7-run mean ± std)

Backs the `output_guidance` tunable defaults in §8.1. ΔQ thresholds:
≥ 0.030 (Anthropic, opus judge); ≥ 0.065 (Gemini, gpt-4o judge).

| Model        | Base Q          | Concise Q        | ΔQ      | Token savings |
|--------------|-----------------|------------------|---------|---------------|
| Sonnet       | 0.831 ± 0.010   | 0.842 ± 0.065    | +0.011  | −85.5%        |
| Haiku        | 0.856 ± 0.010   | 0.851 ± 0.049    | −0.005  | −74.3%        |
| Gemini-flash | 0.900 ± 0.025   | 0.740 ± 0.153    | −0.160  | −9.8%         |

Sonnet +0.011 and Haiku −0.005 sit inside the opus minimum
detectable difference (0.030) and are statistical nulls, delivering
token savings at null quality delta. Gemini −0.160 exceeds the
gpt-4o noise floor (0.065) and is a genuine regression, motivating
the `auto` default in Table 7.

### Table 7. Auto output guidance routing (due_diligence pipeline, 3 FINE warmup + 7 forced-compound runs per cell)

Per-group routing decisions on the due_diligence pipeline. Output-only
tokens are compared to the forced-compound no-guidance baseline from
Table 6.

| Model        | Auto out tok | Savings        | Routing (research / analysis / synthesis) |
|--------------|--------------|----------------|-------------------------------------------|
| Sonnet       | 9,395        | −63%           | none / concise / concise                  |
| Haiku        | 5,219        | −64%           | none / concise / concise                  |
| Gemini-flash | 3,738        | −3% (neutral)  | none / none / none                        |

No deployer is worse off under `auto`; the Gemini deployer is
strictly better off than under forced concise.

### Table 8. Context injection strategy (predecessor_only vs full)

Backs the `context_injection_strategy = "predecessor_only"` default
in §8.2. Top block: Sonnet, opus judge (min detectable diff 0.030).
Bottom block: P-3 long-chain validation, gpt-4o judge (min
detectable diff 0.065).

| Pipeline                   | Strategy        | Quality | ΔQ      | ΔTok   | Note               |
|----------------------------|-----------------|---------|---------|--------|--------------------|
| due_diligence              | full            | 0.788   | —       | —      | 2-agent groups     |
| due_diligence              | predecessor     | 0.821   | +0.033  | —      | above floor        |
| code_review                | full            | 0.858   | —       | —      | 3-agent chain      |
| code_review                | predecessor     | 0.829   | −0.029  | —      | noise floor        |
| long_chain_research (Son.) | full            | 0.715   | —       | —      | 4+3+1 chain        |
| long_chain_research (Son.) | predecessor     | 0.713   | −0.002  | −2.8%  | noise floor        |
| long_chain_research (Hai.) | full            | 0.673   | —       | —      | 4+3+1 chain        |
| long_chain_research (Hai.) | predecessor     | 0.762   | +0.089  | −2.0%  | above MDD          |

### Table 9. Budgeted structural hint (research group, standard compound, 7-run mean ± std)

Backs the `merged_output_structure = "budgeted"` default in §8.3.

| Model                  | Baseline        | Budgeted        | ΔQ                  |
|------------------------|-----------------|-----------------|---------------------|
| Sonnet                 | 0.702 ± 0.055   | 0.709 ± 0.122   | +0.007 (neutral)    |
| Haiku                  | 0.500 ± 0.000   | 0.709 ± 0.046   | +0.209              |
| gemini-2.5-flash-lite  | 0.483 ± 0.052   | 0.887 ± 0.087   | +0.404              |

Haiku +0.209 and Gemini-flash +0.404 are far above their respective
MDDs (0.030 / 0.065) and carry the decision. Sonnet +0.007 is a
statistical null — adoption as default imposes no cost where the
hint is not needed.

### Table 10. Cache-aligned prompts (Sonnet, 7-run mean ± std)

Backs the `cache_aligned_prompts = True` default in §8.4. Min
detectable diff 0.030 (opus judge).

| Pipeline      | Baseline        | Cache-aligned   | ΔQ      |
|---------------|-----------------|-----------------|---------|
| due_diligence | 0.839 ± 0.020   | 0.883 ± 0.066   | +0.044  |
| code_review   | 0.873 ± 0.048   | 0.934 ± 0.033   | +0.061  |

Both deltas exceed the 0.030 detection threshold in the positive
direction.

---

## §9 — Quality Gate

### Table 11. LLM judge reliability (7 reps, 3 quality levels)

Calibration baseline for every other quality number in the paper.
The std is intra-rater consistency; the minimum detectable
difference is the quality delta a comparison must clear before it
counts as signal.

| Judge        | Mean std | Min detectable diff | Sufficiency      |
|--------------|----------|---------------------|------------------|
| GPT-4o       | 0.032    | 0.065               | 7 runs sufficient |
| Claude-opus  | 0.012    | 0.030               | 7 runs sufficient |

Calibration gap: opus scores ~0.17 lower than gpt-4o on near-identical
inputs (0.725 vs ~0.90 on a calibration pair). Within-provider
relative comparisons are valid; cross-provider absolute comparisons
are not.

### Table 12. Escalation ladder validation (C2)

Backs **C2** in [CLAIMS.md](../CLAIMS.md). Code-review review
group, Sonnet aggressive sensitivity, n=7, claude-opus-4-6 judge.

| Configuration       | Quality ± std    | Tokens   | Compound fires |
|---------------------|------------------|----------|----------------|
| escalation = False  | 0.313 ± 0.137    | 189,632  | 1 / 7          |
| escalation = True   | 0.724 ± 0.068    | 170,734  | 5 / 7          |
| Δ                   | +0.411           | −10%     | +4 / 7         |

The +0.411 improvement is ~14× the opus judge MDD (0.030); tokens
drop 10%, latency drops 15%. Recovered quality lands 0.026 below
the 0.75 floor — exactly one MDD, statistically indistinguishable
from the floor. The improvement decomposes into tier rescue
(two-phase restores tool access, ≈ +0.18) plus controller
stabilisation (the rolling-mean gate stops reverting compound to
FINE on noise).

### Table 13. COMPOUND quality ceiling by model and execution mode (C3)

Backs **C3** in [CLAIMS.md](../CLAIMS.md) — the paper's
load-bearing operational claim (oracle-equivalent routing). Quality
floor = `0.75`. **Bold** passes the floor.

| Model                  | Mode             | research      | analysis      | synthesis     | Passes floor?            |
|------------------------|------------------|---------------|---------------|---------------|--------------------------|
| GPT-4o-mini            | auto             | 0.725         | 0.683         | **0.825**     | synthesis                |
| GPT-4o-mini            | sequential       | **0.883**     | **0.808**     | **0.833**     | all                      |
| Gemini-flash           | auto             | 0.608         | 0.733         | **0.842**     | synthesis                |
| GPT-4o                 | standard         | 0.583         | 0.700         | 0.742         | none                     |
| Haiku                  | any              | 0.717         | 0.608         | 0.742         | none                     |
| Sonnet                 | auto / standard  | **0.775**     | 0.675         | **0.833**     | synthesis; research border |
| Sonnet                 | sequential       | **0.833**     | **0.783**     | **0.833**     | all                      |

Synthesis groups (no tools, single aggregation agent) reliably pass
the floor in standard mode. Research and analysis require sequential
compound to clear the floor, and only on mid-tier models (Sonnet,
GPT-4o-mini). Haiku fails in all modes — root cause is the BSI
metric: tool calls in COMPOUND / tool calls in FINE = 0.0 (Haiku
suppresses tool use entirely under compound framing).

#### Figure 5 — Quality ceiling heatmap (visualisation of Table 13)

Same data, visualised as a model × group × mode heatmap with
`quality_floor = 0.75` as the green/red threshold. Only sequential
compound on mid-tier models (Sonnet, GPT-4o-mini) achieves
consistent floor passage across all three task groups. The figure
adds no data beyond Table 13.

### Table 14. Schema-compliance delta (C4)

Backs **C4** in [CLAIMS.md](../CLAIMS.md). Forced compound
(`compose_at = 0.20`) vs FINE baseline. FINE = 1.000 by construction
(each agent emits its own section), so compound values measure how
much required structure naive compound drops. Directionally
consistent with the LLM-judge results in Table 13 but not
comparable in magnitude.

| Model        | Group     | Compliance drop                |
|--------------|-----------|--------------------------------|
| Haiku        | research  | −0.396 (0.604 vs 1.000)        |
| Haiku        | analysis  | −0.392                         |
| GPT-4o       | research  | −0.417                         |
| GPT-4o-mini  | research  | −0.275                         |
| Sonnet       | analysis  | −0.325                         |
| GPT-4o-mini  | synthesis | 0.000 (preserves structure)    |
| Sonnet       | synthesis | 0.000 (preserves structure)    |

The two measurement regimes (LLM-judge in §9.4 / schema-compliance
here) agree on the partition: the same groups pass under LLM-judge
quality and preserve structure under schema compliance. Regime (a)
is the primary evidence; regime (b) is a structural consistency
check.

---

## §11 — Infrastructure Impact

### Table 17. Token savings under auto mode (Pareto-optimal `compose_at`)

Backs §11.1. Savings are realized only when the quality gate
passes (default `quality_floor = 0.75`).

| Model         | Auto savings | Sequential + concise | Gate-adjusted realized                 |
|---------------|--------------|----------------------|----------------------------------------|
| Haiku         | 53–75%       | 74%                  | gate blocks; effective 0%              |
| GPT-4o-mini   | 15–30%       | —                    | synthesis: 30% realized                |
| Gemini-flash  | 13–33%       | —                    | synthesis: 33% realized                |
| Sonnet        | 43–48%       | 85.5%                | all groups via sequential + concise    |

A model that saves 75% of tokens but fails the quality gate has
effective savings of zero, because the compound path is never
committed to. The quality gate is what makes the achievable-savings
number a trustworthy upper bound on realized savings rather than a
measurement of degraded output.

### Table 18. FINE vs COMPOUND latency reduction

Backs §11.2. Synthesis groups (1 agent) show negative reduction:
compound adds overhead with no compression benefit.

| Model         | Group     | FINE (ms)  | COMPOUND (ms) | Reduction  |
|---------------|-----------|------------|---------------|------------|
| Sonnet        | research  | 118,651    | 101,290       | 14.6%      |
| Sonnet        | analysis  | 309,223    | 101,290       | 67.2%      |
| Sonnet        | synthesis | 17,613     | 50,645        | −187.5%    |
| GPT-4o-mini   | research  | 16,887     | 13,434        | 20.4%      |
| GPT-4o-mini   | analysis  | 21,272     | 13,434        | 36.8%      |
| GPT-4o-mini   | synthesis | 5,177      | 6,717         | −29.8%     |

Latency benefit is model-tier dependent. Sonnet's per-call
wall-clock is 17,361–207,933 ms (mean 96,131 ms): a large per-call
fixed cost is paid once instead of N times in compound, so analysis
groups benefit dramatically. Synthesis groups already issue a
single call in FINE, so compound adds overhead without benefit.

---

## What is *not* in this file

Per the public-repo strategy, the following stay in the private
operational repo and are available on request to
**research@anindaray.com**:

- Raw per-cell probe traces and overnight run logs that produced
  these table values.
- Gap-audit batch outputs and recovery-pattern notes (the
  iterative work that motivated each default).
- Full Pareto sweeps (only the Pareto-optimal points are summarised
  here).
- Per-call capturing-adapter probes used to instrument the BSI tool
  behaviour metric.
