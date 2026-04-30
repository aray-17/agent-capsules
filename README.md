# agent-capsules

A runtime-adaptive Python framework for multi-agent LLM pipelines.
Observes per-group coordination overhead, switches between
fine-grained and compound execution per group, and gates every
mode switch on rolling-mean output quality so cost reductions
never silently degrade results.

You declare the pipeline once. The runtime handles compilation,
mode selection, cross-provider adaptation, quality shadowing, and
escalation.

> **From the paper?** This codebase is the artifact for
> *Agent Capsules: Quality-Gated Granularity Control for
> Multi-Agent LLM Pipelines*. The exact code state cited in the
> paper is tagged
> [`v1.0-arxiv`](https://github.com/aray-17/agent-capsules/releases/tag/v1.0-arxiv);
> `main` may have evolved since.

## Headline results

Head-to-head against two representative alternatives on the same
pipelines, same models, same judge:

| Baseline | Pipeline | Agent Capsules wins by | Quality delta |
|---|---|---|---|
| Hand-tuned LangGraph | 14-agent competitive intelligence (Haiku, 15 runs/cell) | 51% fewer fine-mode input tokens, 42% fewer compound-mode input tokens | +0.020 / +0.017 |
| Uncompiled DSPy | 5-agent due diligence (Sonnet, 7 tasks) | 19% fewer total tokens | parity (+0.012, within judge noise floor) |
| DSPy with MIPROv2 compilation | 5-agent due diligence (Sonnet, 7 tasks) | 68% fewer total tokens | +0.052 |

Full methodology, per-cell data, and statistical caveats:
[`paper/paper.pdf`](paper/paper.pdf) §11 and [`CLAIMS.md`](CLAIMS.md).

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
from agentic_capsules import Pipeline, Tool
from agentic_capsules.adapters.anthropic import AnthropicAdapter

search = Tool(
    "web_search",
    "Search the web for current information.",
    input_schema={"query": "str"},
    fn=lambda args: {"results": "..."},
)

pipeline = (
    Pipeline("research")
    .group("research")
        .agent("researcher", "Find key facts about the topic.", tools=[search])
        .agent("verifier",   "Cross-check the findings for accuracy.")
    .group("writing")
        .agent("writer", "Draft a clear 200-word summary.")
        .agent("editor", "Improve clarity and conciseness.")
)

result = pipeline.run(
    "AI safety challenges at scale",
    adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
)

print(result.output)
print(result.mode_used)        # {"research": "fine", "writing": "fine"}
print(result.recommendation)   # {"research": "COMPOSE", "writing": "MAINTAIN"}
```

## How it adapts

Each group is observed independently. The runtime computes a
behavioral *composition score* from coordination overhead, agent
count, tool-call rate, and dependency depth. When the score clears
the configured threshold and the rolling-mean output quality
remains above the floor, the group is compiled into a compound
call (one LLM invocation for several agents). If quality dips, the
controller escalates through *standard → two-phase → sequential*
compound, and finally reverts to fine-grained execution. No model-
or pipeline-specific configuration is required for the controller
to make these decisions; defaults work across Anthropic, OpenAI,
and Google adapters out of the box.

The full mode ladder, the composition-score formula, and the
quality-gate dynamics are in the paper (§4–§9).

## Examples

- [`examples/research_pipeline.py`](examples/research_pipeline.py) — sequential research/writing pipeline
- [`examples/code_review_pipeline.py`](examples/code_review_pipeline.py) — fan-out + converge over reviewers
- [`examples/competitive_analysis.py`](examples/competitive_analysis.py) — multi-source brief
- [`examples/content_creation.py`](examples/content_creation.py) — small writing pipeline
- [`examples/advanced/`](examples/advanced/) — calibration, custom policies, per-group overrides

## Demo app

A Streamlit app that runs three pipelines through the controller
and visualizes the mode decisions and per-group telemetry:

```bash
streamlit run demo/app.py
```

The demo's scripted adapter requires no API keys; the live tab
uses the Anthropic, OpenAI, or Google adapter if a corresponding
key is set.

## Tests

```bash
pytest
```

The test suite runs offline against scripted adapters — no API
keys required. Live evaluation is reserved for separate
benchmarking and is not part of CI.

## API documentation

Build the Sphinx docs locally:

```bash
cd docs && make html
open _build/html/index.html
```

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

The arXiv URL will be added here once the preprint is live.

## Issues and pull requests

This is a single-maintainer research project. Bug reports are
triaged within ~2 weeks; pull requests are reviewed within
~3 weeks. Best effort, not SLA. See [CONTRIBUTING.md](CONTRIBUTING.md)
for what falls in scope. Forks are welcome — Apache 2.0 explicitly
permits forking and divergence.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Independent author. Correspondence: research@anindaray.com.
ORCID: [0009-0007-3029-8265](https://orcid.org/0009-0007-3029-8265).
