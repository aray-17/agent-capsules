# Contributing to Agent Capsules

Agent Capsules is a single-maintainer research project. Bug reports
are triaged within ~2 weeks; pull requests are reviewed within
~3 weeks. Best effort, not SLA. Forks are welcome — Apache 2.0
explicitly permits forking and divergence without coordination.

## Scope

### Welcome — please send a PR

- Bug fixes with a reproduction (failing test or minimal script)
- Documentation, typo, and example fixes
- New tests for existing behavior
- Additional model adapters that follow the existing adapter
  contract in `src/agentic_capsules/adapters/`

### Open an issue first — wait for acknowledgment before sending a PR

- API changes (anything visible to pipeline authors)
- New framework features
- Changes that touch the composition score, quality gate, or
  escalation ladder. These behaviors back specific claims in the
  paper and require careful review.
- New evaluation methodology or new benchmarks

### Out of scope — will be closed

- Changes that would invalidate published claims without a clear
  upgrade path or revised evaluation
- Dependencies on private datasets or paid APIs in the test path
- Production infrastructure unrelated to the framework's
  programming model

## Tests

PRs must keep the offline test suite green:

```
pip install -e ".[dev]"
pytest
```

The offline suite uses scripted adapters — no API keys required.
Live evaluation against real models is reserved for separate
benchmarking and is not part of CI.

## Eval methodology and runtime internals

Some evaluation methodology, per-call probes, overnight harnesses,
and architectural deep-dives are intentionally maintained outside
this repository. If you want to discuss them, contact
**research@anindaray.com**.

## Citing the paper

If your contribution is in the context of academic work, please
cite the paper this framework is described in. See `CLAIMS.md` for
the citation block.
