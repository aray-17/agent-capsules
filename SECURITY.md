# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in Agent Capsules,
please email **research@anindaray.com** rather than filing a public
GitHub issue. Include:

- A description of the issue and the affected component
- Steps to reproduce
- Any proof-of-concept code (please keep it minimal)

You can expect an acknowledgement within **7 days**. Please give
the maintainer reasonable time to investigate and prepare a fix
before public disclosure.

## Scope

Agent Capsules is a Python framework that orchestrates calls to
third-party LLM APIs. It exposes no network listener and stores no
credentials of its own. Most vulnerability reports in this scope
will fall into one of:

- Prompt-injection paths that bypass quality-gate logic
- Resource exhaustion (token, memory, or persistence-layer abuse)
- Issues in the optional Redis persistence backend

Issues in upstream LLM providers, in user code that calls the
framework, or in deployments that expose the framework over a
network are out of scope for this policy and should be reported to
the relevant party.
