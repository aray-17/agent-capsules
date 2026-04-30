"""
Code-review pipeline — eval template for T-028 (eval breadth).

Exercises a different pipeline topology from the due-diligence baseline:
  - Higher agent count per group (3 reviewers in review group)
  - Tool-heavy fan-out pattern within a group
  - Different domain (code quality vs. financial analysis)

Pipeline shape:
  review     (3 agents, 2 tools) — security_reviewer, performance_reviewer,
              style_reviewer — all independently examine a code diff with tools.
              Fan-out pattern: each agent reads the same diff via tools.
  assessment (2 agents, no tools) — severity_classifier, impact_assessor.
              Sequential: impact_assessor reads severity_classifier output.
  report     (1 agent, no tools) — review_writer synthesises findings.

Score profile expected with new weights (0.45, 0.25, 0.00, 0.25, 0.05):
  review:     3 agents (w2=0.75→capped 1.0), 2 tool calls/agent (w4=0.667),
              overhead ~0.07 → score ~0.45×0.07 + 0.25×0.75 + 0.25×0.667
              ≈ 0.031 + 0.1875 + 0.167 = 0.385 → above balanced compose_at=0.23 ✓
  assessment: 2 agents (w2=0.5), 0 tool calls → ~0.25×0.5 = 0.125 — stays FINE
  report:     1 agent (w2=0.25), 0 tool calls → ~0.0625 — stays FINE

T-028 ref: phase_13_weight_redesign.md § Eval breadth.
"""
from __future__ import annotations

import random
import re

from agentic_capsules import Pipeline, Tool


# ---------------------------------------------------------------------------
# Scripted adapter — offline / CI mode, no API key needed
# ---------------------------------------------------------------------------

class CodeReviewScriptedAdapter:
    """
    Deterministic adapter for the code review pipeline.

    Returns realistic, domain-appropriate responses for each reviewer and
    assessor agent. Verbose enough to produce meaningful token counts.
    """
    context_window = 200_000

    _RESPONSES: dict[str, str] = {
        "SECURITY_REVIEWER": (
            "Security review findings:\n"
            "  CRITICAL: SQL injection risk in UserRepository.findByEmail() — "
            "raw string interpolation of user input into query at line 47. "
            "Must parameterise before merge.\n"
            "  HIGH: Authentication bypass possible in middleware/auth.py — "
            "JWT expiry check uses local time without timezone awareness. "
            "Token replay window up to 3600s in DST transition periods.\n"
            "  MEDIUM: Logging sensitive data — request_logger.py logs full request "
            "body including Authorization header. GDPR risk for EU users.\n"
            "  LOW: Dependency pinning missing in requirements.txt — "
            "3 transitive dependencies lack version bounds. Supply chain risk.\n"
            "OWASP Top 10 coverage: A03 Injection (critical), A07 Auth failures (high). "
            "Lint check showed 2 hardcoded secrets in test fixtures (should use env vars)."
        ),
        "PERFORMANCE_REVIEWER": (
            "Performance review:\n"
            "  BLOCKING: N+1 query in OrderService.get_orders_with_items() — "
            "queries DB once per order item. For a user with 100 orders × 10 items "
            "= 1001 queries. Fix: use JOIN or prefetch_related.\n"
            "  WARNING: Missing index on orders.user_id FK — full table scan on "
            "every user dashboard load. Estimated 4× slowdown at 100k rows.\n"
            "  INFO: Synchronous HTTP call to payment provider in request path — "
            "p99 latency will be dominated by provider SLA (~800ms). "
            "Recommend async task queue for non-blocking checkout flow.\n"
            "  INFO: 3 redundant cache misses in session middleware — "
            "cache.get() called 3× for same key per request. Memoize within request scope.\n"
            "Lint check: 0 linting errors, 3 complexity warnings (cyclomatic > 10)."
        ),
        "STYLE_REVIEWER": (
            "Code style review:\n"
            "  14 files modified, 847 lines added, 213 lines removed.\n"
            "  Violations: 8 PEP-8 line-length (>120 chars), 3 missing docstrings "
            "on public methods, 2 inconsistent naming (camelCase in snake_case module).\n"
            "  Test coverage: 68% on new code (target 80%). Missing coverage in "
            "payment/webhook_handler.py (0%) and auth/session.py (41%).\n"
            "  Architecture: new PaymentService introduces a circular import with "
            "UserService — violates layered architecture. Recommend injecting "
            "UserService as a dependency rather than importing directly.\n"
            "  Positive: consistent error handling pattern, good use of dataclasses, "
            "type annotations complete on all new public APIs.\n"
            "Overall style score: 6/10 — functional but needs cleanup before merge."
        ),
        "SEVERITY_CLASSIFIER": (
            "Severity classification:\n"
            "  BLOCK (must fix before merge):\n"
            "    - SQL injection (SECURITY_REVIEWER, line 47): severity CRITICAL\n"
            "    - N+1 query in OrderService (PERFORMANCE_REVIEWER): severity HIGH, "
            "will cause production outage at current growth rate within 30 days\n"
            "  BEFORE_RELEASE (fix before next release):\n"
            "    - JWT timezone bypass (SECURITY_REVIEWER): HIGH\n"
            "    - Missing DB index on orders.user_id (PERFORMANCE_REVIEWER): HIGH\n"
            "    - Circular import PaymentService/UserService (STYLE_REVIEWER): MEDIUM\n"
            "  FOLLOW_UP (track in backlog):\n"
            "    - Sensitive data logging: MEDIUM\n"
            "    - Async payment calls: INFO\n"
            "    - Style/coverage items: LOW\n"
            "Total: 2 blockers, 3 before-release, 4 follow-up."
        ),
        "IMPACT_ASSESSOR": (
            "Business impact assessment:\n"
            "  SQL injection (CRITICAL): full database compromise risk. "
            "Regulatory exposure: SOC 2 Type II audit failure, potential PCI-DSS "
            "violation. Estimated remediation: 2 hours. Must block merge immediately.\n"
            "  N+1 query: at current growth (40% MoM), will hit 100k orders in ~45 days. "
            "Without fix: order dashboard will time out for top 5% of users. "
            "Revenue impact: ~$12k/month in failed upsell impressions. "
            "Estimated remediation: 4 hours.\n"
            "  JWT bypass: exploitable only during DST transitions (2× per year). "
            "Low immediate risk but HIGH audit risk if discovered. Fix: 1 hour.\n"
            "  Overall merge recommendation: BLOCK. Two fixes required (SQL injection + N+1). "
            "Estimated combined effort: 6 engineering hours before merge is safe."
        ),
        "REVIEW_WRITER": (
            "CODE REVIEW SUMMARY\n\n"
            "Decision: REQUEST CHANGES — merge blocked pending 2 critical fixes.\n\n"
            "Critical issues:\n"
            "1. SQL injection (UserRepository:47) — parameterise query before merge. "
            "Full DB compromise risk.\n"
            "2. N+1 query (OrderService.get_orders_with_items) — will cause "
            "production timeout at current growth rate. Use JOIN or prefetch_related.\n\n"
            "Required before release:\n"
            "3. JWT timezone bypass — use UTC throughout, validate with timezone-aware "
            "datetime.\n"
            "4. Add index on orders.user_id — 4× query speedup on dashboard load.\n"
            "5. Break circular import PaymentService/UserService via DI.\n\n"
            "Nice to have (non-blocking):\n"
            "- Disable sensitive data logging in request_logger.py\n"
            "- Move payment provider call to async task queue\n"
            "- Improve test coverage: webhook_handler.py (0% → target 80%)\n\n"
            "Estimated effort: 6h critical + 4h required-before-release = 1.5 dev-days."
        ),
    }

    def complete(self, messages: list, tools=None) -> str:
        combined = " ".join(
            getattr(m, "content", str(m)) for m in messages
        )
        keys = re.findall(r"([A-Z][A-Z_]+_OUTPUT)", combined)
        seen: set[str] = set()
        parts: list[str] = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:  # type: ignore[func-returns-value]
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{self._RESPONSES.get(agent, 'Review complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tools() -> tuple[Tool, Tool]:
    """
    Stub tools for the code review pipeline.

    get_code_diff: returns a fixed patch representing the PR diff.
    lint_check:    returns a fixed lint report for the changed files.
    """
    get_code_diff = Tool(
        name="get_code_diff",
        description=(
            "Retrieve the git diff for a pull request. Returns a unified diff "
            "of all changed files with line numbers and context."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pr_id":  {"type": "string", "description": "Pull request ID"},
                "files":  {"type": "array",  "items": {"type": "string"},
                           "description": "Specific files to diff (optional)"},
            },
            "required": ["pr_id"],
        },
        fn=lambda args: {
            "pr_id":       args.get("pr_id", "pr-42"),
            "files_changed": 14,
            "lines_added":   847,
            "lines_removed": 213,
            "diff_excerpt": (
                "--- a/auth/repository.py\n"
                "+++ b/auth/repository.py\n"
                "@@ -44,7 +44,7 @@ class UserRepository:\n"
                "-    query = f\"SELECT * FROM users WHERE email = '{email}'\"\n"
                "+    query = f\"SELECT * FROM users WHERE email = '{email}' -- TODO: parameterise\"\n"
                "     return self.db.execute(query).fetchone()\n"
                "\n"
                "--- a/services/order_service.py\n"
                "+++ b/services/order_service.py\n"
                "@@ -88,10 +88,12 @@ class OrderService:\n"
                "     def get_orders_with_items(self, user_id: int):\n"
                "+        orders = Order.objects.filter(user_id=user_id)\n"
                "+        for order in orders:\n"
                "+            order.items = OrderItem.objects.filter(order_id=order.id)\n"
                "+        return orders"
            ),
        },
    )

    lint_check = Tool(
        name="lint_check",
        description=(
            "Run static analysis and linting on changed files. Returns violation "
            "counts, complexity metrics, and test coverage delta."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pr_id":      {"type": "string", "description": "Pull request ID"},
                "tool":       {"type": "string", "description": "Linter to use (flake8, pylint, mypy)"},
                "fail_level": {"type": "string", "description": "Minimum severity to report"},
            },
            "required": ["pr_id"],
        },
        fn=lambda args: {
            "violations": {
                "error":   2,
                "warning": 8,
                "info":    14,
            },
            "complexity_warnings": [
                {"file": "services/payment.py",  "function": "process_payment",  "cyclomatic": 12},
                {"file": "auth/middleware.py",    "function": "validate_token",   "cyclomatic": 11},
                {"file": "services/order_service.py", "function": "checkout",    "cyclomatic": 10},
            ],
            "type_errors":       0,
            "test_coverage_delta": -2.3,
            "new_code_coverage":  68.1,
            "target_coverage":    80.0,
            "secrets_detected": [
                {"file": "tests/fixtures/auth.py", "line": 12, "type": "api_key"},
                {"file": "tests/fixtures/auth.py", "line": 28, "type": "secret"},
            ],
        },
    )

    return get_code_diff, lint_check


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

TASK_TEMPLATE = "Review pull request {pr_id}"

PIPELINE_DESCRIPTION = (
    "review (3 agents, 2 tools) → assessment (2 agents) → report (1 agent)"
)


def build_pipeline_with_judge(
    sensitivity:                  str   = "balanced",
    judge_adapter                       = None,
    quality_floor:                float = 0.75,
    compound_execution_model:     str   = "standard",
    merged_output_structure:      str   = "none",
    output_guidance:              str   = "none",
    sequential_context_strategy:  str   = "full",
    cache_aligned_prompts:        bool  = False,
    escalation_enabled:           bool  = False,
):
    """
    Build the code review pipeline with an optional LLM judge evaluator.

    Mirrors the interface of evals.shared.pipeline.build_pipeline_with_judge
    for use in Track A gate runs on the P-2 pipeline.

    Args:
        sequential_context_strategy: S-1 — "full"|"predecessor_only".
        cache_aligned_prompts:       C-1 — Anthropic prefix caching restructure.
        escalation_enabled:          E-1 — quality-driven execution model escalation.

    Returns:
        ``(pipeline, evaluator)`` tuple. ``evaluator`` is None when
        ``judge_adapter`` is None.
    """
    from agentic_capsules import LLMJudgeEvaluator, ControllerPolicy
    from agentic_capsules.controller.policy import policy_for

    base_policy = policy_for(sensitivity)
    policy = ControllerPolicy(
        compose_at=base_policy.compose_at,
        decompose_at=base_policy.decompose_at,
        confidence=base_policy.confidence,
        min_observations=base_policy.min_observations,
        window_size=base_policy.window_size,
        score_weights=base_policy.score_weights,
        error_rate_threshold=base_policy.error_rate_threshold,
        context_util_threshold=base_policy.context_util_threshold,
        latency_threshold_ms=base_policy.latency_threshold_ms,
        quality_floor=quality_floor if judge_adapter is not None else None,
        compound_execution_model=compound_execution_model,
        merged_output_structure=merged_output_structure,
        output_guidance=output_guidance,
        sequential_context_strategy=sequential_context_strategy,
        cache_aligned_prompts=cache_aligned_prompts,
        escalation_enabled=escalation_enabled,
    )

    evaluator = LLMJudgeEvaluator(judge_adapter) if judge_adapter is not None else None

    get_code_diff, lint_check = _tools()
    pipeline = (
        Pipeline("code_review", policy=policy)
        .group("review")
            .agent("security_reviewer",
                "Review the pull request for security vulnerabilities. Use get_code_diff "
                "to examine changed files and lint_check to identify hardcoded secrets or "
                "insecure patterns. Focus on OWASP Top 10: injection, auth failures, "
                "sensitive data exposure, and dependency vulnerabilities. "
                "Rate each finding CRITICAL/HIGH/MEDIUM/LOW with exact file and line.",
                tools=[get_code_diff, lint_check])
            .agent("performance_reviewer",
                "Review the pull request for performance issues. Use get_code_diff to "
                "examine query patterns, loop complexity, and I/O patterns. Use lint_check "
                "to identify high cyclomatic complexity. Flag N+1 query patterns, missing "
                "indexes, synchronous I/O in hot paths, and redundant cache misses. "
                "Estimate performance impact with concrete numbers where possible.",
                tools=[get_code_diff, lint_check])
            .agent("style_reviewer",
                "Review the pull request for code style, architecture, and test coverage. "
                "Use get_code_diff for a complete file-level change summary. Use lint_check "
                "for violation counts and coverage metrics. Flag naming inconsistencies, "
                "missing docstrings, circular imports, and test coverage gaps. "
                "Give an overall style score 1–10 with a brief rationale.",
                tools=[get_code_diff, lint_check])
        .group("assessment")
            .agent("severity_classifier",
                "Classify each finding from the review group into: BLOCK (must fix before "
                "merge), BEFORE_RELEASE (fix before next release), or FOLLOW_UP (backlog). "
                "For each BLOCK item state the specific risk and estimated fix effort.")
            .agent("impact_assessor",
                "Assess the business impact of each BLOCK and BEFORE_RELEASE item. "
                "Quantify revenue, compliance, or reliability risk where possible. "
                "Provide a final merge recommendation (MERGE / REQUEST_CHANGES / BLOCK) "
                "with total estimated engineering effort.")
        .group("report")
            .agent("review_writer",
                "Write a concise code review summary (200–300 words): decision "
                "(MERGE / REQUEST_CHANGES / BLOCK), the 2–3 critical issues with specific "
                "file/line references, required-before-release items, and optional "
                "nice-to-have improvements. Include estimated effort in hours.")
    )

    return pipeline, evaluator


def build_pipeline(sensitivity: str = "balanced") -> Pipeline:
    """
    Build the code review evaluation pipeline.

    Fan-out topology: 3 agents in the review group independently examine
    the same diff using tools. Higher agent count and tool usage than the
    due-diligence pipeline, exercises controller behavior at scale.

    Args:
        sensitivity: One of "conservative", "balanced", "aggressive".
    """
    get_code_diff, lint_check = _tools()

    return (
        Pipeline("code_review", sensitivity=sensitivity)
        .group("review")
            .agent(
                "security_reviewer",
                "Review the pull request for security vulnerabilities. Use get_code_diff "
                "to examine changed files and lint_check to identify hardcoded secrets or "
                "insecure patterns. Focus on OWASP Top 10: injection, auth failures, "
                "sensitive data exposure, and dependency vulnerabilities. "
                "Rate each finding CRITICAL/HIGH/MEDIUM/LOW with exact file and line.",
                tools=[get_code_diff, lint_check],
            )
            .agent(
                "performance_reviewer",
                "Review the pull request for performance issues. Use get_code_diff to "
                "examine query patterns, loop complexity, and I/O patterns. Use lint_check "
                "to identify high cyclomatic complexity. Flag N+1 query patterns, missing "
                "indexes, synchronous I/O in hot paths, and redundant cache misses. "
                "Estimate performance impact with concrete numbers where possible.",
                tools=[get_code_diff, lint_check],
            )
            .agent(
                "style_reviewer",
                "Review the pull request for code style, architecture, and test coverage. "
                "Use get_code_diff for a complete file-level change summary. Use lint_check "
                "for violation counts and coverage metrics. Flag naming inconsistencies, "
                "missing docstrings, circular imports, and test coverage gaps. "
                "Give an overall style score 1–10 with a brief rationale.",
                tools=[get_code_diff, lint_check],
            )
        .group("assessment")
            .agent(
                "severity_classifier",
                "Classify each finding from the review group into: BLOCK (must fix before "
                "merge), BEFORE_RELEASE (fix before next release), or FOLLOW_UP (backlog). "
                "For each BLOCK item state the specific risk and estimated fix effort.",
            )
            .agent(
                "impact_assessor",
                "Assess the business impact of each BLOCK and BEFORE_RELEASE item. "
                "Quantify revenue, compliance, or reliability risk where possible. "
                "Provide a final merge recommendation (MERGE / REQUEST_CHANGES / BLOCK) "
                "with total estimated engineering effort.",
            )
        .group("report")
            .agent(
                "review_writer",
                "Write a concise code review summary (200–300 words): decision "
                "(MERGE / REQUEST_CHANGES / BLOCK), the 2–3 critical issues with specific "
                "file/line references, required-before-release items, and optional "
                "nice-to-have improvements. Include estimated effort in hours.",
            )
    )
