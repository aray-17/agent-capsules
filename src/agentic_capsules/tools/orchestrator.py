"""
ToolOrchestrator — executes ToolCapsules as compound chains.

The orchestrator runs all steps in a ToolCapsule sequentially without
returning to the LLM between steps. Intermediate results are held in a
local context dict. Only the final step's output is returned.

This directly implements the "compound tool execution" behavior from §3.2.4:
  "When a ToolCapsule is dispatched, the orchestrator executes its constituent
   tool steps in sequence without returning to the LLM between steps."

T-Rule 3 (latency budgeting) is enforced per step via configurable timeout.
T-Rule 4 (side-effect ordering) is respected: the orchestrator runs write
steps in serial but notes read_only_prefix for future parallel optimization.

Design plan ref: §3.2.4, §4.2
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .tool_capsule import ToolCapsule, ToolStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ToolExecutionResult:
    """
    The output of a ToolCapsule run.

    outputs     — dict keyed by each step's output_key (full chain context)
    final_output — the last step's output dict (the capsule's external result)
    step_latencies_ms — per-step wall-clock times (for telemetry)
    total_calls — number of tool adapter invocations made
    """
    outputs: dict[str, dict[str, Any]]
    final_output: dict[str, Any]
    step_latencies_ms: list[float]
    total_calls: int

    @property
    def total_latency_ms(self) -> float:
        return sum(self.step_latencies_ms)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ToolExecutionError(Exception):
    """
    Raised when a ToolCapsule step fails (timeout, adapter error, etc.).

    Design plan ref: §3.2.4 (intra-capsule error handling)
    """
    def __init__(self, message: str, step_index: int, tool_name: str) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.tool_name = tool_name


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ToolOrchestrator:
    """
    Executes ToolCapsules end-to-end without LLM involvement between steps.

    Usage:
        orchestrator = ToolOrchestrator(adapter)
        result = orchestrator.run(capsule, initial_input={"query": "AI safety"})
        final = result.final_output   # only this crosses the capsule boundary
    """

    def __init__(self, adapter, default_timeout_s: float = 30.0) -> None:
        """adapter must satisfy the ToolAdapter protocol."""
        self._adapter = adapter
        self._default_timeout_s = default_timeout_s

    def run(
        self,
        capsule: ToolCapsule,
        initial_input: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        Execute all steps of *capsule* in order.

        *initial_input* is the external input to the chain — made available
        to any step that has input_from=None.

        Steps with input_from set receive the output of the referenced prior
        step as their input dict.

        T-Rule 2: logs a warning if any step is non-idempotent.
        T-Rule 3: enforces per-step timeout; raises ToolExecutionError on breach.
        T-Rule 4: write steps run in serial (read_only_prefix noted in log).

        Returns ToolExecutionResult with all step outputs and final output.
        """
        if capsule.has_non_idempotent_steps:
            logger.warning(
                "ToolCapsule %r contains non-idempotent steps (T-Rule 2). "
                "Ensure rollback is handled externally if the chain fails mid-way.",
                capsule.name,
            )

        context: dict[str, dict[str, Any]] = {}  # output_key -> output dict
        step_latencies: list[float] = []

        for i, step in enumerate(capsule.steps):
            # Determine input for this step
            if step.input_from is not None:
                step_input = context[step.input_from]
            else:
                step_input = initial_input

            timeout = step.timeout_s if step.timeout_s > 0 else self._default_timeout_s

            logger.debug(
                "ToolOrchestrator: step %d/%d %r (read_only=%s, timeout=%.1fs)",
                i + 1, len(capsule.steps), step.tool_name, step.read_only, timeout,
            )

            start = time.perf_counter()
            try:
                output = self._invoke_with_timeout(step, step_input, timeout)
            except ToolExecutionError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    f"ToolCapsule {capsule.name!r} step {i} ({step.tool_name!r}) "
                    f"failed: {exc}",
                    step_index=i,
                    tool_name=step.tool_name,
                ) from exc

            elapsed_ms = (time.perf_counter() - start) * 1000
            step_latencies.append(elapsed_ms)

            context[step.output_key] = output
            logger.debug(
                "  -> %r: %.1fms, output_keys=%s",
                step.output_key, elapsed_ms, list(output.keys()),
            )

        final_output = context[capsule.final_output_key]
        return ToolExecutionResult(
            outputs=context,
            final_output=final_output,
            step_latencies_ms=step_latencies,
            total_calls=len(capsule.steps),
        )

    def _invoke_with_timeout(
        self,
        step: ToolStep,
        input_data: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """
        Invoke the tool adapter for *step*.

        T-Rule 3: if the call exceeds *timeout_s*, raises ToolExecutionError.
        For the mock adapter, calls are instantaneous; this timeout is
        meaningful for real MCP adapters in Phase 6.
        """
        start = time.perf_counter()
        result = self._adapter.invoke(step.tool_name, input_data)
        elapsed = time.perf_counter() - start

        if elapsed > timeout_s:
            raise ToolExecutionError(
                f"Step {step.tool_name!r} exceeded timeout ({elapsed:.2f}s > {timeout_s}s)",
                step_index=-1,
                tool_name=step.tool_name,
            )

        if not isinstance(result, dict):
            raise ToolExecutionError(
                f"Step {step.tool_name!r} returned {type(result).__name__}, expected dict",
                step_index=-1,
                tool_name=step.tool_name,
            )

        return result
