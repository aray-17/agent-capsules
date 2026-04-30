"""
Offline scripted adapter for the agentic-capsules demo app.

Returns realistic-looking but deterministic responses without any API calls.
Each agent name maps to a canned response; unknown agents receive a generic fallback.
Artificial latency (configurable) simulates realistic response times for demo effect.

Usage:
    adapter = DemoScriptedAdapter()                     # 80–350 ms per call
    adapter = DemoScriptedAdapter(min_latency=0, max_latency=0)  # instant (tests)
"""
from __future__ import annotations

import random
import re
import time


# ---------------------------------------------------------------------------
# Canned responses keyed by agent name (lowercased)
# ---------------------------------------------------------------------------

_CANNED: dict[str, str] = {
    "summarizer": (
        "This text presents a structured overview of the subject matter. "
        "The core argument is well-supported with concrete examples. "
        "In brief: the content describes a modular approach with three primary "
        "components, each fulfilling a distinct role in the overall system."
    ),
    "extractor": (
        "Key facts extracted:\n"
        "• The system employs a modular, composable architecture\n"
        "• Three primary components are described with distinct responsibilities\n"
        "• Performance metrics indicate measurable efficiency gains at scale\n"
        "• The methodology is reproducible and clearly documented\n"
        "• Forty-three participating entities have signed the voluntary framework"
    ),
    "analyzer": (
        "Analysis:\n"
        "The central theme is optimization balanced against interpretability. "
        "The approach navigates trade-offs between throughput and latency effectively. "
        "Strengths: modularity, clear separation of concerns, and runtime adaptability. "
        "Areas for improvement: edge-case handling and adversarial robustness."
    ),
    "researcher": (
        "Research findings:\n"
        "Recent literature strongly supports the proposed approach for multi-agent "
        "coordination. Key developments include advances in dynamic task allocation, "
        "improved context-window utilization strategies, and emergent tool-use patterns. "
        "The field is actively evolving, with open problems in cross-agent grounding "
        "and long-horizon planning remaining unresolved."
    ),
    "critic": (
        "Critical gaps identified:\n"
        "• Scalability under high concurrency has not been empirically validated\n"
        "• Edge cases in the composition logic require further stress testing\n"
        "• Evaluation benchmarks should include adversarial and out-of-distribution inputs\n"
        "• The proposed methodology lacks ablation studies isolating individual components"
    ),
    "synthesizer": (
        "Synthesis:\n"
        "Integrating the research findings with the critical analysis, the proposed "
        "approach is technically sound and addresses the core coordination problem. "
        "The gaps identified are tractable with targeted engineering effort. "
        "Recommended next steps: (1) deploy at moderate scale to gather production "
        "telemetry, (2) run ablation studies on each composition axis, "
        "(3) establish a community benchmark suite for reproducible comparison."
    ),
    "fact_checker": (
        "Fact check results:\n"
        "• Claim 1: Verified — supported by multiple independent sources ✓\n"
        "• Claim 2: Partially supported — core assertion holds, "
        "but the stated figure requires an updated citation\n"
        "• Claim 3: Verified — consistent with established methodology ✓\n"
        "• Claim 4: Unverified — insufficient evidence in available literature"
    ),
}

_GENERIC = (
    "Output: The assigned task has been completed based on the provided input. "
    "The analysis identified three key points relevant to the query. "
    "Results are consistent with prior context and ready for downstream processing."
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DemoScriptedAdapter:
    """
    Scripted LLM adapter for offline demo mode.

    Parses expected output keys from the compiled prompt, returns canned content
    per agent name, and sleeps for a configurable duration to simulate latency.

    Satisfies the LLMAdapter Protocol (complete + count_tokens + context_window).
    """

    context_window: int = 200_000

    def __init__(
        self,
        min_latency: float = 0.08,
        max_latency: float = 0.35,
        seed: int | None = None,
    ) -> None:
        self._min_latency = min_latency
        self._max_latency = max_latency
        self._rng = random.Random(seed)

    def complete(self, messages: list, tools=None) -> str:
        """
        Return a canned response for each output key found in the prompt.

        Keys are detected by the pattern ``\\w+_OUTPUT`` which matches the
        heading markers inserted by PromptCompiler.
        """
        if self._min_latency < self._max_latency:
            time.sleep(self._rng.uniform(self._min_latency, self._max_latency))
        elif self._min_latency > 0:
            time.sleep(self._min_latency)

        full_text = " ".join(m.content for m in messages)
        # Deduplicate while preserving order
        seen: set[str] = set()
        keys: list[str] = []
        for k in re.findall(r"(\w+_OUTPUT)", full_text):
            if k not in seen:
                seen.add(k)
                keys.append(k)

        if not keys:
            return _GENERIC

        parts: list[str] = []
        for key in keys:
            agent_name = key.replace("_OUTPUT", "").lower()
            content = _CANNED.get(agent_name, _GENERIC)
            parts.append(f"{key}:\n{content}")

        return "\n\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def __repr__(self) -> str:
        return (
            f"DemoScriptedAdapter("
            f"latency={self._min_latency:.2f}–{self._max_latency:.2f}s)"
        )


# ---------------------------------------------------------------------------
# DemoToolAdapter — mock ToolAdapter for offline tool-chain demo
# ---------------------------------------------------------------------------

class DemoToolAdapter:
    """
    Scripted ToolAdapter for offline demo mode.

    Returns deterministic canned outputs for each tool in the demo research
    chain (web_search → web_fetch → extract_text → summarize).  Unknown tools
    receive a generic fallback dict.

    Satisfies the ToolAdapter protocol: invoke(tool_name, input_data) -> dict.
    """

    def invoke(self, tool_name: str, input_data: dict) -> dict:
        if tool_name == "web_search":
            query = input_data.get("query", "topic")
            return {
                "results": (
                    f"Top results for '{query}':\n"
                    "1. https://example.com/article-a — Overview of key developments\n"
                    "2. https://example.com/article-b — In-depth analysis and benchmarks\n"
                    "3. https://example.com/article-c — Recent advances and open problems"
                )
            }

        if tool_name == "web_fetch":
            results = input_data.get("results", "")
            return {
                "content": (
                    "Page content retrieved. "
                    "The article covers modular system design with three primary components, "
                    "each fulfilling a distinct role. "
                    "Performance benchmarks show a 3× efficiency improvement at scale. "
                    "The methodology is reproducible and aligns with current best practices."
                )
            }

        if tool_name == "extract_text":
            content = input_data.get("content", "")
            return {
                "text": (
                    "Extracted key passages:\n"
                    "• Modular design with three primary components\n"
                    "• 3× efficiency improvement demonstrated at scale\n"
                    "• Reproducible methodology aligned with best practices\n"
                    "• Open problems remain in adversarial robustness"
                )
            }

        if tool_name == "summarize":
            text = input_data.get("text", "")
            return {
                "summary": (
                    "A modular three-component system achieves 3× efficiency gains "
                    "at scale using reproducible, best-practice methodology, with "
                    "adversarial robustness identified as the key remaining open problem."
                )
            }

        return {"result": f"Tool '{tool_name}' executed successfully on input: {list(input_data.keys())}"}

    def __repr__(self) -> str:
        return "DemoToolAdapter()"
