"""
Content Creation Pipeline — v2 SDK

Real-world example: three groups, seven agents, two tools.
Models an SEO content production workflow: research → write → review.

Pipeline:
  research group — topic_researcher (web_search, trending_topics) → keyword_analyst
  writing group  — outline_creator → content_writer → seo_optimizer
  review group   — editor → fact_checker

Use conservative sensitivity here — content quality is more important than
minimising LLM calls. The controller waits for strong evidence (90% confidence,
5+ observations) before switching groups to COMPOUND mode.

This also shows mode="observe" for the first run to understand baseline
overhead before letting the controller manage switching automatically.

Run offline:
    python -m examples.content_creation

Run live:
    ANTHROPIC_API_KEY=... python -m examples.content_creation --live --topic "LLM fine-tuning"

Show observe mode:
    python -m examples.content_creation --observe
"""

from __future__ import annotations

import argparse
import re

from agentic_capsules import Pipeline, Tool, PipelineResult


# ---------------------------------------------------------------------------
# Scripted adapter
# ---------------------------------------------------------------------------

class ScriptedAdapter:
    context_window = 200_000

    _RESPONSES = {
        "TOPIC_RESEARCHER": (
            "Research findings: 12 authoritative sources identified. "
            "Key angles: technical depth, practical applications, common misconceptions. "
            "Top sources: OpenAI blog, Hugging Face docs, Anthropic research papers."
        ),
        "KEYWORD_ANALYST": (
            "Primary keyword: 'LLM fine-tuning guide' (8,100 monthly searches, KD 42). "
            "Secondary: 'fine-tune language model' (3,600/mo), 'LoRA fine-tuning' (2,900/mo). "
            "Recommended H2 topics: cost comparison, dataset preparation, evaluation metrics."
        ),
        "OUTLINE_CREATOR": (
            "Outline:\n"
            "  1. What is fine-tuning and when to use it\n"
            "  2. Choosing the right base model\n"
            "  3. Dataset preparation and quality\n"
            "  4. Training techniques: full fine-tune vs LoRA vs QLoRA\n"
            "  5. Evaluation and iteration\n"
            "  6. Cost and compute considerations\n"
            "  7. Production deployment"
        ),
        "CONTENT_WRITER": (
            "Fine-tuning a large language model lets you adapt a general-purpose model "
            "to your specific domain, tone, or task without training from scratch. "
            "This guide covers everything from dataset preparation to deployment, "
            "with practical advice on choosing between full fine-tuning, LoRA, and QLoRA "
            "based on your compute budget and quality requirements. "
            "[Article continues for ~1,800 words covering all outline sections...]"
        ),
        "SEO_OPTIMIZER": (
            "SEO enhancements applied:\n"
            "  - Primary keyword in H1, first paragraph, and meta description\n"
            "  - 3 internal link opportunities identified\n"
            "  - Schema markup recommended: HowTo for sections 3–5\n"
            "  - Estimated reading time: 8 min (optimal for target audience)"
        ),
        "EDITOR": (
            "Editorial notes:\n"
            "  - Tightened section 4 (was 380 words, now 290) — removed redundant examples\n"
            "  - Added concrete cost estimates to section 6 (reader request from comments)\n"
            "  - Corrected terminology: 'RLHF' → 'RLHF/DPO' in section 5\n"
            "  - Overall reading level: Grade 12 (appropriate for technical audience)"
        ),
        "FACT_CHECKER": (
            "Fact-check complete — 3 issues resolved:\n"
            "  1. LoRA paper citation updated to correct arXiv ID\n"
            "  2. GPU memory claim revised: A100 80GB → A100 40GB for 7B model fine-tune\n"
            "  3. Pricing figure updated to 2025 rates\n"
            "  All other factual claims verified against primary sources."
        ),
    }

    def complete(self, messages, tools=None) -> str:
        combined = messages[0].content + messages[-1].content
        keys = re.findall(r"(\w+_OUTPUT)", combined)
        seen: set[str] = set()
        parts = []
        for key in [k for k in keys if not (k in seen or seen.add(k))]:
            agent = key.replace("_OUTPUT", "")
            parts.append(f"{key}:\n{self._RESPONSES.get(agent, 'Task complete.')}")
        return "\n\n".join(parts) if parts else "OUTPUT:\nDone."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tools() -> tuple[Tool, Tool]:
    web_search = Tool(
        name="web_search",
        description="Search for authoritative articles, documentation, and research papers on a topic.",
        input_schema={"query": "str", "num_results": "int"},
        fn=lambda args: {
            "results": [
                {"title": "Official documentation", "url": "https://example.com", "domain_authority": 92},
                {"title": "In-depth tutorial", "url": "https://example.com/2", "domain_authority": 78},
            ]
        },
    )
    trending_topics = Tool(
        name="trending_topics",
        description="Get trending search queries and questions related to a topic from the last 7 days.",
        input_schema={"topic": "str"},
        fn=lambda args: {
            "trending": [
                "how to fine-tune llm without gpu",
                "llm fine-tuning vs rag comparison",
                "best open source models to fine-tune 2025",
            ],
            "questions": [
                "How much does it cost to fine-tune GPT-4?",
                "Is LoRA better than full fine-tuning?",
            ],
        },
    )
    return web_search, trending_topics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    web_search, trending_topics = _tools()

    return (
        # Conservative: quality over speed — wait for strong evidence before switching
        Pipeline("content_creation", sensitivity="conservative")
        .group("research")
            .agent(
                "topic_researcher",
                "Find authoritative sources and identify the key angles for the topic. "
                "Use web_search to find recent, high-authority content.",
                tools=[web_search, trending_topics],
            )
            .agent(
                "keyword_analyst",
                "Identify primary and secondary SEO keywords. "
                "Recommend H2 topics based on search intent and volume.",
            )
        .group("writing")
            .agent(
                "outline_creator",
                "Create a detailed article outline covering all key angles. "
                "Structure for both SEO and reader engagement.",
            )
            .agent(
                "content_writer",
                "Write a comprehensive, authoritative article following the outline. "
                "Target 1,500–2,000 words. Write for a technical but non-expert audience.",
            )
            .agent(
                "seo_optimizer",
                "Apply on-page SEO optimisations: keyword placement, meta description, "
                "internal link opportunities, and schema markup recommendations.",
            )
        .group("review")
            .agent(
                "editor",
                "Edit for clarity, concision, and appropriate reading level. "
                "Ensure the article delivers on its title's promise.",
            )
            .agent(
                "fact_checker",
                "Verify all factual claims, statistics, and citations against primary sources. "
                "Correct any errors and note the changes made.",
            )
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_offline(topic: str, observe_first: bool = False) -> None:
    adapter  = ScriptedAdapter()
    pipeline = _build_pipeline()

    print(f"\nContent Creation Pipeline — offline mode")
    print(f"Topic: {topic!r}")
    print(f"Groups: research (2 agents, 2 tools) → writing (3 agents) → review (2 agents)")
    print(f"Sensitivity: conservative\n")

    if observe_first:
        print("Run 1 (observe mode — baselining, controller will not switch):")
        result = pipeline.run(topic, adapter=adapter, mode="observe")
        print(f"  mode_used:      {result.mode_used}")
        print(f"  recommendation: {result.recommendation}")
        print(f"  (would switch to: {result.recommendation} — but observe mode prevents it)\n")

        print("Run 2 (auto mode — controller now uses accumulated observations):")
        result = pipeline.run(topic, adapter=adapter, mode="auto")
    else:
        result = pipeline.run(topic, adapter=adapter)

    print(f"  mode_used:      {result.mode_used}")
    print(f"  confidence:     {{{', '.join(f'{k}: {v:.0%}' for k, v in result.confidence.items())}}}")
    print(f"  recommendation: {result.recommendation}")
    print(f"  token_usage:    {result.token_usage}")
    print(f"  latency_ms:     {result.latency_ms}")

    print(f"\nStep outputs: {list(result.step_outputs.keys())}")
    print(f"\nFinal output (editor+fact_checker):\n{result.output}")


def run_live(topic: str) -> None:
    from agentic_capsules.adapters.anthropic import AnthropicAdapter
    pipeline = _build_pipeline()
    result: PipelineResult = pipeline.run(
        topic,
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
    )
    print(result.output)
    print(f"\nAll agent outputs: {list(result.step_outputs.keys())}")
    print(f"mode_used: {result.mode_used}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Creation Pipeline — v2 SDK")
    parser.add_argument("--topic",   type=str, default="LLM fine-tuning: a practical guide")
    parser.add_argument("--observe", action="store_true",
                        help="Run first iteration in observe mode to baseline overhead")
    parser.add_argument("--live",    action="store_true")
    args = parser.parse_args()

    if args.live:
        run_live(args.topic)
    else:
        run_offline(args.topic, observe_first=args.observe)
