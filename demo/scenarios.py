"""
Pre-built demo scenarios for the agentic-capsules demo app.

Each Scenario bundles:
  - A list of AgentConfig entries (name, system_prompt, depends_on)
  - A default task input string
  - A short description shown in the sidebar

Call scenario.build_hierarchy() to get a fresh CapsuleHierarchy ready for
execution. A new hierarchy must be built before each run because CapsuleExecutor
mutates leaf state during execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentic_capsules.core.capsule import AgentStepCapsule
from agentic_capsules.core.hierarchy import AgentLeaf, CapsuleHierarchy, CompoundCapsule
from agentic_capsules.core.types import Schema
from agentic_capsules.runtime.scheduler import compute_order
from agentic_capsules.tools.registry import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single agent within a scenario."""
    name: str
    system_prompt: str
    depends_on: list[str] = field(default_factory=list)
    input_fields: dict[str, str] = field(default_factory=lambda: {"text": "str"})
    output_fields: dict[str, str] = field(default_factory=lambda: {"result": "str"})


@dataclass
class Scenario:
    """A named pipeline configuration with a default task input."""
    name: str
    description: str
    default_input: str
    agents: list[AgentConfig]

    def build_hierarchy(self) -> CapsuleHierarchy:
        """
        Construct a fresh CapsuleHierarchy from this scenario's agent configs.

        Must be called before each run — CapsuleExecutor mutates capsule state
        (IDLE → RUNNING → COMPLETE) so a new object is required per execution.
        """
        leaves = {
            cfg.name: AgentLeaf(
                capsule=AgentStepCapsule(
                    name=cfg.name,
                    system_prompt=cfg.system_prompt,
                    input_schema=Schema("input", fields=cfg.input_fields),
                    output_schema=Schema("output", fields=cfg.output_fields),
                )
            )
            for cfg in self.agents
        }
        dep_edges = {
            cfg.name: cfg.depends_on
            for cfg in self.agents
            if cfg.depends_on
        }
        root = CompoundCapsule(
            name="pipeline",
            children=list(leaves.values()),
            dependency_edges=dep_edges,
        )
        compute_order(root)
        return CapsuleHierarchy(
            name=self.name.lower().replace(" ", "_"),
            root=root,
        )


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {

    "Summarizer": Scenario(
        name="Summarizer",
        description=(
            "Single-agent pipeline. One LLM call produces a concise summary "
            "of the provided text. Good for comparing FINE vs COMPOUND overhead "
            "on a minimal baseline."
        ),
        default_input=(
            "Agentic capsules is a Python framework for dynamic granularity "
            "composition across agents, data, and tools. It exposes three "
            "composition axes — iteration-space, computation-space, and "
            "tool-space — that let developers trade off LLM call count against "
            "coordination overhead at runtime. The executor supports FINE mode "
            "(one call per agent), COMPOUND mode (one merged call per compound), "
            "and ITERATION mode (one call per batch of K items)."
        ),
        agents=[
            AgentConfig(
                name="summarizer",
                system_prompt=(
                    "You are a concise summarizer. "
                    "Summarize the given text in 2–3 clear sentences."
                ),
                input_fields={"text": "str"},
                output_fields={"summary": "str"},
            ),
        ],
    ),

    "Document Analysis": Scenario(
        name="Document Analysis",
        description=(
            "Two-agent pipeline: an extractor pulls key facts, then a summarizer "
            "writes a structured summary. In COMPOUND mode both agents run in a "
            "single LLM call; in FINE mode they run sequentially."
        ),
        default_input=(
            "The 2025 AI Safety Report highlights three critical risk categories: "
            "misalignment of objectives, lack of interpretability, and adversarial "
            "robustness gaps. The report recommends mandatory red-teaming for "
            "frontier models and improved transparency standards for training data "
            "and evaluation benchmarks. Forty-three countries have signed the "
            "voluntary framework, but enforcement mechanisms remain undefined."
        ),
        agents=[
            AgentConfig(
                name="extractor",
                system_prompt=(
                    "Extract the key facts from the document as a concise bullet-point list."
                ),
                input_fields={"text": "str"},
                output_fields={"facts": "str"},
            ),
            AgentConfig(
                name="summarizer",
                system_prompt=(
                    "Write a structured summary of the document. "
                    "Use the extracted facts as supporting context."
                ),
                depends_on=["extractor"],
                input_fields={"text": "str"},
                output_fields={"summary": "str"},
            ),
        ],
    ),

    "Research Pipeline": Scenario(
        name="Research Pipeline",
        description=(
            "Three-agent pipeline: a researcher gathers findings, a critic "
            "identifies gaps, and a synthesizer produces the final report. "
            "Demonstrates 3× call reduction in COMPOUND mode."
        ),
        default_input=(
            "What are the main challenges and recent advances "
            "in multi-agent LLM systems?"
        ),
        agents=[
            AgentConfig(
                name="researcher",
                system_prompt=(
                    "Research the given topic thoroughly. "
                    "Report key findings in a structured format with clear sub-headings."
                ),
                input_fields={"question": "str"},
                output_fields={"findings": "str"},
            ),
            AgentConfig(
                name="critic",
                system_prompt=(
                    "Review the researcher's findings. "
                    "Identify gaps, limitations, and open questions."
                ),
                depends_on=["researcher"],
                input_fields={"question": "str"},
                output_fields={"critique": "str"},
            ),
            AgentConfig(
                name="synthesizer",
                system_prompt=(
                    "Synthesize the research findings and the critique into a "
                    "cohesive final report with concrete recommendations."
                ),
                depends_on=["critic"],
                input_fields={"question": "str"},
                output_fields={"report": "str"},
            ),
        ],
    ),
}

SCENARIO_NAMES: list[str] = list(SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Paper evaluation pipeline configurations (for Simulation tab presets)
# ---------------------------------------------------------------------------
# These mirror the actual pipeline topologies used in the ACM paper evaluation.
# Each entry is a list of group dicts matching the Simulation tab format:
#   {"agents": int, "tools": int, "independent": bool}

PAPER_PIPELINES: dict[str, dict] = {
    "Due Diligence (5 agents, 3 groups)": {
        "description": (
            "3 sequential groups: research (2 agents, 2 tools each), "
            "analysis (2 agents, no tools), synthesis (1 agent). "
            "The primary evaluation pipeline across all experiments."
        ),
        "agents_total": 5,
        "groups": [
            {"agents": 2, "tools": 2, "independent": True},   # research
            {"agents": 2, "tools": 0, "independent": True},   # analysis
            {"agents": 1, "tools": 0, "independent": True},   # synthesis
        ],
        "group_names": ["Research", "Analysis", "Synthesis"],
    },
    "Code Review (6 agents, 3 groups)": {
        "description": (
            "3 groups: review (3 parallel agents, 2 tools each), "
            "assessment (2 agents), report (1 agent). "
            "Fan-out pattern — 3 independent reviewers examine the same diff."
        ),
        "agents_total": 6,
        "groups": [
            {"agents": 3, "tools": 2, "independent": True},   # review (fan-out)
            {"agents": 2, "tools": 0, "independent": True},   # assessment
            {"agents": 1, "tools": 0, "independent": True},   # report
        ],
        "group_names": ["Review", "Assessment", "Report"],
    },
    "Long-Chain Research (8 agents, 3 groups)": {
        "description": (
            "3 groups: gather (4 sequential agents, 1 tool each), "
            "analyze (3 agents), report (1 agent). "
            "Longest sequential chain — exercises predecessor-only context strategy."
        ),
        "agents_total": 8,
        "groups": [
            {"agents": 4, "tools": 1, "independent": True},   # gather
            {"agents": 3, "tools": 0, "independent": True},   # analyze
            {"agents": 1, "tools": 0, "independent": True},   # report
        ],
        "group_names": ["Gather", "Analyze", "Report"],
    },
    "Multi-Source Brief (14 agents, 6 groups)": {
        "description": (
            "1 scoping agent → 4 parallel research arms × 3 extractors each → 1 briefer. "
            "14 agents across 6 groups. The LangGraph comparison benchmark — "
            "AC achieves 1.69× fine / 2.51× compound of hand-tuned baseline at quality parity."
        ),
        "agents_total": 14,
        "groups": [
            {"agents": 1, "tools": 0, "independent": True},   # scoping
            {"agents": 3, "tools": 0, "independent": True},   # competitive arm
            {"agents": 3, "tools": 0, "independent": True},   # product arm
            {"agents": 3, "tools": 0, "independent": True},   # financial arm
            {"agents": 3, "tools": 0, "independent": True},   # risk arm
            {"agents": 1, "tools": 0, "independent": True},   # briefer
        ],
        "group_names": ["Scoping", "Competitive", "Product", "Financial", "Risk", "Briefer"],
    },
}

PAPER_PIPELINE_NAMES: list[str] = list(PAPER_PIPELINES.keys())


# ---------------------------------------------------------------------------
# Tool-using scenario (T-014)
# ---------------------------------------------------------------------------

def build_tool_using_hierarchy() -> tuple[CapsuleHierarchy, ToolRegistry]:
    """
    Build a 1-agent hierarchy where the agent has two tools declared.

    The agent calls web_search and summarise during its own reasoning turn.
    Both tools are marked independent=True — their inputs come from the original
    task, not from each other.

    Returns (hierarchy, registry) — pass both to CapsuleExecutor so it can
    resolve the tool definitions when the LLM issues tool calls.

    T-014: makes tool boundaries visible in the Live Run tab.
    """
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="web_search",
        description="Search the web for recent facts, news, and documentation on a topic.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        callable=lambda args: {
            "results": [
                {"title": "Latest advances overview",   "snippet": f"Key facts about {args.get('query', 'the topic')}..."},
                {"title": "Research paper summary",     "snippet": "Peer-reviewed evidence supports the main hypothesis."},
                {"title": "Industry analysis Q1 2025",  "snippet": "Market adoption growing 34% YoY across enterprise."},
            ],
        },
        independent=True,  # input is fully determined from task — safe for TOOL_CHAIN
    ))
    registry.register(ToolDefinition(
        name="summarise",
        description="Condense a long passage of text into the three most important bullet points.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        callable=lambda args: {
            "summary": (
                "• Key finding 1: strong empirical evidence across multiple studies.\n"
                "• Key finding 2: practical applications emerging in production systems.\n"
                "• Key finding 3: open challenges remain around scalability and safety."
            ),
        },
        independent=True,
    ))

    leaf = AgentLeaf(
        capsule=AgentStepCapsule(
            name="researcher",
            system_prompt=(
                "You are a research assistant with access to web_search and summarise. "
                "Use web_search to find recent information on the topic, then use summarise "
                "to condense the most relevant passage. "
                "Produce a concise 3-bullet research note using the tool results."
            ),
            input_schema=Schema("input", fields={"question": "str"}),
            output_schema=Schema("output", fields={"research_note": "str"}),
            tools=["web_search", "summarise"],
        )
    )
    root = CompoundCapsule(name="tool_pipeline", children=[leaf], dependency_edges={})
    compute_order(root)
    hierarchy = CapsuleHierarchy(name="tool_using_agent", root=root)
    return hierarchy, registry


def get_scenario(name: str) -> Scenario:
    """Return the Scenario for *name*. Raises KeyError for unknown names."""
    return SCENARIOS[name]


# ---------------------------------------------------------------------------
# Iteration-space demo helpers
# ---------------------------------------------------------------------------

#: Short documents used as per-item inputs in the iteration-space demo.
ITERATION_DOCUMENTS: list[str] = [
    "Climate report 2024: global CO2 levels reached 423 ppm, the highest recorded in 3 million years.",
    "AI safety researchers found that RLHF-trained models exhibit unexpected goal generalization in novel environments.",
    "Quantum computing milestone: 1 000-qubit processor demonstrated with sub-0.1% error rate in lab conditions.",
    "New antibiotic candidate showed efficacy against drug-resistant bacteria in controlled lab trials.",
    "Renewable energy now supplies 38% of global electricity, up from 30% in 2022 according to the IEA.",
    "Brain-computer interface trial enabled a paralysed patient to type 40 words per minute using thought alone.",
]


def build_iteration_hierarchy() -> CapsuleHierarchy:
    """
    Build a single-agent hierarchy with a TagSpace for iteration-space demo.

    The hierarchy contains one 'summarizer' leaf and a TagSpace dimensioned
    over the indices of ITERATION_DOCUMENTS.  Pass this to a CapsuleExecutor
    with CompositionLevel.ITERATION to process all documents in batches.

    A fresh hierarchy must be returned on each call.
    """
    from agentic_capsules.core.tag import TagDimension, TagSpace

    leaf = AgentLeaf(
        capsule=AgentStepCapsule(
            name="summarizer",
            system_prompt="Summarize the given document in exactly one sentence.",
            input_schema=Schema("input", fields={"text": "str"}),
            output_schema=Schema("output", fields={"summary": "str"}),
        )
    )
    root = CompoundCapsule(
        name="iteration_pipeline",
        children=[leaf],
        dependency_edges={},
    )
    compute_order(root)

    tag_space = TagSpace(
        agent_name="summarizer",
        dimensions=[
            TagDimension("doc_id", list(range(len(ITERATION_DOCUMENTS))))
        ],
    )
    return CapsuleHierarchy(
        name="iteration_demo",
        root=root,
        tag_space=tag_space,
    )


# ---------------------------------------------------------------------------
# Tool-space demo helpers
# ---------------------------------------------------------------------------

def build_tool_chain():
    """
    Build a 4-step ToolCapsule chain: web_search → web_fetch → extract_text → summarize.

    Pass this to a ToolOrchestrator (with DemoToolAdapter) to run 4 tool steps
    without a single LLM call.  Compare with a naïve 4-LLM-call baseline.
    """
    from agentic_capsules.tools.tool_capsule import ToolCapsule, ToolStep

    return ToolCapsule(
        name="research_chain",
        description="4-step research chain: web_search → web_fetch → extract_text → summarize",
        steps=[
            ToolStep(
                tool_name="web_search",
                input_keys=["query"],
                output_key="search_results",
                input_from=None,
                read_only=True,
            ),
            ToolStep(
                tool_name="web_fetch",
                input_keys=["results"],
                output_key="page_content",
                input_from="search_results",
                read_only=True,
            ),
            ToolStep(
                tool_name="extract_text",
                input_keys=["content"],
                output_key="extracted",
                input_from="page_content",
                read_only=True,
            ),
            ToolStep(
                tool_name="summarize",
                input_keys=["text"],
                output_key="final_summary",
                input_from="extracted",
                read_only=True,
            ),
        ],
    )
