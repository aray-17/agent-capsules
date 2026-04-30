agentic-capsules
================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Overview
--------

**agentic-capsules** is a Python adaptive execution runtime for multi-agent
pipelines. It scores coordination overhead after each run, selects an
appropriate compound execution strategy (standard, two-phase, or sequential),
and gates every mode-switching decision on empirical quality to prevent
silent regressions.

You define a pipeline once using the fluent :class:`~agentic_capsules.api.builder.Pipeline`
builder. The runtime observes token overhead and output quality after each
run and automatically switches each group between fine-grained mode (one LLM
call per agent) and compound mode (one or fewer calls per group) once it
has sufficient confidence.

Against hand-tuned LangGraph on a 14-agent pipeline, the runtime uses 51%
fewer fine-mode input tokens and 42% fewer compound-mode input tokens at
+0.020 and +0.017 quality respectively. Against DSPy on a 5-agent pipeline
it uses 19% fewer tokens than uncompiled DSPy at parity quality and 68%
fewer tokens than DSPy-MIPROv2 at +0.052 quality. See the paper for the
full competitive benchmark.

Quickstart
----------

.. code-block:: python

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
   print(result.mode_used)       # {"research": "fine", "writing": "fine"}
   print(result.recommendation)  # {"research": "COMPOSE", "writing": "MAINTAIN"}

A ``Pipeline`` instance is **reusable** across multiple calls to ``.run()``. The
per-group state accumulates observations across runs and switches modes automatically.


Pipeline builder
----------------

:class:`~agentic_capsules.api.builder.Pipeline` is the fluent builder and execution
entry point. Chain ``.group()`` and ``.agent()`` calls to define the pipeline
structure, then call ``.run()`` to execute.

.. code-block:: python

   pipeline = (
       Pipeline("my_pipeline", sensitivity="balanced")
       .group("research")
           .agent("searcher", "Find relevant sources.", tools=[web_search])
           .agent("verifier", "Check each source for credibility.")
       .group("writing")
           .agent("analyst", "Identify the three key insights.")
           .agent("writer",  "Write a 200-word summary.")
   )

   result = pipeline.run("topic", adapter=adapter)

Agents within a group form a linear chain by default — each agent receives
the prior agent's output as context. Pass ``depends_on`` to declare an
explicit dependency set and build fan-out, diamond, or parallel-converge
topologies. Every name in ``depends_on`` must refer to an agent already
declared earlier in the same group; the builder rejects self-loops, forward
references, and cross-group references at declaration time.

.. code-block:: python

   pipeline = (
       Pipeline("review")
       .group("reviewers")
           .agent("seed",  "Summarise the diff.")
           .agent("sec",   "Review for security.",    depends_on=["seed"])
           .agent("perf",  "Review for performance.", depends_on=["seed"])
           .agent("style", "Review for style.",       depends_on=["seed"])
           .agent("synth", "Merge all reviews.",
                  depends_on=["sec", "perf", "style"])
   )


Tool
----

:class:`~agentic_capsules.api.tool.Tool` wraps a Python callable so an agent can
invoke it during reasoning.

.. code-block:: python

   from agentic_capsules import Tool

   web_search = Tool(
       name="web_search",
       description="Search the web for recent articles on a topic.",
       input_schema={"query": "str", "num_results": "int"},
       fn=lambda args: {"results": [{"title": "Example", "url": "https://example.com"}]},
   )

Tools are dispatched by the runtime when the LLM issues a tool call. Results are
injected back into the same agent's context before it produces its final output.


PipelineResult
--------------

:class:`~agentic_capsules.api.result.PipelineResult` is the return value of
``Pipeline.run()``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``output``
     - ``str``
     - Final text output from the last agent in the last group.
   * - ``step_outputs``
     - ``dict[str, str]``
     - Output of every agent, keyed by agent name.
   * - ``mode_used``
     - ``dict[str, str]``
     - Composition mode actually used per group: ``"fine"`` or ``"compound"``.
   * - ``recommendation``
     - ``dict[str, str]``
     - Controller recommendation per group: ``"COMPOSE"``, ``"DECOMPOSE"``, or ``"MAINTAIN"``.
   * - ``confidence``
     - ``dict[str, float]``
     - Confidence score (0.0–1.0) per group at the time of this run.
   * - ``token_usage``
     - ``int``
     - Total token count for the pipeline run.
   * - ``latency_ms``
     - ``int``
     - Wall-clock execution time in milliseconds.


Sensitivity presets
-------------------

Pass ``sensitivity=`` to ``Pipeline`` to select a switching profile:

.. list-table::
   :header-rows: 1

   * - Preset
     - compose_at
     - confidence
     - min observations
   * - ``"aggressive"``
     - 0.18
     - 65%
     - 2
   * - ``"balanced"`` (default)
     - 0.23
     - 80%
     - 3
   * - ``"conservative"``
     - 0.35
     - 90%
     - 5

Choose based on your latency tolerance:

* **aggressive** — switches after 2 consistent observations. Best when call overhead
  is the dominant cost and you want fast adaptation.
* **balanced** — needs 3 observations at 80% confidence. Good default for most
  production pipelines.
* **conservative** — needs 5 observations at 90% confidence. Prefer this for
  editorial or high-stakes pipelines where quality matters more than call reduction.


Controller modes
----------------

Pass ``mode=`` to ``Pipeline.run()`` to override per-run behaviour:

.. list-table::
   :header-rows: 1

   * - Mode
     - Behaviour
   * - ``"auto"`` (default)
     - Controller observes and switches each group when confident.
   * - ``"observe"``
     - Shadow mode: records observations but never switches. Use to baseline
       a new pipeline without triggering any switches.
   * - ``"fine"``
     - Locked FINE for all groups: one LLM call per agent.
   * - ``"compound"``
     - Locked COMPOUND for all groups: one merged LLM call per group.


Compound execution strategies
------------------------------

When a group enters COMPOUND mode, ``compound_execution_model`` controls which
strategy is used:

.. list-table::
   :header-rows: 1

   * - Strategy
     - Behaviour
   * - ``"standard"``
     - One merged LLM call per group combining all agent prompts.
   * - ``"two_phase"``
     - Phase A: per-agent tool calls. Phase B: single merged reasoning call
       over all tool results. Best for tool-heavy groups.
   * - ``"sequential"``
     - Per-agent calls with accumulated context from prior agents injected.
       Best quality for verbose models; 74–86% token savings with
       ``output_guidance="concise"``.
   * - ``"auto"``
     - Framework selects per group based on topology (tool presence, agent
       count, verbosity signal).


ControllerPolicy
----------------

:class:`~agentic_capsules.controller.policy.ControllerPolicy` provides raw threshold
control. Most applications should use ``sensitivity=`` on ``Pipeline`` instead.

.. note::
   The balanced preset (``sensitivity="balanced"``) sets ``compose_at=0.23``. The
   ``ControllerPolicy`` code default is 0.40 — higher than any preset. Always use
   ``sensitivity=`` or construct a policy from a preset via ``policy_for("balanced")``.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Parameter
     - Default
     - Description
   * - ``compose_at``
     - ``0.40``
     - Composition score threshold to switch to COMPOUND.
   * - ``decompose_at``
     - ``0.15``
     - Composition score threshold to revert to FINE.
   * - ``confidence``
     - ``0.80``
     - Fraction of rolling window that must exceed threshold before switching.
   * - ``min_observations``
     - ``3``
     - Minimum run count before any switching is considered.
   * - ``window_size``
     - ``10``
     - Number of recent observations in the rolling window.
   * - ``quality_floor``
     - ``None``
     - If set, revert to FINE if rolling-mean quality drops below this value.
   * - ``compound_execution_model``
     - ``"standard"``
     - Compound strategy: ``standard``, ``two_phase``, ``sequential``, or ``auto``.
   * - ``merged_output_structure``
     - ``"budgeted"``
     - Anti-compression hint for standard compound: ``none``, ``budgeted``, ``reinforced``.
   * - ``output_guidance``
     - ``"auto"``
     - Output length guidance for sequential mode: ``none``, ``concise``, ``moderate``, ``auto``. ``auto`` applies concise only when per-agent output exceeds a verbosity threshold.
   * - ``sequential_context_strategy``
     - ``"predecessor_only"``
     - Context injection for sequential mode: ``full`` or ``predecessor_only``.
   * - ``cache_aligned_prompts``
     - ``True``
     - Align system prompt prefixes for Anthropic prompt caching (90% input discount).
   * - ``escalation_enabled``
     - ``True``
     - Auto-escalate execution tier (standard → two_phase → sequential) on quality failures. Requires ``quality_floor`` and an evaluator to have any effect.
   * - ``escalation_min_failures``
     - ``2``
     - Consecutive below-floor readings before escalating.

**Production example**

.. code-block:: python

   from agentic_capsules import Pipeline, ControllerPolicy

   # A tuned policy; every field below is already the default except
   # compose_at (balanced preset value) and quality_floor (opt-in).
   policy = ControllerPolicy(
       compose_at=0.23,
       quality_floor=0.75,
       compound_execution_model="sequential",
       output_guidance="auto",
       merged_output_structure="budgeted",
   )

   pipeline = Pipeline("my_pipeline", policy=policy)

Prefer ``sensitivity="balanced"`` with an explicit ``quality_floor`` for
most deployments — it expands to ``compose_at=0.23`` and leaves the other
defaults in place.


How it adapts
-------------

.. code-block:: text

   Run 1 (FINE)  →  controller observes overhead
   Run 2 (FINE)  →  controller observes overhead
   Run 3 (FINE)  →  confidence crosses threshold  →  auto-switch to COMPOUND
   Run 4+ (COMPOUND) — fewer LLM calls per group

Each group switches independently based on its own overhead observations.
A research group with many tool-calling agents may switch on run 3; a single-agent
synthesis group may never switch.


Adapters
--------

Pass any adapter to ``Pipeline.run(adapter=...)``. An adapter must implement the
``LLMAdapter`` protocol: ``complete(messages, tools) → str``.

**Anthropic**

.. code-block:: python

   from agentic_capsules.adapters.anthropic import AnthropicAdapter

   adapter = AnthropicAdapter(model="claude-sonnet-4-6")

**OpenAI**

.. code-block:: python

   from agentic_capsules.adapters.openai import OpenAIAdapter

   adapter = OpenAIAdapter(model="gpt-4o")

**Gemini**

.. code-block:: python

   from agentic_capsules.adapters.gemini import GeminiAdapter

   adapter = GeminiAdapter(model="gemini-2.5-flash")

**Scripted (offline / testing)**

.. code-block:: python

   class ScriptedAdapter:
       context_window = 200_000

       def complete(self, messages, tools=None) -> str:
           return "Scripted response."

       def count_tokens(self, text: str) -> int:
           return max(1, len(text) // 4)


Production: Redis persistence
------------------------------

By default state is in-memory. In production, use ``RedisBackend`` so observations
survive restarts and are shared across workers:

.. code-block:: python

   from agentic_capsules import Pipeline
   from agentic_capsules.runtime.backends.redis_backend import RedisBackend

   store    = RedisBackend(host="redis", port=6379, db=0)
   pipeline = Pipeline("content_pipeline", store=store)

Redis keys follow the pattern::

   agentic_capsules:{pipeline_name}:{group_name}:controller_state


Examples
--------

The ``examples/`` directory contains runnable examples ordered by complexity.
All examples support offline mode (no API key needed):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Description
   * - ``examples/research_pipeline.py``
     - Entry-level: 2 groups, 4 agents, 1 tool. Run with ``--runs 5`` to watch
       mode adaptation.
   * - ``examples/code_review_pipeline.py``
     - 2 groups (static analysis → review synthesis), no tools.
   * - ``examples/competitive_analysis.py``
     - 3 groups, 7 agents, 3 tools.
   * - ``examples/content_creation.py``
     - 3 groups with ``sensitivity="conservative"`` for an editorial workflow.

Run offline::

   python -m examples.research_pipeline --runs 5


Key components
--------------

Public:

* :mod:`agentic_capsules.api.builder` — ``Pipeline`` builder
* :mod:`agentic_capsules.api.tool` — ``Tool`` dataclass
* :mod:`agentic_capsules.api.result` — ``PipelineResult`` dataclass
* :mod:`agentic_capsules.api.state` — ``PipelineState``, ``GroupControllerState``
* :mod:`agentic_capsules.controller.policy` — ``ControllerPolicy``, sensitivity presets
* :mod:`agentic_capsules.controller.pareto` — Pareto threshold sweep
* :mod:`agentic_capsules.evaluation.base` — ``QualityEvaluator`` protocol

Internal (advanced use):

* :mod:`agentic_capsules.runtime.executor` — ``CapsuleExecutor``
* :mod:`agentic_capsules.runtime.topology` — topology classifier
* :mod:`agentic_capsules.runtime.prompt_compiler` — ``PromptCompiler``
* :mod:`agentic_capsules.runtime.checkpoint` — ``CheckpointStore``
* :mod:`agentic_capsules.runtime.backends.redis_backend` — ``RedisBackend``
* :mod:`agentic_capsules.controller.granularity` — ``GranularityController``
* :mod:`agentic_capsules.controller.telemetry` — ``TelemetryCollector``
* :mod:`agentic_capsules.tools.registry` — ``ToolRegistry``


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
