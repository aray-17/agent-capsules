"""
Prompt Compiler — merges N agent prompts into a single compound prompt.

This is the core mechanism by which computation-space composition (§3.2.3)
is realized. When agents are merged into a CompoundCapsule, their individual
system prompts are combined into one structured prompt that instructs a single
LLM invocation to perform all constituent reasoning steps sequentially.

Phase-marker format (from design plan §3.2.3):

    == PHASE 1: Research ==
    [researcher system prompt, adapted]
    Produce your output under the heading: RESEARCH_OUTPUT

    == PHASE 2: Fact Verification ==
    [fact-checker system prompt, adapted]
    Use RESEARCH_OUTPUT from Phase 1 as your input.
    Produce your output under the heading: VERIFIED_OUTPUT

    == PHASE 3: Summarization ==
    ...

Rule 6 (Context Budget Feasibility) is enforced here using the adapter's
exact count_tokens() — this is the production check; rules.py does a
cheaper heuristic estimate at definition time.

Design plan ref: §3.2.3, decision D-1 (phase-marker strategy), decision D-2
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.capsule import AgentItemCapsule, AgentTagCapsule
from ..core.hierarchy import AgentLeaf, CompoundCapsule, IterationCapsule
from ..core.types import CompositionError, LLMAdapter, LLMMessage, OutputKey


# ---------------------------------------------------------------------------
# Compiled prompt result
# ---------------------------------------------------------------------------

@dataclass
class CompiledPrompt:
    """
    The result of compiling a capsule (compound or iteration batch).

    `messages` is ready to pass directly to adapter.complete().
    `output_keys` lists the expected output headings in order —
    the executor uses these to parse the LLM's response.
    `coordination_tokens` counts overhead tokens (delimiters, headings,
    instructions) — used by TelemetryCollector to compute overhead_ratio.
    """
    messages: list[LLMMessage]
    output_keys: list[OutputKey]       # e.g. ["RESEARCH_OUTPUT", "ITEM_1_OUTPUT"]
    estimated_tokens: int
    coordination_tokens: int = 0       # Phase 2+: overhead token count


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class PromptCompiler:
    """
    Merges constituent agent prompts into a single compound prompt.

    Two compile modes:
      compile_compound() — for CompoundCapsules (computation-space composition)
      compile_single()   — for a standalone AgentLeaf (no merging needed)

    Design plan ref: §3.2.3
    """

    # Preamble injected at the top of a compound prompt
    _COMPOUND_PREAMBLE = (
        "You are executing a compound task with {n} sequential phases.\n"
        "Complete each phase fully before moving to the next.\n"
        "Use the exact output headings specified — the system parses them.\n"
    )

    # Phase section template
    _PHASE_HEADER = "== PHASE {n}: {name} =="
    _OUTPUT_INSTRUCTION = "Produce your output under the heading: {key}"
    _INPUT_INSTRUCTION = "Use {prev_key} from Phase {prev_n} as your input."

    # T-045 compact framing — anonymous step markers, no role labels, no "compound task" framing.
    # Removes structural cues that cause haiku to shift into synthesis/coordination mode.
    _COMPACT_PREAMBLE = (
        "Complete the following steps in order.\n"
        "Use the exact output headings specified — the system parses them.\n"
    )
    _COMPACT_STEP_HEADER = "--- Step {n} ---"
    _COMPACT_INPUT_INSTRUCTION = "Use {prev_key} from step {prev_n} as your input."

    # PE (prompt-economy, 2026-04-09) — parallel-independent framing.
    #
    # When every leaf in the compound has depends_on=[] (i.e. they are
    # parallel-independent extractors operating on the same source
    # material), the "sequential phases" framing is actively harmful:
    #
    #   1. "Use <PREV>_OUTPUT from Phase N" chain lines tell the model to
    #      derive later sections from earlier sections, which is
    #      semantically wrong when the leaves do not depend on each other.
    #   2. The "compound task with N sequential phases" preamble causes
    #      behavioral collapse on haiku: the model fuses the leaf
    #      personas into a single "I'm writing the brief" narrator and
    #      produces prose paragraphs instead of the structured extraction
    #      lists the per-leaf instructions ask for.
    #   3. Per-phase M-1 budgeted hints (800 words × N sections) multiply
    #      output ~5× versus natural parallel-extraction output.
    #
    # This framing fixes all three at once:
    #   - parallel preamble that frames sections as independent
    #   - no per-phase input-reference lines
    #   - explicit "do not write a preamble" directive (borrowed from
    #     LangGraph's _MERGED_ARM_INSTRUCTION)
    #   - global M-1 hint emitted once with a parallel-mode budget
    #   - shared-prefix hoisting (see _extract_shared_prefix below)
    _PARALLEL_PREAMBLE = (
        "You will produce {n} independent extractions on the shared "
        "source material below. The sections are independent — each one "
        "must be produced from the source material directly, NOT from "
        "the previous section's output. Use the exact output headings "
        "specified — the system parses them. Do not write a preamble; "
        "begin your reply directly with the first output heading.\n"
    )
    _PARALLEL_SECTION_HEADER = "--- Section {n}: {name} ---"
    _PARALLEL_SHARED_CONTEXT_HEADER = "SHARED CONTEXT:"

    # PE: in parallel-independent mode, the M-1 budget is drastically
    # smaller than the linear-mode default (800). Parallel extractors
    # naturally produce ~200-250 tokens of structured output per section;
    # the linear-mode budget over-fires by ~4×. This target was validated
    # by the 2026-04-09 probe (V3 variant matched LangGraph compound
    # output within 1.23× at 250 words/section).
    _PARALLEL_BUDGETED_WORDS = 250

    def __init__(self, adapter: LLMAdapter) -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # PE (prompt-economy) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_parallel_independent(compound: CompoundCapsule) -> bool:
        """
        True iff every leaf in *compound* declares ``depends_on=[]``.

        Uses ``compound.dependency_edges`` as the source of truth (set by
        ``_PipelineCompiler._compile_group`` from each agent's explicit
        ``depends_on`` list). A leaf not present in ``dependency_edges``
        has no explicit declaration and is treated as linear (implicit
        agent-N depends-on-agent-N-1), which is NOT parallel-independent.

        A single-leaf compound is treated as NOT parallel-independent
        (the current behaviour is fine and there is nothing to gain).
        """
        order = compound.serialization_order
        if len(order) < 2:
            return False
        edges = compound.dependency_edges
        for leaf in order:
            # Missing entry → implicit linear (not parallel-independent)
            if leaf.name not in edges:
                return False
            if len(edges[leaf.name]) > 0:
                return False
        return True

    @staticmethod
    def _extract_shared_prefix(
        leaves: list, *, min_chars: int = 200,
    ) -> tuple[str, list[str]]:
        """
        Return ``(shared_prefix, per_leaf_tails)`` — the longest common
        prefix across all leaves' ``system_prompt`` strings, snapped to
        the nearest paragraph boundary (``\\n\\n``).

        If the longest common prefix is shorter than ``min_chars``,
        returns ``("", [full_system_prompt per leaf])`` — no hoisting.

        Paragraph-boundary snapping avoids splitting a sentence or a
        structured block in half. The common case on shared-source
        pipelines is that every leaf begins with ``SOURCE MATERIAL:\\n``
        + bundle + ``\\n\\nTASK:\\n`` + instruction — the natural
        boundary is the blank line between SOURCE MATERIAL and TASK, so
        ``\\n\\n`` snapping produces a clean hoist of just the SOURCE
        MATERIAL block.
        """
        if not leaves:
            return "", []
        texts = [leaf.capsule.system_prompt for leaf in leaves]
        if len(texts) < 2:
            return "", list(texts)

        # Longest common prefix
        shortest = min(texts, key=len)
        lcp_len = 0
        for i, ch in enumerate(shortest):
            if any(t[i] != ch for t in texts):
                break
            lcp_len = i + 1

        if lcp_len < min_chars:
            return "", list(texts)

        # Snap back to the last paragraph boundary within the LCP
        snap = texts[0].rfind("\n\n", 0, lcp_len)
        if snap == -1 or snap < min_chars:
            # No clean paragraph break — fall back to no hoist rather
            # than split mid-sentence.
            return "", list(texts)

        shared = texts[0][:snap]
        tails = [t[snap + 2:] for t in texts]  # skip the "\n\n" separator
        return shared, tails

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_compound(
        self,
        compound: CompoundCapsule,
        task_input: str,
        prior_outputs: dict[OutputKey, str] | None = None,
        tool_contexts: dict[str, str] | None = None,
        min_output_words: int | None = None,
        compact_framing: bool = False,
        merged_output_structure: str = "none",
        per_agent_budgets: dict[str, int] | None = None,
        output_guidance: str = "none",
        mean_fine_tokens: int | None = None,
        guidance_threshold: int | None = None,
        cache_aligned_prompts: bool = False,
    ) -> CompiledPrompt:
        """
        Build the merged prompt for *compound*.

        `task_input` is the user-facing task description injected as the
        first user message.

        `prior_outputs` carries any ItemCapsule data that flows into this
        compound from outside (i.e. the external inputs at the capsule
        boundary).

        `tool_contexts` maps agent leaf name → pre-gathered tool data string
        (T-038 two_phase mode). When set, each matching phase section gets a
        "Pre-gathered tool data" block injected before the output instruction.

        `min_output_words` appends a depth hint to each phase output instruction
        ("Aim for a comprehensive response of at least N words."). Reduces
        compression for small models such as haiku (T-038 Layer 2).

        `compact_framing` uses anonymous ``--- Step N ---`` section markers
        instead of ``== PHASE N: Name ==`` headers (T-045 Approach B). Removes
        structural cues (role labels, "compound task" framing) that cause some
        models (haiku) to shift from research mode to synthesis mode. Does not
        affect output key parsing — output headings are unchanged.

        `merged_output_structure` (M-1 Track A): adds per-phase output pressure
        after the output instruction to resist context compression.

        - ``"budgeted"`` — fixed "Target approximately 800 words."
        - ``"budgeted_adaptive"`` — per-agent token target from ``per_agent_budgets``
          (dict of leaf.name → token budget); falls back to 800 if the agent is
          not in the dict.
        - ``"reinforced"`` — attention-redirect back to agent's own instructions.
        - ``"none"`` — no hint (default).

        `per_agent_budgets` (M-1 ``"budgeted_adaptive"``): maps leaf.name to an
        integer token budget (80% of that agent's mean FINE output tokens).
        Computed in ``_PipelineCompiler._execute_group()`` from
        ``GroupControllerState.mean_avg_output_tokens_fine()``.

        **PE (prompt-economy, 2026-04-09):** when every leaf in *compound*
        declares ``depends_on=[]``, the compiler switches to a
        parallel-independent framing: shared-prefix hoisting, a parallel
        preamble (no phase chaining), no per-leaf ``Use <PREV>_OUTPUT``
        lines, and a single global M-1 hint with a parallel-mode budget.
        This eliminates the essay-mode behavioral collapse observed on
        haiku and reduces output tokens ~5× on shared-source pipelines.
        Linear-dependency compounds keep the historical sequential
        framing unchanged. See the 2026-04-09 probe entry in
        ``evals/last_eval.md`` for measurements.

        Raises CompositionError (Rule 6) if the compiled prompt exceeds
        the adapter's context window.
        """
        order = compound.serialization_order
        if not order:
            from .scheduler import compute_order
            order = compute_order(compound)

        # G-3 (AC↔LG parity, Phase 2 Batch B, 2026-04-10): single-leaf
        # compound → compile_single.
        #
        # A compound with exactly one AgentLeaf is structurally a single
        # call. The legacy path still wrapped it in the "compound task
        # with 1 sequential phases" preamble, a "== PHASE 1: Name ==" header,
        # a "Produce your output under the heading: X" instruction that
        # compile_single already emits, and (when M-1 is active) a
        # per-phase "Target approximately 800 words. All 1 sections should
        # be equally detailed" hint — all pure overhead on a single call.
        # In multi_source_brief both `scoping` and `synthesis` are
        # single-leaf groups; offline probe showed +430 / +396 chars
        # respectively versus LG's equivalent calls which have none of it.
        #
        # The fix delegates to compile_single, which produces exactly the
        # same output contract as fine mode would (same output_key, same
        # prior_outputs handling, same user-text shape). Two_phase tool
        # injection (`tool_contexts`) is the one path that could need
        # compound framing even with a single leaf — skip the shortcut
        # when tool_contexts is set so the legacy "Pre-gathered tool data
        # for this phase" block is still emitted. ToolLeaf children (no
        # capsule.system_prompt) also take the legacy path.
        from ..core.hierarchy import AgentLeaf as _AgentLeaf
        if (
            len(order) == 1
            and not tool_contexts
            and isinstance(order[0], _AgentLeaf)
        ):
            # T-059 adjacent fix (2026-04-23): forward output_guidance +
            # observations to compile_single so the auto-concise gate can
            # fire on single-leaf compound calls. Before this fix, the
            # shortcut silently dropped these params — any single-leaf
            # compound (e.g. due_diligence synthesis group, multi_source_brief
            # scoping/briefer) bypassed T-058's auto-concise mechanism.
            return self.compile_single(
                order[0],
                task_input,
                prior_outputs=prior_outputs,
                output_guidance=output_guidance,
                mean_fine_tokens=mean_fine_tokens,
                guidance_threshold=guidance_threshold,
                cache_aligned_prompts=cache_aligned_prompts,
            )

        # PE: branch on parallel-independent detection — if every leaf
        # has depends_on=[], we route to the new framing entirely and
        # return early. The legacy sequential code below is preserved
        # verbatim for linear-dependency compounds (Track A M-1 wins on
        # due_diligence/code_review rely on it).
        if self._is_parallel_independent(compound):
            return self._compile_compound_parallel_independent(
                compound=compound,
                order=order,
                task_input=task_input,
                prior_outputs=prior_outputs,
                tool_contexts=tool_contexts,
                min_output_words=min_output_words,
                merged_output_structure=merged_output_structure,
                per_agent_budgets=per_agent_budgets,
            )

        if compact_framing:
            preamble = self._COMPACT_PREAMBLE
            step_header_tpl = self._COMPACT_STEP_HEADER
            input_instr_tpl = self._COMPACT_INPUT_INSTRUCTION
        else:
            preamble = self._COMPOUND_PREAMBLE.format(n=len(order))
            step_header_tpl = self._PHASE_HEADER
            input_instr_tpl = self._INPUT_INSTRUCTION

        # PE P0 (linear path extension, 2026-04-09): shared-prefix hoisting.
        #
        # If every agent leaf in this linear compound shares a common
        # system-prompt prefix (≥200 chars, snapped to \n\n), emit it once
        # under SHARED CONTEXT: and use per-leaf tails in the phase loop.
        # This is pure dedup — semantically identical, no framing change,
        # no per-phase M-1 change, no phase-chaining change. Linear
        # compounds with genuinely different per-phase prompts (the common
        # case: due_diligence, code_review, long_chain_research) see no
        # hoisting and get the legacy output byte-for-byte — the helper
        # returns ("", [full_prompts]) when no LCP ≥ min_chars exists.
        #
        # Skipped when any leaf is not an AgentLeaf (ToolLeaf has no
        # system_prompt). See parallel path for the same guard.
        from ..core.hierarchy import AgentLeaf as _AgentLeaf
        linear_shared_prefix = ""
        linear_tails: list[str] = [
            leaf.capsule.system_prompt for leaf in order
        ]
        if len(order) >= 2 and all(isinstance(l, _AgentLeaf) for l in order):
            linear_shared_prefix, linear_tails = self._extract_shared_prefix(
                order, min_chars=200,
            )

        system_parts: list[str] = [preamble]
        # Emit the hoisted shared context (if any) once, right after the
        # preamble and before the first phase header.
        if linear_shared_prefix:
            system_parts.append(self._PARALLEL_SHARED_CONTEXT_HEADER)
            system_parts.append(linear_shared_prefix)
            system_parts.append("")  # blank line separator
        output_keys: list[OutputKey] = []

        for i, leaf in enumerate(order, start=1):
            capsule = leaf.capsule
            key = capsule.output_key
            output_keys.append(key)

            # Phase/step header
            if compact_framing:
                system_parts.append(step_header_tpl.format(n=i))
            else:
                system_parts.append(step_header_tpl.format(n=i, name=capsule.name.title()))

            # Agent's own system prompt (or the leaf-specific tail when
            # shared-prefix hoisting extracted a common bundle).
            system_parts.append(linear_tails[i - 1])

            # Input reference (all phases after the first read from previous output)
            if i > 1:
                prev_key = output_keys[i - 2]
                system_parts.append(
                    input_instr_tpl.format(prev_key=prev_key, prev_n=i - 1)
                )

            # T-038: inject pre-gathered tool data for this phase (two_phase mode)
            if tool_contexts and leaf.name in tool_contexts:
                system_parts.append(
                    "Pre-gathered tool data for this phase:\n" + tool_contexts[leaf.name]
                )

            # Output instruction
            system_parts.append(self._OUTPUT_INSTRUCTION.format(key=key))

            # T-038: depth hint to reduce compression in small models
            if min_output_words:
                system_parts.append(
                    f"Aim for a comprehensive response of at least {min_output_words} words."
                )

            # M-1 (Track A): anti-compression output pressure
            if merged_output_structure == "budgeted":
                system_parts.append(
                    f"Target approximately 800 words for this section. "
                    f"All {len(order)} sections should be equally detailed — "
                    f"do not abbreviate or shorten later sections."
                )
            elif merged_output_structure == "budgeted_adaptive":
                budget = (per_agent_budgets or {}).get(leaf.name, 800)
                system_parts.append(
                    f"Target approximately {budget} tokens for this section. "
                    f"All {len(order)} sections should be equally detailed — "
                    f"do not abbreviate or shorten later sections."
                )
            elif merged_output_structure == "reinforced":
                system_parts.append(
                    "Fully address all requirements from your instructions above. "
                    "Do not omit any requested dimension. "
                    "Your response for this section should be as detailed as if you "
                    "were the only agent completing this task."
                )

            system_parts.append("")  # blank line between phases

        system_text = "\n".join(system_parts)

        # Build user message — inject task input and any prior boundary outputs
        user_parts: list[str] = [task_input]
        if prior_outputs:
            for out_key, out_value in prior_outputs.items():
                user_parts.append(f"\n{out_key}:\n{out_value}")
        user_text = "\n".join(user_parts)

        messages = [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ]

        # Rule 6: exact token check
        full_text = system_text + user_text
        token_count = self._adapter.count_tokens(full_text)
        if token_count > self._adapter.context_window:
            raise CompositionError(
                6,
                f"Compiled compound prompt for {compound.name!r} is {token_count} tokens, "
                f"exceeding context window {self._adapter.context_window}. "
                f"Reduce composition level or shorten agent prompts.",
            )

        # Coordination tokens = preamble + phase/step headers + input/output instructions
        # (excludes agent system_prompt bodies and actual task_input content)
        if compact_framing:
            overhead_parts: list[str] = [self._COMPACT_PREAMBLE]
            for i, leaf in enumerate(order, start=1):
                overhead_parts.append(self._COMPACT_STEP_HEADER.format(n=i))
                if i > 1:
                    prev_key = output_keys[i - 2]
                    overhead_parts.append(
                        self._COMPACT_INPUT_INSTRUCTION.format(prev_key=prev_key, prev_n=i - 1)
                    )
                overhead_parts.append(self._OUTPUT_INSTRUCTION.format(key=output_keys[i - 1]))
        else:
            overhead_parts = [self._COMPOUND_PREAMBLE.format(n=len(order))]
            for i, leaf in enumerate(order, start=1):
                overhead_parts.append(self._PHASE_HEADER.format(n=i, name=leaf.capsule.name.title()))
                if i > 1:
                    prev_key = output_keys[i - 2]
                    overhead_parts.append(
                        self._INPUT_INSTRUCTION.format(prev_key=prev_key, prev_n=i - 1)
                    )
                overhead_parts.append(self._OUTPUT_INSTRUCTION.format(key=output_keys[i - 1]))
        coordination_tokens = self._adapter.count_tokens("\n".join(overhead_parts))

        return CompiledPrompt(
            messages=messages,
            output_keys=output_keys,
            estimated_tokens=token_count,
            coordination_tokens=coordination_tokens,
        )

    # ------------------------------------------------------------------
    # PE — parallel-independent compilation path
    # ------------------------------------------------------------------

    def _compile_compound_parallel_independent(
        self,
        compound: CompoundCapsule,
        order: list,
        task_input: str,
        prior_outputs: dict[OutputKey, str] | None,
        tool_contexts: dict[str, str] | None,
        min_output_words: int | None,
        merged_output_structure: str,
        per_agent_budgets: dict[str, int] | None,
    ) -> CompiledPrompt:
        """
        Build a compound prompt for a parallel-independent group.

        Called from ``compile_compound`` when every leaf has
        ``depends_on=[]``. See the PE block at the top of this class for
        the full rationale and the 2026-04-09 probe entry in
        ``evals/last_eval.md`` for measurements.

        Differences from the legacy sequential framing:

        - **Shared-prefix hoisting.** If all leaves share a common
          system-prompt prefix ≥200 chars, that prefix is emitted once at
          the top under ``SHARED CONTEXT:``. Each leaf section then emits
          only its tail. On shared-source pipelines (e.g. the
          ``multi_source_brief`` arms), this hoists the source bundle
          out of the N×duplication the legacy path would produce.
        - **Parallel preamble** (``_PARALLEL_PREAMBLE``): explicitly
          frames sections as independent, tells the model to produce
          each one from the source directly (not from the previous
          section), and forbids a leading preamble.
        - **No per-leaf ``Use <PREV>_OUTPUT`` lines.** These caused
          false chaining in the legacy path.
        - **Global M-1 hint.** The budgeted/reinforced hint is emitted
          once in the preamble with a parallel-mode word budget
          (``_PARALLEL_BUDGETED_WORDS``, default 250), not per-section.
          ``budgeted_adaptive`` uses the average of the per-agent
          budgets when any are present; otherwise falls back to the
          parallel default. ``reinforced`` is preserved as a single
          global attention-redirect.
        - **Section headers** use compact ``--- Section N: <name> ---``
          markers, not ``== PHASE N: <Name> ==``. Output keys and the
          parser are unchanged.
        """
        # Shared-prefix hoist — only across AgentLeaf children, skip ToolLeaf
        from ..core.hierarchy import AgentLeaf
        agent_leaves = [leaf for leaf in order if isinstance(leaf, AgentLeaf)]
        if len(agent_leaves) == len(order) and len(agent_leaves) >= 2:
            shared_prefix, per_leaf_tails = self._extract_shared_prefix(
                agent_leaves, min_chars=200,
            )
        else:
            # Mixed AgentLeaf + ToolLeaf — skip hoisting (ToolLeaf has no
            # system_prompt, so the concept does not apply).
            shared_prefix = ""
            per_leaf_tails = [
                leaf.capsule.system_prompt for leaf in order
            ] if all(isinstance(l, AgentLeaf) for l in order) else [""] * len(order)

        n = len(order)

        # ---- Preamble: parallel framing + optional shared context + global M-1 ----
        system_parts: list[str] = [self._PARALLEL_PREAMBLE.format(n=n)]

        if shared_prefix:
            system_parts.append(self._PARALLEL_SHARED_CONTEXT_HEADER)
            system_parts.append(shared_prefix)
            system_parts.append("")  # blank line separator

        # Global M-1 hint — emitted ONCE in the preamble, with a
        # parallel-mode budget. In parallel-independent mode, each
        # section is naturally short (~200-250 tokens of structured
        # output), so the 800-word linear default over-fires ~4×.
        global_m1_hint: str | None = None
        if merged_output_structure == "budgeted":
            global_m1_hint = (
                f"Target approximately {self._PARALLEL_BUDGETED_WORDS} words "
                f"per section. All {n} sections should be equally detailed — "
                f"do not abbreviate or shorten later sections."
            )
        elif merged_output_structure == "budgeted_adaptive":
            # Average the per-agent budgets when provided; otherwise use
            # the parallel default. Adaptive budgets were tuned on FINE
            # runs which do not suffer the compound essay-mode problem,
            # so their 800-token defaults are the wrong target for
            # parallel-compound. Clamp the average down to the parallel
            # ceiling to avoid re-introducing the bug.
            if per_agent_budgets:
                vals = [per_agent_budgets.get(leaf.name, 800) for leaf in order]
                avg_tokens = sum(vals) / len(vals)
                # Convert ~tokens to ~words (3/4 words per token) and
                # clamp to the parallel ceiling.
                avg_words = int(avg_tokens * 0.75)
                target = min(avg_words, self._PARALLEL_BUDGETED_WORDS)
            else:
                target = self._PARALLEL_BUDGETED_WORDS
            global_m1_hint = (
                f"Target approximately {target} words per section. "
                f"All {n} sections should be equally detailed — "
                f"do not abbreviate or shorten later sections."
            )
        elif merged_output_structure == "reinforced":
            global_m1_hint = (
                "Fully address all requirements from each section's instructions. "
                "Do not omit any requested dimension. "
                "Each section should be as detailed as if it were the only "
                "extraction being performed."
            )
        # "none" → no hint (parallel mode's natural output is already tight)

        if global_m1_hint is not None:
            system_parts.append(global_m1_hint)
            system_parts.append("")

        # ---- Per-leaf sections ----
        output_keys: list[OutputKey] = []
        for i, (leaf, tail) in enumerate(zip(order, per_leaf_tails), start=1):
            capsule = leaf.capsule
            key = capsule.output_key
            output_keys.append(key)

            system_parts.append(
                self._PARALLEL_SECTION_HEADER.format(n=i, name=capsule.name.title())
            )
            # When shared_prefix hoisted, `tail` is the leaf-specific
            # tail; otherwise it equals the full system_prompt.
            system_parts.append(tail)

            # T-038: pre-gathered tool data stays per-section (two_phase)
            if tool_contexts and leaf.name in tool_contexts:
                system_parts.append(
                    "Pre-gathered tool data for this section:\n" + tool_contexts[leaf.name]
                )

            system_parts.append(self._OUTPUT_INSTRUCTION.format(key=key))

            # T-038 min-output-words stays per-section when set (it is a
            # per-leaf depth hint, not an anti-compression hint)
            if min_output_words:
                system_parts.append(
                    f"Aim for a comprehensive response of at least {min_output_words} words."
                )

            system_parts.append("")  # blank line between sections

        system_text = "\n".join(system_parts)

        # User message is the same as the sequential path
        user_parts: list[str] = [task_input]
        if prior_outputs:
            for out_key, out_value in prior_outputs.items():
                user_parts.append(f"\n{out_key}:\n{out_value}")
        user_text = "\n".join(user_parts)

        messages = [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ]

        # Rule 6: exact token check
        full_text = system_text + user_text
        token_count = self._adapter.count_tokens(full_text)
        if token_count > self._adapter.context_window:
            raise CompositionError(
                6,
                f"Compiled compound prompt for {compound.name!r} is {token_count} tokens, "
                f"exceeding context window {self._adapter.context_window}. "
                f"Reduce composition level or shorten agent prompts.",
            )

        # Coordination tokens: preamble + section headers + output
        # instructions + optional global M-1 hint. Shared-prefix block
        # is NOT coordination — it carries the actual source material,
        # which counts as payload, same as the sequential path treats
        # per-leaf system_prompts as payload.
        overhead_parts: list[str] = [self._PARALLEL_PREAMBLE.format(n=n)]
        if global_m1_hint is not None:
            overhead_parts.append(global_m1_hint)
        for i, key in enumerate(output_keys, start=1):
            overhead_parts.append(
                self._PARALLEL_SECTION_HEADER.format(
                    n=i, name=order[i - 1].capsule.name.title()
                )
            )
            overhead_parts.append(self._OUTPUT_INSTRUCTION.format(key=key))
        coordination_tokens = self._adapter.count_tokens("\n".join(overhead_parts))

        return CompiledPrompt(
            messages=messages,
            output_keys=output_keys,
            estimated_tokens=token_count,
            coordination_tokens=coordination_tokens,
        )

    # O-1 (Track A): output length guidance strings for compile_single().
    _OUTPUT_GUIDANCE_CONCISE  = "Be concise. Aim for 300–400 words."
    _OUTPUT_GUIDANCE_MODERATE = "Aim for 500–600 words."
    _OUTPUT_GUIDANCE_BRIEF    = "Be brief. Aim for 200 words."

    def compile_single(
        self,
        leaf: AgentLeaf,
        task_input: str,
        prior_outputs: dict[OutputKey, str] | None = None,
        output_guidance: str = "none",
        mean_fine_tokens: int | None = None,
        guidance_threshold: int | None = None,
        cache_aligned_prompts: bool = False,
    ) -> CompiledPrompt:
        """
        Build a simple (non-compound) prompt for a single AgentLeaf.

        Used when the executor runs at FINE granularity or in sequential compound.

        `output_guidance` (O-1 Track A): injects a length hint after the output
        instruction to reduce token usage without quality regression.

        - ``"auto"`` — T-058 observations-based gate. Applies concise when
          ``mean_fine_tokens >= guidance_threshold``; otherwise applies no
          guidance. Falls back to no guidance when either input is None
          (e.g. FINE mode without per-group observations).
        - ``"concise"`` — fixed 300–400 word target (explicit override).
        - ``"moderate"`` — fixed 500–600 word target.
        - ``"brief"`` — fixed 200 word target.
        - ``"none"`` — no guidance.

        `mean_fine_tokens`: mean FINE avg_output_tokens for this group, used by
        ``"auto"``. Pass None to disable auto-gating (falls back to no guidance).

        `guidance_threshold`: token threshold used by ``"auto"``. Sourced from
        ``ControllerPolicy.verbosity_guidance_threshold``. None disables gating.

        `cache_aligned_prompts` (C-1 Track A): when True, restructures messages so
        the shared task context is the first system block (marked ephemeral for
        Anthropic prompt caching) and per-agent instructions follow as a second
        system block. Prior outputs and the output instruction move to the user
        message. Consecutive sequential calls share the cached task prefix,
        yielding a 90% input-token discount on the shared portion after the first
        call. Ignored by non-Anthropic adapters (no-op with correct output).
        """
        capsule = leaf.capsule
        system_text = capsule.system_prompt

        # O-1 guidance suffix (shared between standard and cache-aligned paths)
        guidance_parts: list[str] = []
        if output_guidance == "auto":
            # T-058: observations-based gate. Only apply concise when the group's
            # mean FINE output is verbose enough to benefit; otherwise stay silent
            # (avoids the quality regression seen on already-concise models).
            # Fall back to no guidance if either input is missing — conservative
            # default matches the legacy "none" behaviour for FINE calls that do
            # not yet plumb per-group observations.
            if (
                mean_fine_tokens is not None
                and guidance_threshold is not None
                and mean_fine_tokens >= guidance_threshold
            ):
                guidance_parts.append(self._OUTPUT_GUIDANCE_CONCISE)
        elif output_guidance == "concise":
            guidance_parts.append(self._OUTPUT_GUIDANCE_CONCISE)
        elif output_guidance == "moderate":
            guidance_parts.append(self._OUTPUT_GUIDANCE_MODERATE)
        elif output_guidance == "brief":
            guidance_parts.append(self._OUTPUT_GUIDANCE_BRIEF)

        if cache_aligned_prompts:
            # C-1: move task_input to a cacheable system prefix; per-agent
            # instructions stay as the second system block; user carries only
            # the per-call dynamic content (prior outputs + output instruction).
            user_parts: list[str] = []
            if prior_outputs:
                for out_key, out_value in prior_outputs.items():
                    user_parts.append(f"{out_key}:\n{out_value}")
            user_parts.append(
                f"Produce your output under the heading: {capsule.output_key}"
            )
            user_parts.extend(guidance_parts)
            user_text = "\n\n".join(user_parts)

            messages = [
                LLMMessage(
                    role="system",
                    content=task_input,
                    cache_control={"type": "ephemeral"},
                ),
                LLMMessage(role="system", content=system_text),
                LLMMessage(role="user", content=user_text),
            ]
        else:
            user_parts = [task_input]
            if prior_outputs:
                for out_key, out_value in prior_outputs.items():
                    user_parts.append(f"\n{out_key}:\n{out_value}")
            user_parts.append(
                f"\nProduce your output under the heading: {capsule.output_key}"
            )
            user_parts.extend(guidance_parts)
            user_text = "\n".join(user_parts)

            messages = [
                LLMMessage(role="system", content=system_text),
                LLMMessage(role="user", content=user_text),
            ]

        token_count = self._adapter.count_tokens(system_text + user_text)

        # Coordination overhead = the output instruction appended to every
        # FINE-mode call ("Produce your output under the heading: X").
        # This mirrors the approach used in compile_compound and fixes T-005.
        output_instruction = f"\nProduce your output under the heading: {capsule.output_key}"
        coordination_tokens = self._adapter.count_tokens(output_instruction)

        return CompiledPrompt(
            messages=messages,
            output_keys=[capsule.output_key],
            estimated_tokens=token_count,
            coordination_tokens=coordination_tokens,
        )

    # ------------------------------------------------------------------
    # Iteration batch compilation (Phase 2)
    # ------------------------------------------------------------------

    # Batch prompt templates
    _BATCH_PREAMBLE = (
        "You are processing a batch of {k} items. "
        "Process each item fully and independently before moving to the next.\n"
        "Use the exact output headings specified — the system parses them.\n"
    )
    _ITEM_HEADER = "== ITEM {n} =="
    _ITEM_OUTPUT_INSTRUCTION = "Produce your output for this item under the heading: {key}"

    def compile_iteration_batch(
        self,
        iteration_capsule: IterationCapsule,
        item_contents: list[str],
    ) -> CompiledPrompt:
        """
        Build a batch prompt for one IterationCapsule.

        `item_contents` is a parallel list to `iteration_capsule.tags` — one
        string of input text per item in the batch.

        The compiled prompt instructs the LLM to process all K items in
        sequence, each under its own == ITEM N == heading, producing output
        under ITEM_N_OUTPUT headings that the parser can extract.

        Rule 6 enforced: raises CompositionError if the batch prompt exceeds
        the adapter's context window.

        Design plan ref: §5.2 Phase 2, §3.2.3
        """
        if len(item_contents) != len(iteration_capsule.tags):
            raise ValueError(
                f"item_contents length ({len(item_contents)}) must match "
                f"batch size ({len(iteration_capsule.tags)})."
            )

        k = len(item_contents)
        capsule = iteration_capsule.leaf.capsule

        system_parts = [
            self._BATCH_PREAMBLE.format(k=k),
            capsule.system_prompt,
            "",
        ]

        output_keys: list[OutputKey] = []
        user_parts: list[str] = []

        for i, (content, tag) in enumerate(
            zip(item_contents, iteration_capsule.tags), start=1
        ):
            item_key: OutputKey = f"ITEM_{i}_OUTPUT"
            output_keys.append(item_key)

            user_parts.append(self._ITEM_HEADER.format(n=i))
            user_parts.append(content)
            user_parts.append(self._ITEM_OUTPUT_INSTRUCTION.format(key=item_key))
            user_parts.append("")  # blank line between items

        system_text = "\n".join(system_parts)
        user_text = "\n".join(user_parts)

        messages = [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ]

        # Rule 6: exact token check
        full_text = system_text + user_text
        token_count = self._adapter.count_tokens(full_text)
        if token_count > self._adapter.context_window:
            raise CompositionError(
                6,
                f"Iteration batch prompt for agent {capsule.name!r} "
                f"with k={k} items is {token_count} tokens, "
                f"exceeding context window {self._adapter.context_window}. "
                f"Reduce batch size k.",
            )

        # Coordination tokens = delimiters + headers + output instructions
        # (everything except the actual item content)
        overhead_text = system_text + "".join(
            self._ITEM_HEADER.format(n=i) + self._ITEM_OUTPUT_INSTRUCTION.format(key=f"ITEM_{i}_OUTPUT")
            for i in range(1, k + 1)
        )
        coordination_tokens = self._adapter.count_tokens(overhead_text)

        return CompiledPrompt(
            messages=messages,
            output_keys=output_keys,
            estimated_tokens=token_count,
            coordination_tokens=coordination_tokens,
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_outputs(
        response_text: str,
        output_keys: list[OutputKey],
    ) -> dict[OutputKey, str]:
        """
        Extract phase outputs from the LLM response.

        Looks for each heading in *output_keys* and captures the text
        between consecutive headings (or end-of-string for the last one).

        Returns a dict {output_key: extracted_text}.
        If a heading is not found, its value is the full response (fallback).
        """
        results: dict[OutputKey, str] = {}

        for i, key in enumerate(output_keys):
            start_marker = f"{key}:"
            next_markers = [f"{k}:" for k in output_keys[i + 1:]]

            start_idx = response_text.find(start_marker)
            if start_idx == -1:
                # Heading not found — use full response as fallback
                results[key] = response_text.strip()
                continue

            content_start = start_idx + len(start_marker)

            # Find the start of the next heading
            end_idx = len(response_text)
            for next_marker in next_markers:
                candidate = response_text.find(next_marker, content_start)
                if candidate != -1 and candidate < end_idx:
                    end_idx = candidate

            results[key] = response_text[content_start:end_idx].strip()

        return results
