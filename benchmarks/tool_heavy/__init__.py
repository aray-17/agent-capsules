# Benchmark 3 — Tool-Heavy Pipeline (Tool Space Composition)
# Task: Complex queries requiring 10–30 tool invocations across 3–5 MCP servers.
# Composition levels: fine-grained (one tool call per round-trip) | chain-composed | fully-batched
# Expected: chain composition reduces round-trips by 2–5×; schema caching cuts token overhead 10–20%.
# Novel axis — no direct analog in original Capsules paper.
# See design plan §5.2 Phase 4, §6.2 Benchmark 3
