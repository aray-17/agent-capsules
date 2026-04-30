# Developer-facing decorators — application layer entry point (design plan §3.1).
# @agent(name, prompt, tools)  — registers a function as an AgentLeaf in the hierarchy
# @tool(name, schema)          — registers an MCP tool endpoint in the ToolCapsule registry
# @pipeline(name, flow)        — declares a CapsuleHierarchy from a data-flow description
# Composition decisions are NOT made here; this layer only expresses intent.
