"""
Tests for ToolDefinition and ToolRegistry (Phase 10).
"""
import pytest
from agentic_capsules.tools.registry import ToolDefinition, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool(name: str = "search") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Search the web.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"]},
        callable=lambda inp: {"results": f"results for {inp['query']}"},
    )


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------

class TestToolDefinition:
    def test_fields(self):
        t = _make_tool("web_search")
        assert t.name == "web_search"
        assert "Search" in t.description
        assert t.input_schema["type"] == "object"

    def test_callable_invoked(self):
        t = _make_tool()
        result = t.callable({"query": "test"})
        assert "results for test" in result["results"]

    def test_repr(self):
        t = _make_tool("calc")
        assert "calc" in repr(t)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        t = _make_tool("search")
        reg.register(t)
        assert reg.get("search") is t

    def test_get_missing_raises_key_error(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_key_error_lists_available(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get("nonexistent")

    def test_definitions_for_returns_in_order(self):
        reg = ToolRegistry()
        t_a = _make_tool("a")
        t_b = _make_tool("b")
        reg.register(t_a)
        reg.register(t_b)
        result = reg.definitions_for(["b", "a"])
        assert result == [t_b, t_a]

    def test_definitions_for_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.definitions_for(["missing"])

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(_make_tool("x"))
        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        assert len(reg) == 2

    def test_register_overwrites(self):
        reg = ToolRegistry()
        t1 = _make_tool("search")
        t2 = _make_tool("search")
        reg.register(t1)
        reg.register(t2)
        assert reg.get("search") is t2

    def test_repr(self):
        reg = ToolRegistry()
        reg.register(_make_tool("alpha"))
        assert "alpha" in repr(reg)
