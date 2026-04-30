"""Tests for api/tool.py."""
import pytest
from agentic_capsules.api.tool import Tool


def _make_tool(**kwargs):
    defaults = dict(
        name="web_search",
        description="Search the web.",
        input_schema={"query": "str"},
        fn=lambda args: {"results": "ok"},
    )
    defaults.update(kwargs)
    return Tool(**defaults)


def test_tool_construction():
    t = _make_tool()
    assert t.name == "web_search"
    assert t.description == "Search the web."
    assert t.input_schema == {"query": "str"}
    assert callable(t.fn)


def test_tool_fn_callable():
    t = _make_tool()
    result = t.fn({"query": "test"})
    assert isinstance(result, dict)


def test_tool_empty_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        _make_tool(name="")


def test_tool_whitespace_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        _make_tool(name="   ")


def test_tool_non_callable_fn_raises():
    with pytest.raises(ValueError, match="fn must be callable"):
        _make_tool(fn="not_a_function")


def test_tool_input_schema_is_dict():
    t = _make_tool(input_schema={"query": "str", "limit": "int"})
    assert isinstance(t.input_schema, dict)
    assert t.input_schema["limit"] == "int"
