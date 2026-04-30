"""Tests for tools/tool_capsule.py and tools/tool_adapter.py and tools/schema_cache.py"""

import pytest

from agentic_capsules.tools.tool_capsule import ToolCapsule, ToolStep
from agentic_capsules.tools.tool_adapter import MockToolAdapter
from agentic_capsules.tools.schema_cache import ToolSchemaCache


# ---------------------------------------------------------------------------
# ToolStep
# ---------------------------------------------------------------------------

def test_toolstep_repr():
    step = ToolStep("web_search", ["query"], "search_results")
    assert "web_search" in repr(step)
    assert "search_results" in repr(step)


def test_toolstep_input_from_repr():
    step = ToolStep("fetch_page", ["url"], "page_content", input_from="search_results")
    assert "search_results" in repr(step)


# ---------------------------------------------------------------------------
# ToolCapsule — construction and T-Rule 1
# ---------------------------------------------------------------------------

def _search_chain() -> ToolCapsule:
    return ToolCapsule(
        name="search_and_fetch",
        steps=[
            ToolStep("web_search", ["query"], "search_results"),
            ToolStep("fetch_page", ["url"], "page_content", input_from="search_results"),
            ToolStep("extract_text", ["html"], "extracted_text", input_from="page_content"),
        ],
    )


def test_toolcapsule_len():
    assert len(_search_chain()) == 3


def test_toolcapsule_final_output_key():
    assert _search_chain().final_output_key == "extracted_text"


def test_toolcapsule_repr():
    tc = _search_chain()
    assert "search_and_fetch" in repr(tc)
    assert "web_search" in repr(tc)


def test_toolcapsule_empty_raises():
    with pytest.raises(ValueError):
        ToolCapsule(name="empty", steps=[])


def test_trule1_valid_chain_no_error():
    # Should not raise
    _search_chain()


def test_trule1_bad_input_from_raises():
    with pytest.raises(ValueError, match="T-Rule 1"):
        ToolCapsule(
            name="broken",
            steps=[
                ToolStep("web_search", ["query"], "search_results"),
                ToolStep("fetch_page", ["url"], "page_content", input_from="nonexistent_key"),
            ],
        )


def test_trule1_first_step_can_have_no_input_from():
    # input_from=None on first step is always valid
    tc = ToolCapsule(name="single", steps=[
        ToolStep("web_search", ["query"], "search_results", input_from=None),
    ])
    assert len(tc) == 1


# ---------------------------------------------------------------------------
# T-Rule 2 and T-Rule 4 metadata
# ---------------------------------------------------------------------------

def test_trule2_all_idempotent_by_default():
    assert not _search_chain().has_non_idempotent_steps


def test_trule2_detects_non_idempotent():
    tc = ToolCapsule(name="write_chain", steps=[
        ToolStep("read_db", ["id"], "record", read_only=True, idempotent=True),
        ToolStep("write_db", ["data"], "write_result", input_from="record",
                 read_only=False, idempotent=False),
    ])
    assert tc.has_non_idempotent_steps


def test_trule4_read_only_prefix_all_reads():
    tc = _search_chain()
    prefix = tc.read_only_prefix
    assert len(prefix) == 3  # all steps are read_only=True


def test_trule4_read_only_prefix_stops_at_write():
    tc = ToolCapsule(name="mixed", steps=[
        ToolStep("read_a", ["q"], "a", read_only=True),
        ToolStep("read_b", ["q"], "b", read_only=True),
        ToolStep("write_c", ["q"], "c", input_from="b", read_only=False),
    ])
    prefix = tc.read_only_prefix
    assert len(prefix) == 2
    assert all(s.read_only for s in prefix)


# ---------------------------------------------------------------------------
# MockToolAdapter
# ---------------------------------------------------------------------------

def test_mock_adapter_invoke_registered():
    adapter = MockToolAdapter()
    adapter.register("web_search", {"url": "https://example.com"})
    result = adapter.invoke("web_search", {"query": "test"})
    assert result == {"url": "https://example.com"}


def test_mock_adapter_invoke_unregistered_echo():
    adapter = MockToolAdapter()
    result = adapter.invoke("unknown_tool", {"query": "hello"})
    assert "result" in result


def test_mock_adapter_call_count():
    adapter = MockToolAdapter()
    adapter.invoke("web_search", {})
    adapter.invoke("web_search", {})
    adapter.invoke("fetch_page", {})
    assert adapter.call_counts["web_search"] == 2
    assert adapter.call_counts["fetch_page"] == 1
    assert adapter.total_calls == 3


def test_mock_adapter_reset():
    adapter = MockToolAdapter()
    adapter.invoke("web_search", {})
    adapter.reset()
    assert adapter.total_calls == 0


def test_mock_adapter_get_schema_registered():
    adapter = MockToolAdapter()
    adapter.register("web_search", {}, schema={"input": {"query": "str"}, "output": {"url": "str"}})
    schema = adapter.get_schema("web_search")
    assert "input" in schema
    assert "output" in schema


def test_mock_adapter_get_schema_default():
    adapter = MockToolAdapter()
    schema = adapter.get_schema("any_tool")
    assert "input" in schema
    assert "output" in schema


# ---------------------------------------------------------------------------
# ToolSchemaCache
# ---------------------------------------------------------------------------

def test_schema_cache_fetches_once():
    adapter = MockToolAdapter()
    cache = ToolSchemaCache(adapter)
    cache.get("web_search")
    cache.get("web_search")
    # get_schema is not tracked by call_count in MockToolAdapter, but
    # cache should only have 1 entry
    assert len(cache) == 1
    assert "web_search" in cache.cached_tools


def test_schema_cache_prefetch():
    adapter = MockToolAdapter()
    cache = ToolSchemaCache(adapter)
    cache.prefetch(["web_search", "fetch_page", "extract_text"])
    assert len(cache) == 3


def test_schema_cache_validate_chain_passes():
    adapter = MockToolAdapter()
    # Register schemas whose output fields match the next step's input_keys
    adapter.register("web_search", {}, schema={
        "input": {"query": "str"},
        "output": {"url": "str", "snippet": "str"},
    })
    adapter.register("fetch_page", {}, schema={
        "input": {"url": "str"},
        "output": {"html": "str"},
    })
    adapter.register("extract_text", {}, schema={
        "input": {"html": "str"},
        "output": {"text": "str"},
    })
    cache = ToolSchemaCache(adapter)
    # Should not raise for a fully schema-compatible chain
    cache.validate_chain(_search_chain())
