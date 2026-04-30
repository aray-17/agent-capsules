"""Tests for adapters/openai.py"""

import pytest

from agentic_capsules.adapters.openai import (
    OpenAIAdapter,
    _CONTEXT_WINDOWS,
    _FALLBACK_CONTEXT_WINDOW,
)


def test_known_model_uses_known_context_window():
    try:
        adapter = OpenAIAdapter(model="gpt-4.1")
        assert adapter.context_window == _CONTEXT_WINDOWS["gpt-4.1"]
    except ImportError:
        pytest.skip("openai package not installed")


def test_unknown_model_uses_fallback_context_window():
    try:
        adapter = OpenAIAdapter(model="gpt-5")
        assert adapter.context_window == _FALLBACK_CONTEXT_WINDOW
    except ImportError:
        pytest.skip("openai package not installed")


def test_unknown_model_with_custom_context_window():
    try:
        adapter = OpenAIAdapter(model="gpt-5")
        adapter._context_window = 500_000
        assert adapter.context_window == 500_000
    except ImportError:
        pytest.skip("openai package not installed")


def test_count_tokens_heuristic_fallback():
    """count_tokens heuristic works without any SDK installed."""
    try:
        adapter = OpenAIAdapter()
        # Force heuristic path by clearing the encoding
        adapter._encoding = None
        count = adapter.count_tokens("hello world")
        assert count >= 1
    except ImportError:
        pytest.skip("openai package not installed")


def test_repr():
    try:
        adapter = OpenAIAdapter(model="gpt-4.1")
        assert "gpt-4.1" in repr(adapter)
    except ImportError:
        pytest.skip("openai package not installed")
