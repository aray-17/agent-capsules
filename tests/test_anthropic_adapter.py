"""Tests for adapters/anthropic.py — model flexibility."""

import pytest

from agentic_capsules.adapters.anthropic import (
    AnthropicAdapter,
    _CONTEXT_WINDOWS,
    _FALLBACK_CONTEXT_WINDOW,
)


def test_known_model_uses_known_context_window():
    """Known models use their entry from _CONTEXT_WINDOWS, not the fallback."""
    try:
        adapter = AnthropicAdapter(model="claude-opus-4-6")
        assert adapter.context_window == _CONTEXT_WINDOWS["claude-opus-4-6"]
    except ImportError:
        pytest.skip("anthropic package not installed")


def test_unknown_model_uses_fallback_context_window():
    """A future/unknown model should be accepted and use the fallback window."""
    try:
        adapter = AnthropicAdapter(model="claude-opus-5")
        assert adapter.context_window == _FALLBACK_CONTEXT_WINDOW
    except ImportError:
        pytest.skip("anthropic package not installed")


def test_unknown_model_with_custom_context_window():
    """Caller can override context window for an unknown model."""
    try:
        adapter = AnthropicAdapter(model="claude-opus-5")
        adapter._context_window = 500_000  # caller sets explicitly if they know
        assert adapter.context_window == 500_000
    except ImportError:
        pytest.skip("anthropic package not installed")
