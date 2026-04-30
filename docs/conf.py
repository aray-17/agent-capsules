"""
Sphinx configuration for agentic-capsules documentation.
"""

import os
import sys

# Ensure the package is importable without installation
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "agentic-capsules"
copyright = "2026, agentic-capsules contributors"
author = "agentic-capsules contributors"
release = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # generates API docs from docstrings
    "sphinx.ext.napoleon",      # Google/NumPy-style docstring support
    "sphinx.ext.viewcode",      # [source] links next to each symbol
    "sphinx.ext.intersphinx",   # links to Python stdlib docs
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc: show type annotations, preserve order from source
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"  # include both class and __init__ docstrings

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
