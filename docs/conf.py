"""Sphinx configuration for the MultipleIntegrate documentation."""

from __future__ import annotations

project = "MultipleIntegrate"
author = "Bhuvanesh Bhatt"
copyright = "2026, Bhuvanesh Bhatt"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_title = "MultipleIntegrate"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
