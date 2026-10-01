"""Sphinx configuration for streameucliddes."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "streameucliddes"
copyright = "2026, MatthieuPe"
author = "MatthieuPe"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary --------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
# Render "Attributes:" sections inside the class description rather than as
# standalone py:attribute entries, which would collide with autodoc's own
# entries for @dataclass fields ("duplicate object description").
napoleon_use_ivar = True

# dollarmath: $...$ inline and $$...$$ display math in Markdown pages;
# amsmath: \begin{aligned} and friends. Rendered by MathJax (sphinx.ext.mathjax).
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

# -- intersphinx --------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output --------------------------------------------------------
html_theme = "furo"
html_title = "streameucliddes"
