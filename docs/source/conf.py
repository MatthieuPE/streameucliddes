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
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- lean CI builds: mock the heavy/data-dependent imports -------------
# Set only when STREAMEUCLIDDES_DOCS_LEAN is in the environment (the CI
# workflow sets it; a normal local build, with the real packages actually
# installed, does not) -- autodoc_mock_imports unconditionally substitutes
# a fake stand-in for these module names whenever it's set at all, even if
# the real package happens to be installed too, so this must stay opt-in
# rather than always-on or it would silently degrade a real local build.
#
# torch/streamobs/ugali/gala/astropy/scipy/matplotlib/hyrax are the
# packages that drive CI install time (hyrax alone pulls in
# mlflow/chromadb/onnxruntime and more) and/or need real external
# survey/isochrone data to do anything beyond a bare import -- neither of
# which a docs build needs. numpy/pandas/healpy/pyyaml are NOT mocked even
# in lean mode: they're small, fast, dependency-light, and mocking numpy
# specifically breaks `x: np.random.Generator | None = ...`-style
# parameter annotations (Sphinx's mock objects don't support `|`).
#
# Same limitation applies to any mocked type: a module whose signatures
# use e.g. a bare `torch.Tensor | None` annotation ends up with an empty
# API page in lean mode. A local build (this env var unset) documents it
# fully.
if os.environ.get("STREAMEUCLIDDES_DOCS_LEAN"):
    autodoc_mock_imports = [
        "torch",
        "streamobs",
        "ugali",
        "gala",
        "astropy",
        "scipy",
        "matplotlib",
        "hyrax",
    ]

# -- HTML output --------------------------------------------------------
html_theme = "furo"
html_title = "streameucliddes"
