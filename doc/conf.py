# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import fairseq2

fairseq2._DOC_MODE = True

# ------------------------------------------------------------
# Project Information
# ------------------------------------------------------------

project = "fairseq2"

version = fairseq2.__version__

release = fairseq2.__version__

author = "Fundamental AI Research (FAIR) at Meta"

copyright = "Meta Platforms, Inc. and affiliates"

# ------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------

needs_sphinx = "5.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

primary_domain = "py"

highlight_language = "python3"

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented_params"

autosectionlabel_prefix_document = True

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["templates"]

bibtex_bibfiles = ["../bibliography.bib"]

# ------------------------------------------------------------
# HTML Output Options
# ------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

html_static_path = ["static"]
