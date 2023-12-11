# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fairseq2n

fairseq2n.DOC_MODE = True

import fairseq2

# ------------------------------------------------------------
# Project Information
# ------------------------------------------------------------

project = "fairseq2"

version = fairseq2.__version__

release = fairseq2.__version__

author = "Fundamental AI Research (FAIR) at Meta"

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

html_show_copyright = False

html_static_path = ["static"]
