# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# ------------------------------------------------------------
# General Configuration
# ------------------------------------------------------------

needs_sphinx = "7.4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_favicon",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_parser",
    "nbsphinx",
]

myst_enable_extensions = ["colon_fence"]

primary_domain = "py"

highlight_language = "python3"

autoclass_content = "both"
autodoc_class_signature = "mixed"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"

autosectionlabel_prefix_document = True

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["templates"]

bibtex_bibfiles = ["_static/bibliography.bib"]

# ------------------------------------------------------------
# HTML Output Options
# ------------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = project
html_theme = "furo"
html_logo = "_static/img/logo.svg"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#008080",
        "color-brand-content": "#008080",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/facebookresearch/fairseq2",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
html_show_copyright = False
html_static_path = ["_static"]
html_title = "fairseq2 Documentation"

favicons = [
    {"href": "img/logo.svg"},  # => use `_static/img/logo.svg`
]

# ------------------------------------------------------------
# Doctest Configuration
# ------------------------------------------------------------

# Configure doctest options
doctest_default_flags = (
    0
    | __import__("doctest").ELLIPSIS
    | __import__("doctest").IGNORE_EXCEPTION_DETAIL
    | __import__("doctest").DONT_ACCEPT_TRUE_FOR_1
)

# Global setup for doctests
doctest_global_setup = """
# Common imports for doctests
import sys
import os

# Mock some imports that might not be available during doc build
try:
    from fairseq2.assets import get_asset_store
    from fairseq2.models import load_model
except ImportError:
    raise ImportError("Please install fairseq2 to run doctests.")
"""
