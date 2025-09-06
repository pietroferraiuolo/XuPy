"""
Sphinx configuration for XuPy documentation.

Uses MyST for Markdown, AutoAPI for import-free API docs, and Furo theme.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Paths --------------------------------------------------------------------

# Project root (two levels up from this file)
ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, ROOT)

# -- Project information -------------------------------------------------------

project = "XuPy"
author = "Pietro Ferraiuolo"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# Avoid importing the package at build time (it has import side effects)
# Read version from pyproject.toml or fallback constant.
_version = "1.1.1"
try:
	import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
	tomllib = None

pyproject_path = os.path.join(ROOT, "pyproject.toml")
if tomllib and os.path.exists(pyproject_path):
	try:
		with open(pyproject_path, "rb") as f:
			data = tomllib.load(f)
			_version = data.get("project", {}).get("version", _version) or _version
	except Exception:
		pass

version = _version
release = _version

# -- General configuration -----------------------------------------------------

extensions = [
	"myst_parser",
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",
	"sphinx.ext.intersphinx",
	"autoapi.extension",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# MyST (CommonMark + Sphinx directives in Markdown)
myst_enable_extensions = [
	"colon_fence",
	"deflist",
	"html_image",
	"substitution",
	"tasklist",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Intersphinx mapping for external references
intersphinx_mapping = {
	"python": ("https://docs.python.org/3", None),
	"numpy": ("https://numpy.org/doc/stable/", None),
	# CuPy docs (optional; may 404 if offline)
	"cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# -- AutoAPI configuration -----------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [os.path.join(ROOT, "xupy")]  # Parse directly from source without importing
autoapi_keep_files = True
autoapi_add_toctree_entry = False  # We'll link to autoapi/index from our API page
autoapi_root = "autoapi"

# -- Options for HTML output ---------------------------------------------------

html_theme = "furo"
html_title = f"XuPy {release} documentation"
html_static_path = ["_static"]

html_theme_options = {
	"sidebar_hide_name": False,
	"light_logo": "",
	"dark_logo": "",
}

