# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add source path to sys.path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "heraldo"
copyright = "2025, Sadeq Ismailzadeh"
author = "Sadeq Ismailzadeh"
release = "0.1.0"

# Mock external dependencies so docs can build without compiled/heavy libraries
autodoc_mock_imports = [
    "strawberryfields",
    "numba",
    "thewalrus",
    "scipy",
    "sympy",
    "pandas",
    "tqdm",
]

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML Output theme settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]