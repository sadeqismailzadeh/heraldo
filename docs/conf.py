import os
import sys

# Add project root to python path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# --- Project Information ---
project = 'heraldo'
copyright = '2026, Sadeq Ismailzadeh'
author = 'Sadeq Ismailzadeh'
release = '0.1.0'

# --- General Configuration ---
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Source suffix for MyST Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# --- MyST Parser Settings ---
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

# --- Napoleon Settings (Docstring parser) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# --- Autodoc Settings ---
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_mock_imports = [
    'scipy',
    'numpy',
    'strawberryfields',
    'numba',
    'pandas',
    'thewalrus',
    'sympy',
    'tqdm',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# --- HTML Output Options ---
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
