import os
import re
import sys

# Add project root to python path for autodoc
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If Sphinx is executed in an environment without project dependencies installed,
# attempt to locate and add the local virtual environment (.venv) site-packages.
try:
    import scipy
except ImportError:
    venv_site_packages = None
    if sys.platform == "win32":
        candidate = os.path.join(project_root, ".venv", "Lib", "site-packages")
        if os.path.exists(candidate):
            venv_site_packages = candidate
    else:
        lib_dir = os.path.join(project_root, ".venv", "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                candidate = os.path.join(lib_dir, entry, "site-packages")
                if os.path.exists(candidate):
                    venv_site_packages = candidate
                    break

    if venv_site_packages and venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)

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

# --- User Agent for External Intersphinx Inventories ---
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

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

# --- LaTeX Output Configuration ---
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath,amssymb}
\providecommand{\ket}[1]{\left|#1\right\rangle}
\providecommand{\bra}[1]{\left\langle#1\right|}
\providecommand{\braket}[2]{\left\langle#1\middle|#2\right\rangle}
''',
}


def setup(app):
    """Sphinx extension setup hook."""
    def process_docstring(app, what, name, obj, options, lines):
        """Format bra-ket and pipe notation in docstrings to prevent docutils substitution warnings."""
        for i, line in enumerate(lines):
            if '|' in line:
                # Convert unescaped bra-ket expressions to Sphinx math roles
                new_line = re.sub(
                    r'\|<([^|]+)\|([^>]+)>\|\^2',
                    r':math:`|\\langle \1 | \2 \\rangle|^2`',
                    line
                )
                new_line = re.sub(
                    r'\|\\langle\s*([^|]+)\s*\|\s*([^\\>]+)\\rangle\|\^2',
                    r':math:`|\\langle \1 | \2 \\rangle|^2`',
                    new_line
                )
                new_line = re.sub(
                    r'\|<([^|]+)\|([^>]+)>\|',
                    r':math:`|\\langle \1 | \2 \\rangle|`',
                    new_line
                )
                # Escape any remaining raw unescaped vertical bars outside math/code inline blocks
                if '|' in new_line and '`' not in new_line:
                    new_line = re.sub(r'(?<!\\)\|', r'\|', new_line)
                lines[i] = new_line

    app.connect('autodoc-process-docstring', process_docstring)
