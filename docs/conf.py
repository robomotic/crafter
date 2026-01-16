# Configuration file for Sphinx documentation builder.

import os
import sys

# Add the parent directory to sys.path so Sphinx can find the oak module
sys.path.insert(0, os.path.abspath('..'))

project = 'OaK (Options and Knowledge) Framework'
copyright = '2026, Richard Sutton'
author = 'Richard Sutton'

release = '1.0'
version = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon (for Google/NumPy style docstrings)
extensions.append('sphinx.ext.napoleon')
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autosummary options
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# MathJax options
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
