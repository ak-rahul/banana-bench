# Configuration file for the Sphinx documentation builder.

project = 'banana-bench'
copyright = '2026, Ak Rahul'
author = 'Ak Rahul'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Auto-generate GitHub-style slugged anchors for headings up to this depth, so
# plain Markdown links like [ackley](#ackley) (used throughout
# BENCHMARK_FUNCTIONS.md's summary table) resolve without needing explicit
# MyST cross-reference targets.
myst_heading_anchors = 4

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
