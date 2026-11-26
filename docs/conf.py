# Configuration file for the Sphinx documentation builder.

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

project = 'banana-bench'
copyright = '2026, Ak Rahul'
author = 'Ak Rahul'

# Read the version from the installed package metadata (pyproject.toml is the
# single source of truth) instead of hardcoding it here, where it previously
# drifted out of sync. Read the Docs installs the package before building, so
# the metadata is available there.
try:
    release = _pkg_version('banana-bench')
except PackageNotFoundError:
    release = 'unknown'
version = '.'.join(release.split('.')[:2])

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
