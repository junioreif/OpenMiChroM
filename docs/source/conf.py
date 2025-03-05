# Configuration file for the Sphinx documentation builder.
#
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Assuming conf.py is in docs/source/, this points to project_root/
sys.path.insert(0, os.path.abspath("../.."))

# Mock imports for autodoc when dependencies aren't installed
autodoc_mock_imports = [
    "simtk", "h5py", "numpy", "scipy", "itertools", "pandas", "os",
    "sys", "time", "random", "six", "sklearn"
]

# -- Project information -----------------------------------------------------

project = 'OpenMiChroM'
copyright = '2020-2025 The Center for Theoretical Biological Physics (CTBP) - Rice University'
author = 'Antonio B. Oliveira Jr. & Vinícius G. Contessoto'
version = '1.1.0'
release = '1.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "jupyter_sphinx",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["Reference/OpenMiChroM.bib"]

templates_path = ['_templates']
source_suffix = ".rst"
master_doc = "index"

exclude_patterns = [
    "_build",
    "_templates",
]

show_authors = True
pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Static files (e.g., for logo and favicon)
html_static_path = ['_static']  # Add this to include static files
html_logo = "_static/images/Open-MichroM_chr1_A549.png"  # Adjusted path
html_favicon = "_static/images/Open-MichroM_chr1_A549_icon.png"  # Adjusted path

html_show_sourcelink = True

# Don’t execute notebooks during build (ensure they’re pre-rendered)
nbsphinx_execute = 'never'