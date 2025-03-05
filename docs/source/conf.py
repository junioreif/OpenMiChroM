# Configuration file for the Sphinx documentation builder.

import os
import sys

# Point to project root for Python modules
sys.path.insert(0, os.path.abspath("../.."))

# Mock imports for autodoc
autodoc_mock_imports = [
    "simtk", "h5py", "numpy", "scipy", "itertools", "pandas", "os",
    "sys", "time", "random", "six", "sklearn"
]

# Project information
project = 'OpenMiChroM'
copyright = '2020-2025 The Center for Theoretical Biological Physics (CTBP) - Rice University'
author = 'Antonio B. Oliveira Jr. & Vinícius G. Contessoto'
version = '1.1.0'
release = '1.1.0'

# General configuration
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

exclude_patterns = ["_build", "_templates"]

# Dynamically add Tutorials folder to Sphinx source path
def setup(app):
    tutorials_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Tutorials'))
    app.srcdir = tutorials_path  # Add Tutorials to source directory
    app.add_config_value('extra_source_dir', tutorials_path, 'env')

show_authors = True
pygments_style = "sphinx"
todo_include_todos = False

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = "_static/images/Open-MichroM_chr1_A549.png"
html_favicon = "_static/images/Open-MichroM_chr1_A549_icon.png"
html_show_sourcelink = True

nbsphinx_execute = 'never'