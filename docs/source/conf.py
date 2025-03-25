# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../.."))

autodoc_mock_imports = ["simtk","h5py","numpy","scipy","itertools","pandas","os","sys","time","random","six","sklearn",
                        "openmm"]

# Project information
project = 'OpenMiChroM'
copyright = '2020-2025 The Center for Theoretical Biological Physics (CTBP) - Rice University'
author = 'Antonio B. Oliveira Jr. & Vinícius G. Contessoto'

# The full version, including alpha/beta/rc tags
version = '1.1.1'
release = '1.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ".rst"

master_doc = "index" 

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = [
    "_build",
    "_templates",
]

show_authors = True
pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = "images/Open-MichroM_chr1_A549.png" 
html_favicon = "images/Open-MichroM_chr1_A549_icon.png"

html_static_path = []

html_show_sourcelink = True

nbsphinx_execute = 'never'
