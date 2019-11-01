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
_path_to_ecogdata = os.path.abspath('../../')
sys.path.insert(0, _path_to_ecogdata)
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'ecogdata'
copyright = '2019, M Trumpis'
author = 'M Trumpis'

# The full version, including alpha/beta/rc tags
release = '0.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme'
]

# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-member': False
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False #True
napoleon_use_keyword = True


# Some nbshpinx options are copied here.. others (for notebook widgets and latex build?)
# might be relevant: https://github.com/spatialaudio/nbsphinx/blob/master/doc/conf.py

# Allow longish runtimes for some examples. Can alternatively add this to notebook metadata
# "nbsphinx": {
#   "timeout": 60
# },
#
nbsphinx_timeout = 60

# allow errors in notebooks, but continue docs building
nbsphinx_allow_errors = True

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Make links to notebook files in the epilog
nbsphinx_epilog = """
----

Generated by nbsphinx_ from a Jupyter_ notebook: thisbook_.

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Jupyter: https://jupyter.org/
.. _thisbook: https://gabilan2.egr.duke.edu/dummy-path/{{ env.doc2path(env.docname, base=None) }}
"""

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'style.css',
]