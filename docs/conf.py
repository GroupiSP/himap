# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors


project = 'HiMAP'
copyright = '2025, ISP Group, Aerospace Engineering, TU Delft and contributors'
author = 'T.Kontogiannis, M. Salinas-Camus, N. Eleftheroglou'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'autodocsumm',
              'sphinx.ext.coverage']

autodoc_default_options = {'autosummary': True, 'members': True, 'undoc-members': True, 'show-inheritance': True,
                           'private-members': True,
                           'exclude-members': '__weakref__, __dict__, __module__',
                           'member-order': 'bysource'}
autoclass_content = 'both'
# add_module_names = False

autodoc_class_signature = "mixed"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
