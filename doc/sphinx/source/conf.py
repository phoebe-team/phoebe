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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'Phoebe'
copyright = '2023, Phoebe team'
author = 'Phoebe team'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ 'sphinx.ext.mathjax', 'sphinx.ext.autosectionlabel','sphinx_rtd_theme']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "collapse_navigation" : False,
    "sticky_navigation": True,
    'navigation_depth': 4,
}

# Build doxygen docs from here (needed for read the docs)
import subprocess
subprocess.call('cd ../../doxygen ; doxygen', shell=True)
subprocess.call('cp -r ../../doxygen/html ./doxygen', shell=True)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static','doxygen']

html_logo = '_static/light-logo.png'
html_favicon = '_static/icon.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# add a css sheet.
# if this isn't done this way, changes to the css sheet
# aren't added to the build when it's updated.
def env_get_outdated(app, env, added, changed, removed):
    return ['index']
def setup(app):
    app.add_css_file('custom.css')
    app.connect('env-get-outdated', env_get_outdated)
