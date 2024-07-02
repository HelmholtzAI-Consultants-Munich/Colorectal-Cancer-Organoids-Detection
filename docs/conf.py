import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = "example_project"
author = "Jane Doe"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

html_theme = 'alabaster'
