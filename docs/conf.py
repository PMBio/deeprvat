# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime
from pathlib import Path

import yaml


def get_authors(cff_file):
    with open(cff_file) as cff_data:
        authors = yaml.safe_load(cff_data).get("authors")
        return ", ".join(
            [f"{a['family-names']}, {a['given-names'][0]}." for a in authors]
        )


cff_path = Path(__file__).parent.resolve() / "../CITATION.cff"
author_list = get_authors(cff_file=cff_path)


project = "DeepRVAT"
copyright = f"{datetime.now().year}, {author_list}"
author = f"{author_list}"
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autodoc2", "myst_parser", "sphinx_copybutton"]
autodoc2_packages = [
    "../deeprvat",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
