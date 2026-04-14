"""Sphinx configuration for sparsekit documentation."""

project = "SparseKit"
author = "Ayoub Ghriss"
release = "0.1.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["triton", "triton.language"]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}

# exclude_patterns = ["_build"]
html_static_path = ["_static"]
html_logo = "_static/sparsekitlogo.png"
html_favicon = "_static/sparsekitlogo.png"
html_css_files = ["custom.css"]

