"""Settings of the documentation site, built by Sphinx from the guides, the README's sections and the docstrings."""

import tomllib
from pathlib import Path

_PROJECT = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]

project = _PROJECT["name"]
release = _PROJECT["version"]
author = ", ".join(person["name"] for person in _PROJECT["authors"])
html_baseurl = _PROJECT["urls"]["Documentation"]
html_theme = "furo"
html_title = project
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = False
# A reference that names nothing documented fails the build, except the standard library's
# names in signatures and the stream's type variable.
nitpicky = True
nitpick_ignore = [("py:class", name) for name in (
    "collections.abc.Iterable", "collections.abc.Callable", "collections.abc.Generator",
    "threading.Event", "_Delta")]


def _outside_the_navigation(app, docname, source):
    """Mark each decision record and the archive as a page the navigation does not list."""
    if docname.startswith("decisions/") and docname != "decisions/README":
        source[0] = "---\norphan: true\n---\n" + source[0]


def setup(app):
    app.connect("source-read", _outside_the_navigation)
