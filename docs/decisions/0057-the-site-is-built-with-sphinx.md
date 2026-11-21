# 0057. The documentation site is built with Sphinx

Date: 2026-11-20

## Context

The site was built with MkDocs and the Material theme, with the API reference
from mkdocstrings and a build hook that showed a docstring's cross-reference
as the name of the member. The theme's makers ended its feature development
and fix only critical bugs and security faults until 5 May 2027 (their
announcement, read 2026-11-20). Their successor, Zensical, reads an existing
`mkdocs.yml` but runs no build hooks, and its strict mode checks links only.

## Decision

- The site is built by Sphinx: the guides and records in Markdown through
  MyST, the API reference by autodoc with Google style docstrings read by
  napoleon, the Furo theme, and URLs of the form `guide/actions/`
  (`dirhtml`), the form the README links to.
- `python -m sphinx -W --keep-going -E -b dirhtml docs site` builds it, every
  warning an error and every page read again; `hooks/pre-push` and the docs
  workflow build the same into a fresh folder, where `-E` changes nothing.
- Docstrings keep their Sphinx cross-references, which Sphinx reads itself,
  so the build hook is gone. A reference must name a documented member: the
  build runs nitpicky, so one that names nothing documented fails it, except
  the standard library's names in signatures and the stream's type variable.
- The overview and install pages include the README's sections between the
  comments `<!-- start: NAME -->` and `<!-- end: NAME -->`.
- The exported constants and type aliases are listed by the module that
  defines them, where their docstrings are; the typed dictionaries are listed
  with their keys.
- `docs/conf.py` reads the project's name, version, author and site address
  from `pyproject.toml`.

## Consequences

- Every warning fails the build, not only broken links: a reference to an
  undocumented name included.
- The docs workflow writes Sphinx's own cache outside the published folder.
- The decision records and the archive are built but kept out of the
  navigation: `docs/conf.py` marks each as a page no table of contents
  lists, so a guide left out of the navigation still fails the build.

## Alternatives rejected

- Staying on MkDocs with the Material theme: no fixes after May 2027.
- Zensical: no build hooks, and a strict mode that checks links only.
- pdoc: the API reference alone, without the guides.
