# 0051. A release is a version tag, released on GitHub

Date: 2026-11-20

## Context

The package stated a version in `pyproject.toml` but had no tags and no
release, so a user could install only the newest commit or a checkout. The
package is published on GitHub alone.

## Decision

- The version in `pyproject.toml` is the one place it is stated. A release is
  the tag `v` followed by that version, on the commit that sets it. Earlier
  versions are tagged where `pyproject.toml` first stated them.
- Pushing a version tag runs `.github/workflows/release.yml`: it refuses a tag
  that differs from the stated version, builds the source package and the
  wheel, checks that the wheel holds the add-on files and the type marker,
  and makes a GitHub release of the tag with both attached.
- The README installs the current release from GitHub by its tag;
  `tests/test_release.py` keeps that tag equal to the stated version.
- Version 3.0.0 is the first release made this way.
- No changelog file is kept; the release's generated notes, its commits and
  the decision records say what changed.
- The package ships `py.typed`, so a caller's type checker reads the
  library's annotations.
- The workflows hold read rights only, each job given what it writes, and
  name each action by the commit of the version they use.

## Consequences

- `pip install "git+https://github.com/...@v3.0.0"` installs a fixed release.
  The engine is still built from source on the installing machine, with the
  `CMAKE_ARGS` the README's install section gives.
- GitHub makes no push event for any tag where more than three are pushed at
  once, so the version tag is pushed on its own, after the earlier ones.
  The earlier tags hold no release workflow and make no release.
- The release workflow runs no tests: a tag is pushed from a machine whose
  `hooks/pre-push` runs the whole suite on it first, and the suite needs the
  engine built for that machine.
- The security policy's private reports work once private vulnerability
  reporting is switched on in the repository's settings.
- An action pinned by its commit is updated by hand, with the version in the
  comment beside it.

## Alternatives rejected

- Publishing on PyPI: the package is published on GitHub only.
- Tags without a release: a release carries the built package for a user
  without git.
- A version read from the tag at build time: two places would state it during
  development.
