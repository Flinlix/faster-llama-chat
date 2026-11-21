# 0053. Dependencies are bounded to the releases the checks are run on

Date: 2026-11-20

## Context

`pyproject.toml` allowed any `jinja2` from 3.0 on, although the template
sandbox is the library's security boundary and its tests check the sandbox's
remembered answers on 3.1 only (`tests/test_template.py`). The lint, type,
test and documentation tools had no upper bound, although the lint and type
checks run as tests: a new release of one could fail a push with no change
to the code.

## Decision

- `jinja2>=3.1.6,<3.2`. 3.1.6 is the first release that closes the sandbox
  escapes known for 3.1 (from the release notes, not checked again here), and
  the tests check the sandbox on the 3.1 line only.
- The `test`, `dev` and `docs` extras are bounded below the next major
  version of each tool, `ruff` below its next minor version, which may add
  rules to the selected sets, and the site's theme, versioned by date, below
  the next year. Each floor is one that installs on every Python version the
  package supports (Sphinx 9.1 needs 3.12, so the floor is 9.0).
- The site build is checked with every warning an error before every push
  (0057).
- A bound is raised in a commit of its own, after the suite passes on the new
  release.

## Consequences

- A clean environment installs the releases the checks were run on, within
  their line.
- The docs workflow installs the `docs` extra as `pyproject.toml` states it,
  so the site is built with the same bounds.

## Alternatives rejected

- A lock file of exact versions: one more file to keep in step, for a library
  whose callers choose their own versions.
- No upper bounds: a tool release could fail a push unasked.
