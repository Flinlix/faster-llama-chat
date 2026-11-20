# 0054. What is public, and what a version number promises

Date: 2026-11-20

## Context

The package was at its third major version within a few weeks, marked as
stable, with no statement of what a caller may rely on or what a version
number says, and nothing noticed a change to a public signature.

## Decision

- `docs/guide/compatibility.md` states what is public: the names
  `llama_cpp_conversation` exports, `llama_cpp_conversation.testing`, what
  `conversation()` returns and `begin` takes, the add-on file format, the cache
  context protocol and the two logger names. Everything else, the wording of
  messages included, is not.
- Versions follow `MAJOR.MINOR.PATCH`: a major release may change anything
  public, a minor one only adds, a patch only fixes. A fault in what a rule
  promises is fixed in a patch.
- `tests/test_public_api.py` compares every public signature with
  `tests/public_api.txt`, read in a fresh interpreter since the suite's own
  checks wrap some of the names. A change to that file is made with the
  signature it follows.
- The exported constants and type aliases carry their own docstrings, so the
  API reference lists every exported name; it also lists `FakeContext` and
  the invented templates.

## Consequences

- A change to a public signature fails the suite until the list is written
  anew, which puts the change in the commit's diff for review.

## Alternatives rejected

- A test of the exported names alone: a changed argument or default would pass
  it.
