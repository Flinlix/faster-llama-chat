# 0049. The README and the guides keep to what a caller relies on; the reasons live in the decision records

Date: 2026-11-20

## Context

The README held passages on how the library works inside, a table of modules,
and what each test file checks; code comments argued their case at length.
Both drifted from the code and from each other.

## Decision

- The README is short: what the library does, why, how to install it, a
  quick start, the models its tests cover, and links to the documentation.
- The guides in `docs/guide/` state what a caller may rely on, one page per
  task: what to call (one line per call, its details as a paragraph after),
  streams and threads, tool calling, reasoning, keeping a conversation,
  configuration, add-on files, models without a usable template, a cache
  context of one's own, and how to verify. Passages a docstring points to are
  headings of a guide.
- The API reference is built from the docstrings (`docs/reference.md`), and
  the guides, the reference and the records form one site, built with
  Sphinx with every warning an error (0057). The site's overview and install
  pages take their text from the README's marked sections, so the two never
  differ.
- The README's list of supported models is the table the test templates'
  files give, and a test keeps it equal to that table, row for row. It is the one place outside
  the install section where the README names models (rule 3).
- Each module's docstring says what the module is; each test file's docstring
  says what it tries to break. An internal docstring is a summary line, with
  `Args`, `Returns` and `Raises` only where the signature leaves a question
  open.
- The reason for a line, the alternatives and the measurements live in the
  decision records; the code keeps what it does and a pointer to the record.
- The contract names, for each rule, the test files that try to break it.
- The README's quick start and the demo use the public interface alone and run
  on every test template: each injects a question and answers it with
  `reply()`, a test running the same steps on every template.
- Every Python example in the README and the guides runs as written in
  `tests/test_examples.py`, on the stand-in model in place of a model file,
  the examples of one page in order as a reader runs them (2026-11-20).

## Consequences

- A reader of the README meets only promises; a reader of the code finds the
  reason one pointer away.
- The README's links point to the documentation site on GitHub Pages, built
  from `main` after every push, so that they work wherever the README is read
  (2026-11-20; they pointed to the files on GitHub before the site was
  hosted).
- The tests that read the documentation (the exported names, the logging
  example) read the README and the guides together.

## Alternatives rejected

- A README that explains the mechanism: two documents to keep in step.
- One README of about a thousand lines holding every guide: long to read on
  GitHub, and without an API reference.
- The site's pages written apart from the README: the overview and install
  text would drift between the two.
- A demo that subclasses the engine context to count decodes: it now reads the
  same from `begin`, `inject` and `n_prefilled_tokens`.
