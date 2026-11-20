# 0047. The tests try to break the rules, on real templates, with expectations the code under test does not supply

Date: 2026-11-19

## Context

The tests once leaned on a hand-written renderer, on copies of names the code
defines, on the library's own readings of a template and on its private
fields, so a wrong reading passed: against a copy of the library that reads no
reasoning switch, 122 of 130 reasoning tests passed.

## Decision

- One stand-in, `FakeContext` (shipped as `llama_chat.testing`), with one
  tokenizer, one cache and one scripted model, renders real Jinja templates
  through the renderer a model loads with: the test templates of real families
  in `tests/templates/` and small invented ones. Every conversation test runs
  on a tokenizer of one token per character and on a merging one, which joins
  a space with the letter after it. The stand-in reads only the tags each
  template's file lists and can write a reply's tags as their tokens.
- What a test expects of a template stands in that template's file
  (`render_variables`, tags, call opening, which properties it has, its card's
  sampling); `tests/test_real_templates.py` checks each stated value against
  the template, and a file may hold only the keys the tests read. Tests take
  their cases from those lists instead of skipping.
- Checks after every test: the cache holds the template's render token for
  token, a tag set apart from its letters, each generated turn placed among
  its own messages (shown to catch a break by `tests/test_cache_check.py`); no
  generated turn spells a later tool result; what `conversation()` gives is
  taken by `begin`; every library error says what went wrong.
- One property for a failed action: every public action fails at each cache
  operation in turn, on three kinds of cache (`tests/test_failures.py`).
- Tests read private state only through `tests/internals.py`. Tests that bound
  work count renders or the thread's processor time, not clock time.
- Random action sequences run as a Hypothesis state machine
  (`tests/test_sequences.py`); each failing sequence found is kept in
  `tests/found_sequences/`, recorded by choice index and template fingerprint,
  and replayed every run.
- `tests/mutation_run.py` runs mutation testing so that a mutant is tested
  again wherever a test that reached its function changed, each test function
  identified by a digest of its syntax tree, its module's other code and the
  shared test code.

## Consequences

- Measured: the suite went from 285 s to 55 s with no check dropped, by
  compiling each template once and remembering 200 renders per template.
- A sample of mutation survivors held real test gaps, which became the checks
  after every test above.

## Alternatives rejected

- A hand-written renderer: no test may pass on render behaviour only it has.
- Expectations computed by the library under test.
- Following which helper each test calls, for the mutation digests: it needs
  the tests traced and still misses data a helper reads.
- One pattern for every tag shape: it took `<IMPORTANT>`, plain text in eight
  templates' instructions, for a tag no vocabulary holds.
- A list of known library faults marked in the template files: it went once
  the faults it held were fixed.
