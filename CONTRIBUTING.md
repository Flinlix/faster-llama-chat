# Contributing

The rules every change keeps, and the commands that test them, are in
[AGENTS.md](AGENTS.md). Each rule names the tests that try to break it; a
change that needs a rule changed says so, with the reason.

Set up an environment with the test and development tools:

```bash
./install.sh --dev
```

This builds the engine for the machine, installs the package editable with
its `test`, `dev` and `docs` extras, and switches on the check that runs the tests
before every push.

A change that makes or revises a design decision comes with its decision
record in [docs/decisions/](docs/decisions/README.md), in the same commit.

## Adding a model family

A family not covered yet gets a test template: the file
`tests/templates/STEM.json`, which every template-wide test runs on.

1. `python -m tests.new_template path/to/model.gguf publisher/repository STEM`
   writes the keys the model file tells: its template, its end-of-sequence
   tag and the tags the template writes, with a first guess at the rest. It
   reads the vocabulary alone and loads no weights.
2. `python -m pytest -q tests/test_real_templates.py -k STEM` checks each key
   against what the template does; a failure names the key. Correct the file
   until it passes. The keys every file states are `REQUIRED_KEYS` in
   `tests/chat_templates.py`; each key a file may state besides, with what it
   means and its value where the file leaves it out, is listed in `_UNSTATED`
   beside it.
3. Run the whole suite. A template the load check refuses is refused with its
   cause (rule 2): where that cause is right, the family is not supported.
4. What the template cannot tell, such as the sampling the model's card
   recommends or a call format, goes into an add-on file in
   `llama_cpp_conversation/addons/`, matched by the template's fingerprint
   (rule 3, and the [add-on file guide](docs/guide/add-on-files.md)).
5. Add the family's row to the README's table of supported models, in its
   alphabetical place. `tests/test_neutral.py` compares the table with the
   rows the test templates' files give.

## What the suite refuses

Besides the rules' own tests, the suite refuses a change that breaks one of
these. Each is found by the test named, which says where.

| Check | Test |
|---|---|
| No em dash anywhere, no dash between two clauses in prose, no path of one machine outside the decision records | `tests/test_prose.py` |
| No model family or vendor named outside the files rule 3 lists | `tests/test_neutral.py` |
| Every `while` loop of the library gives the measure it makes smaller (rule 12) | `tests/test_loops.py` |
| A plain reply runs only the functions listed as serving every template (rule 5); a function a plain reply needs is added to `_PLAIN_REPLY_CODE` | `tests/test_wrapper.py` |
| Every error names the argument and the cause | `tests/conftest.py` |
| The linter and the type checker find nothing | `tests/test_static.py` |
| Every file a rule names exists, and every rule a file cites is one of the contract's | `tests/test_contract.py` |

Report a bug or ask for a feature through the issue forms. Report a
vulnerability privately, as [SECURITY.md](SECURITY.md) describes.
