# 0002. No model is special: what one model needs lives in that model's own files

Date: 2026-11-19

## Context

The library is meant to serve every model that ships a chat template, a future
one included. Code that names a model, a family or a vendor is code that serves
one model, and it hides whether the general path works.

## Decision

- What a model needs beyond its chat template comes from its add-on file
  (0011). Where a template itself shows what it does, the library reads it
  from the template.
- A fault found on one model is fixed in the general code wherever its cause
  is a kind of template any model may have. Only what is known from one model
  alone goes into that model's add-on file.
- No file names a model, a family or a vendor, except the model's own files (a
  test template in `tests/templates/`, an add-on file in
  `llama_cpp_conversation/addons/`), the decision records, the README's install
  section, `install.sh` and `pyproject.toml`. llama.cpp and llama-cpp-python
  may be named anywhere.
- The tests name no model either, and write no tag of a real template. A test
  chooses its templates by what they do, from properties stated in each test
  template's file (0047). Real model files are named by the environment
  (`LLAMA_CPP_CONVERSATION_TEST_MODELS`), never in the tree.
- `tests/test_neutral.py` enforces this over every text file of the project,
  except hidden folders, `__pycache__`, `*.egg-info` and the result files a
  mutation run writes (names beginning `mutmut-`). It reads the families,
  publishers and chat formats from the model files themselves, so it names
  none. A name is split at `_`, `-`, `.`, before capitals and between letters
  and digits, and every name before a slash in a file's `source` counts as a
  publisher. In `_backend.py` and its tests the engine's own function names
  are allowed. No other file may write a tag longer than three characters
  that a test template writes.
- A family, publisher or chat format whose name is also an everyday word is
  listed under `everyday_words` in its template file. Such a word passes as a
  part of a longer name, and is refused standing alone or followed by a
  version, since that is how prose names a model.
- The history form is named after its API, "Chat Completions", not after its
  vendor.

## Consequences

- A new family needs a test template in `tests/templates/`, and an add-on file
  only for what its template cannot show.
- Every fix applies to every model of the same template shape.

## Alternatives rejected

- A rule for one family's tags alone: a chat format's tags are shared by
  several families and identify the format as its name does.
- Giving the neutrality test a list of common model names: such a list would
  itself name vendors in a file the rule keeps free of them.
- Marking only publishers as everyday words: compound identifiers made of a
  family's word were refused.
