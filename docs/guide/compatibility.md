# Compatibility

What a caller may rely on from one release to the next, and what a version
number says about it.

## What is public

- Every name `llama_cpp_conversation` exports, with its signature, as the
  [API reference](../reference.md) lists them.
- `llama_cpp_conversation.testing`: `FakeContext` and the invented templates,
  for tests of code that drives a `ChatWrapper`.
- What `conversation()` returns and `begin` takes, the add-on file format,
  and the cache context protocol (`CacheContext`).
- The logger names `llama_cpp_conversation` and `llama_cpp_conversation.engine`.

Everything else is not public: every other module and name, names that begin
with an underscore, the text of log lines and error messages (their type is
public, their wording is not), and the tokens a reply is generated as.

## What a version number says

A release is `MAJOR.MINOR.PATCH`:

- **A major release** may change or remove anything public. Its release
  notes name each change.
- **A minor release** adds to what is public and changes nothing that was.
- **A patch release** fixes a fault and changes nothing public.

A fault in what the rules promise, such as a cache that does not hold the
template's render, is fixed in a patch release, even where a caller noticed
the faulty behaviour.

## Saved conversations

What `conversation()` returned is taken by `begin` in every later release of
the same major version. Its `generated_turns` are only meaningful to the
model that wrote them; on another model, the conversation is written again
from its messages.

## How it is checked

`tests/test_public_api.py` compares the signature of every public name with
the list in `tests/public_api.txt`. A change to that list is made in the
commit that changes the signature, and names the kind of release it needs.
