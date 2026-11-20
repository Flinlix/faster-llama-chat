# 0056. The project is named llama-cpp-conversation

Date: 2026-11-20

## Context

The project was named `faster-llama-chat` and imported as `llama_chat`. The
name made a claim ("faster") with nothing to compare against, read as a chat
for one model family although the library takes any model whose file ships a
chat template, and gave one thing three names: the repository, the import and
the README's title.

## Decision

- The repository, the distribution, the documentation site and the folder
  are named `llama-cpp-conversation`; the import, the logger names and the
  prefix of the test settings read from the environment are
  `llama_cpp_conversation` and `LLAMA_CPP_CONVERSATION_`.
- The rename was made before the first release, version 3.0.0, so that no
  release carries the old name.

## Consequences

- A caller changes its import once. An environment that installed the old
  distribution uninstalls it; nothing else claims the new import name.
- GitHub forwards the old repository address to the new one.
- `docs/decisions/archive/design_history.md` keeps the old names, as it is
  kept unchanged.

## Alternatives rejected

- `conversation-optimized-llama-cpp`: reads as a modified build of llama.cpp,
  "optimized" is a claim, and the import would be 32 characters long.
- `llama-kv-chat`, `llama-cpp-session` and others: the name chosen says what a
  caller gets, a conversation layer for llama.cpp, without a claim.
