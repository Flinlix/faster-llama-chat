# 0019. llama-cpp-python 0.3.23 is the oldest binding supported; the package installs as faster-llama-chat

Date: 2026-11-13

## Context

The adapter looked every C symbol up under each name llama.cpp had given it
since 0.3.0, with fallbacks for builds without saved states. Read against each
release's `llama_cpp.py`: 0.3.10 adds the memory API, and 0.3.23 the
`llama_state_seq_*_ext` calls with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` that
saved states rest on; every other name is present from 0.3.9 on.

## Decision

- The floor is llama-cpp-python 0.3.23, in `pyproject.toml`, `install.sh` and
  the README. Every symbol is required under its one current name; the
  fallbacks and `Backend.can_edit_cache` and `Backend.can_save_state` are
  gone. The lookup under several names stays for the next rename.
- The penalties call differs inside the range (four parameters up to 0.3.34,
  five from 0.3.35, the vocabulary size first), so the adapter reads which one
  it has. An unknown signature warns at the first generation, which then runs
  unpenalized rather than refusing to start.
- `install.sh` checks the build by constructing the adapter, which raises where
  a symbol is missing. The message for a missing engine package names 0.3.23,
  that it is built from source for the machine, and the README's install
  section, which gives `./install.sh` and the build by hand.
- The distribution is named `faster-llama-chat` and is imported as
  `llama_chat`. An environment that installed the old distribution name must
  uninstall it first, as both claim `llama_chat`.
- The engine is imported when the first model file is loaded, not with the
  package: a caller with a context of its own, `llama_chat.testing`
  and the tests that use no model run without the engine built (2026-11-20). A
  `model_path` that names no file is refused as a setting before the engine is
  imported, so the refusal is the same with or without it.
- Python 3.11 is the version the package requires; no claim is made of the
  oldest version it is tested on. `jinja2` is a declared dependency (0004).

## Consequences

- One code path per engine call within the supported range.

## Alternatives rejected

- Keeping the fallbacks: they could answer nothing but yes within any build
  that has saved states.
- Installing the prebuilt wheel in the missing-package message: `install.sh`
  builds the engine instead.
