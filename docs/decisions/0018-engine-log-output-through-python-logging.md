# 0018. The engine's log output goes through Python's logging, and each level means one thing

Date: 2026-11-14

## Context

`log_level` set the level of llama-cpp-python's own logger, whose callback
printed to stderr past every Python handler. Measured on llama-cpp-python
0.3.36, recording the engine's raw log calls while gemma-3-1b loaded on the
processor and a corrupt file failed to load: the engine numbers its levels as
`ggml.h` does (1 debug, 2 info, 3 warn, 4 error, 5 a continuation), while the
binding maps them by an older numbering. The engine's errors were filed as
`DEBUG`, its warnings as `ERROR` and its 194 lines of metadata listing as
`WARNING`, so no level showed real errors while hiding the listings.

## Decision

- `_backend.py` installs its own callback through `llama_log_set` when it is
  imported, held at module level since the engine keeps only a raw pointer.
  Each line becomes a record of the logger `llama_chat.engine`: debug to
  `DEBUG`, info and a line without a level to `INFO`, warn to `WARNING`, error
  to `ERROR`. A continuation goes on with the open line at its level, a call of
  several lines becomes a record per line, and blank lines are dropped.
- `log_level` and `LOG_LEVELS` are removed.
- Across the library each level means one thing: `INFO` for a change of the
  conversation the caller did not ask for, `DEBUG` for a slower path (the
  cache decoded again, history going for room during a running
  conversation), `WARNING` for something to fix (an add-on file not used, a
  result that cannot be written, an engine whose penalty signature differs).
- The README's logging example adds `logging.basicConfig()`, since without a
  handler Python shows only warnings and errors.

## Consequences

- Measured on the same load: 1,922 `DEBUG` records, 195 `INFO`, 2 `WARNING`
  and 1 `ERROR` (the CUDA initialization, with no device visible).
  Unconfigured, Python shows only the last three.
- The routing is process-wide by nature: the engine has one log sink, so
  importing the package routes the engine's output for the whole process, a
  model loaded by other code included.

## Alternatives rejected

- Keeping a level setting beside it: a second way to do what `logging` does.
- Patching the binding's level map: the output would still print past
  Python's handlers.
