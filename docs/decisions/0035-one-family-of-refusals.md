# 0035. Every refusal is one family, raised before anything changes and naming its field; every other fault is an InternalError

Date: 2026-11-19

## Context

The library signalled failure with six exception types and no pattern: a
wrong type was a `TypeError` in some calls and a `ValueError` in others, only
settings named the field they refused, and raw Python errors escaped with no
cause. A server had to check each request field beforehand to say which was
wrong.

## Decision

- Every refusal is a `RefusalError`, a `ValueError` raised before the action
  changes anything, whose `field` names the argument or setting. Its kinds say
  what the caller does next: `InvalidSettingError` and `InvalidArgumentError`
  for a value, `UnsupportedError` for what the model cannot do,
  `TemplateRefusedError` for what the template does not write,
  `OutOfOrderError` for an action that needs another first (a closed wrapper
  and an action a tool asks for included), and `ContextOverflowError`, which
  carries `needed_tokens` and `available_tokens` and ends with one wording of
  what makes room.
- A wrong type is an `ArgumentTypeError`, a `TypeError`, naming the argument,
  so that any other `TypeError` is a fault. The engine's failures while it
  decodes, edits or generates are an `EngineError`, a `RuntimeError`.
- One entry point, `raising_library_errors_only`, wraps every public method
  and property of `ChatWrapper`, and `ReplyStream` wraps its own steps. An
  exception not in `LIBRARY_ERRORS` is raised as an `InternalError` naming the
  action and the cause.
- Inputs are checked where they enter: text (a lone surrogate refused by
  name), message forms, tool specs (which check themselves when built),
  generated turns. A history is read once, so a generator works.
- A refused history names the first message whose history up to it the
  template will not write, by role and its first words, with the template's own
  reason; it is found by halving. "Two user turns in a row" is said only where
  the template writes no two in a row.
- Every library error carries a message of at least three words; a check after
  every test enforces it.

## Consequences

- A caller can catch `ValueError` and read `field` to say which of its own
  fields was wrong.
- `tests/test_public_errors.py` checks that every public member goes through
  the entry point.

## Alternatives rejected

- Six unrelated exception types, and a type error that is sometimes a
  `ValueError`.
- An `InternalError` partway through a drop counted as a refused action: it is
  a fault of the library, but it is undone at once all the same (0033).
