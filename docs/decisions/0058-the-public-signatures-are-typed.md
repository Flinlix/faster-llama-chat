# 0058. The public signatures are typed

Date: 2026-11-20

## Context

`begin` took and `conversation` gave bare dictionaries, and `reply` gave a
`ReplyStream` with no type for its pieces, so a caller's type checker could
check neither a history nor `begin(**chat.conversation())`.

## Decision

- `llama_cpp_conversation/conversation_types.py` holds the types of a
  Chat Completions message (`ChatMessage`, with `ChatToolCall` and
  `ChatFunction`) and of a whole conversation (`Conversation`), as typed
  dictionaries, and `GeneratedTurns`, a JSON-ready dictionary only the
  model that wrote it reads. The package exports them.
- `conversation()` gives a `Conversation`; `begin` takes any iterable of
  `ChatMessage` or of dictionaries, so that a history of plain dictionaries,
  read from JSON, type-checks as well; `reply()` gives a
  `ReplyStream[str | ToolCall]`.
- `ChatMessage` holds only the keys `begin` takes: a text or `None` as the
  content, the thought under `reasoning_content`, calls whose arguments are
  JSON text, and a result's `tool_call_id`. A form `begin` refuses, such as a
  list of content parts or a `name`, does not type-check either.
- The types change nothing at run time: `begin` checks a history as before.
- `tests/typed_caller.py`, a caller of the public interface, is checked with
  mypy's strict settings by `tests/test_static.py`: `begin(**conversation())`,
  a history of dictionaries and a reply's pieces.

## Consequences

- `ReplyStream.finish()` gives `Reply | None`, `None` only for a stream closed
  before it started, so a strictly checked caller handles that case.

## Alternatives rejected

- `begin` taking only `ChatMessage`: a history read from JSON is a
  `dict[str, Any]`, which a type checker does not take as a typed dictionary.
- One overload of `reply()` per call policy: the union type states what a
  stream may yield, and an overload per policy would make the policy part of
  the signature twice.
