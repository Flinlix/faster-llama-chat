# 0042. A call keeps the model's id where it is new, else gets one of nine characters

Date: 2026-11-15

## Context

A template that pairs a result with its call by id needs every id unique
within the conversation. Some templates refuse any id that is not exactly 9
alphanumeric characters, or refuse a short one.

## Decision

- `ToolCall` carries its `id`, wherever a call is handed out: the stream,
  `Reply.tool_calls`, `pending_calls` and `conversation()`.
- A call keeps the id the model wrote where that id is new to the conversation
  and the template writes the call under it; otherwise it gets an id of its
  own, logged rather than refused, since the call itself was read.
- The library's ids are `call` and five base-36 digits, nine characters, for
  every model; the length needs no add-on key.
- Only ids of exactly five base-36 digits after `call` count towards the next
  number. The next is one past the highest, or, where that leaves fewer than
  1,000 numbers below 36 to the power 5, the lowest run of 1,000 numbers no id
  holds.
- The count only goes up within one wrapper; a failed `begin` puts it back.
  Ids of put-back turns and of left-out calls count. A history that gives one
  id to two calls is refused.
- `ToolCall` hashes by its name and id; equality compares every field.

## Consequences

- About 60 million calls per conversation before an id would pass nine
  characters.
- A server can read each call's id from what the library returns instead of
  keeping its own list.

## Alternatives rejected

- Numbering from the count of calls the conversation held: after a `begin`
  that dropped older exchanges, the next call took an id a kept call had.
- Five decimal digits: the 100,000th call was ten characters long.
- Setting `__hash__` to `None`: the owner chose hashing.
