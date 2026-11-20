# 0029. A history has two forms: Chat Completions messages, and beside them the generated turns

Date: 2026-11-20

## Context

A caller, typically a server, keeps the conversation and begins again from it
after a restart or a settings change. One mixed form, chat messages plus this
library's extras, made every caller convert both ways and copy the library's
rules, and the turns kept as generated were lost on every restart, since no
template writes them back exactly.

## Decision

- `begin` takes, and `conversation()` gives, Chat Completions messages: the
  system message first (an empty one kept, distinct from none, since some
  templates write a default prompt only where none is given); `user`;
  `assistant` with text, or `null` where it only calls; `tool` with its
  `tool_call_id`; every call with an id and its arguments as a JSON string.
  `reasoning_content` is the one field beyond the official form, the name
  other compatible servers use. A result is text; a value from tool code is
  written as JSON at once.
- Only the smallest part of the form is taken: a list of content parts, the
  `developer` role, `name`, `refusal`, `audio` and a `custom` call are
  refused, naming where they stand. Whatever `begin` takes `conversation()`
  gives back unchanged, but for the arguments' spelling (compact JSON in their
  own order) and an empty reply's text.
- The checks are in one place (`messages.as_messages`): a result answers the
  oldest call still owed since the newest question, by its id or with none; a
  result out of order or naming another function, and a reply between a call
  and its result, are refused. A call whose arguments are no JSON object is
  unreadable and left out with its result.
- `generated_turns` is the second form: each turn kept as generated, as its
  token ids and its own messages through the end of its turn (a thought, an
  unreadable call and the library's answer to it included), with the model's
  fingerprint (a digest of the GGUF's description, its chat templates, its
  vocabulary and its file size, taken at load where the model can call tools).
  Each record carries a digest of the rest; a record changed since is ignored
  and `generated_turns_ignored` says so. It is trusted input.
- A record is found by a digest of its Chat Completions messages with the
  question before them, from one conversion of the whole conversation;
  records are placed newest first. On the same model its messages and tokens
  are put back; on another model, or without it, every turn is written as the
  template writes it, an unreadable call left out with its result.
- `conversation()` holds everything `begin` takes (`messages`, `tools`,
  `reasoning`, `reasoning_level`, `generated_turns`), so
  `begin(**chat.conversation())` gives the same conversation.

## Consequences

- A server begins again from its own list of messages, appended to and
  trimmed, without losing exact turns.
- Turns whose saved messages differ from what a client sends back fall back
  to the template's render. Measured: a begin with 400 saved turns over 1,600
  messages matched them in about 0.02 s, against 3.4 to 6.8 s before the
  digests per width were worked out once.

## Alternatives rejected

- One mixed form: the server's copy of the unreadable-call rule had drifted,
  and a saved turn with a thought was never found.
- `(role, text)` pairs and a step form of tool turns: second ways to write a
  history.
- One sealed record of the whole conversation: lost by any edit a client makes.
- Text and tag positions in place of token ids: only ids are exact where a
  model splits a word otherwise than its tokenizer.
- Restoring an unreadable call as text spliced into the template's call frame:
  it needed a rule per template shape and failed on two.
- Reordering results given out of order: it writes a conversation the caller
  did not send.
