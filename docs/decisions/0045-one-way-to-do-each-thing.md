# 0045. The public interface has one way to do each thing

Date: 2026-11-14

## Context

The interface grew second ways of doing one thing: three reads of the
conversation, a message callback beside the messages the caller already has,
two cache type spellings, a declaration call beside `begin`, a settings change
beside `begin`. Each second way is one more rule a caller must learn and one
more pair that can disagree.

## Decision

- One read of the conversation: `conversation()`. One read of the calls still
  owed: `pending_calls`. Whether a reply without new text has something to
  answer: `awaits_reply`. What a model can do: `capabilities(tools=...)`.
- One way to start, start again or change the tools or the reasoning:
  `begin`. One way to ask: `reply`, its call policy explicit (0039). One way
  to add a message without a reply: `inject`. One way to answer a call:
  `inject_tool_result`, which takes the id of the call it answers.
- No message callback: a caller already has every message it sends and every
  reply it reads, and refuses a message before calling.
- One spelling per setting: `kv_cache_type` always holds the type the cache is
  built with.
- Names say what they are: `pending_calls` (the calls still waiting for a
  result), `turn_open` (a tool call left the newest turn open).
- Types that serve only the library are not exported: `ModelReader`,
  `GenerationAccumulator`, `CallBoundary`, `ToolCallFragments`, `CallBody`,
  `result_text`. The test stand-in ships as `llama_cpp_conversation.testing`,
  so a server's suite can run on it.
- A wrong state is refused at once by `reply()`, against the published copy
  (0044), not at the first read of its stream.

## Consequences

- A caller cannot pair the system prompt of one `begin` with the history of
  another.
- A server keeps only its own call ids and reads everything else from the
  library.

## Alternatives rejected

- `history()` and `system_prompt` beside `conversation()`: two reads could
  pair halves of two begins.
- `declare_tools` and `change_settings`: `begin` with the conversation does
  both at the cost of a compare with the cache (0030).
- Removing `max_history_exchanges` as a second eviction trigger: the owner
  kept it.
