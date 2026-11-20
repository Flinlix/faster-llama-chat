# 0010. The caller sees only the answer the template marks as one

Date: 2026-11-20

## Context

A reasoning model writes a thought before its answer; a channel template
writes a header naming the answer's reader. Filtering the stream by text alone
hid an answer that explained the thought tag, showed a tag the model wrote
inside its answer, and could end a reply at a tag that closed a thought the
filter had not yet seen.

## Decision

- A reply's text runs through the thought filter, the header filter and the
  stop strings, in that order (`ReplyText`).
- A thought's tags are matched without the whitespace around them, which a
  model does not always reproduce.
- Before any answer text, a thought tag's text opens a thought whether written
  as the tag or as its letters. Once answer text has passed, a thought opens
  only where one of the model's own tag tokens stands within the opening, and
  any tag token of the model's inside the answer is left out of the stream,
  `Reply.text` and the conversation.
- The header a channel template writes before an answer is worked out once at
  load per reasoning mode, and neither of its ends may divide one of the
  model's tags.
- Where a prompt without reasoning ends in an empty thought and a past reply's
  tag holds more, that rest is the answer's opening, which the model writes and
  the answer leaves out.
- Whether a turn's end closes a thought is read from every token generated,
  through a thought filter of its own, so no hold in the stream changes where a
  reply ends.
- A call made inside an open thought is noted; the reply after its result
  starts its filter inside the thought. A `begin` reads that note from the
  kept tokens.
- `Reply.thought` holds the thought's text, joined across the generations of
  one request with a blank line.
- `max_thought_tokens` (in `Config` and per reply, `None` by default) closes a
  thought that has taken that many tokens with the template's closing tag;
  each thought is capped on its own.

## Consequences

- A gpt-oss thought under about 35 characters no longer loses the answer.
- North Mini Code's stray `<|START_TEXT|>` before every answer without
  reasoning is now the answer's opening and is not shown or kept.
- A reply may still hold several thoughts where the model writes each opening
  as its tag.

## Alternatives rejected

- Filtering by text alone: an answer that quoted the tag lost the rest of
  itself into a hidden thought.
- Working the header out per generation from both openings: on cohere-2 and
  inkling it ended inside a tag.
