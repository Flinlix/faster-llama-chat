# 0025. After each reply the conversation is decoded up to the next question, so a question decodes only its own tokens

Date: 2026-11-13

## Context

The wait before the first word is what a voice assistant feels. Everything a
template writes before the next question's text can be decoded while the
previous reply is still being read.

## Decision

- After each reply, the conversation is rendered with a stand-in where the
  next question goes, and decoded up to it.
- A question takes the stand-in's place where the template writes that
  question's own text exactly so, checked on the question alone with the probe
  rendered once per conversation start. Only its tokens and the generation
  prompt are then decoded before the reply. Other questions, such as ones with
  surrounding spaces or a tag, are rendered when they come.
- An injected question is spliced within the room left for a reply, as
  `reply(text)` splices, and its eviction follows its reply; `reply()` after it
  goes straight to the generation.
- One probe at the conversation's start asks whether the template writes a
  question that repeats the probe's question, its reply or the system prompt
  as any other. Only where it does not is a repeating question rendered whole,
  and a template whose text then depends on another message's is refused
  before the reply.

## Consequences

- Measured through the owner's server on Gemma 4 26B at 8k, 36 questions of
  about 280 tokens: the first word with the context full took 182 to 184 ms
  before the injected path was fixed and 97 to 98 ms after (90 to 99 ms for
  the piece-built wrapper). The injected path had taken 224 ms against 84 ms
  for `reply(text)`.
- The test that a question waits only for its own tokens counts every render
  and drives the injected path, after several exchanges.

## Alternatives rejected

- A template-wide splice probe of three questions at `begin`: it added no
  guarantee and kept a template that failed one probe question from splicing
  any.
- Rendering the whole conversation before every spliced question: it would
  undo the windows (0026).
- Saving a state after the question for later shortening: see 0024.
