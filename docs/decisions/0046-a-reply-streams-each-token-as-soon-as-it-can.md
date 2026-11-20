# 0046. A reply streams each token as soon as nothing could still change it, at a cost that does not grow with the reply

Date: 2026-11-19

## Context

The generation held back the last `held_chars - 1` characters of every reply
on templates that define a call opening or a reply ending (9 to 62 characters
on the test templates), and the answer behind stop strings held back
`max(len(stop)) - 1`. Several buffers were extended with `+=` on an
attribute, which CPython copies whole each time, so the cost per token grew
with the reply.

## Decision

- The generation holds back only the longest end of the text that is the start
  of a call's opening or of the reply's ending, and releases it with the token
  that shows it is neither. A plain-text opening is held only where a call may
  open; the closing is searched for only inside a call. Stop strings hold back
  only the longest end of the answer that starts one. Both use one function in
  `reply_text.py`.
- The text buffer keeps only what is not yet shown and where it ends without
  trailing whitespace; the token a position falls in is found by binary search;
  accumulated texts are lists joined when read, the answer once per read.
- Bytes run through an incremental UTF-8 decoder, so a character split across
  tokens is never broken.
- Where a decode returns only once it is done (`decode_blocks`: the processor
  alone), a token's text is yielded first and the token decoded before the
  next draw or the stream's end. On a GPU, where a decode only queues the
  work, the order stays. A token shown is in the cache before the stream
  yields again.
- Where a reply goes on with an open turn, it reads what the model wrote in
  that turn so far and starts with a blank line where the two texts would
  join without whitespace; one function says what a stream shows between two
  generations, for the stream and for a later cut.

## Consequences

- Measured on an i9-9900K (6 threads, context 4096, batch 512, greedy): on
  Llama-3.2-3B the first delta came after 2 to 10 sampled tokens, median
  0.18 s, before; after the first sampled token, 0.051 to 0.053 s, after. On
  gemma-3-1b with two stop strings, median 0.11 s before and 0.022 to 0.025 s
  after.
- On the stand-in, 8,000 generated tokens on a template whose call opens with
  plain text took 1.70 s instead of 0.03 s before; 80,000 tokens now take 0.28
  to 0.40 s on every template. An 8,000-word thought streamed in 4.9 s under
  the profiler before, 0.9 s after. A test counts bytecode steps per token,
  which does not depend on the machine.
- On the processor the first word comes one decode earlier.

## Alternatives rejected

- A fixed hold of the longest marker text: the first words waited for up to
  ten tokens.
