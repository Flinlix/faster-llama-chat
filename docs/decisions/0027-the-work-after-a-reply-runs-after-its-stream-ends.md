# 0027. The work after a reply runs on a thread of its own once the stream has ended, and gives way to a waiting action

Date: 2026-11-20

## Context

After a reply, the conversation is evicted where it is over a limit and
decoded ahead of the next question. A stream that ended only after that work
made a server that speaks the last sentence at the stream's end speak it
later.

## Decision

- A reply marks that work due. At the stream's end the wrapper starts a
  thread, not a daemon, that waits 5 ms so the reader takes in the end first,
  then takes the wrapper's lock, for at most one second, and does it. Every
  action and `wait_idle` does a due job first, so the order of the work is as
  before. The thread takes no lock where an action already did the work.
- An action that waits for the lock sets an event the cache reads between the
  parts of a long decode (one micro-batch, at least 512 tokens), not before a
  short one. The work stands as far as it got, the cache left behind its
  tokens, and the action finishes the rest.
- A failure in that work has no caller: it is logged at `WARNING`, the
  conversation is as it was before the work, and the next action decodes the
  cache again. What the work evicted and decoded is counted by the next
  request's `Reply` or `Report`. A `begin` forgets what was left undecoded.
- A question asked with `reply` before that work began is answered first,
  where the cache holds everything it is to hold, no tools are declared, no
  exchange limit is passed, and the question with its tags and a reply of its
  cap fit in the context beside the cache. This happens once in a row; the
  drop follows its reply. `inject` defers nothing.

## Consequences

- The rule that the cache is the render allows the cache to hold less than
  the render until the interrupting action decodes the rest, before the model
  reads it.
- Measured on the stand-in at 2,048 tokens with a reply cap of 120, every
  other question asked at once: Llama 3.2's template went from 736 to 95
  tokens at most before the first word.
- The tests do the work at once on the reader's thread unless a test asks for
  the thread.

## Alternatives rejected

- Putting interrupted work back and decoding it again: three to four times the
  cost of waiting (measured).
- A daemon thread: a script that ended right after a reply crashed at exit, 3
  of 3 on the Ministral 8B GGUF.
- Answering first without a reply cap, or with tools declared: the reply would
  get only the room left above the threshold, or the results' room is
  unbounded.
- Deferring after `inject`: the next action decoded ahead a second user turn,
  which failed the cache check on 19 of 53 test templates.
- Sizing the decode parts by time: it would need a rate measured on the
  machine; 512 tokens are about 150 ms on a GPU.
