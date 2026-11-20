# 0044. Actions take turns under one lock, readers see a published copy, and the thread that reads a reply cannot wait for itself

Date: 2026-11-19

## Context

One cache sequence serves every generation, so two actions whose decodes
interleave would corrupt it. A reply holds the wrapper while it is read, so an
action on the reading thread, such as a tool the reply runs calling back into
the wrapper, waited for a lock it already held and never returned.

## Decision

- The wrapper's lock is held for a whole action, a generation and the tool
  runs between its rounds included, so no other action comes between the
  rounds.
- The conversation publishes a copy once per outermost action. Readers without
  the lock read only it: `conversation()`, `pending_calls`, `turn_open`,
  `total_tokens`, `awaits_reply`, `capabilities()`. A count read while a reply
  streams is the one from before that reply.
- `reply()` checks the published state at once, without the lock; that check
  is advisory, and the reply is checked again under the lock. The policy and
  budget a reply goes on with are chosen under the lock as it starts.
- The thread that takes each step of a stream is noted as the holder, as the
  thread object. An action on that thread ends its own unread reply at once;
  `close()` there ends every unfinished reply of the wrapper first, recording
  what each produced, and then frees the model even where ending a reply
  raises. From a tool the reply runs, an action is refused with an
  `OutOfOrderError` and the tool is answered as one that raised. Another
  thread waits, reading the cancel every 0.05 s. A tool that reads its own
  reply's stream is refused.
- A stream is driven from one thread at a time.
- `close()` copies the set of replies under a lock of its own; a begin's checks
  that read the model run under a lock `close()` takes before it frees the
  model.

## Consequences

- `tests/test_thread_orders.py` holds the work after a reply at each point
  where it hands over and runs each kind of action there.
- In `with ChatWrapper(...) as chat:`, an error raised while a reply was read
  now comes through unchanged, and the model is freed.

## Alternatives rejected

- A locked read of the calls still owed: it would block a thread that reads it
  while its own stream holds the lock.
- Taking the lock when `reply()` is called: an unread stream never runs its
  `finally`, and a thread whose own stream is paused would deadlock on its
  next `reply()`.
- The thread's number as the holder: a thread that has ended can pass its
  number to one started later.
