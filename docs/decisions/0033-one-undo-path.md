# 0033. One undo path: a failed, refused or cancelled action leaves the conversation and the cache as they were

Date: 2026-11-19

## Context

Ten places put the conversation back after a failure or a cancel, from three
different copies of "before the question", under three repair policies. Three
faults came from it: lost exchanges after an early failure, a cancel that kept
its question and recorded an empty reply, and a failed recording that kept a
failed reply's call policy.

## Decision

- The conversation has one state object, which a rollback fills in place; the
  parts that evict, cut results, publish and render windows are made with it,
  and the record of unasked changes, the turn's policy and call budget, and
  what answers an unreadable call live in it.
- Every action saves the state and the cache's tokens and puts both back where
  it raises. Every generation opens one attempt before its question is
  written; recording the reply ends it. Every exit before that (a cancel or a
  failure before the first word, a failure while the reply is recorded) puts
  the snapshot back whole: the exchanges the question evicted, the cut
  results, the call policy, the kept conversations.
- A fault after the reply is recorded keeps the reply: a failed reply keeps
  what it produced.
- A request taken back with its question because the template writes neither
  the reply nor an empty one goes back to the snapshot its first generation
  took.
- The policy and the budget a reply leaves its turn are noted only once its
  generation has started.
- A request whose cancel is set when it starts, or while its question is
  decoded, ends with `"interrupted"` and changes nothing. A long decode reads
  the cancel between parts. A later round of a turn cancelled before its first
  word leaves the turn open with its results. A cancel is read before each
  registered call.
- The cache is brought back at once where it is known; one a failure left
  unknown is brought back by the next action, and a failed repair is tried once
  more by decoding the whole conversation.
- A fault of the library partway through a drop is undone at once.

## Consequences

- `tests/test_failures.py` makes every public action fail at each of its cache
  operations in turn, on every kind of cache, and checks that everything is as
  before and usable.
- A cancelled request no longer decodes the whole kept history to roll back
  (1,958 tokens before, at most 512 after, on a cache that cannot shift).

## Alternatives rejected

- Storing a placeholder answer such as "error during reply": the model would
  later read it as its own words.
- Joining a question whose reply failed with the next one: it changes what was
  sent.
- Restoring the old policy on each failure path: every path would have to do
  it, while one point where the generation starts covers all of them.
- Giving each part its own copy of the state: a second rollback path.
- Keeping a cancelled question: the cancel would then drop history for good
  and decode a whole question nobody waits for.
