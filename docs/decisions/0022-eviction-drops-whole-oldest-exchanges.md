# 0022. Eviction drops whole oldest exchanges, never the system turn, so a chat runs indefinitely

Date: 2026-11-15

## Context

When the context fills, something must go. A conversation that opens on a
reply is a shape many templates refuse or write otherwise, and a removal must
not decode the survivors again where the cache can avoid it.

## Decision

- Two triggers: more than `evict_above_fraction` times `context_size` tokens,
  and more than `max_history_exchanges` exchanges (off by default). An
  exchange is one user message and everything after it up to the next; what a
  history holds before its first user message counts as one more.
- Whole exchanges go, oldest first, and never the system turn; the survivors
  open on a user turn. Under a tiny limit the newest reply can go with its
  exchange.
- On a cache that shifts, evicting `[a, b)` is
  `llama_memory_seq_rm(seq, a, b)` and
  `llama_memory_seq_add(seq, b, -1, -(b - a))`. The shift fixes the
  RoPE positions but only marks the cells; the K-shift is built into the next
  decode. So the assistant turn's closing is decoded after the eviction that
  follows the reply, and that decode carries the shift instead of the next
  question's prefill. A test pins the order.
- Exchanges are dropped before the decode ahead of the next question, for the
  same reason.
- The strategy is chosen at construction from `llama_memory_can_shift()` and a
  cut tried on the empty cache. A cache that refuses the cut holds a running
  state and is never shifted, whatever it reports (a hybrid cache reports its
  attention part's shift). Where the cache cannot shift, see 0023 and 0024.
- The count of exchanges to drop is searched: the first try drops every
  exchange that can go; then each try drops the count at which the lengths
  measured so far, read as a straight line, reach the bound, and a try that
  does not halve the range is followed by one that does.
- A request evicts before it is refused. The exchange limit holds at rest
  after each action; a question is added with only the token room enforced and
  the limit is restored when its reply settles; `inject` evicts first.
- Where the conversation still renders over the threshold after every
  exchange that can go has gone (replies no question follows), the action is
  refused with a `ContextOverflowError` saying so.
- Eviction records how many messages went and why where it drops them.

## Consequences

- The reuse is lossy, the StreamingLLM trade-off: the survivors' K and V were
  computed while attending to the removed tokens, so quality drifts slightly.
- On Gemma 4 26B-A4B at 8k, dropping after the decode ahead made the first
  word wait 500 ms instead of 200 ms.
- A long question into a context full of short exchanges stalled 85 s at 64k
  on the stand-in with one render per dropped exchange; the search takes
  1.0 s, about two dozen renders.

## Alternatives rejected

- Single-message eviction: it could open a conversation on a reply, which 6
  of 50 test templates refused and 4 rendered otherwise than the cache held.
- A limit counted in messages: whole exchanges go, so a limit of 3 kept 2 and
  a limit of 1 kept nothing between turns.
- Removing the exchange limit as a second trigger: the owner kept it.
- Evicting before every question at the limit: it would skip the question
  decoded ahead on every turn and put the eviction before each reply.
