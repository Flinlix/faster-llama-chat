# 0031. The caches of conversations a begin switched away from are kept, within a RAM bound that defaults to none

Date: 2026-11-19

## Context

A server that serves several conversations on one model switches between them
with `begin`, and each switch decoded the other conversation again from the
start. The owner asked for a RAM bound, and that a bound of 0 switch the
feature off.

## Decision

- `kept_conversation_bytes` bounds the host RAM the caches of conversations a
  begin switched away from may take. The default, 0, keeps none and switches
  the feature off.
- Where a begin leaves more than 256 tokens of the cache behind, the whole
  cache is saved (`save_whole_state`, in the protocol) with its tokens. Where a
  kept one shares more than 256 tokens more of the new conversation than the
  cache does, it is restored instead and what differs from it is decoded. The
  entry is taken out of the store before anything else is kept.
- The oldest kept goes first; one that alone exceeds the bound is not kept. The
  bound holds once they are kept; while one is saved it takes its own size
  beside them. A whole state is not saved again for a cache at least as long
  as one whose state did not fit.
- Only a begin keeps or switches; an eviction never does.
- A cache that can neither shift nor cut keeps no conversation.
- A failed action puts the kept conversations back as they were.

## Consequences

- Switching back to a kept conversation decodes only what differs from it.
- The RAM is spent only where the caller sets a bound.

## Alternatives rejected

- Keeping conversations on a cache that can neither shift nor cut: going back
  restored the whole cache but not the partial states it returns to, so the
  next change near its end decoded everything again (2,299 tokens against 36
  in one measured run). Keeping those states with each conversation is left
  until such a model needs it.
