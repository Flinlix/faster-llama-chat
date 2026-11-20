# 0021. One object brings the cache to a target built from the render alone

Date: 2026-11-19

## Context

Decodes, generations, removals and cuts were made from several places, so the
cache and the record of what it held could drift apart until the next
`begin`. A target taken partly from the cache would carry any earlier error
forward.

## Decision

- `CacheSync` holds every decode, generation, removal and cut, and owns the
  tokens the cache should hold. Bringing the cache to a target is
  `CacheSync.hold`: it compares the prefix on token lists (in blocks of 1,024
  tokens), searches for a stretch the target no longer holds, removes it and
  shifts the rest where the cache can shift, cuts, and decodes from the first
  difference. The text form of the tokens is built only for the stretch
  search.
- The target is built from the render alone, never taken from the cache, so a
  wrong match in the stretch search costs decoding only.
- A cache that shifts or cuts takes each change at once. One that cannot is
  brought up once before the next generation or rest, from the newest saved
  state that still holds a beginning of the target, else by decoding again
  (0024).
- `CacheSync.version` changes with every change to the token list. Records
  made ahead (the render for the next question, the splice note) carry the
  conversation and cache version they were made for and are used only while
  both stand.
- A decode or edit that fails part-way marks the cache unknown; the next
  action brings it back, and a repair that fails is tried once more by decoding
  the whole conversation from nothing. Putting back unchanged tokens changes
  nothing.
- A stretch the template writes into the prompt before a reply but leaves out
  of the past reply (an empty thought, say) is removed and the rest shifted,
  where the cache can shift, the stretch holds no message's text, and the cache
  goes on as the render does after it. The prompt range it is searched in is
  kept with the cache and moved down with every removal before it.
- A call's closing tag is decoded without logits, since a result follows.
  What an interrupted decode left is decoded with what the next action adds,
  in one batch, where the cache can shift or cut. An action copies the token
  list once.

## Consequences

- The cache's tokens are the render exactly. Where a stretch is shifted out,
  the entries after it were computed with that stretch in view; that is the
  same reuse an eviction accepts, accepted by the owner for stretches that
  hold no message text.
- Before the shift-out, a template that writes an empty thought into the
  prompt but not into a past reply made the cache cut there and decode the
  reply again after every reply.

## Alternatives rejected

- A recorded map of the cache, which every edit would have to keep in step:
  the search is a decoding shortcut, not a placement.
- The cache's length as its version: two different targets of one
  conversation can have the same length.
- Converting both token lists to text for the comparison: 24 ms against 15 ms
  on token lists at 128k tokens.
