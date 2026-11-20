# 0030. A begin compares with the cache, keeps the newest history that fits, and is the one way to change settings

Date: 2026-11-19

## Context

A `begin` emptied the cache and decoded everything again on every regenerate,
resume or change of tools or reasoning: 85 of 85 tokens on a short
conversation, all 14,019 at 64k. A separate action for changing settings was
a second way to do what a begin does.

## Decision

- A `begin` compares its render with the cache and decodes only from the first
  difference. It puts the conversation's own messages in the place of those
  its history opens with that say the same, and runs the load's checks and the
  room check again only where the system turn, the tools or the reasoning
  changed.
- It keeps the newest exchanges that fit: within half the context, or up to
  the eviction threshold where the cache already holds the oldest part of what
  it keeps. The search starts from the count of exchanges the cache holds,
  doubles and halves, and tries the whole history once the measured rate says
  it fits. `max_history_exchanges` applies, keeping at least the newest
  exchange.
- Changing the tools or the reasoning is a `begin` with the conversation and
  the new settings.
- A begin closes a turn a later question follows while it still owes results
  as the question closed it live, joins two questions in a row where the
  template writes no two, allows a newest reply or call the room it was
  generated in, and drops what is over the context as the work after that
  reply would.
- A history that ends on an owed call is decoded only as far as its render
  agrees with the render that has a stand-in result for each owed call, so the
  results only append.
- Checks that depend only on the arguments run before the begin takes the
  wrapper, so a begin refused for its arguments ends no reply.

## Consequences

- A begin of 800 messages takes about 40 renders and 0.03 s on the stand-in.
- A begin of a different conversation still decodes it whole, unless a kept
  conversation shares it (0031).

## Alternatives rejected

- Dropping one exchange at a time and rendering after each: 716 renders and
  5.4 s for 800 messages.
- Emptying the cache at every begin, and keeping only half the context every
  time: each settings switch lost history (24 messages fell to 16 after 30
  turns on a 2,000-token context) and decoded everything again.
- `change_settings` and `declare_tools`: second ways to change what a begin
  changes.
- Refusing a history whose call turn is left open before a later question: it
  blocked a server's resume.
