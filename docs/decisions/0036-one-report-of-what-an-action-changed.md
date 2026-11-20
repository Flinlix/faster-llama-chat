# 0036. Every action reports, and logs once, what it changed beyond what was asked

Date: 2026-11-14

## Context

Six places dropped history and said so in different ways: `begin` returned how
many items it kept, the other actions how many they dropped, and none said
why. A cut result was reported only in the next reply, a reply the template
refused vanished while the stream still showed it, and a result that did not
fit even cut to nothing closed the turn and returned 0, as a success does.

## Decision

- `begin`, `inject` and `inject_tool_result` return a `Report`: the messages
  that went and why (`EVICTION_CAUSES`: no room, the exchange limit, or a
  begin's share of the context); for a begin, how many it kept, the unreadable
  calls it left out and the generated turns it restored or found to be another
  model's; the results cut; whether the action closed the open turn; and
  whether a question was joined to an unanswered one.
- A `Reply` carries the same where it concerns a reply, and also whether the
  request was taken back and the answer as kept where the template keeps
  another text than was shown. A stream closed early holds the summary of the
  generation it was closed in.
- Eviction records how many messages went and why where it drops them, and
  every report reads only that record. Counts count messages, a result
  included.
- Each such change is logged once at the level its kind takes (0018). A
  `begin` logs one line only where it left something out or could not use the
  generated turns it was given, for example "begin kept 120 of 200 history
  messages; left out the 80 oldest so that half of the 8192-token context stays
  free".

## Consequences

- A caller learns of every change it did not ask for at the action that made
  it.
- Eviction during a running conversation logs at `DEBUG`, since once the
  context is full it happens every turn.

## Alternatives rejected

- Working the count out by subtraction: done in nine places, and a written
  result reported -1.
- A Python warning for an engine whose penalty signature differs: now a
  `WARNING` record like every other thing to fix.
