# 0023. Where the cache cannot shift, an eviction drops the conversation well below the threshold

Date: 2026-11-19

## Context

A cache that cannot shift (a recurrent or hybrid model's running state, a
compact sliding-window cache) must decode the survivors again after every
drop. Dropping just enough would do that on every turn, which the rule that
the memory rolls forbids. The same holds where the template writes the rest
otherwise once the oldest exchange goes, as one that numbers its messages
does.

## Decision

- There, an eviction brings the conversation down to
  `EVICT_TO_FRACTION_WITHOUT_SHIFT` (0.5) of the room under the threshold that
  remains after what the conversation takes with no history and a
  one-character question. Declarations that move with the newest question are
  not counted in the system turn.
- An eviction for `max_history_exchanges` leaves the same fraction of the
  limit, at least one exchange.
- For a long result, where the system turn and the open turn alone pass the
  usual bound, the bound is what they take plus half the room after them, so
  only the history it needs goes.
- A drop in windows goes down to the newest exchange alone where the newest
  two are still past the bound.
- A template that writes the rest otherwise only around turns that called
  tools drops this far only where the conversation holds one.
- The fraction is a constant, not a setting. It must lie in (0, 1].

## Consequences

Measured on Qwen 3.5 9B Q4_K_M on an RTX 3090, context 8,192, six tools, 70
questions of about 185 tokens, replies capped at 40 tokens:

| System turn | Version | Questions kept | Evictions | Tokens decoded |
|---|---|---|---|---|
| 987 | before the pinned state | 10 | 3 | 34,677 |
| 987 | pinned state | 10 | 3 | 31,716 |
| 987 | target from the room after the system turn | 13 | 4 | 35,780 |
| 3,517 | before the pinned state | 2 | 5 | 45,039 |
| 3,517 | pinned state | 2 | 5 | 27,463 |
| 3,517 | target from the room after the system turn | 6 | 8 | 34,222 |

The first word waited 55 to 57 ms in every run, 56 to 58 ms on turns that
evicted: the decode of the kept history falls after the previous reply, never
before the first word. On the stand-in with a limit of 4 exchanges, the cache
is decoded again every third reply (337 tokens, 75 between).

## Alternatives rejected

- Refusing models whose cache cannot shift: it would have lost Qwen 3.5.
  Only a model that would need the whole conversation decoded every turn is
  refused.
- A setting for the fraction: it can be chosen well only by measuring the
  decode it trades against the history kept, so no caller could set it better
  than its default.
- 0.4: by calculation it would bring the short system turn back to the old
  total and history.
- A fraction of the whole threshold: a system turn of two thirds of it left no
  exchange after the first eviction.
- Measuring the system turn from the system message alone: it missed
  declarations written only once a question follows (26 tokens counted where
  the conversation opened with 256 or more on one test template).
