# 0041. A result too long for the room left evicts, then is cut with a note, then closes the turn

Date: 2026-11-19

## Context

A tool result too long for the context was cut silently, so the model read
cut-off text as whole, and where not even an empty result fitted the call
stayed owed and every later action was refused.

## Decision

In this order:

1. The oldest exchanges are evicted first, as for a new question, the turn the
   result answers kept whole; on a cache that cannot shift, down to well below
   the room (0023).
2. A result still too long is cut to the longest start that fits, with
   `RESULT_CUT_NOTE` after it, which tells the model how many of how many
   characters it got. Its call is listed in `tool_results_cut` of the reply
   and the report. Room is kept for each result still owed after it, as long
   as the note a result cut to nothing carries; where a later result fits not
   even cut to nothing, the newest result written whole earlier in the round
   is cut to make room for it. A result whose note would be no shorter than
   itself stays whole. A result cut again keeps the total it was first cut
   from.
3. Where not even a result cut to nothing fits, the turn closes as the reply
   cap closes one: the reply kept as the model wrote it, without the calls
   that could not be answered, their results taken out, and `"token_limit"`.
   Calls whose tools ran are reported `"result_dropped"`. Closing is one
   action, so a failure puts everything back with the call still owed.
   `inject_tool_result` returns rather than raises.

## Consequences

- The model always knows when it read part of a result.
- The cut is searched outward from what the result's first tokens spell within
  the room: 200,000 characters cost 35 ms and 13 window renders.

## Alternatives rejected

- A silent cut: the model read 417 of 2,410 characters as if whole.
- Raising and leaving the call owed: every later action was refused until a
  new `begin`.
- Reserving room for an empty later result: a later result that does not fit
  whole is written with its note, so that room would not take it.
