# 0040. Every call the model makes is answered and reported, an unreadable one included

Date: 2026-11-19

## Context

A call left without a result leaves a history no template produces, which the
model reads back on every later turn. A call missing from the reply's report
leaves the caller unable to tell what the model asked for and what ran.

## Decision

- Every call is answered. An unreadable call is answered at once by a result
  under the name `unknown` saying it failed and could not be read, so nothing
  is owed and the turn goes on. It spends a call of the budget like any failed
  call, and one reply reads at most `MAX_UNREADABLE_CALLS_IN_A_ROW` (3) such
  calls in a row under every policy; the generation after the third cannot open
  a call. Under `"yield"` the reply generates again after such a call, within
  one `max_reply_tokens`.
- The answer to a failed call names the failure and ends in the one
  instruction not already implied: `try again` where a call the model writes
  next is read, else `do not call again`. The choice is made after the whole
  round, since its last call settles the budget. A tool's error result is told
  the same way whether the library ran the tool or the caller handed it over.
- Every call is reported on `Reply.tool_calls` with a `status`: `"complete"`,
  `"unreadable"` (with the `text` the model wrote between the call's opening
  and closing, and no name, arguments or id), `"unfinished"` (cut short, or
  not run because it spent the reply cap), and `"result_dropped"` (its tool
  ran but the turn closed for want of room).
- A call a registry runs is handed over in the stream just before it runs. A
  stream closed at a call, or a tool stopped by anything raised, writes the
  results of the calls of the round that ran before it and leaves the rest
  owed.
- A result JSON cannot write (`NaN`, infinity, a lone surrogate, a value whose
  own methods raise) reaches the model as an `unwritable_result` failure,
  wherever it comes from; `inject_tool_result` logs a warning for it.
- A turn ended inside a call closed by a tag has that tag decoded and the call
  read, as one closed by text is.
- Answers not yet written when the engine fails are kept and written by the
  next action within its own change.

## Consequences

- A model that cannot write a readable call gives up after a few tries and
  answers in words.
- Measured on the stand-in before the limit of three: with every call
  unreadable and no cap, 59 tries and 1,534 tokens at a context of 8,192.

## Alternatives rejected

- Writing nothing for an unreadable call: a history no template produces.
- Handing an unreadable call over as an event, or closing the turn as
  `"refuse"` does under `"yield"`: the first leaves every caller a case to
  handle, the second takes from the model the chance to correct its call.
- Saying `try again` in every case: it asks for a call that cannot be read.
