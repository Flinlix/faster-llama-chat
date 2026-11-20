# 0028. A turn that called tools keeps the tokens it was generated as

Date: 2026-11-17

## Context

A template writes a past call in its own spelling and spacing, which is often
not how the model generated it. Writing the template's version in place of a
turn still running, or just finished, would change the prompt the model has
already read and decode the turn again on every round.

## Decision

- A turn that called tools during this conversation is kept as the tokens it
  was generated as, from the first tag after the content of the message before
  it to the end of its last call's closing or its final reply text. Both the
  live reply and the render ask one function for that start.
- The target (0021) is the template's render with each such turn's tokens in
  place of the template's version. The turn is held on the message it ends
  with, with how many messages it spans, and goes when its messages are
  dropped.
- A call's tags are the single tokens a generated call holds; its arguments
  are never read as tags.
- The template is given each calling message inside a kept turn without its
  text, which the turn's tokens already hold.
- A reply after a result that a stop string ended leaves the turn kept up to
  its call, so the cache holds the answer as the caller read it.
- A change of how the template writes the conversation that would leave a
  kept turn unplaceable refuses that action. A turn closed for want of room
  that cannot be placed, or would stand for none of the template's text, is
  held as the template writes it. A reply shortened to nothing after a tool
  turn cuts the turn back to the closing right before the template's text up
  to the end of the newest result.
- After a reply, a kept turn is checked against the render only where a reply
  can move it (0003).
- A plain reply is settled to the template's render live, so it needs no
  such keeping.

## Consequences

- The model reads its own turn exactly as it wrote it while the conversation
  runs; a `begin` on the same model restores the same tokens (0029).
- The cache check after every test places each kept turn among its own
  messages and fails where a kept turn spells a later tool result.

## Alternatives rejected

- Placing a kept turn by matching the tokens after it: fails where rounds
  repeat, as nine identical calls do.
- Tracking a kept turn by the message objects it starts and ends on: filtered
  by hand in four files, and three faults came from it.
- Cutting the kept tokens back to where a stop string began: a token may hold
  both the end of the answer and the start of the stop string.
- Searching backwards for the call's closing text when cutting: on a template
  whose call and result end with the same tag it stopped at the result's end
  and held the result twice.
