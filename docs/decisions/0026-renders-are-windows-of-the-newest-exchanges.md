# 0026. Actions render windows of the newest exchanges after an anchor, where the template writes them alike

Date: 2026-11-20

## Context

Rendering the whole conversation on every action costs time that grows with
the square of the messages (0001). A render of the system message and the
newest exchanges alone is identical to the whole render from a known tag on,
for most templates, so the rest can be taken from the cache.

## Decision

- The state keeps anchors: where in the cache the tag that opens the exchange
  after a message stands, recorded with that message, its place among the
  messages, the position, and the generated turns up to it. At most the two
  newest are kept, and the newest exchange's anchor after an eviction.
- Any target, length, placement of a generated turn or refusal is worked out
  from the cache's tokens up to the newest anchor that still stands and a
  render of the system message and the window after it: the anchor's own
  exchange and at most three more.
- A window is used only where:
  - the load's probes showed that the template writes the newest exchanges
    alike in a render of them alone (`windows_are_exact`, checked on five
    probe exchanges with the conversation's system message, tools and
    reasoning), and with exchanges that called tools too where the window holds
    one (worked out only for a template that can call tools);
  - a question follows the anchor's reply, and the reply closed its turn;
  - no question after the anchor repeats a text before it, on a template that
    writes a repeated question otherwise, neither now nor when the cache up to
    the anchor was written. A template that writes its declarations before
    every question equal to the newest writes an earlier question otherwise
    while a later one repeats it, and again otherwise once the later one is
    joined by the next question; the anchor records whether such a repeat
    stood when it was set (found by a random sequence on 2026-11-20).
  Where the tag the window opens with is not the one the cache holds there,
  the older anchor is tried, else the whole conversation is rendered.
- Templates whose declarations move with the newest question use windows
  too: a window opens before the stretch the old declarations leave, and the
  cache search removes that stretch.
- On a cache that shifts, an eviction drops in windows of the oldest
  exchanges, only where the dropped exchange ends on a reply that closes its
  turn, and moves the anchors along.

## Consequences

- Measured on the Gemma 4 test template at 240 messages: the call from 395 to
  4 ms, the result from 200 to 3 ms, the reply after it from 800 to 7.5 ms, the
  next plain reply from 413 to 6 ms, a begin with the same conversation from
  630 to 6 ms. Mistral Small 3.2 and Ministral 2410 with tools went from 4
  whole renders per reply and 14 per tool turn to none, the tokens decoded
  unchanged.
- A fresh begin still renders the whole conversation about four times, since
  every token of it is decoded anyway.
- Two or more unanswered questions joined in one turn are decoded again with
  each further one, since the text between two tags is tokenized as one run.

## Alternatives rejected

- Rendering the whole conversation each turn (0001).
- Switching windows off for templates whose declarations move: the window
  check no longer fails there.
- Trusting the cache before an anchor whatever followed its reply: a second
  reply or a client's own call can make the template write the reply's closing
  otherwise.
- Treating the join of two questions as a token boundary: a guess on a merging
  tokenizer.
