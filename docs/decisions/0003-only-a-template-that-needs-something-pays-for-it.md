# 0003. Only a template that needs something pays for it

Date: 2026-11-14

## Context

Many features serve only some templates: a clock, reasoning, tool calls,
declarations that move with the newest question, a template that compares
message texts. Checking for them on every reply made every model pay for
features it does not have.

## Decision

Whatever only some templates need is read once, at load or where a
conversation begins, or stated in the add-on file. No question, reply or token
of another template pays for it. Instances:

- A question that repeats an earlier message is rendered whole only where one
  probe pair at the conversation's start shows that the template writes a
  repeated question, reply or system prompt otherwise (0025).
- The clock's renders run only where the template's text names `strftime_now`
  (0007).
- The opening's length for a pinned state, and states before a reply, are
  worked out only on a cache that goes back to saved states (0024).
- A question is measured inside the system turn's render only where its own
  token count comes within 64 tokens of the room; otherwise it is counted
  beside the system turn measured once at `begin`. An injected message is
  counted the same way.
- After a reply, a turn kept as generated is checked against the render only
  where a reply can move it: where the reply adds to such a turn, or where a
  probe at `begin` shows that the template writes a past tool turn otherwise
  once another exchange follows.
- Whether the template writes a result is read from the newest exchange, not
  from two renders of the whole conversation.
- Detecting a call after text, and following whether the generation is inside
  a thought for it, runs only for templates that write a call there (0037).
- Load checks that read the same thing are worked out once per set of
  template variables.

`tests/test_wrapper.py` traces a whole exchange on a template without
reasoning, tools or a clock with the interpreter's profile hook, and lists the
functions such a reply may run, so that code new to it is flagged.

## Consequences

- Measured on the test stand-in with 20 exchanges: the first word after a
  repeated "Thanks!" went from 1.15 to 1.21 ms to 0.38 ms, the same as a new
  question. The past tool turn check took 1.1 to 1.8 s after every reply on
  Gemma 4 at 31,000 tokens before it was limited. Measuring a question with 50
  tools cost 1 to 8 ms per question. Renders at load over the test templates
  went from 2,448 to 1,964.
- A template that reads a question's text against earlier ones in a way the
  probe does not cover (by length, say) would still pass unseen.

## Alternatives rejected

- Reading the names a template calls through the template parser to decide
  whether it reads the clock: a compile per load, which tripled the test time.
  A text search suffices, since a template that does not name the function
  cannot call it.
- A trace test listing the code that must not run: replaced by the list of
  code that may run, which flags new code without anyone adding it.
