# 0032. Every question gets a reply: nothing a conversation's state holds blocks a question

Date: 2026-11-19

## Context

The owner's rule: "None should block: the model should always be able to
respond to a question." A question was refused while a turn a call left open
waited, after a question no reply followed on templates that write no two in a
row, and where the eviction threshold lay above the room a reply needs.
Probing found more: a `begin` that left no room for any question, raw
Python errors, replies that ended empty.

## Decision

- A question on an open turn closes the turn first: each call still owed gets
  a result saying it was not run, and the question follows where the template
  writes it there; where the template needs a reply between, an empty one is
  added; where it writes neither, the unfinished turn is taken out.
- A question after an unanswered one is joined to it, separated by a blank
  line, where the template writes no two in a row. Where the two joined do not
  fit, the unanswered one goes, counted as evicted, and the new one is asked
  alone. Joining needs a template that writes no two in a row.
- A question too long for the room the system turn leaves under the eviction
  threshold keeps as much of its end as fits, possibly none of it, with
  `QUESTION_CUT_MARK` in place of its beginning and a warning logged. The cut
  is found by a binary search on the start.
- A question is refused only where the template does not write it as its text;
  the refusal leaves the next question free.
- `begin` refuses a system prompt and declarations that leave no room for a
  short question (the cut mark and `?`) and `min_reply_tokens` under the
  threshold, on the longest date (0007).
- A reply evicts down to the room it needs, whatever the threshold. It always
  keeps room for the shortest reply that holds an answer: what the model
  writes before its answer, one token of answer, and the closing.
- `min_answer_tokens` (default 1): until the answer holds text or a call, a
  drawn token that ends the turn is drawn again from a second sampler chain
  that withholds every ending token. `answer_room_tokens` (default 1024): the
  thought's closing is fed in once the generation reaches its room less the
  answer's share and the closing's length.
- Where the room left is smaller than the shortest call to a declared tool,
  the call's opening token is withheld for that generation. Where a call opens
  with no token of its own, a reply the room cuts at such an opening ends
  inside the call, reported `"unfinished"`.
- A reply nobody reads holds an action at most `unread_reply_timeout_s`
  (default 10 s), then it is ended as closing its stream would. A registry tool
  that has not returned within `tool_timeout_s` (default 60 s) is answered as
  `tool_timeout`. `max_calls` (default 10) applies where a request sets none.
- A model whose template marks calls but writes no result block reads no call
  under the default policy, since none could be answered.
- A `begin` after `reply()` and before the reply is read makes reading it an
  `OutOfOrderError`.

## Consequences

- No state is reachable in which the model is ready but cannot answer. The
  failure sweep and the random sequences follow every state they reach with a
  question.
- The answer guard's second chain sees only its own draws, so the two chains'
  repetition penalties differ by those few tokens; the owner chose to keep it.
- What stays refused: a closed wrapper, a conversation never begun, and
  declaring tools or injecting an assistant message on an open turn.

## Alternatives rejected

- Refusing a question for the conversation's state.
- Refusing a too-long question: the question likely stands at its end.
- Cutting a question to the whole context: one question could leave the reply
  only `min_reply_tokens`.
- Refusing at load a model whose own system prompt is too large: it refused
  models that work with a caller's system prompt.
- Refusing system prompts sized for the costliest character, or over a fixed
  fraction of the context: both refuse setups that work.
- Withholding ending tokens in the first chain throughout: an engine call per
  token. Counting the thought's own tokens to close it: it missed openings the
  model writes as letters.
