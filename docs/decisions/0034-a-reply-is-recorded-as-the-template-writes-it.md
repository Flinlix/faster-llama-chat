# 0034. A reply is recorded whole or not at all, as the template writes it

Date: 2026-11-15

## Context

Some templates cannot write what a model produced: an answer holding the
template's own thought tags (a Qwen 3 template writes only what follows
`</think>`), an empty reply on templates that refuse one, a text the template
refuses. Recording such a reply and only then failing to render left the
conversation refusing every later question until the next `begin`.

## Decision

- Recording a reply is one step, published once at its end, so a reader never
  sees a call that is withdrawn in the same step.
- An answer holding the template's thought opening or closing is kept as the
  answer itself where the template writes it as its text, given with the
  reply's thought; otherwise as the part of it the template does write as its
  text, trimmed at most; otherwise as an answer with no text.
- A reply that produced nothing is kept where the template writes an empty
  reply; taken out alone where the template writes two questions in a row, so
  its question waits for the next; otherwise taken back with its question and
  everything after it, a tool turn's calls and results included.
- A reply whose text the template refuses is taken back with its question,
  read from a render of the system message and the newest exchange.
- The caller's stream and `Reply.text` keep what the model wrote; the
  conversation holds what the model will read, and the `Reply` reports the
  answer as kept where it differs.
- `truncate_last_reply` is refused where the template writes the shortened
  conversation neither as it is nor with a question after it, naming that more
  text must be kept. The reply that ends a tool turn is cut at the last whole
  token before the kept text's end, by the tokens' bytes.
- Replies in a row whose calls could not be read are joined.

## Consequences

- Only an answer holding those tags pays the extra renders.
- A tool that ran in a reply taken back has run all the same; its result
  leaves the conversation with the question.

## Alternatives rejected

- Taking every such reply back with its question: it loses an answer the user
  saw in full, and a first version did so on 13 of 31 reasoning templates.
- Storing the whole answer: the next render could not place it and raised
  after the reply had streamed.
