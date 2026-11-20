# 0001. The cache holds the template's render of the conversation, decoded only where it differs

Date: 2026-11-15

## Context

A KV-cache chat wrapper must hand the model the prompt the model was trained
on. The first version built each message from pieces recovered from the
template with sentinel probes (a prefix and a suffix per role), kept a table of
the token slice each message held, and assembled the cache from those pieces.
Every template feature the pieces did not cover needed code of its own: a
system prompt folded into the first user turn, declarations that move with the
newest question, a past reply written without its thought, messages the
template numbers. A missed feature made the cache differ from the render
without any error.

The owner asked whether a model could work every time, a future one included.
A prototype that rendered the whole conversation and kept the cache equal to
that render was built beside the old wrapper and then made the library.

## Decision

- The conversation is the list of messages a chat template is given. The cache
  is the template's render of them, tokenized as the model reads it.
- Each action renders its target, compares it with the cache token by token,
  and decodes only from the first token that differs. A stretch the target no
  longer holds is removed and the rest shifted where the cache allows it
  (0021, 0022).
- Pieces are still read from the template at load, but only for what a
  generation needs: the thought tags, the end of a reply, the call format, and
  what the template does to message text (0006).
- A model whose GGUF ships no chat template is refused.
- Where an action interrupts the work after a reply, the cache may hold less
  than the render until that action decodes the rest, before the model reads
  it (0027).
- The limits count the render without the stand-in question used for decoding
  ahead (0025).

## Consequences

- Encoding as the template renders holds by construction for every model with
  a template, without code particular to one model.
- Every action renders the template, so the cost of a render must not grow
  with the conversation; windows (0026) and decoding ahead (0025) keep it
  bounded.
- A turn that called tools is the one exception to the render: it keeps the
  tokens it was generated as (0028).
- Measured with the prototype on 8 real models (Gemma 3 1B, Llama 3.2 3B,
  Qwen 3.5 9B, Ministral 3 8B, MiniCPM 5 2B, Gemma 4 26B-A4B, gpt-oss 20B, an
  R1 distill 8B; Q4 or MXFP4 on an RTX 3090): the cache equalled the
  tokenized render before each of 12 replies, at 1,024 and at 512 tokens of
  context. On Gemma 4 26B-A4B at 8k, every turn evicting, the wait before the
  first word was 199 ms against 194 ms for the piece-built wrapper.

## Alternatives rejected

- Building messages from template pieces: every template shape needed its own
  code, and a missed shape broke the encoding silently.
- Rendering the whole conversation every turn, without windows: on an
  i9-9900K with Gemma 4's template at an 8k context, one render took 234 ms
  for 394 short messages, 51 ms for 178 medium ones and 9 ms for 64 long ones
  (median of 15), and twice the messages took four to five times as long. On
  a Raspberry Pi 5 that was estimated at 0.5 to 0.8 s more per turn for short
  exchanges. Windows of constant size replaced it (0026).
