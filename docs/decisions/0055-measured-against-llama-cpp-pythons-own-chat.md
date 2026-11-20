# 0055. Measured against llama-cpp-python's own chat interface

Date: 2026-11-20

## Context

The README's reason for the library, that a kept cache saves prefill and
lets a chat run on, rested on the design alone: no record held a measurement
against the plain way of chatting with the same engine.

## Decision

`benchmarks/compare.py` is run on both sides, each in its own process, one
after the other, and the results are kept here. The plain side is
llama-cpp-python's `Llama.create_chat_completion` with the whole history
given each turn, which reuses the start the request shares with the one
before, flash attention on and every layer on the GPU.

Machine and settings: one Nvidia RTX 3090 (24 GB), Gemma 4 26B A4B instruct,
`UD-Q4_K_M`, llama-cpp-python 0.3.36, a context of 16384 tokens, sampling
`temperature 1.0, top_k 64, top_p 0.95, seed 1234` on both sides, the
library's own defaults otherwise.

### Ten short questions on a history

Each run begins on a history filling part of the context, then asks ten
questions of up to 160 tokens of answer. The library decodes the history at
`begin`; the plain side decodes it with the first question.

| History | Side | Begin | Time to first text, first question | Later questions (median) | Tokens per second (median) |
|---|---|---|---|---|---|
| none | library | 21 ms | 35 ms | 29 ms | 133 |
| none | plain | | 630 ms | 51 ms | 130 |
| 40 % (6.9k tokens) | library | 1.30 s | 41 ms | 31 ms | 128 |
| 40 % (6.9k tokens) | plain | | 1.43 s | 68 ms | 104 |
| 60 % | library | 0.73 s | 49 ms | 31 ms | 127 |
| 60 % | plain | | 0.84 s | 74 ms | 96 |

At 60 % the library's `begin` kept 8.3k of the 10.0k tokens, about half the
context, as a begin keeps (0030); the plain side held all of them.

### One long conversation

Each turn gives a passage of about 600 tokens to summarize.

| Side | Turns | Tokens given and generated | Most held | Time to first text (median, last turn) | Tokens per second (first five, last five turns) |
|---|---|---|---|---|---|
| plain | 21, then its context was full | 15.5k | 15.5k | 206 ms, 282 ms | 124, 88 |
| library | 214, stopped at the run's limit | 150k | 12.3k | 190 ms, 146 ms | 130, 123 |

The library evicted the oldest exchanges in 187 of the 214 turns, each time
shifting the rest. No turn decoded the history again: no time to first text
was above 210 ms, where decoding the 12k tokens held would take over a second
(6.9k took 1.3 s in the run above).

## Consequences

- A question after the first waits about half as long as on the plain side,
  since only the question is decoded; the first question after a `begin`
  waits for nothing the begin already decoded.
- The plain side's generation slows as its history grows (130 to 96 tokens
  per second at 60 %); the library's held cache stays near its eviction
  threshold, so its speed stays near its start.
- The plain side ends when its context is full; the library's chat runs on.
- Not measured: the same runs on the processor alone, and on other models.

## Alternatives rejected

- Measuring both sides in one process: the second would find the GPU and the
  file cache warmed by the first.
