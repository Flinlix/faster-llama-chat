# 0016. Loading warms the engine: one micro-batch where layers run on a GPU, then two tokens and one

Date: 2026-11-19

## Context

The engine's first decode of a given shape prepares it, a one-time cost. Left
to the first request, it lands on the first `begin` with a real system prompt
or on the first word. The owner's server starts the model with an empty
`begin`, too short to trigger the large-batch preparation, so that cost fell
on the first `begin` with a persona.

## Decision

- At the end of loading, the engine context decodes, where the model has
  layers on a GPU (`gpu_layers` is not 0 and the build can offload), one
  micro-batch (`micro_batch_tokens`, else llama.cpp's 512, at most
  `context_size`), then a batch of two tokens and a single token, as
  llama.cpp's own tools do, and clears the cache after each.
- On the processor alone it decodes only the two tokens and the single one.
- Warming is the engine context's own business; the cache-context protocol
  asks for no warm-up member.

## Consequences

- Measured on an RTX 3090 with Gemma 4 26B A4B Q4_K_M, all layers on the card,
  batch 512, flash attention, q8_0 cache: the full batch costs 0.68 s of the
  load and takes the first `begin` from 0.59 s to 0.017 s; the first word and
  one decode step are the same either way. In the server's order of steps the
  first `begin` with a persona took 0.61 to 0.93 s without it and 0.04 s with
  it.
- Measured on an i9-9900K (6 threads, context 4096, batch 2048): a full batch
  took 8.3 s of gemma-3-1b's load and 20.4 s of Llama-3.2-3B's, and saved
  nothing, the first `begin` and the first word being the same with the small
  warm-up and without any. The small warm-up costs 0.03 to 0.15 s.
- Not measured: a model with only part of its layers on the GPU.

## Alternatives rejected

- A full batch on the processor: 8 to 20 s of every load at the defaults, an
  estimated 45 s on a Raspberry Pi 5, for no gain.
- No warm-up at all: as fast on the processor, but the card's first word was
  9 ms slower.
- A full `max_batch_tokens` batch on the GPU: the engine computes a batch one
  micro-batch at a time and the first prepares it; one micro-batch saves about
  0.26 s per load (estimated).
- A warm-up sized by a setting, which no caller asked for, and the same
  warm-up in the server's start alone, which would help one caller.
- A `warmup` member in the protocol called at the first `begin`: only one
  context needs it.
