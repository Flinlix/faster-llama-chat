# 0015. The engine keeps its own micro-batch unless the caller sets one

Date: 2026-11-17

## Context

`max_batch_tokens` sets the batch, the most tokens handed to the engine in one
decode call. The micro-batch is the most tokens the engine computes at once,
the size its compute buffer is reserved for. It was set to the batch.

## Decision

- `micro_batch_tokens` defaults to `None`, which leaves llama.cpp's own
  micro-batch (512). A caller who wants a long prefill read faster sets it, at
  most `max_batch_tokens`, and pays the device memory.
- The engine's out-of-memory advice names `micro_batch_tokens`.

## Consequences

Measured on an i9-9900K (8 threads, context 4096, batch 2048, a history of
about 1,040 tokens, interleaved pairs on a shared machine): the speed was the
same within the noise, and the compute buffer a quarter of the size.

| Model | Micro-batch | Compute buffer | Prefill, tokens/s |
|---|---|---|---|
| gemma-3-1b Q4_K_M | 2048 | 2,125 MiB | 301.9, 327.4, 217.5 |
| gemma-3-1b Q4_K_M | 512 | 531 MiB | 328.0, 322.0, 213.3 |
| Llama-3.2-3B Q4_K_M | 2048 | 1,162 MiB | 139.5, 136.1, 103.1 |
| Llama-3.2-3B Q4_K_M | 512 | 291 MiB | 138.9, 114.1, 104.0 |

Measured with gemma-4-26B-A4B Q4_K_M on an RTX 3090 (context 8192, all layers
on the card, q8_0 cache, flash attention), a 5,221-token question and a begin
of 240 messages, three runs each:

| Micro-batch | Question | First token | Begin | Card memory, peak |
|---|---|---|---|---|
| 512 | 4,583 to 4,693 tok/s | 1.11 to 1.14 s | 1.89 to 1.91 s | 18,274 MiB |
| 1024 | 5,180 to 5,547 tok/s | 0.94 to 1.01 s | 1.72 to 1.73 s | 18,838 MiB |
| 2048 | 5,298 to 6,051 tok/s | 0.86 to 0.99 s | 1.60 s | 19,966 MiB |

On the card a larger micro-batch reads a long question 13 to 15 percent
faster and takes 564 or 1,692 MiB more.

## Alternatives rejected

- The micro-batch equal to the batch always: 0.9 to 1.6 GB more RAM on the
  processor at the default batch for no gain there, and the card's memory for
  every caller.
- No setting at all: rejected once the owner asked for one after the card was
  measured.
