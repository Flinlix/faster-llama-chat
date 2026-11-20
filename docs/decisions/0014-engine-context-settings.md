# 0014. The engine context follows llama.cpp's own placement, uses the unified cache, and frees itself

Date: 2026-11-19

## Context

The bundled cache context creates the llama.cpp context. Several of its
defaults departed from llama.cpp's without a reason, and one setting
(`gpu_layers=0`) did not mean what its name says.

## Decision

- The context is created with `kv_unified`: attention runs over the cache's
  rows, so a layer with a sliding window reads only its window, while the
  cache keeps its full size and can still shift.
- `kv_cache_type` defaults to `"f16"` and is always passed; `None` is not
  accepted.
- `kv_cache_on_gpu` defaults to `True`, llama.cpp's `offload_kqv`: each
  layer's cache sits on the device that layer runs on. `False` forces all of
  it into host RAM to free VRAM.
- `flash_attention` defaults to `None`, llama.cpp's automatic mode, except
  under a quantized `kv_cache_type`, which works only with flash attention and
  forces it on; `False` with a quantized cache is refused. On a GPU, flash
  attention with a `q8_0` cache roughly halves the cache's memory at a small
  cost in quality.
- `gpu_layers` defaults to `-1`, all layers, as llama.cpp does; a model larger
  than the card fails at load with the engine's cause. `gpu_layers=0` loads
  with an empty device list, so llama.cpp uses no GPU at all.
- The context registers a finalizer when it loads, which frees the engine
  context and then the model when the context is collected, and which
  `close()` runs at once; the engine is freed exactly once. A construction
  that fails after the load frees what it loaded; a context of one's own is
  left open, since the caller holds it.

## Consequences

- Measured before the unified cache, engine alone at 6,000 tokens deep: 103.4
  tokens/s on the GPU without it, 118.6 with it (the windowed cache gave
  117.3); 16.3 and 17.9 on the processor. Memory use is the same.
- With `gpu_layers=0` before the change, a visible card still took every batch
  of 32 tokens or more (prompt reading 594 tokens/s on the processor alone,
  2,100 to 3,600 with the card), and the first long prompt stalled about
  0.6 s.
- A dropped wrapper of a 1B model on the processor kept about 0.9 GB until the
  program ended before the finalizer.
- Not built: multi-token prediction, measured with llama.cpp's speculative
  example, greedy: Gemma 4 26B on an RTX 3090 reached 205 to 209 tokens/s with
  2 to 4 drafted tokens (60 to 71 percent accepted) against 146; Gemma 4 E2B
  on an i9-9900K 38.2 tokens/s against 25.4.

## Alternatives rejected

- The windowed cache: it cannot shift, so every drop would decode the history
  again.
- Forcing flash attention on: the automatic mode turns it off where the
  device or model does not support it.
- `__del__` instead of a finalizer: a finalizer holds no reference to the
  context and runs at most once, also at interpreter exit.
