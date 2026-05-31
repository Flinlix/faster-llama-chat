# llama_chat — a KV-cache scalpel for llama.cpp

A thin Python wrapper around llama.cpp whose one job is to **prefill as little as
possible**. Prefill (processing the prompt) is the expensive part of inference;
the KV cache holding that work can be reused across turns. This wrapper keeps the
cache alive, prefills only genuinely new tokens, and when the cache fills it
*removes the oldest messages and shifts the survivors down to close the gap* —
reusing their KV instead of re-prefilling. Like a scalpel.

## Three actions

```python
from llama_chat import ChatWrapper, Config

chat = ChatWrapper(Config(model_path="model.gguf", n_ctx=4096, threshold_pct=0.8))

# begin: reset + prefill the system prompt and as much recent history as fits
chat.begin("You are a helpful assistant.",
           [("user", "Hi"), ("assistant", "Hello!")])

# inject: prefill one message as context, WITHOUT generating
chat.inject("Reference: the deadline is Friday.")

# request: prefill only the new request text, then generate
turn = chat.request("What's the deadline?")
print(turn.text)

# stream: same, but yield text deltas as they are generated (low latency).
# Ideal for a voice pipeline — synthesize sentence n while n+1 is still writing.
for delta in chat.stream("Summarize the plan."):
    print(delta, end="", flush=True)
```

- **`begin`** — full reset for a restart / conversation switch. Prefills the
  system prompt plus the most recent history that fits under the threshold;
  older excess is dropped.
- **`inject`** — adds context (e.g. retrieved documents) with no generation.
  Evicts the oldest messages first if needed to stay under the threshold.
- **`request`** — prefills just the request text and generates, evicting oldest
  messages if over threshold and capping generation so the total never exceeds
  `n_ctx`. The reply is recorded so the next turn reuses it for free.
- **`stream`** — the generator form of `request`: yields visible text deltas as
  tokens are sampled and returns the same `Turn` summary on completion. Bytes are
  decoded incrementally, so a UTF-8 codepoint split across tokens is never
  mangled. Abandoning the stream early (barge-in) still records exactly the
  tokens that reached the cache, so the next turn stays valid. Cache-mutating
  actions are serialized with a lock, safe to drive from a threaded server.

## How eviction works (the scalpel)

The cache for sequence 0 is a contiguous run `[0, total)`; each message owns a
slice. To evict the oldest non-system message at `[a, b)`:

```
llama_memory_seq_rm(seq, a, b)               # drop those tokens
llama_memory_seq_add(seq, b, total, -(b-a))  # shift survivors down to close the gap
```

`seq_add` also fixes the RoPE positions, so the survivors stay coherent without
recomputation. This reuse is *lossy* (the survivors' cached K/V were computed
while attending to the now-removed tokens), which is the standard StreamingLLM
trade-off — fast, with slight quality drift. The system prompt is never evicted.
(`_backend.py` resolves these symbols defensively: the modern `llama_memory_*`
handle API when present, falling back to the older `llama_kv_self_*` /
`llama_kv_cache_*` names across llama.cpp versions.)

Caches that can't shift in place (compact sliding-window, recurrent/Mamba) are
detected at load via `llama_memory_can_shift()`; for those the wrapper drops the
oldest messages and **rebuilds** the cache from the survivors in one prefill —
correct, just not free. The fast in-place path above is used everywhere it can
be (including Gemma 4).

## Install (same package, per-machine build)

```bash
# Raspberry Pi 5 (CPU) — builds llama-cpp-python from source for the ARM
# dot-product kernels, then installs llama_chat + test deps and verifies:
./install.sh

# Equivalent manual build:
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python llama-cpp-python
pip install -e ".[test]"

# i9 + RTX 3090 (CUDA offload)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -e ".[test]"
```

Only `Config` differs between targets: `model_path`, `n_ctx`, `threshold_pct`,
and `n_gpu_layers` (`0` on the Pi, `-1` to offload everything on the 3090). The
wrapper code is identical — `seq_rm`/`seq_add` work on both backends.

## Chat template

The template must match the model, because its turn-terminator is what lets the
model stop generating; a mismatched template tokenizes as plain text, so the
model never emits its end-of-turn token and runs to the generation cap.

By default the template is **auto-detected** from the model's built-in chat
template — Gemma 4 (`<|turn>`), Gemma 2/3 (`<start_of_turn>`) and ChatML
(`<|im_start|>`) are recognised. The chosen terminator is validated against the
model's vocabulary at load time, so a mismatch fails loudly instead of degrading
silently. To override, pass a preset or a custom `TemplateConfig`:

```python
from llama_chat import Config, TemplateConfig
Config(model_path="model.gguf", template=TemplateConfig.preset("gemma4"))
```

## Verify

```bash
# Pure bookkeeping + end-to-end logic, no model needed:
python -m pytest -q

# Against a real model (logs prefill reuse + eviction stress):
python examples/demo.py /path/to/model.gguf            # CPU
python examples/demo.py /path/to/model.gguf --gpu-layers -1
```

The test suite uses a model-free fake backend whose `prefill` asserts
`start_pos == len(cache)` — i.e. the wrapper can only ever *append*, never
re-prefill a survivor. That assertion is the design guarantee, checked
mechanically.

## Layout

| File | Role |
|------|------|
| `llama_chat/config.py` | `Config` / `TemplateConfig` — the only per-machine knobs |
| `llama_chat/messages.py` | `MessageTable` — position bookkeeping, evict + shift (pure) |
| `llama_chat/template.py` | per-message chat-template fragments |
| `llama_chat/_backend.py` | version-stable adapter over `llama_cpp.*` (handles API churn) |
| `llama_chat/context.py` | `KVContext` — tokenize / prefill / streaming generate / evict |
| `llama_chat/wrapper.py` | `ChatWrapper` — `begin` / `inject` / `request` / `stream` |
