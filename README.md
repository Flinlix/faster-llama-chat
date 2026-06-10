# llama_chat: a KV-cache manager for llama.cpp

A thin Python wrapper around llama.cpp whose goal is to **prefill as little as
possible**. Prefill (processing the prompt) can be an expensive part of inference;
the KV cache holding that work can be reused across turns. This wrapper keeps the
cache alive, prefills only genuinely new tokens, and when the cache fills it
*removes the oldest messages and shifts the survivors down to close the gap* -
reusing their KV instead of re-prefilling.

## Goals


1. **Minimum prefill:** only ever process genuinely *new* tokens. The KV cache
   from previous turns must survive and be reused.
2. **Rolling eviction:** when the context fills, old messages are dropped
   automatically so the chat runs indefinitely without erroring on a full
   context.

Naive prompt trimming defeats goal 1: removing old messages from the start of
the sequence shifts every later token's position and **invalidates the entire
cached prefix → full reprefill every turn.** llama-server's `cache_prompt` has
the same problem - the moment any non-prefix message is dropped, the common
prefix collapses to the system prompt. The correct mechanism is **in-place KV
surgery**: remove the oldest message's token span and *shift the survivors' RoPE
positions down* to close the gap, so kept messages are never re-processed.

## Quick start

```python
from llama_chat import ChatWrapper, Config

# Config defaults to Gemma 4. Change it to use other models.
chat = ChatWrapper(Config(model_path="llama/models/gemma-4-E2B-it-Q4_K_M.gguf"))

# begin: reset + prefill the system prompt and as much recent history as fits
chat.begin("You are a helpful assistant.",
           [("user", "Hi"), ("assistant", "Hello!")])

# inject: prefill one message as context, WITHOUT generating
chat.inject("Reference: the deadline is Friday.")

# request: prefill only the new request text, then generate
turn = chat.request("What's the deadline?")
print(turn.text)

# stream: same, but yield text deltas as they are generated (low latency).
for delta in chat.stream("Summarize the plan."):
    print(delta, end="", flush=True)
```

### The four actions

- **`begin(system_prompt, history)`** - full reset for a restart / conversation
  switch. Prefills the system prompt (pinned at position 0, never evicted) plus
  the most recent history that fits under the threshold in a single decode;
  older excess is dropped.
- **`inject(text, role="user")`** - adds context (e.g. retrieved documents) with
  no generation. Evicts the oldest messages first if needed to stay under the
  threshold. Returns the number of messages evicted.
- **`request(text)`** - prefills just the request text and generates, evicting
  oldest messages if over threshold and capping generation so the total never
  exceeds `n_ctx`. The reply is recorded so the next turn reuses it for free.
  Returns a `Turn` (`text`, `n_prefilled`, `n_generated`, `n_evicted`,
  `stop_reason`). If `min_answer_tokens` is set and the prompt would leave less
  than that for the reply, it raises `ContextOverflowError` before touching the
  cache.
- **`stream(text)`** - the generator form of `request`: yields text
  deltas as tokens are sampled and returns the same `Turn` summary on completion
  (via `StopIteration.value`). Raises `ContextOverflowError` on the same
  `min_answer_tokens` headroom check as `request`. Bytes are decoded
  incrementally, so a UTF-8 codepoint split across tokens is never mangled. Each
  token is committed to the cache *before* its text is surfaced, so abandoning
  the stream early (barge-in via `gen.close()`) still records exactly the tokens
  that reached the cache, and the next turn stays valid.

Cache-mutating actions are serialized with a reentrant lock, so the wrapper is
safe to drive from a threaded HTTP server.

## How eviction works

A pure-Python `MessageTable` is the single source of truth: the cache for
sequence 0 is a contiguous run `[0, total)`, each message owns a `[start, stop)`
slice, and the table recomputes every position after any structural change. To
evict the oldest non-system message at `[a, b)`:

```
llama_memory_seq_rm(seq, a, b)               # drop those tokens
llama_memory_seq_add(seq, b, total, -(b-a))  # shift survivors down to close the gap
```

`seq_add` also fixes the RoPE positions, so the survivors stay coherent without
recomputation. This reuse is *lossy* (the survivors' cached K/V were computed
while attending to the now-removed tokens), which is the standard StreamingLLM
trade-off - fast, with slight quality drift. The system prompt is never evicted.

The strategy is chosen automatically at construction from
`llama_memory_can_shift()`:

- **shift** (default, the fast path) - the in-place edit above, applied
  incrementally as messages are dropped. Used everywhere it can be, including
  Gemma 4 under the default full-size SWA cache.
- **rebuild** (caches that can't shift - compact sliding-window, recurrent /
  Mamba state, where dropping tokens loses the position info shifting needs) -
  drop the oldest messages from the bookkeeping, then re-prefill the surviving
  conversation once. Correct, just not free; a non-shiftable model degrades
  gracefully instead of corrupting the cache.

The backend (`_backend.py`) resolves the cache symbols defensively across
llama.cpp API churn: the modern `llama_memory_*` handle API when present,
falling back to the older `llama_kv_self_*` / `llama_kv_cache_*` names.

## Install

llama-cpp-python is built **from source per machine** so it uses that CPU's own
kernels (Armv8.2 dot-product on the Pi, AVX2 on x86) or the CUDA backend - the
prebuilt wheel targets a baseline ISA and ships no CUDA.

```bash
# Auto-detect: CUDA if nvidia-smi is present, else a native-CPU build.
# Builds llama-cpp-python from source, installs llama_chat (editable), and
# verifies the llama.cpp symbols the eviction core relies on.
./install.sh

./install.sh --cpu      # force the native-CPU build
./install.sh --cuda     # force the Nvidia GPU build
./install.sh --dev      # add test dependencies (append to any of the above)
```

Equivalent manual build:

```bash
# Raspberry Pi 5 / CPU
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python "llama-cpp-python>=0.3.0"

# Nvidia GPU (CUDA offload)
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.3.0"

pip install -e ".[test]"
```

Only `Config` differs between targets - `model_path`, `n_ctx`, `threshold_pct`,
and `n_gpu_layers` (`0` on the Pi, `-1` to offload everything on a GPU). The
wrapper code is identical; `seq_rm` / `seq_add` work on both backends. On a GPU,
`flash_attn=True` with `kv_cache_type="q8_0"` roughly halves KV-cache memory at
near-zero quality cost (a quantized `kv_cache_type` requires flash attention).

## Chat template

The template must match the model, because its turn-terminator is what lets the
model stop generating. A mismatched template tokenizes as plain text, so the
model never emits its end-of-turn token and runs to the generation cap every
turn.

The template is expressed as per-role **fragments** on `Config`: the literal text
wrapped around each message, so one turn renders as `prefix + text + suffix`.
Content is stripped first when `trim_content` is set (the default), matching
templates that apply Jinja `| trim` such as Gemma 4; set it `False` for templates
that emit content verbatim. The defaults are **Gemma 4**
(`<|turn>{role}\n … <turn|>\n`). To target another model, set the
fragment fields:

```python
from llama_chat import Config

# ChatML, for example:
Config(
    model_path="model.gguf",
    system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
    user_prefix="<|im_start|>user\n",     user_suffix="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n", assistant_suffix="<|im_end|>\n",
)
```

Three load-time checks make a mismatch fail loudly instead of degrading silently:

- **At construction** the turn-terminator (`assistant_suffix`) is validated
  against the model's vocabulary - if it does not tokenize to a single special
  token (the silent-mismatch trap above), construction raises.
- **At construction** the `user`/`assistant` fragments are checked against the
  model's own chat template read from the GGUF: a fixed probe (with surrounding
  whitespace, so a wrong `trim_content` is caught too) is rendered both ways and
  must tokenize identically, so mis-typed tags are caught before they corrupt the
  cache. Disable with `validate_against_model_template=False`; it is
  skipped automatically for models that ship no chat template. The `system`
  fragment is not probed - chat templates handle a system role inconsistently
  (Gemma 4 rejects it; others fold it into the first user turn).
- **At `begin`** the wrapper asserts that per-message prefill tokenizes
  identically to a one-shot render of the whole conversation. If the template
  tokenizes differently across message boundaries (which would make incremental
  `inject` / `request` prefill diverge from a full render), it raises.

## Verify

```bash
# Pure bookkeeping + end-to-end logic, no model needed:
python -m pytest -q

# Against a real model (logs prefill reuse + eviction stress):
python examples/demo.py /path/to/model.gguf                  # CPU
python examples/demo.py /path/to/model.gguf --gpu-layers -1  # GPU
```

The test suite uses a model-free fake backend whose `prefill` asserts
`start_pos == len(cache)` - i.e. the wrapper can only ever *append*, never
re-prefill a survivor. That assertion is the design guarantee, checked
mechanically. The demo proves the same claim against a real model by logging how
many tokens each action prefills, and stresses eviction with a tiny context to
show the oldest messages being cut while the system prompt survives and the total
never crosses `n_ctx`.

## Layout

| File | Role |
|------|------|
| `llama_chat/config.py` | `Config` - the hardware- and model-specific configuration |
| `llama_chat/messages.py` | `MessageTable` - position bookkeeping, evict + shift (pure Python) |
| `llama_chat/template.py` | `TemplateFormatter` - per-message chat-template fragments |
| `llama_chat/_backend.py` | version-stable adapter over `llama_cpp.*` (handles API churn) |
| `llama_chat/context.py` | `KVContext` - tokenize / prefill / streaming generate / evict |
| `llama_chat/wrapper.py` | `ChatWrapper` - `begin` / `inject` / `request` / `stream` |
