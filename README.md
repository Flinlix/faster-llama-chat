# llama_chat: a KV-cache manager for llama.cpp

A thin Python wrapper around llama.cpp whose goal is to **prefill as little as
possible**. Prefill (processing the prompt) can be an expensive part of inference;
the KV cache holding that work can be reused across turns. This wrapper keeps the
cache alive, prefills only genuinely new tokens, and when the cache fills it
*removes the oldest messages and shifts the survivors down to close the gap* -
reusing their KV instead of re-prefilling.

## Overview

- [**Goals**](#goals) - what the wrapper optimizes for, and why naive prompt
  trimming fails to achieve it.
- [**Comparison with llama.cpp**](#comparison-with-llamacpps-built-in-context-shift) -
  how the policy differs from llama.cpp's built-in context shift.
- [**Install**](#install) - building llama-cpp-python from source for CPU or CUDA.
- [**Quick start**](#quick-start) - the API in a few lines, plus the four actions
  (`begin` / `inject` / `request` / `stream`).
- [**How eviction works**](#how-eviction-works) - the in-place KV surgery and the
  shift / rebuild strategies.
- [**Persistence**](#persistence) - keeping the full history beyond the cache window.
- [**Chat template**](#chat-template) - per-role fragments and the load-time
  validation that catches a mismatched template.
- [**Verify**](#verify) - running the tests and the real-model demo.
- [**Layout**](#layout) - a file-by-file map of the package.

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

## Comparison with llama.cpp's built-in context shift

llama.cpp's server has its own context-shift mechanism. Both use the same
`seq_rm` + `seq_add` cache operations; the differences are in policy.

**What gets removed**
- *llama.cpp*: a fixed token count starting at `n_keep`. Cuts mid-message,
  mid-template-tag - leaves malformed turn fragments in the cache.
- *llama_chat*: whole messages only. The surviving cache is always a clean
  sequence of intact turns.

**System-prompt protection**
- *llama.cpp*: `--keep N` where `N` is a hand-counted token offset. Default
  is ~0 (only BOS survives). Wrong or stale `N` silently corrupts the prompt.
- *llama_chat*: the system message is structurally non-evictable - no number
  to maintain.

**When eviction fires**
- *llama.cpp*: when the context is physically full (decode would overflow).
- *llama_chat*: against a soft threshold (`eviction_threshold × context_size`,
  default 0.75). After each turn the cache rests ≤ threshold; `min_reply_tokens`
  guarantees reply headroom or raises `ContextOverflowError` up front.

**SWA / Gemma compatibility**
- *llama.cpp*: context shift is disabled or unreliable on iSWA models in
  default memory-saving mode. `--swa-full` restores it but forfeits the memory
  benefit.
- *llama_chat*: probes `llama_memory_can_shift()` at startup and falls back
  to the rebuild path (drop + re-prefill) on non-shiftable caches - correct
  on the exact models where the built-in fails.

**Bookkeeping**
- *llama.cpp*: no message-level record; client and cache can diverge silently.
- *llama_chat*: `MessageTable` mirrors the cache exactly, invariants asserted
  after every change, `snapshot()` exposes the live layout.

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

Only `Config` differs between targets - `model_path`, `context_size`,
`eviction_threshold`, and `gpu_layers` (`0` on the Pi, `-1` to offload everything
on a GPU). The wrapper code is identical; `seq_rm` / `seq_add` work on both
backends. On a GPU, `flash_attention=True` with `kv_cache_type="q8_0"` roughly
halves KV-cache memory at
near-zero quality cost (a quantized `kv_cache_type` requires flash attention).

## Quick start

```python
from llama_chat import ChatWrapper

# Default config (Gemma 4, CPU, 4096-token context):
chat = ChatWrapper()

# Or override specific fields - kwargs map directly to Config fields:
chat = ChatWrapper(
    model_path="llama/models/gemma-4-E2B-it-Q4_K_M.gguf",
    context_size=8192,    # context window in tokens
    gpu_layers=-1,        # -1 = offload all layers to GPU; 0 = CPU only
    flash_attention=True, # required for quantized KV cache
    kv_cache_type="q8_0", # halves KV memory; requires flash_attention=True
)

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
  older excess is dropped. The first call also warms the generation graph (a
  throwaway decode, immediately cleared) so the first reply has no cold start.
  A system prompt that alone exceeds the threshold raises `ValueError` before
  the previous conversation is touched.
- **`inject(text, role="user")`** - adds context (e.g. retrieved documents) with
  no generation. Evicts the oldest messages first if needed to stay under the
  threshold. Returns the number of messages evicted. A message that could never
  fit is rejected (or truncated, per `oversize_policy`) *before* anything is
  evicted, so a rejected inject leaves the conversation untouched.
- **`request(text)`** - prefills just the request text and generates, evicting
  oldest messages if over threshold and capping generation so the total never
  exceeds `context_size`. The reply is recorded so the next turn reuses it for
  free. Returns a `Turn` (`text`, `n_prefilled`, `n_generated`, `n_evicted`,
  `stop_reason`). If `min_reply_tokens` is set and the prompt would leave less
  than that for the reply, it raises `ContextOverflowError` before touching the
  cache.
- **`stream(text)`** - the generator form of `request`: yields text
  deltas as tokens are sampled and returns the same `Turn` summary on completion
  (via `StopIteration.value`). Raises `ContextOverflowError` on the same
  `min_reply_tokens` headroom check as `request`. Bytes are decoded
  incrementally, so a UTF-8 codepoint split across tokens is never mangled. Each
  token is committed to the cache *before* its text is surfaced, so abandoning
  the stream early (barge-in via `gen.close()`) still records exactly the tokens
  that reached the cache, and the next turn stays valid.

Cache-mutating actions are serialized with a lock, so the wrapper is safe to
drive from a threaded environment. `close()` (also available via `with
ChatWrapper(...) as chat:`) waits for any in-flight turn, frees the model, and
makes later actions raise; it is idempotent and thread-safe.

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

## Persistence

The KV cache only holds the recent window that fits `context_size`; eviction drops older
turns. To keep the *full* history, `PersistentChat` wraps `ChatWrapper` and mirrors
every message (user, assistant, and injected context - everything but the system
prompt) into a durable store, then replays the prior turns at `begin`:

```python
from llama_chat import PersistentChat, InMemoryStore

chat = PersistentChat(InMemoryStore(), context_size=8192)  # kwargs forward to Config here as well
chat.begin("conversation-42", "You are a helpful assistant.")  # loads prior turns
chat.request("What did we decide last time?")                  # auto-persisted
```

A store is any object satisfying the `ConversationStore` protocol - two methods
keyed by conversation id:

```python
class ConversationStore(Protocol):
    def load(self, conversation_id: str) -> list[tuple[str, str]]: ...  # (role, text), oldest first
    def append(self, conversation_id: str, role: str, text: str) -> None: ...
```

`InMemoryStore` is a reference implementation (not durable); back it with SQLite,
Redis, or files by implementing those two methods. Capture is driven by an optional
`on_message(role, text)` hook on `ChatWrapper` itself, so you can persist to your
own sink without `PersistentChat` if you prefer.

## Chat template

The template must match the model, because its turn-terminator is what lets the
model stop generating. A mismatched template tokenizes as plain text, so the
model never emits its end-of-turn token and runs to the generation cap every
turn.

The template is expressed as per-role **fragments**: the literal text wrapped
around each message, so one turn renders as `prefix + text + suffix`. Content is
stripped first when `trim_content` is set, matching templates that apply Jinja
`| trim` such as Gemma 4; it is `False` for templates that emit content verbatim.

**The fragments are derived from the model automatically.** At construction the
wrapper reads the model's own chat template from the GGUF
(`tokenizer.chat_template`) and recovers the per-role fragments from it. It works by rendering
short probes through the model's template with sentinel content and splitting the
render on the sentinels to recover each role's `prefix`/`suffix` (the inverse of
the validation probe below), plus a whitespace-padded probe to detect
`trim_content`. The `system` role has no portable ground truth - some models have
no distinct system role (Gemma 4 rejects it; others fold it into the first user
turn) - so when the system probe does not survive, the system fragment falls back
to the user tags, the same convention llama.cpp uses.

For a model that ships **no** embedded chat template, pass the tags explicitly via
`ChatWrapper(fragments=...)`:

```python
from llama_chat import ChatWrapper, Config, Fragments

# ChatML, for example:
w = ChatWrapper(
    Config(model_path="model.gguf"),
    fragments=Fragments(
        system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",     user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n", assistant_suffix="<|im_end|>\n",
    ),
)
```

Two load-time checks make a mismatch fail loudly instead of degrading silently:

- **At construction** the turn-terminator (`assistant_suffix`) is validated
  against the model's vocabulary - if it does not tokenize to a single special
  token (the silent-mismatch trap above), construction raises.
- **At construction** the `user`/`assistant` fragments are checked against the
  model's own chat template read from the GGUF: a fixed probe (with surrounding
  whitespace, so a wrong `trim_content` is caught too) is rendered both ways and
  must tokenize identically - a self-consistency check that the recovered (or
  explicitly supplied) fragments round-trip through the model's template before
  they corrupt the cache. The fragment side is tokenized exactly as real prefill
  is (tags special-on, content special-off), so this also confirms the two agree
  across the tag/content boundaries. It is skipped automatically for models that
  ship no chat template. The `system` fragment is not probed - chat templates
  handle a system role inconsistently (Gemma 4 rejects it; others fold it into
  the first user turn).

**Special tokens in message content.** *Message content* is tokenized with
special-token parsing **off**, while the structural template tags (the fragment
prefix/suffix) are tokenized with it **on**. So a user typing a literal template
tag (e.g. `<end_of_turn>`) yields ordinary text tokens that can never become a
control token, while the wrapping tags still tokenize to their real control
tokens. This closes token-level prompt injection (forging turn boundaries) at
the source: it is robust even on models that mis-type a control-looking token
in the GGUF or on llama.cpp builds lacking the `llama_vocab_is_control` symbol.

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
never crosses `context_size`.

## Layout

| File | Role |
|------|------|
| `llama_chat/config.py` | `Config` - the hardware- and model-specific configuration |
| `llama_chat/messages.py` | `MessageTable` - position bookkeeping, evict + shift (pure Python) |
| `llama_chat/template.py` | `Fragments` + `TemplateFormatter` - per-message chat-template fragments |
| `llama_chat/template_extract.py` | `extract_fragments` - recover the fragments from the model's GGUF template |
| `llama_chat/_backend.py` | version-stable adapter over `llama_cpp.*` (handles API churn) |
| `llama_chat/context.py` | `KVContext` - tokenize / prefill / streaming generate / evict |
| `llama_chat/wrapper.py` | `ChatWrapper` - `begin` / `inject` / `request` / `stream` |
