# llama_chat: a KV-cache manager for llama.cpp

A thin Python wrapper around llama.cpp whose goal is to **prefill as little as
possible**. Prefill (processing the prompt) can be an expensive part of inference;
the KV cache holding that work can be reused across turns. This wrapper keeps the
cache alive, prefills only genuinely new tokens, and when the cache fills it
*removes the oldest messages and shifts the survivors down to close the gap*,
reusing their KV instead of re-prefilling.

It also calls tools, on any model whose own chat template describes them. The
call tags and the argument spelling are recovered from that template, so no
model's call format is written down in this library.

## Overview

- [**Goals**](#goals): what the wrapper optimizes for, and why naive prompt
  trimming fails to achieve it.
- [**Comparison with llama.cpp**](#comparison-with-llamacpps-built-in-context-shift):
  how the policy differs from llama.cpp's built-in context shift.
- [**Install**](#install): building llama-cpp-python from source for CPU or CUDA.
- [**Quick start**](#quick-start): the API in a few lines, plus what
  `begin` / `inject` / `generate` / `generate_stream` each do.
- [**Configuration**](#configuration): every `Config` field, with its default.
- [**How eviction works**](#how-eviction-works): the in-place KV surgery and the
  shift / rebuild strategies.
- [**Persistence**](#persistence): keeping the full history beyond the cache window.
- [**Chat template**](#chat-template): per-role fragments and the load-time
  validation that catches a mismatched template.
- [**Tool calling**](#tool-calling): registering functions, the loop
  (`answer` / `answer_stream`) and driving it yourself.
- [**Verify**](#verify): running the tests and the real-model demo.
- [**Layout**](#layout): a file-by-file map of the package.

## Goals


1. **Minimum prefill:** only ever process genuinely *new* tokens. The KV cache
   from previous turns must survive and be reused.
2. **Rolling eviction:** when the context fills, old messages are dropped
   automatically so the chat runs indefinitely without erroring on a full
   context.

Naive prompt trimming defeats goal 1: removing old messages from the start of
the sequence shifts every later token's position and **invalidates the entire
cached prefix → full reprefill every turn.** llama-server's `cache_prompt` has
the same problem: the moment any non-prefix message is dropped, the common
prefix collapses to the system prompt. The correct mechanism is **in-place KV
surgery**: remove the oldest message's token span and *shift the survivors' RoPE
positions down* to close the gap, so kept messages are never re-processed.

## Comparison with llama.cpp's built-in context shift

llama.cpp's server has its own context-shift mechanism. Both use the same
`seq_rm` + `seq_add` cache operations; the differences are in policy.

**What gets removed**
- *llama.cpp*: a fixed token count starting at `n_keep`. Cuts mid-message,
  mid-template-tag, leaving malformed turn fragments in the cache.
- *llama_chat*: whole messages only. The surviving cache is always a clean
  sequence of intact turns.

**System-prompt protection**
- *llama.cpp*: `--keep N` where `N` is a hand-counted token offset. Default
  is ~0 (only BOS survives). Wrong or stale `N` silently corrupts the prompt.
- *llama_chat*: the system message is structurally non-evictable, with no
  number to maintain.

**When eviction fires**
- *llama.cpp*: when the context is physically full (decode would overflow).
- *llama_chat*: against a soft threshold (`evict_above_fraction × context_size`,
  default 0.75), and optionally against a message count as well
  (`max_history_messages`). After each turn the cache rests inside both;
  `min_reply_tokens` guarantees reply headroom or raises `ContextOverflowError`
  up front.

**Sliding-window compatibility**
- *llama.cpp*: context shift is disabled or unreliable on iSWA models in
  default memory-saving mode. `--swa-full` restores it but forfeits the memory
  benefit.
- *llama_chat*: probes `llama_memory_can_shift()` at startup and falls back
  to the rebuild path (drop + re-prefill) on non-shiftable caches, which is
  correct on the exact models where the built-in fails.

**Bookkeeping**
- *llama.cpp*: no message-level record; client and cache can diverge silently.
- *llama_chat*: `MessageTable` mirrors the cache exactly, invariants asserted
  after every change, `snapshot()` exposes a copy of the conversation and its layout.

## Install

llama-cpp-python is built **from source per machine** so it uses that CPU's own
kernels or the CUDA backend. The prebuilt wheel targets a baseline instruction
set and ships no CUDA.

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
# CPU, compiled for this machine's own instruction set
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python "llama-cpp-python>=0.3.0"

# Nvidia GPU (CUDA offload)
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.3.0"

pip install -e ".[test]"
```

Only `Config` differs between targets: `model_path`, `context_size`,
`evict_above_fraction`, and `gpu_layers` (`0` for a CPU-only build, `-1` to
offload every layer on a GPU). The wrapper code is identical; `seq_rm` /
`seq_add` work on both backends. On a GPU, `flash_attention=True` with
`kv_cache_type="q8_0"` roughly halves KV-cache memory at near-zero quality cost
(a quantized `kv_cache_type` requires flash attention). `kv_cache_on_gpu`
picks where the KV cache lives: `True` places it in VRAM for fastest
generation, `False` keeps it in host RAM, trading speed for VRAM headroom.
[Configuration](#configuration) lists every field.

## Quick start

```python
from llama_chat import ChatWrapper

# model_path is required; everything else has a default (CPU, 4096 tokens):
chat = ChatWrapper(model_path="models/your-model.gguf")

# Or override specific fields. Kwargs map directly to Config fields:
chat = ChatWrapper(
    model_path="models/your-model.gguf",
    context_size=8192,    # context window in tokens
    gpu_layers=-1,        # -1 = offload all layers to GPU; 0 = CPU only
    kv_cache_on_gpu=True,  # KV cache in VRAM; False keeps it in host RAM
    flash_attention=True, # required for quantized KV cache
    kv_cache_type="q8_0", # halves KV memory; requires flash_attention=True
    decode_threads=12,      # see the note below
    prefill_threads=16,
)

# on_load_progress: an optional callback for startup progress feedback, invoked
# with a fraction 0.0..1.0 while the model weights load. It is keyword-only
# (not a Config field) and must not raise.
chat = ChatWrapper(
    on_load_progress=lambda f: print(f"\rloading model {f * 100:3.0f}%", end="", flush=True)
)

# begin: reset + prefill the system prompt and as much recent history as fits
chat.begin("You are a helpful assistant.",
           [("user", "Hi"), ("assistant", "Hello!")])

# inject: prefill one message as context, WITHOUT generating
chat.inject("Reference: the deadline is Friday.")

# generate: prefill only the new request text, then generate
reply = chat.generate("What's the deadline?")
print(reply.text)

# generate_stream: the same, yielding text deltas as they are sampled.
for delta in chat.generate_stream("Summarize the plan."):
    print(delta, end="", flush=True)
```

**`decode_threads` and `prefill_threads` are separate on purpose.** Prefill is
one brief compute-bound burst and wants every core. Decode is often faster with
fewer, because filling the pool oversubscribes llama.cpp's spin-waiting workers.
What carries over between machines is that the two should differ, not what they
are: measure decode throughput on the host once, and set `decode_threads` below
`prefill_threads` if that is what it shows.

### The core actions

- **`begin(system_prompt, messages=None, tools=None)`**: full reset for a restart /
  conversation switch. Prefills the system prompt (pinned at position 0, never
  evicted) plus the most recent history that fits in a single decode, under the
  token threshold and `max_history_messages` alike; older excess is dropped.
  The first call also warms the generation graph (a throwaway decode,
  immediately cleared) so the first reply
  has no cold start. A system prompt that alone exceeds the threshold raises
  `ValueError` before the previous conversation is touched. `tools` declares
  functions to the model in the same decode; see
  [Tool calling](#tool-calling).
- **`inject(text, role="user")`**: adds context (e.g. retrieved documents) with
  no generation. Evicts the oldest messages first if needed to stay inside both
  eviction limits. Returns the number of messages evicted. A message that could
  never fit is rejected (or truncated, per `oversize_policy`) *before* anything is
  evicted, so a rejected inject leaves the conversation untouched.
- **`generate(text)`**: prefills just the request text and generates, evicting
  oldest messages if either limit is crossed and capping generation so the total
  never exceeds `context_size`. The reply is recorded so the next generation reuses
  it. Returns a `Reply` (`text`, `n_prefilled_tokens`, `n_generated_tokens`,
  `n_evicted_messages`, `stop_reason`, `tool_calls`). If `min_reply_tokens` is
  set and the prompt would leave less than that for the reply, it raises
  `ContextOverflowError`
  before touching the cache. `max_reply_tokens=` and `stop=` override the config for
  one call; the stop strings are added to the configured ones rather than
  replacing them.
- **`generate_stream(text)`**: the generator form of `generate`. It yields text
  deltas as tokens are sampled and returns the same `Reply` summary on completion
  (via `StopIteration.value`). Raises `ContextOverflowError` on the same
  `min_reply_tokens` headroom check as `generate`. Bytes are decoded
  incrementally, so a UTF-8 codepoint split across tokens is never mangled. Each
  token is committed to the cache *before* its text is surfaced, so abandoning
  the stream early (barge-in via the generator's `close()`) still records exactly the tokens
  that reached the cache, and the next generation stays valid. To interrupt from
  *another* thread (a generator must only be driven from its consuming thread),
  pass `generate_stream(text, cancel=event)`: setting the `threading.Event` stops
  generation within one token and the stream ends normally with
  `stop_reason == "interrupted"`.

`stop_reason` is one of:

| | |
|------|------|
| `"end_of_turn"` | the model ended its own turn |
| `"stop_string"` | one of the configured stop strings matched; the reply is trimmed just before it |
| `"token_limit"` | `max_reply_tokens` ran out, or the remaining context did |
| `"interrupted"` | a `cancel` event was set, or the stream was closed early |
| `"tool_call"` | the reply stopped on a call, and the turn is left open |
| `"unreadable_tool_call"` | it stopped on a call whose body this model's recovered format could not read; the turn is closed rather than left open with nothing to close it |

Both stop early if the model emits a tool call, whether or not you declared any:
the reply ends with `stop_reason == "tool_call"`, the call is on
`Reply.tool_calls`, and the turn is left **open** so the result can go inside it.
Until `resume_events()` closes that turn, `generate`, `inject` and
`declare_tools` refuse; `begin()` discards it. On a model whose template
describes tool calls this can happen from a hallucinated call tag alone, so a
caller not using tools should check `stop_reason`. [Tool
calling](#tool-calling) describes the whole mechanism.

Cache-mutating actions are serialized with a lock, so the wrapper is safe to
drive from a threaded environment. `close()` (also available via `with
ChatWrapper(...) as chat:`) waits for any in-flight generation, frees the model, and
makes later actions raise; it is idempotent and thread-safe. It closes a context
passed to the constructor as well, so a wrapper frees the model either way.

### What to reach for, and when

`ChatWrapper` and `PersistentChat` are both laid out in exactly these sections,
in this order, so the code reads in the order of this table.

| doing what | reach for |
|------|------|
| **setting up** | `ChatWrapper(model_path=...)`, then `begin(system_prompt, messages, tools)`. Pass `tools=registry` and it is declared and remembered |
| **adding context** | `inject(text, role="user")`: prefilled, no reply |
| **asking: one generation** | `generate(text)` drained, `generate_stream(text)` streaming. Stops at a tool call and leaves the turn open, with the call on `Reply.tool_calls` |
| **asking: until it answers** | `answer(text)` drained, `answer_stream(text)` streaming. Runs whatever the model calls and keeps going until a generation ends without one, so the turn always closes. Returns one `Reply` summing every generation. Needs no tools to exist |
| **tools: driving the loop yourself** | `generate_events(text)` -> `inject_tool_result(name, result)` -> `resume_events()`; `declare_tools(specs)` adds one mid-conversation |
| **introspection** | the table below; none of it touches the cache |
| **lifecycle** | `close()`, or `with ChatWrapper(...) as chat:` |

Building what the model may call is the registry's job, not the wrapper's:
`@tool(...)`, `registry.register(fn)`, `registry.declare(name)`,
`registry.call(call)`. And every row above works the same through
`PersistentChat`, which mirrors the wrapper method for method.

`begin` is a reset: it drops the history, the registry it was given, and any
turn a call left open.

### Asking the wrapper where it stands

None of these touch the cache, and all are safe to call from another thread.

| | |
|------|------|
| `total_tokens` | how many tokens the cache currently holds |
| `context_size` | the hard limit those tokens count against |
| `evict_above_tokens` | the total above which the oldest messages are evicted, already multiplied out of `evict_above_fraction` |
| `max_history_messages` | the second eviction trigger, or `None` where the token threshold is the only one |
| `snapshot()` | the conversation and its cache layout: one `{"role", "text", "n_tokens", "pos_start", "pos_end"}` row per message |
| `busy` | whether a generation is running. Advisory: it can be stale the moment it returns, and correctness rests on the lock every action takes anyway |
| `wait_idle(timeout_s=None)` | block until no generation is running; `False` on timeout. Advisory in the same way |
| `can_shift` | whether eviction can shift surviving KV in place. `False` (compact SWA, recurrent caches) means eviction re-prefills the survivors instead: correct, but a latency cliff the first time history crosses the threshold |
| `can_call_tools` | whether tools can be used with this model: its template describes a call *and* the argument spelling was recovered |
| `awaiting_tool_result` | whether a call is still owed a result |
| `reply_open` | whether a reply is unfinished, so `generate`, `inject` and `declare_tools` refuse. Outlasts `awaiting_tool_result`: the last result empties the calls while the turn stays open until `resume_events()` |

`snapshot()` copies its rows under the table's own lock, so the positions in a
row set are always mutually consistent even when another thread is mid-generation.
`text` is what was submitted: a message truncated to fit the threshold keeps its
original text beside the tokens that survived, so the two can disagree, and the
tokens are what the model saw.

## Configuration

`Config` is the only thing that differs between deployment targets. Pass one
whole, or pass its fields as keyword arguments to `ChatWrapper` and it builds
one; passing both raises `TypeError`. Everything but `model_path` has a default.

**Model and sampling**

| field | default | meaning |
|------|------|------|
| `model_path` | `None` | Path to the GGUF file. Loading refuses without it |
| `temperature` | `1.0` | `<= 0` selects greedy decoding: `top_k`, `top_p`, `min_p` and `seed` then have no effect, while the three penalties still apply |
| `top_k` | `64` | Top-k cutoff; `<= 0` disables |
| `top_p` | `0.95` | Nucleus cutoff |
| `min_p` | `0.0` | Drop tokens below `min_p` times the top token's probability; `0` disables |
| `repetition_penalty` | `1.0` | Applied over `penalty_window_tokens`; `1.0` disables |
| `presence_penalty` | `0.0` | Flat penalty on any token already in the window; `0` disables |
| `frequency_penalty` | `0.0` | Penalty scaled by how often a token occurs in the window; `0` disables |
| `penalty_window_tokens` | `64` | How far back the three penalties look |
| `seed` | `None` | Sampling RNG seed in `[0, 2**32 - 2]`, for reproducible generation. `None` draws a fresh one per run |

**Context management**

| field | default | meaning |
|------|------|------|
| `context_size` | `4096` | The hard limit in tokens: generation never crosses it |
| `evict_above_fraction` | `0.75` | Fraction of `context_size` above which the oldest messages are evicted. Must be in `(0, 1]` |
| `max_history_messages` | `None` | The most non-system messages to keep, OR'd with `evict_above_fraction` as a second eviction trigger. Counts one per request, one per reply, one per `inject` and one per `declare_tools`. `None` leaves the token threshold as the only trigger; otherwise `>= 1` |
| `max_reply_tokens` | `1024` | Cap on tokens per generation, capped again so the total stays inside `context_size` |
| `oversize_policy` | `"reject"` | What happens to a message that cannot fit even with everything but the system prompt evicted. `"reject"` raises; `"truncate"` clips the content and re-appends the turn-terminator tag, so the clipped turn still closes |
| `min_reply_tokens` | `32` | Refuse a request with `ContextOverflowError` when fewer than this many tokens would be left for the reply. `0` disables the guard |

**Generation**

| field | default | meaning |
|------|------|------|
| `stop_strings` | `[]` | Strings that end generation. A `stop=` passed to one request adds to these rather than replacing them |

**Backend and hardware**

| field | default | meaning |
|------|------|------|
| `gpu_layers` | `0` | Layers offloaded to GPU. `0` for a CPU-only build, `-1` for all of them |
| `kv_cache_on_gpu` | `False` | `True` puts the KV cache in VRAM (llama.cpp's own default, fastest generation); `False` keeps it in host RAM, freeing a large amount of VRAM on long contexts. Ignored on a CPU-only build |
| `decode_threads` | `None` | CPU threads for decode; `None` lets llama.cpp decide |
| `prefill_threads` | `None` | CPU threads for prefill; `None` means as many as `decode_threads`. See the note under [Quick start](#quick-start) |
| `max_batch_tokens` | `512` | Most tokens handed to one `llama_decode`; a longer prefill is chunked to this |
| `flash_attention` | `False` | Set explicitly either way, so a build's own default never overrides the caller. Required by a quantized KV cache |
| `kv_cache_type` | `None` | KV-cache quantization: `"f16"`, `"q8_0"`, `"q5_1"`, `"q5_0"`, `"q4_1"` or `"q4_0"`, or `None` for llama.cpp's own default. Every one but `"f16"` is quantized and requires `flash_attention=True`; `"f16"` is the default type spelled out, and needs nothing |

**Logging**

| field | default | meaning |
|------|------|------|
| `log_level` | `"error"` | llama.cpp's own verbosity: `"debug"`, `"info"`, `"warn"`, `"error"` or `"none"`. Applied process-wide before the model loads. llama.cpp tags its routine loader banners at WARN, so `"warn"` still prints the whole load-time log; `"none"` silences it entirely |

A `Config` validates itself on construction, so a bad value fails there rather
than at first use: a fraction outside `(0, 1]`, a `context_size`,
`max_batch_tokens` or `max_reply_tokens` that is not positive, a
negative `min_reply_tokens`, a
`repetition_penalty` of zero or below, a seed outside its range, an unknown
`oversize_policy`, `log_level` or `kv_cache_type`, and a quantized
`kv_cache_type` without flash attention.

The rest of what `ChatWrapper` takes is keyword-only and deliberately outside
`Config`, because none of it is a value to tune:

| | |
|------|------|
| `fragments=` | the per-role template tags, for a model that ships none. See [Chat template](#chat-template) |
| `call_body=` | how a call spells its arguments, for a model whose shape the probes cannot recover. See [Tool calling](#tool-calling) |
| `formatter=` | a `TemplateFormatter` subclass, where opening or closing a turn takes more than the tag. See [A resumed turn needs its generation prompt back](#a-resumed-turn-needs-its-generation-prompt-back) |
| `on_message=` | called with `(role, text)` for every genuine new message. See [Persistence](#persistence) |
| `on_load_progress=` | called with a fraction while the weights load. It must not raise: an exception there aborts the load |
| `context=` | a cache context of your own, in place of the one built from `Config`. See [Layout](#layout) |

## How eviction works

Eviction has two triggers, and they are OR'd: the cache is trimmed when it
crosses `evict_above_fraction × context_size`, and again when it holds more than
`max_history_messages` non-system messages. Whichever binds first evicts, so a
verbose conversation is bounded by its tokens and a long one by its message
count. The second is off by default.

A pure-Python `MessageTable` is the single source of truth: the cache for
sequence 0 is a contiguous run `[0, total)`, each message owns a `[start, stop)`
slice, and the table recomputes every position after any structural change. To
evict the oldest non-system message at `[a, b)`:

```
llama_memory_seq_rm(seq, a, b)               # drop those tokens
llama_memory_seq_add(seq, b, total, -(b-a))  # shift survivors down to close the gap
```

`seq_add` also fixes the RoPE positions, so the survivors stay coherent without
re-prefilling them. It does not do that work where you call it: `seq_add` only
marks the cells as shifted, and the RoPE K-shift over the whole cache is built
into the graph of the **next** `llama_decode` (the shift is expensive only for
a quantized cache). That cost is therefore paid by whatever decodes next,
which is why `generate_stream` decodes the assistant turn's closing tag *after* the
post-reply eviction rather than before it: a decode the turn performs anyway
then carries the shift, instead of the next generation's prefill.

This reuse is *lossy* (the survivors' cached K/V were computed
while attending to the now-removed tokens), which is the standard StreamingLLM
trade-off: fast, with slight quality drift. The system prompt is never evicted.

The strategy is chosen automatically at construction from
`llama_memory_can_shift()`:

- **shift** (default, the fast path): the in-place edit above, applied
  incrementally as messages are dropped. Used everywhere it can be, including
  sliding-window models under a full-size cache.
- **rebuild** (caches that cannot shift: compact sliding-window, recurrent or
  Mamba state, where dropping tokens loses the position information shifting
  needs):
  drop the oldest messages from the bookkeeping, then re-prefill the surviving
  conversation once. Correct, just not free; a non-shiftable model degrades
  gracefully instead of corrupting the cache.

The backend (`_backend.py`) resolves the cache symbols defensively across
llama.cpp API churn: the modern `llama_memory_*` handle API when present,
falling back to the older `llama_kv_self_*` / `llama_kv_cache_*` names.

## Persistence

The KV cache only holds the recent window that fits `context_size`; eviction drops older
turns. To keep the *full* history, `PersistentChat` wraps `ChatWrapper` and mirrors
every message (user, assistant, and injected context, so everything but the
system prompt) into a durable store, then replays the prior turns at `begin`:

```python
from llama_chat import PersistentChat, InMemoryStore

chat = PersistentChat(InMemoryStore(), context_size=8192)  # kwargs forward to Config here as well
chat.begin("conversation-42", "You are a helpful assistant.")  # loads prior turns
chat.generate("What did we decide last time?")                  # auto-persisted
```

A store is any object satisfying the `ConversationStore` protocol, which is
two methods keyed by conversation id:

```python
class ConversationStore(Protocol):
    def load(self, conversation_id: str) -> list[tuple[str, str]]: ...  # (role, text), oldest first
    def append(self, conversation_id: str, role: str, text: str) -> None: ...
```

`InMemoryStore` is a reference implementation (not durable); back it with SQLite,
Redis, or files by implementing those two methods. Capture is driven by an optional
`on_message(role, text)` hook on `ChatWrapper` itself, so you can persist to your
own sink without `PersistentChat` if you prefer.

`PersistentChat` forwards the whole wrapper API, tools included:
`begin(cid, prompt, tools=registry)`, then `answer` / `answer_stream`, or the
manual loop. A call and the result it was given are not messages, so a tool turn
is persisted exactly like any other: one user entry, one assistant entry holding
what a reader saw. Declarations are not persisted either: they belong to the
system turn, which is rebuilt on every `begin`.

## Chat template

The template must match the model, because its turn-terminator is what lets the
model stop generating. A mismatched template tokenizes as plain text, so the
model never emits its end-of-turn token and runs to the token cap every
generation.

The template is expressed as per-role **fragments**: the literal text wrapped
around each message, so one turn renders as `prefix + text + suffix`. Content is
stripped first when `trim_content` is set, matching templates that apply Jinja
`| trim`; it is `False` for templates that emit content verbatim.

**The fragments are derived from the model automatically.** At construction the
wrapper reads the model's own chat template from the GGUF
(`tokenizer.chat_template`) and recovers the per-role fragments from it. It works by rendering
short probes through the model's template with sentinel content and splitting the
render on the sentinels to recover each role's `prefix`/`suffix` (the inverse of
the validation probe below), plus a whitespace-padded probe to detect
`trim_content`. The `system` role has no portable ground truth, because some models
have no distinct system role (some reject it, others fold it into the first user
turn). So when the system probe does not survive, the system fragment falls back
to the user tags, the same convention llama.cpp uses.

For a model that ships **no** embedded chat template, pass the tags explicitly via
`ChatWrapper(fragments=...)`:

```python
from llama_chat import ChatWrapper, Config, TemplateFragments

# ChatML, for example:
w = ChatWrapper(
    Config(model_path="model.gguf"),
    fragments=TemplateFragments(
        system_prefix="<|im_start|>system\n", system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",     user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n", assistant_suffix="<|im_end|>\n",
    ),
)
```

For a model that ships no template *and* calls tools, add the call tags to those
fragments with
`TemplateFragments(..., tools=ToolCallFragments(call_open=..., call_close=...))`.
Each is taken as it stands in a render, so either may carry fixed text around the
tag itself; the wrapper narrows it to the one token the tag is.

The fragments say what wraps text that already exists. What is decoded *around
sampling* is the formatter's, and `ChatWrapper(formatter=...)` takes a
`TemplateFormatter` subclass for a model whose generation prompt carries more
than the turn tag, an empty reasoning channel for instance. See
[A resumed turn needs its generation prompt back](#a-resumed-turn-needs-its-generation-prompt-back).

Two load-time checks make a mismatch fail loudly instead of degrading silently:

- **At construction** the turn-terminator (`assistant_suffix`) is validated
  against the model's vocabulary: if it does not tokenize to a single special
  token, which is the silent mismatch described above, construction raises.
- **At construction** the `user`/`assistant` fragments are checked against the
  model's own chat template read from the GGUF: a fixed probe (with surrounding
  whitespace, so a wrong `trim_content` is caught too) is rendered both ways and
  must tokenize identically, a self-consistency check that the recovered (or
  explicitly supplied) fragments round-trip through the model's template before
  they corrupt the cache. The fragment side is tokenized exactly as real prefill
  is (tags special-on, content special-off), so this also confirms the two agree
  across the tag/content boundaries. It is skipped automatically for models that
  ship no chat template. The `system` fragment is not probed, because chat
  templates handle a system role inconsistently (some reject it, others fold it into
  the first user turn).

### Special tokens in message content

*Message content* is tokenized with special-token parsing **off**, while the
structural template tags (the fragment prefix/suffix) are tokenized with it
**on**. So a user typing a literal template
tag (e.g. `<end_of_turn>`) yields ordinary text tokens that can never become a
control token, while the wrapping tags still tokenize to their real control
tokens. This closes token-level prompt injection (forging turn boundaries) at
the source: it is robust even on models that mis-type a control-looking token
in the GGUF or on llama.cpp builds lacking the `llama_vocab_is_control` symbol.

## Tool calling

Register real Python functions; the model calls them by name and the reply
resumes with what they returned. Nothing about any model's call format is
written down here: the tags and the argument spelling are recovered from the
model's own chat template, the same way the per-role fragments above are.

```python
from llama_chat import ChatWrapper, ToolRegistry

def get_weather(city: str) -> dict:
    return {"city": city, "celsius": 18}

registry = ToolRegistry()
registry.register(get_weather, "Current weather for a city.", declare=True)

chat = ChatWrapper(model_path="model.gguf")
chat.begin("You are a terse assistant.", tools=registry)

print(chat.answer("What is the weather in Munich?").text)   # the call runs inside
```

Handing `begin` the whole registry declares what that registry says should be
declared and keeps it, so what the model was told about and what can actually
run cannot drift apart. `answer` is `answer_stream` drained, the way `generate` is
`generate_stream`; both return a `Reply` whose `text` is the reply. Use
`answer_stream` where you want the deltas:

```python
for delta in chat.answer_stream("What is the weather in Munich?"):
    print(delta, end="")     # only prose; the calls happen inside
```

`chat.can_call_tools` says whether the model can do this at all. It is `False`
both for a template that renders no tool call and for one whose argument
spelling could not be recovered, because neither yields a call you can run.

### Registering and declaring are two things

`register` makes a function runnable and returns the name the model calls it
by. That name is also how you refer to the tool afterwards. Whether the model
is *told* about the tool is a separate question, because telling it costs a
prefill, and where that prefill lands depends on when you do it:

| | where the declaration lands | cost |
|------|------|------|
| `begin(..., tools=registry)` | in the system turn | none extra |
| `chat.declare_tools([registry.declare("clock")])` | a message of its own | one prefill |
| neither | nowhere; registered and silent | none |

A tool registered with `declare=False` still runs as soon as something names
it. The only thing withheld is the model's knowledge that it exists. Registering
a name that is already taken replaces it, `declare` state included.

The `@tool` decorator attaches the same declaration to the function itself, so
`register` needs no description of its own:

```python
from llama_chat import tool

@tool("Current weather for a city.")
def get_weather(city: str) -> dict:
    return {"city": city, "celsius": 18}

registry.register(get_weather, declare=True)
```

The parameters come from the annotations either way. A signature that cannot be
described (an unannotated argument, or a type with no JSON Schema equivalent)
raises `TypeError`, and `parameters=` takes an explicit schema instead.

Two consequences worth knowing. A declaration added later is an ordinary message
and can be evicted from a long conversation, so one that has to survive belongs
in `begin`. And `begin` is a reset, so a declaration made before one is thrown
away; `declare_tools` refuses rather than writing something that will vanish.

The registry does not track either of those. `specs_to_declare()` answers what
the model *should* be told, which is what you hand to `begin` or
`declare_tools`; it never claims to know what is still in the conversation, and
it goes on naming a tool whose declaration has since been evicted.

### Which entry point does what

All of them keep a tool call out of the reply text, and all of them hand back a
`Reply`. Two things vary: how far the call is carried, and what the generator
yields on the way.

| how far it carries the exchange | drained | yields text | yields text and calls |
|------|------|------|------|
| one generation, which may stop at a call | `generate(text)` | `generate_stream(text)` | `generate_events(text)` |
| resume the reply a call interrupted | | | `resume_events()` |
| as many generations as it takes, calls run for you | `answer(text)` | `answer_stream(text)` | |

The name says both. The stem is how far it goes: `generate` runs the model once,
`answer` runs it as often as the tools require, `resume` picks up a reply a call
left open. The suffix says what comes out: no suffix returns the `Reply`
directly, `_stream` yields text deltas and returns the `Reply` through
`StopIteration.value`, `_events` yields those deltas and each `ToolCall` as the
model asks for it.

The two empty cells are not gaps. `answer` runs the calls itself, so it has
nothing to surface; and a resumption that ends on another call has to hand that
call back, which is what an event stream is for.

`answer` needs no tools: on a conversation without them the `Reply` is the reply
and no calls, and a call the model invented is answered saying so rather than
raising, so the turn still closes. `generate` is what you reach for when you want
the counts for one generation: `n_prefilled_tokens`, `n_generated_tokens`,
`n_evicted_messages`, `stop_reason`.

### Driving the loop, or driving it yourself

- **`answer(text)`**: `answer_stream` drained, for a caller that does not want
  the deltas.
- **`answer_stream(text, registry=None, max_calls=4)`**: the whole loop.
  Generate, run whatever the model called, feed the results back, and resume
  until it answers. Yields text deltas and returns the summed `Reply` (via
  `StopIteration.value`). `max_calls` limits how many tools one turn may use.
  The last round runs with the call tags withheld, so the model cannot ask for
  another one. Without that the limit would stop the loop but not the turn: a
  turn that ends on a call is deliberately left open, so it would never close.
  The registry defaults to the one `begin` was handed.
- **`generate_events(text)`**: the same generation, yielding `str` deltas and
  then a `ToolCall` for each call the model made. Use this to run the loop
  yourself: answer with `inject_tool_result(name, result)` once per call, in the
  order they were asked, then call `resume_events()`. The order is enforced,
  because the name is what says which call is being answered.
  `chat.awaiting_tool_result` says whether anything is still owed, and
  `chat.reply_open` whether the turn is still open.

`answer` and `answer_stream` return one `Reply` for the whole exchange: the
counts are sums over every generation, `stop_reason` is the last generation's, and
`tool_calls` holds every call the model made in order. `tool_limit_reached` says
whether `max_calls` ran out, so the final generation ran with the call tokens
withheld; it is always `False` where no call was asked for.

Two outcomes are visible only on that summary. A reply that ended on a call this model's
format could not be read from has empty `text`, and only
`stop_reason == "unreadable_tool_call"` tells it apart from a model that said
nothing. And when `tool_limit_reached` is set, the model was cut off from a tool
it wanted rather than choosing to stop.

A turn that stopped on a call stays open until `resume_events()` closes it,
even once every result is in. Until then `generate`, `inject` and `declare_tools`
refuse: a message appended there would sit inside a model turn that never ended.
Finish the turn, or drop it with `begin()`. `reply_open` reports that state, and
it is the one to branch on: `awaiting_tool_result` goes `False` the moment the
last result lands, while the turn is still open.

**Nothing is evicted while a turn is open**, because evicting would cut into the
turn being written. A turn that calls several tools therefore grows past both
eviction limits towards `context_size`, and is trimmed back inside them only
once it closes. Until then `context_size` is the only bound: a result too large
for what is left is clipped to leave `min_reply_tokens` free,
and one whose wrapping alone does not fit raises `ContextOverflowError`. The
wrapping itself is never clipped, because half a result block is something no
template ever produced, so such a turn cannot be continued and `begin()` is the
way out.

A call never appears in the reply. Its tokens reach the cache, because the model
has to see its own call when it reads the turn back, but they are not yielded as
deltas and not part of `Reply.text`. On a voice pipeline that is the difference
between a working assistant and one that reads its own function calls aloud.

### What is recovered, and what happens when it cannot be

The call tags are found by probing the model's template, then resolving each
one to the single token it stands for. A tag counts as such if it is either a
control token or **a token added by a fine-tune**. Both are needed: on the
models this was tested against, the tags around a tool call are added tokens,
and `llama_vocab_is_control` returns false for every one of them. A check that
accepts only control tokens therefore turns tool calling off without saying
so.

The argument spelling is recovered from four probes that differ only in their
content, so each delimiter is found by comparing the results. A body that is
JSON is read directly instead, because it already carries the function name and
the arguments. Together the two cover JSON objects and arrays,
`name{key:value}` with quoted strings, `name(k="v")`, attribute-style XML, and
`<function=name><parameter=k>v</parameter>`, including nested objects, lists,
and separators inside quoted values.

What the recovered delimiters look like on two real models, since the field
names alone are hard to picture:

| `CallBody` field | a delimited model | an element-shaped model |
|------|------|------|
| `name_close` | `{` | `>\n<` |
| `key_open` | (empty) | `parameter=` |
| `key_close` | `:` | `>\n` |
| `pair_sep` | `,` | `\n</parameter>\n<parameter=` |
| `body_close` | `}` | `\n</parameter>\n</function>\n` |
| `empty_close` | `}` | `/function>\n` |

A delimiter runs only as far as the probes agree, which is why the element
column splits `<` off its `<parameter=`: the no-argument render shares that
character, so it lands in `name_close` rather than in what follows.

Where the format cannot be recovered, `can_call_tools` is `False`. The call is
still detected, so its syntax never reaches the reply text, but its body is
reported as unreadable and the turn closes with
`stop_reason == "unreadable_tool_call"`. Nothing is guessed at. For such a model, pass
the spelling in yourself with `ChatWrapper(call_body=CallBody(...))`, which makes
`can_call_tools` `True` again, the same way `fragments=` covers a template that
cannot be read.

One limit: a model that writes a batch of calls as a JSON array has all of them
read and each owes a result, but any other shape ends at its closing tag, so
generation stops after one call either way.

### A resumed turn needs its generation prompt back

A model whose generation prompt carries more than the turn tag, an empty
reasoning channel for instance, never has that part prefilled when the reply
resumes after a tool result, because the turn tag must not repeat. Left alone
the model writes it itself and it arrives as reply text. `TemplateFormatter`
has `resume_open()` for exactly that part, empty by default; a formatter that
adds something at turn start adds it here too:

```python
from llama_chat import ChatWrapper, TemplateFormatter

EMPTY_CHANNEL = "<|channel>thought\n<channel|>"   # whatever this model wants

class Channelled(TemplateFormatter):
    def assistant_open(self) -> str:
        return super().assistant_open() + EMPTY_CHANNEL

    def resume_open(self) -> str:
        return EMPTY_CHANNEL

ChatWrapper(model_path="model.gguf", formatter=Channelled)
```

## Verify

```bash
# Pure bookkeeping + end-to-end logic, no model needed:
python -m pytest -q

# Against a real model (logs prefill reuse + eviction stress):
python examples/demo.py /path/to/model.gguf                  # CPU
python examples/demo.py /path/to/model.gguf --gpu-layers -1  # GPU
```

The test suite uses a model-free fake backend whose `prefill` asserts
`start_pos == len(cache)`, so the wrapper can only ever *append*, never
re-prefill a survivor. That assertion is the design guarantee, checked
mechanically. The demo proves the same claim against a real model by logging how
many tokens each action prefills, and stresses eviction with a tiny context to
show the oldest messages being cut while the system prompt survives and the total
never crosses `context_size`.

## Layout

| File | Role |
|------|------|
| `llama_chat/config.py` | `Config`: every runtime setting, field by field in [Configuration](#configuration) |
| `llama_chat/messages.py` | `MessageTable`, `Message`, `Eviction`, `n_newest_that_fit`: position bookkeeping, evict + shift (pure Python) |
| `llama_chat/template.py` | `TemplateFragments`, `TemplateFormatter`, `ToolCallFragments`, `CallBody`: what a turn and a tool call are wrapped in |
| `llama_chat/template_extract.py` | `extract_fragments`, `extract_tool_fragments`, `extract_call_body`: recover all of that from the model's GGUF template |
| `llama_chat/_backend.py` | version-stable adapter over `llama_cpp.*` (handles API churn) |
| `llama_chat/context.py` | `KVContext`: tokenize / prefill / streaming generate / evict |
| `llama_chat/tools.py` | `ToolRegistry`, `ToolSpec`, `ToolCall`: declaring tools, running them, reading calls back |
| `llama_chat/wrapper.py` | `ChatWrapper`, `Reply`, `ContextOverflowError`: `begin` / `inject` / `generate` / `generate_stream` / `answer` / `answer_stream` |
| `llama_chat/persistent_wrapper.py` | `PersistentChat`, `ConversationStore`, `InMemoryStore`: the same API, with every message mirrored to a durable store |

Everything in that table is importable from `llama_chat` itself, apart from the
three extraction functions: the wrapper runs those for you at construction, and
`llama_chat.template_extract` is where to reach them to run against a template
directly. `n_newest_that_fit` is the pure function `begin` uses to decide how
much history fits, exposed so that decision can be made before committing to it.

The bookkeeping types are exported for one purpose: `ChatWrapper(context=...)`
takes a cache context of your own in place of the one built from `Config`, and
`Eviction` is what such a context is handed to apply. No subclassing and no
registration: an object that answers what the wrapper calls on it is a cache
context, and `tests/fake_context.py` is a worked one, over a plain list of ints
with no llama.cpp in it.
