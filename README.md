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
- [**How eviction works**](#how-eviction-works): the in-place cache edit and the
  shift / rebuild strategies.
- [**Install**](#install): building llama-cpp-python from source for CPU or CUDA.
- [**Quick start**](#quick-start): the API in a few lines, plus what `begin` /
  `inject` / `reply_without_tools` / `reply_without_tools_stream` each do.
- [**Configuration**](#configuration): every `Config` field, with its default.
- [**Tool calling**](#tool-calling): registering functions, the loop
  (`reply_with_tools` / `reply_with_tools_stream`) and driving it yourself.
- [**Persistence**](#persistence): keeping the full history beyond the cache window.
- [**Chat template**](#chat-template): per-role fragments and the load-time
  validation that catches a mismatched template.
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
cached prefix, so every turn re-prefills in full.** llama-server's
`cache_prompt` has the same problem: the moment any non-prefix message is
dropped, the common prefix collapses to the system prompt. The correct
mechanism is an **in-place KV cache edit**: remove the oldest message's token
span and *shift the survivors' RoPE positions down* to close the gap, so kept
messages are never re-processed.

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
- *llama_chat*: asks `llama_memory_can_shift()` at startup, proves a cut on
  the empty cache, and falls back to the rebuild path (drop + re-prefill) on
  caches that cannot shift, which is correct on the exact models where the
  built-in fails.

**Bookkeeping**
- *llama.cpp*: no message-level record; client and cache can diverge silently.
- *llama_chat*: `MessageTable` mirrors the cache exactly, invariants asserted
  after every change, `snapshot()` exposes a copy of the conversation and its layout.

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
into the graph of the **next** `llama_decode` (the shift is expensive only for a
quantized cache). That cost is therefore paid by whatever decodes next, which is
why `reply_without_tools_stream` decodes the assistant turn's closing tag
*after* the post-reply eviction rather than before it: a decode the turn
performs anyway then carries the shift, instead of the next generation's
prefill.

This reuse is *lossy* (the survivors' cached K/V were computed
while attending to the now-removed tokens), which is the standard StreamingLLM
trade-off: fast, with slight quality drift. The system prompt is never evicted.

The strategy is chosen automatically at construction from
`llama_memory_can_shift()` and a cut tried on the empty cache. A cache that
refuses the cut holds a running state and is never shifted, whatever it
reports: a hybrid cache reports the shift of its attention part, and takes
every removal that stops short of its end without its running state losing
those tokens.

- **shift** (default, the fast path): the in-place edit above, applied
  incrementally as messages are dropped. Used everywhere it can be, including
  sliding-window models under a full-size cache.
- **rebuild** (caches that cannot shift: compact sliding-window, recurrent or
  Mamba state, where dropping tokens loses the position information shifting
  needs):
  drop the oldest messages from the bookkeeping, then re-prefill the surviving
  conversation once. Correct, and it costs one prefill of the survivors; a
  non-shiftable model degrades gracefully instead of corrupting the cache. Such
  a cache may still take a cut at its end, which a load-time probe settles;
  then the newest reply is shortened by a cut. One that cannot, a recurrent or
  windowed one, has the part of its state a cut cannot reach saved whenever
  the cache comes to rest, the last two kept in RAM, one from before the
  newest reply and one from after it. That part's size is the model's, not
  the conversation's. A reply
  kept in another form than it was generated in, see [Reasoning](#reasoning),
  a cut and a removed reply then restore the earlier state and decode only
  what follows it. Where the build saves no state, or the newest reply lies
  before both, the conversation is rebuilt instead. A cache that cannot shift
  is never asked to shift, not even at load.

`_backend.py` resolves the cache symbols defensively across llama.cpp API
churn: the modern `llama_memory_*` handle API when present, falling back to the
older `llama_kv_self_*` / `llama_kv_cache_*` names.

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
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python "llama-cpp-python>=0.3.0"

pip install -e ".[test]"
```

Only `Config` differs between targets: `model_path`, `context_size`,
`evict_above_fraction`, and `gpu_layers` (`0` for a CPU-only build, `-1` to
offload every layer on a GPU). The wrapper code is identical; `seq_rm` /
`seq_add` work on both backends. On a GPU, `flash_attention=True` with
`kv_cache_type="q8_0"` roughly halves the memory the KV cache occupies, at a
small cost in output quality (a quantized `kv_cache_type` requires flash
attention). `kv_cache_on_gpu` picks where the KV cache lives: `True` places it
in VRAM for fastest generation, `False` keeps it in host RAM, trading speed for
VRAM headroom.
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
    model_path="models/your-model.gguf",
    on_load_progress=lambda f: print(f"\rloading model {f * 100:3.0f}%", end="", flush=True),
)

# begin: reset + prefill the system prompt and as much recent history as fits
chat.begin("You are a helpful assistant.",
           [("user", "Hi"), ("assistant", "Hello!")])

# inject: prefill one message as context, WITHOUT generating
chat.inject("Reference: the deadline is Friday.")

# reply_without_tools: prefill only the new request text, then generate
reply = chat.reply_without_tools("What's the deadline?")
print(reply.text)

# reply_without_tools_stream: the same, yielding deltas as they are sampled
for delta in chat.reply_without_tools_stream("Summarize the plan."):
    print(delta, end="", flush=True)
```

**`decode_threads` and `prefill_threads` are separate on purpose.** Prefill is
one brief compute-bound burst and scales with every core it is given. Decode
is often faster with fewer, because filling the pool oversubscribes llama.cpp's
spin-waiting workers.
What carries over between machines is that the two should differ, not what they
are: measure decode throughput on the host once, and set `decode_threads` below
`prefill_threads` if that is what it shows.

### The core actions

- **`begin(system_prompt, messages=None, tools=None)`**: full reset for a
  restart / conversation switch. Prefills the system prompt (pinned at position
  0, never evicted) plus the most recent history that fits in a single decode,
  under the token threshold and `max_history_messages` alike; older excess is
  dropped. The first call also warms the generation graph (a throwaway decode,
  immediately cleared) so the first reply has no cold start. A system prompt
  that alone exceeds the threshold raises `ValueError` before the previous
  conversation is touched. `tools` declares functions to the model in the same
  decode; see [Tool calling](#tool-calling). Nothing else may run before it:
  `inject` and `reply_without_tools` refuse until a conversation has been
  begun, because only `begin` prefills the sequence opener and the system
  prompt.
- **`inject(text, role="user")`**: adds context (e.g. retrieved documents) with
  no generation. Evicts the oldest messages first if needed to stay inside both
  eviction limits. Returns the number of messages evicted. A message that could
  never fit is rejected (or truncated, per `oversize_policy`) *before* anything is
  evicted, so a rejected inject leaves the conversation untouched.
- **`reply_without_tools(text)`**: prefills just the request text and
  generates, evicting oldest messages if either limit is crossed and capping
  generation so the total never exceeds `context_size`. The reply is recorded so
  the next generation reuses it. Returns a `Reply` (`text`,
  `n_prefilled_tokens`, `n_generated_tokens`, `n_evicted_messages`,
  `stop_reason`, `tool_calls`, `tool_limit_reached`, `n_reasoning_tokens`). If `min_reply_tokens` is set and the prompt
  would leave less than that for the reply, it raises `ContextOverflowError`
  before touching the cache. `max_reply_tokens=` and `stop=` override the config
  for one call; the stop strings are added to the configured ones rather than
  replacing them.
- **`reply_without_tools_stream(text)`**: the streaming form of
  `reply_without_tools`. It returns a `ReplyStream`: iterate it for the text
  deltas as tokens are sampled, then read `stream.reply` for the same summary.
  Raises `ContextOverflowError` on the same `min_reply_tokens` headroom check as
  `reply_without_tools`. Bytes are decoded incrementally, so a UTF-8 codepoint
  split across tokens is never mangled. Each token is committed to the cache
  *before* its text is surfaced, so abandoning the stream early (the
  stream's `close()`) still records exactly the tokens that reached the
  cache, and the next generation stays valid. To interrupt from *another* thread
  (a stream must only be driven from its consuming thread), pass
  `reply_without_tools_stream(text, cancel=event)`: setting the
  `threading.Event` stops generation within one token and the stream ends
  normally with `stop_reason == "interrupted"`.
- **`sampling=Sampling(...)`**, on every `reply_*` and `resume_*` call:
  sampling settings for that one request. `temperature`, `top_p`, `seed`,
  `presence_penalty` and `frequency_penalty` each override the config's value
  where set and leave it where `None`. A request that overrides anything samples
  from a chain of its own, freed when its generation ends; one that overrides
  nothing uses the chain built at load.
- **`truncate_last_reply(keep_text)`**: shortens the newest assistant reply to
  a prefix of its text, for a reply that was cut short where it was being read
  or heard, so the model holds only what the reader got. The cut is at the
  cache's end: the tokens the shortened message shares with the old one are not
  decoded again, and the rest, the turn's closing tag at least, is. A cache that
  cannot shift is instead re-prefilled from the whole conversation on every cut,
  as its eviction is; see "How eviction works". Refused where a reply is
  open, the newest message is not the assistant's or holds a tool call, or
  `keep_text` is not a prefix.
- **`begin(..., reasoning=True)`**: the model reasons before each answer, as
  its own template turns that on. See [Reasoning](#reasoning).

`stop_reason` is one of:

| | |
|------|------|
| `"end_of_turn"` | the model ended its own turn |
| `"stop_string"` | one of the configured stop strings matched; the reply is trimmed just before it |
| `"token_limit"` | `max_reply_tokens` ran out, or the remaining context did |
| `"interrupted"` | a `cancel` event was set, or the stream was closed early |
| `"tool_call"` | the reply stopped on a call, and the turn is left open for a result |
| `"unreadable_tool_call"` | it stopped on a call whose body this model's recovered format could not read. The model is told so and the reply carries on, so this is the final reason only from `reply_until_tool`, or where the template renders no result block to tell the model in |

Both stop early if the model emits a tool call, whether or not you declared any:
on a model whose template describes calls this can happen from a hallucinated
tag alone. The call is kept out of the reply text and reported on
`Reply.tool_calls`. Neither leaves a turn open, so a caller that never mentioned
tools needs no special handling: `reply_without_tools` tells the model the call
failed and that no tool can be run, and it finishes its reply in words, a
blank line parting the text before the call from the text after it where the
two would join without whitespace. Only
`reply_until_tool` holds a turn open, because only it hands you the call to
answer. [Tool calling](#tool-calling) describes the whole mechanism.

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
| **asking: without tools** | `reply_without_tools(text)` drained, `reply_without_tools_stream(text)` streaming. The one entry point where no tool runs, not even one `begin` was given and remembered: a call the model makes anyway is refused with an error and it answers in words instead |
| **asking: with tools** | `reply_with_tools(text)` drained, `reply_with_tools_stream(text)` streaming. Runs whatever the model calls and keeps going until a generation ends without one, so the turn always closes. Returns one `Reply` summing every generation. Needs no tools to exist, and refuses a registry this model could never call |
| **tools: driving the loop yourself** | `reply_until_tool(text)` -> `inject_tool_result(call, result)` -> `resume_until_tool()`; `declare_unknown_tools(registry)` adds one mid-conversation |
| **introspection** | the table below; none of it touches the cache |
| **lifecycle** | `close()`, or `with ChatWrapper(...) as chat:` |

Building what the model may call is the registry's job, not the wrapper's:
`@registry.tool(...)`, `registry.register(fn)`, `registry.spec(name)`,
`registry.call(call)`. And every row above works the same through
`PersistentChat`, which mirrors the wrapper method for method.

`begin` is a reset: it drops the history, the registry it was given, and any
turn a call left open.

### What fails at startup

Whatever can be settled once is settled while the wrapper is being built, so a
deployment that could never work says so at startup rather than several turns
into a conversation. None of it is re-derived per turn.

| settled at startup | what finding it out later would cost |
|------|------|
| every `Config` field, and the combinations no call could succeed under | every `reply_without_tools`, or every `begin`, raising |
| `model_path` names a file | a failure from inside the loader rather than a path you can read |
| each KV-cache edit, performed once on the empty cache | a build that cannot shift fails at the first eviction, a whole conversation in |
| the cache context answers everything the wrapper and its template resolution call on it | the same, for a context of your own |
| the template fragments round-trip through the model's own template, and the turn-terminator is one of its tags | a reply the model never ends, and a cache encoded as it was never trained |
| tool calling, all four steps, as `can_call_tools` | a tool declared, called, and only then found unanswerable |

What is left is what only a conversation can know: whether a request still fits
(`ContextOverflowError`), whether the model calls a tool that exists, and what a
tool does when it runs.

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
| `can_shift` | whether eviction can shift surviving KV in place. `False` (compact SWA, recurrent caches) means eviction re-prefills the survivors instead: correct, but one long prefill the first time history crosses the threshold, and after every reply that is kept in another form than it was generated in |
| `can_call_tools` | whether tools can be used with this model, every step of the way: its template describes a call, it renders the declarations that tell the model a tool exists, the argument spelling was recovered, and it renders a block to answer one in. Settled at construction, and exactly the condition `begin(tools=...)` checks |
| `awaiting_tool_result` | whether a call is still owed a result |
| `reply_open` | whether a reply is unfinished, so `reply_without_tools`, `inject` and `declare_unknown_tools` refuse. Outlasts `awaiting_tool_result`: the last result empties the calls while the turn stays open until `resume_until_tool()` |

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
| `model_path` | `None` | Path to the GGUF file. Loading refuses without one, and refuses a path that names no file |
| `temperature` | the model's | `<= 0` selects greedy decoding: `top_k`, `top_p`, `min_p` and `seed` then have no effect, while the three penalties still apply |
| `top_k` | the model's | Top-k cutoff; `<= 0` disables |
| `top_p` | the model's | Nucleus cutoff |
| `min_p` | the model's | Drop tokens below `min_p` times the top token's probability; `0` disables |
| `repetition_penalty` | the model's | Applied over `penalty_window_tokens`; `1.0` disables |
| `presence_penalty` | `0.0` | Flat penalty on any token already in the window; `0` disables |
| `frequency_penalty` | `0.0` | Penalty scaled by how often a token occurs in the window; `0` disables |
| `penalty_window_tokens` | `64` | How far back the three penalties look, within the generation being sampled: each starts with an empty window, and a turn resumed after a tool result is a generation of its own |
| `seed` | `None` | Sampling RNG seed in `[0, 2**32 - 2]`, which every generation starts from, for reproducible generation. `None` draws a fresh one per generation |

The five fields marked "the model's" are left `None` and take the model's own
recommendation when it loads, unless you set them. That comes from one source,
taken whole: what the GGUF states under `general.sampling`, else the model's
add-on file, a field it leaves out being switched off; with neither,
llama.cpp's defaults (temperature `0.8`, `top_k` `40`, `top_p` `0.95`, `min_p`
`0.05`). An add-on file is JSON: the one next to the GGUF, with its name and
`.json`, or one shipped in `llama_chat/families/` that names the model's chat
template by its SHA-256. It holds `sampling`, with any of the five fields,
`call` (see "Tool calling"), `templates` and `source`, and is checked when the
model loads; anything else in it is refused.

```json
{"templates": ["<sha256 of the chat template>"],
 "source": "Qwen/Qwen3-8B, generation_config.json",
 "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}}
```

**Context management**

| field | default | meaning |
|------|------|------|
| `context_size` | `4096` | The hard limit in tokens: generation never crosses it |
| `evict_above_fraction` | `0.75` | Fraction of `context_size` above which the oldest messages are evicted. Must be in `(0, 1]`, and must come to at least one token |
| `max_history_messages` | `None` | The most non-system messages to keep, OR'd with `evict_above_fraction` as a second eviction trigger. Counts one per request, one per reply, one per `inject` and one per `declare_unknown_tools`. `None` leaves the token threshold as the only trigger; otherwise `>= 1` |
| `max_reply_tokens` | `1024` | Cap on tokens per generation, capped again so the total stays inside `context_size` |
| `oversize_policy` | `"reject"` | What happens to a message that cannot fit even with everything but the system prompt evicted. `"reject"` raises; `"truncate"` clips the content and re-appends the turn-terminator tag, so the clipped turn still closes |
| `min_reply_tokens` | `32` | Refuse a request with `ContextOverflowError` when fewer than this many tokens would be left for the reply. `0` disables the guard. It has to stay below `context_size`, which no request can leave entirely free |

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
| `max_batch_tokens` | `2048` | Most tokens handed to one `llama_decode`; a longer prefill is chunked to this |
| `flash_attention` | `True` | Set explicitly either way, so a build's own default never overrides the caller. Required by a quantized KV cache |
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
`kv_cache_type` without flash attention. Two combinations are refused there as
well, because no call could ever succeed under them: a `min_reply_tokens` at or
above `context_size`, which would make every request raise, and an
`evict_above_fraction` that rounds down to no tokens, which would make every
`begin` raise.

The rest of what `ChatWrapper` takes is deliberately outside `Config`, because
none of it is a value to tune. All of it is keyword-only except `context`, the
second positional parameter:

| | |
|------|------|
| `fragments=` | the per-role template tags, for a model that ships none. See [Chat template](#chat-template) |
| `tool_calls=` | the tags a call sits between, and optionally how it spells its arguments, for a model whose template describes neither. See [Tool calling](#tool-calling) |
| `on_message=` | called with `(role, text)` for every genuine new message. See [Persistence](#persistence) |
| `on_load_progress=` | called with a fraction while the weights load. It must not raise: an exception there aborts the load |
| `context=` | a cache context of your own, in place of the one built from `Config`. See [Layout](#layout) |

## Tool calling

Register real Python functions; the model calls them by name and the reply
resumes with what they returned. Nothing about any model's call format is
written down here: the tags and the argument spelling are recovered from the
model's own chat template, the same way the per-role fragments in
[Chat template](#chat-template) are.

```python
from llama_chat import ChatWrapper, ToolRegistry

registry = ToolRegistry()

@registry.tool("Current weather for a city.")
def get_weather(city: str) -> dict:
    return {"city": city, "celsius": 18}

chat = ChatWrapper(model_path="model.gguf")
chat.begin("You are a terse assistant.", tools=registry)

print(chat.reply_with_tools("What is the weather in Munich?").text)   # the call runs inside
```

Handing `begin` the whole registry declares what that registry says should be
declared and keeps it, so what the model was told about and what can run
cannot drift apart. `reply_with_tools` is `reply_with_tools_stream`
drained, the way `reply_without_tools` is `reply_without_tools_stream`; both
return a `Reply` whose `text` is the reply. Use `reply_with_tools_stream` where
you want the deltas:

```python
for delta in chat.reply_with_tools_stream("What is the weather in Munich?"):
    print(delta, end="")     # only prose; the calls happen inside
```

`chat.can_call_tools` says whether the model can do this at all, and answers
for the whole chain: the template has to describe a call, it has to render the
declarations that tell the model a tool exists, the argument spelling has to
have been recovered, and there has to be a block to answer a call in. A call
the library writes itself, in a probe or in a past turn
rebuilt from its text, carries an id of nine letters and digits, a shape the
templates that check an id's shape accept. All four
are settled at construction, so the answer costs nothing to ask and is available
before the first turn, and it is exactly the condition `begin(tools=...)`
checks.

You do not have to check it, either. **Tools a model could not use are refused
where you hand them over**, by `begin(tools=...)`, by `declare_unknown_tools`,
and by a `reply_with_tools` or `reply_with_tools_stream` given a registry that
holds any, naming which of the four is missing. Declaring tools and handing over
a registry to run them are the same promise, so the chain from the promise to a
result the model can read is checked when the promise is made, rather than at
the first call, where the reply would already be running. `begin` validates
before it resets, so a refused one leaves the previous conversation untouched,
and a refused `reply_with_tools` prefills nothing.

Driving the loop yourself is not refused: `reply_until_tool` and
`inject_tool_result` promise nothing, so a model that can call and be answered
but not be told stays usable that way, with the tools described in the system
prompt in your own words.

`ChatWrapper(tool_calls=...)` reaches that check, so a model whose tags or
spelling the probes could not recover is usable again as soon as you supply
them. A model with
no tool support at all stays perfectly usable without tools; only declaring is
refused.

### Registering, and where the declaration lands

`registry.tool(...)` declares a function and makes it runnable in one step,
where the function is written:

```python
@registry.tool("Current weather for a city.")
def get_weather(city: str) -> dict:
    return {"city": city, "celsius": 18}
```

The function comes back unchanged, so it stays callable as itself. The
module-level `@tool` decorator instead attaches the declaration to the function
without registering it, for a spec handed straight to
`begin(tools=[get_weather.spec])`:

```python
from llama_chat import tool

@tool("Current weather for a city.")
def get_weather(city: str) -> dict:
    return {"city": city, "celsius": 18}
```

The parameters come from the annotations either way. A signature that cannot be
described (an unannotated argument, or a type with no JSON Schema equivalent)
raises `TypeError`, and `parameters=` takes an explicit schema instead.

`registry.register(fn, description)` is the same step for a function written
elsewhere. It makes the function runnable and returns the name the model calls
it by, and that name is also how you refer to the tool afterwards.

A tool the model is never told about could never be called, so the question is
not *whether* to declare it but *where*, and that is what `in_system_prompt`
decides, on `registry.tool` and `register` alike:

| | where the declaration lands | cost |
|------|------|------|
| `in_system_prompt=True` (the default), then `begin(..., tools=registry)` | in the system turn | none extra, and no eviction can take it back |
| `in_system_prompt=False`, then `chat.declare_unknown_tools(registry)` | a message of its own | one prefill, evictable later |
| registered, and neither entry point used | nowhere; the model never hears of it | none |

The tool runs either way from the moment it is registered: placement is only
about what the model knows. Registering a name that is already taken replaces
it, its placement included.

Hand the same registry to both entry points and each takes its own half:
`begin` writes what belongs in the system prompt, `declare_unknown_tools`
writes whatever the conversation does not already declare. So after
`begin(tools=registry)` a `declare_unknown_tools(registry)` writes exactly the
tools marked for later, plus anything registered since, and calling it again
writes nothing at all. What counts as already declared is read off the messages
carrying it, so once a declaration message is evicted its tools count as
unknown again and the next `declare_unknown_tools` writes them back. The system
turn is never evicted, so what `begin` declared is known for good.

A template may instead declare tools right before the newest user turn, so that
they move on with every turn. Then the newest user
turn carries the declarations, and when a newer one comes, the turn that
carried them is decoded again without them, together with everything after it;
what lies before it stays in the cache. A history handed to `begin` carries them
on its last user turn, and `declare_unknown_tools` adds to what the carrier
holds, decoding again from there.

Two consequences worth knowing. A declaration added later is an ordinary message
and can be evicted from a long conversation, so one that has to survive belongs
in `begin`. And `begin` is a reset, so a declaration made before one is thrown
away; `declare_unknown_tools` refuses rather than writing something that will
vanish.

The registry does not track either of those. `specs_for_system_prompt()` and
`specs()` answer what the model *should* be told and where, which is what the
two entry points read; neither claims to know what is still in the
conversation, and both go on naming a tool whose declaration has since been
evicted.

### Which entry point does what

All of them keep a tool call out of the reply text, and all of them hand back a
`Reply`. What varies is **what becomes of a call**, and the name says which it
is.

| what becomes of a tool call | drained | yields text | yields text and calls |
|------|------|------|------|
| refused: the model is told the call failed and answers in words | `reply_without_tools(text)` | `reply_without_tools_stream(text)` | |
| handed to you to run | | | `reply_until_tool(text)`, then `resume_until_tool()` |
| run for you, until the model answers | `reply_with_tools(text)` | `reply_with_tools_stream(text)` | |

What comes out follows the name as well: a name without `_stream` returns the
`Reply` directly, while the `_stream` pair and both `*_until_tool` entry points
return a `ReplyStream`. It yields text deltas, and, from the `*_until_tool`
pair, each `ToolCall` as the model asks for it; the same `Reply` is on
`stream.reply` once it has ended.

```python
with chat.reply_without_tools_stream("Summarize the plan.") as stream:
    for delta in stream:
        print(delta, end="", flush=True)
print(stream.reply.n_generated_tokens, stream.reply.stop_reason)
```

`reply` is `None` until the stream ends of its own accord, so a stream you stop
reading early has none: there is no summary of a reply nobody read to the end.
The turn is closed either way. Closing is what an abandoned stream needs, since
it holds the wrapper's lock until it is collected, which is what the `with`
above is for; exhausting it does the same.

**Being shown the calls is what holds the turn open for them**, so
`reply_until_tool` and `resume_until_tool` are the only entry points that can
leave a turn unfinished. The other two rows settle every call themselves and always come
back with a finished reply, which is why a caller that never intends to use
tools needs no special handling: on a model whose template describes calls, a
hallucinated tag would otherwise leave a turn nothing could close.

Every call is reported on `Reply.tool_calls` whatever became of it, so a refused
one still says what the model asked for.

What `reply_without_tools` guarantees is that **nothing runs**, not that
exactly one generation happens: refusing a call costs a second one. It holds
even for a registry `begin` was given and remembered, and that is the whole
reason the pair exists: a turn that has to be free of side effects, in a
conversation that otherwise uses tools. Where no registry exists anywhere,
`reply_without_tools` and `reply_with_tools` are the same call and it makes no
difference which you write; they part on the turn where something could run, and
the method you chose is what decides it, rather than anything inferred from the
state of the conversation.

### Driving the loop, or driving it yourself

- **`reply_with_tools(text)`**: `reply_with_tools_stream` drained, for a
  caller that does not want the deltas.
- **`reply_with_tools_stream(text, registry=None, max_calls=None)`**: the whole
  loop, and `max_reply_tokens=`, `stop=` and `cancel=` besides, which bound
  every generation of the exchange rather than the exchange itself. Generate, run whatever the model called, feed the results back, and
  resume until it answers. Yields text deltas, and its `reply` sums every
  generation. Where the text before a call and the text after it would join
  without whitespace, a blank line goes between them. The registry defaults to the one `begin` was handed. By default
  the turn may call as often as the model asks, so the only bound is the
  context: each call and its result stay inside the open turn, and a model that
  never stops calling fills it, ending on a `ContextOverflowError` with the turn
  open. `max_calls=n` puts a ceiling on it instead; the last permitted round
  then runs with the call tags withheld, so the model cannot ask for another
  one. Without that withholding a ceiling would stop the loop but not the turn:
  a turn that ends on a call is deliberately left open, so it would never
  close.
- **`reply_until_tool(text)`**: the same generation, yielding `str` deltas and
  then a `ToolCall` for each call the model made. Use this to run the loop
  yourself: answer with `inject_tool_result(call, result)` once per call, in the
  order they were asked (`call` may also be just the call's name), then call `resume_until_tool()`. It returns how many
  messages that cost, as `inject` does, which is none while a turn is open. The
  order is enforced, because the name is what says which call is being
  answered.
  `chat.awaiting_tool_result` says whether anything is still owed, and
  `chat.reply_open` whether the turn is still open.

```python
with chat.reply_until_tool("What is the weather in Munich?") as stream:
    for event in stream:
        if isinstance(event, ToolCall):
            chat.inject_tool_result(event, registry.call(event))
        else:
            print(event, end="", flush=True)

with chat.resume_until_tool() as rest:          # the turn stays open until this
    for delta in rest:
        print(delta, end="", flush=True)
```

The calls arrive after the deltas, and the wrapper's lock is already released by
then, so answering one inside the loop is safe. `registry.call(event)` is one
way to run it; anything that returns a value will do, and a call the model
invented is yours to refuse with an error result of your own.

A history that called tools goes into `begin` as well, for a conversation that
is read in again: each assistant turn that made calls is one `ToolTurn`, its
steps in the order they were written, the model's text as `str`, each call as a
`ToolCall` and each result as a `ToolResult(name, result)`:

```python
chat.begin(system, [
    ("user", "What is the weather in Munich?"),
    ToolTurn(("Let me look.", ToolCall("weather", {"city": "Munich"}),
              ToolResult("weather", {"celsius": 21}), "It is 21 degrees.")),
    ("user", "And tomorrow?"),
], tools=registry)
```

The turn is written as the model's own template writes a past turn, so the
order and the spelling are the template's: rendered with a marker for each
piece of the turn's own text, its text, its string arguments and its results,
which are then tokenized with special-token parsing off, while the template's
tags are tokenized with it on and the call tags become the single tokens a
generated call holds. A history that ends on a result leaves that turn open, as
a result written into a live turn does, and `resume_until_tool()` goes on with
it.

`reply_with_tools` returns one `Reply` for the whole exchange, and
`reply_with_tools_stream` carries the same one on its `reply`: the counts are
sums over every generation, `stop_reason` is the
last generation's, and `tool_calls` holds every call the model made in order.
`tool_limit_reached` says whether a `max_calls` you set ran out, so the final
generation ran with the call tokens withheld; it is always `False` where none
was set and where no call was asked for.

One outcome is visible only on that summary: when `tool_limit_reached` is set,
the model was cut off from a tool it wanted rather than choosing to stop.

A call whose body could not be read is the one the summary cannot show you. The
model is told and answers in words, so the exchange ends on `"end_of_turn"` with
no calls to report. `can_call_tools` is the durable signal: it is `False` for a
model whose format was never recovered, which is when every call takes that
path.

A turn `reply_until_tool()` stopped on a call stays open until
`resume_until_tool()` closes it, even once every result is in. Until then
`reply_without_tools`, `inject` and `declare_unknown_tools`
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
template ever produced, so such a turn cannot be continued: start a new
conversation with `begin()`.

A call never appears in the reply. Its tokens reach the cache, because the model
has to see its own call when it reads the turn back, but they are not yielded as
deltas and not part of `Reply.text`, so a consumer that renders the deltas as
they arrive never renders the call syntax.

### Every failure is told to the model

Whatever goes wrong with a call, the model is handed a result saying so, in its
own template's result block, and the reply carries on from there. The message
has three parts: that the call failed, why, and what to do next.

| what went wrong | the reason it is given |
|------|------|
| the call's body could not be read | fixed text saying so |
| no tool can be run, which is every call under `reply_without_tools` | fixed text saying so |
| the name is not registered | the name it asked for |
| the arguments do not fit the function | Python's own binding error |
| the tool raised | the exception's type and message |
| a `max_calls` ceiling is spent | the limit it ran into |

The instruction is `try again` where another call is still allowed and
`do not call again` where none is. It says only that, because what to do
instead needs no saying: a model that cannot call a tool can only answer.
Which one it gets is decided after every call in the round has been settled,
because the last of them is what settles the budget, so the same failure ends
in `try again` while a call remains and in `do not call again` once a ceiling
is spent. Telling a model to retry where the call tags are about to be withheld
would ask for the one thing it cannot do.

An unreadable call is retryable like any other, and spends the budget like any
other, so under a `max_calls` ceiling a model that cannot write a readable one
gives up after that many attempts rather than looping. Under
`reply_without_tools` the retry is skipped, since a second call could only be
refused as well. It is the one failure whose
result carries no instruction: it is written while the reply is being recorded,
before the round settles the budget, so there is nothing true to say yet.

A result is a failure if it is a dict carrying `error`, so a tool that reports
its own failure that way is presented as one too. Anything else reaches the
model untouched.

Two failures cannot be told, because there is no channel for them: a template
that renders no result block has nowhere to put one, and a context with no room
left cannot hold even an empty one. Both close the turn instead.

### What is recovered, and what happens when it cannot be

The call tags are found by probing the model's template, then resolving each
one to the single token it stands for. A tag counts as such if it is either a
control token or **a token added by a fine-tune**. Both are needed: a fine-tune
commonly adds the tags around a tool call as ordinary added tokens, for which
`llama_vocab_is_control` returns false, so a check that accepts only control
tokens turns tool calling off without saying so.

A call's opening tag is the first tag before the name, and what the template
writes between that tag and the name is fixed text the model writes too; what
it writes before the tag belongs to the reply. Its closing tag is read
from a turn that makes two calls: the last tag between the first call's
argument and the second call's opening. The first tag after the arguments is
not enough, because a template may close each argument with a tag of its own,
`<arg_value>v</arg_value></tool_call>`. Where the opening begins with a tag
written once for all the calls of a turn, `<tool_calls><tool_call>`, the second
call is found by the shortest end of the opening that starts with a tag. Where a
template writes no tag between two calls, the closing tag is the first after
the last argument, which may be the token that ends the turn. A turn that calls
may open without the content tag a reply in words opens with; the call then
begins after the tags the two share.

Where the tag a call opens with also marks every reply, as in a template whose
few tags open and close the message, the answer and the call alike, a call is
recognized by its texts instead: the text it opens with, tags and plain text
together, and the text that ends it, which a call with an argument and one
without both go on with. Such a reply then ends where its closing parts from a
call, once that much of the closing is generated; those tokens leave the cache
again and the closing is written whole.

A call may also open with plain text and no tag at all, such as
`<function name="` in a template whose vocabulary holds no tag for it. Such an
opening counts only at the start of a reply or right after a tag, so a reply
that mentions it in a sentence goes on as text. Where a template writes no two
calls in one turn, the text that ends a call is what a call with an argument and
one without both end with. A plain-text opening that begins with the words of a
reply's own opening, such as a role name in its header, is refused: a model may
write its header in another order than its template does, so the template
cannot tell how a call it generates begins. Where a template names the function
twice, once in the turn's header and again in the call's body, the call opens
with what lies between the two; the header, like any header naming a reader
other than the answer's own, is never shown. Where a model ends its turn inside
such a call, the call ends there, closed as the template closes one. A result
the template writes between quotes is escaped as a JSON string, as the template
escapes it.

Where the template cannot tell how a model writes a call, because the model
writes it in another form than its template shows a past one in, the model's
add-on file says so, under `call`: the `opening` a generated call starts with,
the `name_close` after its name, and the `closing` that ends it, which the
template must write after a call too. The arguments after `name_close` are read
as JSON.

A template that declares tools somewhere other than the system turn, such as in
the first user turn, may have a variable of its own that moves them there. A
variable the template reads qualifies where setting it changes nothing in a
conversation without tools and puts the declarations into the system turn; it is
found when the model loads and passed with every render.

The argument spelling is recovered from four probes that differ only in their
content, so each delimiter is found by comparing the results. A body that is
JSON is read directly instead, because it already carries the function name and
the arguments, under `name` or `tool_name` and `arguments` or `parameters`. Together the two cover JSON objects and arrays,
`name{key:value}` with quoted strings, `name(k="v")`, attribute-style XML,
`<function=name><parameter=k>v</parameter>`, and keys and values in tags of
their own right after the name, `name<arg_key>k</arg_key><arg_value>v</arg_value>`,
including nested objects, lists, and separators inside quoted values. A type
written between a key and its value, `key="k" type="string">v`, is recovered
from where the render of a string argument and that of a number differ. A value
of the string type stands as its text, and any other is read as JSON. A value
closed by a tag that names its own key, `<city>Berlin</city>`, is read up to that
tag, and arguments written after the name as one JSON object are read as JSON.

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
still detected, so its syntax never reaches the reply text, and the model is
told that its call could not be read, so it can write it again or answer in
words instead.
Nothing is guessed at. For such a model, pass
the spelling in yourself with
`ChatWrapper(tool_calls=ToolCallFragments(call_open=..., call_close=...,
body=CallBody(...)))`, which makes `can_call_tools` `True` again, the same way
`fragments=` covers a template that cannot be read.

**Every call is answered, readable or not.** A result block has to name the
function it answers, and here that is exactly what could not be read, so the
block names `unknown` and its message says what happened. Writing nothing would
leave a call with no result in the history, which no template produces and the
model would read back on every later turn.

A model that writes a batch of calls as a JSON array has all of them read, and
each owes a result. The results are written as the template writes the results
of calls made together: one after another, with what closes the block, and the
generation prompt, only after the last. Any other shape ends at its closing
tag, so generation stops after one call either way.

### A resumed turn opens as a new one does

A reply resumed after a tool result opens the way a new reply opens after its
tag, unless the result's wrapping already ends that way. A template whose
generation prompt carries an empty thought, to keep the model from reasoning,
may leave that out after a result; left alone, the model would then open a
thought of its own.

## Reasoning

A template that turns reasoning on through the variable chat templates share
for it, `enable_thinking`, and that marks a thought, is read for both: what the
system turn, its opening and its closing, and the generation prompt are with
reasoning on and off, and the tags around a thought. Where `enable_thinking`
changes nothing, `reasoning_effort` is the switch if its setting changes what
follows the user turn: `"high"` turns reasoning on, and the template's own
default stands for off. The tags around a thought are read from a reply that
holds one; where a template writes no thought into a reply, from the empty
thought that only the generation prompt without reasoning ends in; and where it
removes one from a past reply instead, from the model's own tags: the closing
tag is the one after which a past reply keeps only its answer, the opening tag
the one of the same name without the slash. `ChatWrapper.can_reason` says
whether that succeeded.

Reasoning belongs to the conversation, because a template may switch it in the
system turn, so it is chosen per `begin`:

```python
chat.begin("You are a helpful assistant.", reasoning=True)
reply = chat.reply_without_tools("Why is the sky blue?")
reply.text                  # the answer alone
reply.n_reasoning_tokens    # about how many tokens the thought took
```

Every stream yields the answer alone, with or without reasoning, so a thought
the model has without being asked is kept out too, as is the thought of a
model whose template always opens one and has no switch. Stop strings are
matched against the answer only. A template that routes replies through
channels ends its generation prompt before the channel's header, which the
model writes itself; that header is kept out of the answer as well, and a
thought may be read from either field templates use for it, `reasoning_content`
or `thinking`, or from a content block of type `thinking`. A template that
prints such a block as text instead of reading it marks no thought. A past
reply of a conversation that reasons opens as the template writes it then,
which may differ from its opening without reasoning, such as by an empty
thought. A finished reply is kept as the template
writes a past reply: without its thought where the template leaves a past
thought out, and without what a generation prompt adds, such as an empty
thought, and in the tokens its text tokenizes into, where the model generated
a word in other pieces. A reply whose thought the template keeps stays as it
was generated, thought included, in the tokens its text tokenizes into. Where that differs from what
was generated, the reply is decoded again in that form, from the first token
that differs, before the next request, under the same lock. The post-reply eviction is sized by that form, so a long
thought evicts nothing, and runs first, so the decode carries its shift. A
reply left with no answer, all thought or cut before its first word, is
removed where the template writes two user turns in a row, and kept as an
empty reply where it does not. A turn that called tools keeps the tokens it was
generated as.

## Persistence

The KV cache only holds the recent window that fits `context_size`; eviction drops older
turns. To keep the *full* history, `PersistentChat` wraps `ChatWrapper` and mirrors
every message (user, assistant, and injected context, so everything but the
system prompt) into a durable store, then replays the prior turns at `begin`:

```python
from llama_chat import PersistentChat, InMemoryStore

chat = PersistentChat(InMemoryStore(), context_size=8192)  # kwargs forward to Config here as well
chat.begin("conversation-42", "You are a helpful assistant.")  # loads prior turns
chat.reply_without_tools("What did we decide last time?")   # auto-persisted
```

A store is any object satisfying the `ConversationStore` protocol, which is
three methods keyed by conversation id:

```python
class ConversationStore(Protocol):
    def load(self, conversation_id: str) -> list[tuple[str, str]]: ...  # (role, text), oldest first
    def append(self, conversation_id: str, role: str, text: str) -> None: ...
    def replace_last(self, conversation_id: str, text: str) -> None: ...  # after a cut
```

`InMemoryStore` is a reference implementation (not durable); back it with SQLite,
Redis, or files by implementing those three methods. All are checked where the
store is adopted, because a store is written to mid-turn, and every entry point
that would write a message refuses before prefilling anything until `begin` has
selected a conversation. Capture is driven by an optional
`on_message(role, text)` hook on `ChatWrapper` itself, so you can persist to your
own sink without `PersistentChat` if you prefer.

`PersistentChat` forwards the whole wrapper API, tools included:
`begin(cid, prompt, tools=registry)`, then `reply_with_tools` /
`reply_with_tools_stream`, or the manual loop. A call and the result it was
given are not messages, so a tool turn is persisted exactly like any other: one
user entry, one assistant entry holding what a reader saw. Declarations are not persisted either: they
belong to the system turn, which is rebuilt on every `begin`.

## Chat template

The template must match the model, because its turn-terminator is what lets the
model stop generating. A mismatched template tokenizes as plain text, so the
model never emits its end-of-turn token and runs to the token cap every
generation.

The template is expressed as per-role **fragments**: the literal text wrapped
around each message, so one turn renders as `prefix + text + suffix`. Content is
stripped first when `trim_content` is set, matching templates that apply Jinja
`| trim`; it is `False` for templates that emit content verbatim. A template
that strips a reply or a system prompt differently from a user turn says so in
`reply_strip` or `system_strip`.
The assistant tags are those of a past reply; what the generation prompt holds
where it is more than that tag is `generation_prompt`.

Two user turns in a row, as an injected message or a request after a reply that
was removed leaves them, are written as the template writes them: a template
may close the first with tags of its own, such as the opening of a reply that
never came, and those are written at its end when the second follows, so they
leave with it when it is evicted. A template that refuses such a pair, or
merges it into one turn, gets none: writing one raises `ValueError`.

**The fragments are derived from the model automatically.** At construction the
wrapper reads the model's own chat template from the GGUF
(`tokenizer.chat_template`) and recovers the per-role fragments from it. It works by rendering
short probes through the model's template with sentinel content and splitting the
render on the sentinels to recover each role's `prefix`/`suffix` (the inverse of
the validation probe below), plus whitespace-padded probes to detect
`trim_content`, `reply_strip` and `system_strip`. The user and assistant tags are read from a
conversation of two user turns, because a first turn may carry text the
template writes once at the start, such as a default system block, and a newest
reply may carry what only the newest one does. A past reply may close with a
tag that only a following turn shows; the next user turn then opens with the
longest end of what follows the reply that a first user turn's opening also
ends with. What a template writes once after the last message of a
conversation without a generation prompt, such as an end-of-text token, belongs
to no turn and is left out of every probe. Every probe renders with
reasoning off. A conversation begun with an empty system prompt
has no system message, and opens as the template opens one without, with any
tools declared where the template declares them then. A system prompt is
refused where the template has no system turn of its own and writes one into
the first user turn, because a prompt kept for the whole conversation cannot
be. A system message added later, by `inject(..., "system")` or
`declare_unknown_tools`, takes the tags the template gives a later system turn,
and is refused where it gives none. The `system` role has no portable ground truth, because some models
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

For a model that ships no template *and* calls tools, the call tags are their
own argument:

```python
ChatWrapper(fragments=..., tool_calls=ToolCallFragments(
    call_open="<|tool_call|>", call_close="<|/tool_call|>",
    body=CallBody(...)))     # body optional: recovered when left out
```
 Each tag is taken as it stands in a render, so
either may carry fixed text around the tag itself; the wrapper narrows it to the
one token the tag is.


The load-time checks make a mismatch fail loudly instead of degrading silently:

- **At construction** the turn-terminator (`assistant_suffix`) is validated
  against the model's vocabulary: if it does not tokenize into the model's own
  tags, which is the silent mismatch described above, construction raises. A
  reply ends at the first of those tags, so a model whose closing takes several
  writes none of them into its answer. Inside a thought whose closing begins
  with that tag, the tag is the closing's and the reply goes on.
- **At construction** the fragments are checked against the model's own chat
  template read from the GGUF: a conversation of a system prompt, a past reply
  and a newest user turn, followed by the generation prompt, is rendered both
  ways, in each reasoning mode the model has, and must tokenize identically.
  The content is padded with whitespace, so a wrong `trim_content` is caught
  too, and the fragment side is tokenized exactly as real prefill is (tags
  special-on, content special-off). The system turn is left out where the
  template has none of its own. The check is skipped for models that ship no
  chat template.

### Special tokens in message content

*Message content* is tokenized with special-token parsing **off**, while the
structural template tags (the fragment prefix/suffix) are tokenized with it
**on**. So a user typing a literal template
tag (e.g. `<end_of_turn>`) yields ordinary text tokens that can never become a
control token, while the wrapping tags still tokenize to their real control
tokens. This closes token-level prompt injection (forging turn boundaries) at
the source: it is robust even on models that mis-type a control-looking token
in the GGUF or on llama.cpp builds lacking the `llama_vocab_is_control` symbol.

The plain text a template writes between a tag and the content, such as a
newline, is tokenized together with the content, also with special-token
parsing off, because a tokenizer may merge across that boundary: a system
prompt ending in a space followed by a blank line yields `' \n\n'` as one token
in a render of the whole conversation, and the cache has to hold the same.

## Verify

```bash
# Pure bookkeeping + end-to-end logic, no model needed:
python -m pytest -q

# Against a real model (logs prefill reuse + eviction stress):
python examples/demo.py /path/to/model.gguf                  # CPU
python examples/demo.py /path/to/model.gguf --gpu-layers -1  # GPU
```

`tests/test_real_templates.py` runs a real chat template of every family in
`tests/templates/` through the check a model meets at load, with reasoning on
and off. Each file there names the model file it was read from.

The test suite uses a model-free fake cache context whose `prefill` asserts
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
| `llama_chat/template.py` | `TemplateFragments`, `Reasoning`, `TemplateFormatter`, `ToolCallFragments`, `CallBody`: what a turn and a tool call are wrapped in, and how reasoning is turned on |
| `llama_chat/template_extract.py` | `extract_fragments`, `extract_reasoning`, `extract_tool_fragments`, `extract_call_body`: recover all of that from the model's GGUF template |
| `llama_chat/reasoning.py` | `ThoughtFilter`: telling a reply's thought from its answer as it streams |
| `llama_chat/resolved_template.py` | `ResolvedTemplate`: one model's template, recovered or supplied, validated against it, with the four tool-calling steps settled |
| `llama_chat/_backend.py` | `Backend`: version-stable adapter over `llama_cpp.*`, which renames its symbols across releases. The only file in the package that names it, pinned by a test |
| `llama_chat/context.py` | `KVContext`: tokenize / prefill / streaming generate / evict |
| `llama_chat/tools.py` | `ToolRegistry`, `ToolSpec`, `ToolCall`: declaring tools and running the ones the model calls |
| `llama_chat/call_parsing.py` | `parse_calls`: reading a call back out of what the model generated, JSON or this model's own delimiters |
| `llama_chat/wrapper.py` | `ChatWrapper`, `Reply`, `ReplyStream`, `ContextOverflowError`: `begin` / `inject` / `reply_without_tools` / `reply_without_tools_stream` / `reply_with_tools` / `reply_with_tools_stream` |
| `llama_chat/persistent_wrapper.py` | `PersistentChat`, `ConversationStore`, `InMemoryStore`: the same API, with every message mirrored to a durable store |

Everything in that table is importable from `llama_chat` itself, apart from the
extraction functions, `TemplateFormatter`, `ThoughtFilter` and `ResolvedTemplate`: the wrapper runs those for you at
construction, and `llama_chat.template_extract` is where to reach the extraction
functions to run against a template directly. `n_newest_that_fit` is the pure
function `begin` uses to decide how much history fits, exposed so that decision
can be made before committing to it.

The bookkeeping types are exported for one purpose: `ChatWrapper(context=...)`
takes a cache context of your own in place of the one built from `Config`, and
`Eviction` is what such a context is handed to apply. No subclassing and no
registration: an object that answers what the wrapper calls on it is a cache
context, and `tests/fake_context.py` is a worked one, over a plain list of ints
with no llama.cpp in it. What it has to answer is checked where the context is
adopted, so a missing name is a `TypeError` from the constructor rather than an
`AttributeError` from the first eviction that reaches it.
