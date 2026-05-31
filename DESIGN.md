# llama_chat — Design Rationale

A KV-cache "scalpel" for running a fast, rolling-eviction chat loop on a
Raspberry Pi 5 with llama.cpp.

## Problem / Goals

Run a local chat assistant (voice pipeline: whisper → LLM → piper) on a
Raspberry Pi 5, where **prefill is the bottleneck** — processing the prompt is
slow on CPU, so re-processing history every turn costs seconds.

Two hard goals:

1. **Minimum prefill** — only ever process genuinely *new* tokens. The KV cache
   from previous turns must survive and be reused.
2. **Rolling eviction** — when the context fills, old messages are dropped
   automatically so the chat runs indefinitely without erroring on a full
   context.

The naive approach defeats goal 1: deleting old messages at the client/prompt
level removes tokens from the *start* of the sequence. Because the KV cache is
positional, that shifts every later token's position and **invalidates the
entire cached prefix → full reprefill every turn.** That is the single thing to
avoid.

## Things to consider

- **Prefix caching alone is not enough.** llama-server's `cache_prompt` only
  reuses the *common prefix* of consecutive prompts. The moment any non-prefix
  (old) message is dropped, the common prefix collapses to the system prompt and
  all recent history is re-prefilled — the opposite of the goal.

- **The correct mechanism is in-place KV surgery**, not prompt trimming: remove
  the oldest message's token span from the cache and *shift the survivors' KV
  positions down* (RoPE-corrected) to close the gap. Kept messages are never
  re-processed. In llama.cpp this is `llama_memory_seq_rm` + `llama_memory_seq_add`.

- **`--context-shift` (the llama-server flag) ≠ the memory C API.** The flag is a
  fragile, opt-in server feature (it can corrupt strict chat templates by cutting
  raw tokens). The underlying `llama_memory_seq_*` C API is lower-level and gives
  precise, message-aligned control. This project drives that C API via
  **llama-cpp-python's low-level bindings** (`llama_cpp.llama_memory_seq_*`),
  which keeps the whole pipeline in Python while still operating below the server
  layer. The backend resolves the symbols defensively (`llama_memory_*` →
  `llama_kv_self_*` → `llama_kv_cache_*`) so it survives llama.cpp API churn.

- **Gemma 4 is a hybrid SWA model** (sliding-window local layers + a few global
  full-attention layers; E2B uses a 512-token window). Two consequences:
  - With the **compact/pruned** SWA cache, position info is lost when the window
    slides, so shifting is *not* supported. (This is the limitation earlier
    research flagged as "Gemma 4 is the wrong tool.")
  - With the **default full-size SWA cache** (`swa_full` is the *disable* switch),
    positions are retained and **shifting works** — verified on this model:
    `llama_memory_can_shift()` returns `True` and the logs show `applying K-shift`
    on eviction, with flat per-turn timings.

- **The unavoidable caveat (architecture, not config):** Gemma 4's local layers
  genuinely forget anything older than the 512-token window, no matter how clever
  eviction is. Only the global layers carry older context forward.
  - For a **voice assistant** (short, recent-context turns) this is fine — even
    desirable, since old turns should be forgotten anyway.
  - For **long-range recall** (facts from thousands of tokens ago), a
    full-attention model (e.g. Llama 3.2 1B/3B) retains more. Switch models only
    if long-range memory becomes a requirement.

- **Chat template is model-specific, and a mismatch fails *silently*.** Gemma 4
  uses `<|turn>{role}\n … <turn|>\n` (token ids 105 / 106), **not** the
  `<start_of_turn>` markers of earlier Gemma; `<turn|>` (106) is the
  end-of-generation token. With the wrong markers the model reads them as plain
  text, never emits its end-of-turn token, and runs to the generation cap every
  turn (≈60–75 s on the Pi) instead of stopping after a few tokens. The cost is a
  quiet performance/quality collapse, not a crash — so it must be prevented, not
  merely documented. See the *template handling* point in the solution.

- **Pi 5 specifics:** 4 cores (`n_threads=4`), CPU-only (`n_gpu_layers=0`), a
  small context (2k–4k) keeps each turn's prefill and eviction fast; Q4 quant.
  Build `llama-cpp-python` **from source** with `CMAKE_ARGS="-DGGML_NATIVE=ON"`
  (see `install.sh`) — the generic wheel skips the Cortex-A76 dot-product kernels
  (`DOTPROD`) that are the main prefill speedup. Active cooling matters —
  sustained all-core load will thermal-throttle.

## Solution

A thin Python wrapper around llama.cpp that keeps one KV-cache sequence alive
across turns and exposes three actions, each prefilling as little as possible:

- **`begin(system_prompt, history)`** — full reset (process start / conversation
  switch). Clears the cache, pins the system prompt at position 0, then prefills
  as much recent history as fits under the threshold (oldest excess dropped).

- **`inject(content)`** — prefill one message as context, **without generating**
  (e.g. retrieved documents). Evicts oldest first if needed to stay under
  threshold.

- **`request(content)`** — prefill **only** the new request text, then generate.
  Evicts oldest messages if over threshold; caps generation so the total never
  exceeds `n_ctx`; records the reply so the next turn reuses it for free.
- **`stream(content)`** — the generator form of `request`, central to the voice
  use case: it yields text deltas as tokens are sampled (so TTS can speak
  sentence *n* while *n+1* is still being written), decoding bytes incrementally
  so a codepoint split across tokens is never mangled. Each token is committed to
  the cache *before* its text is surfaced, so abandoning the stream mid-reply
  (barge-in) still records exactly the tokens that reached the cache — the next
  turn stays consistent. Cache-mutating actions are serialized with a reentrant
  lock for use from a threaded server. Long prefills are chunked to `n_batch`.

**Eviction (the scalpel):** the cache for sequence 0 is a contiguous run
`[0, total)`; each message owns a `[start, stop)` span, tracked in a pure-Python
`MessageTable` that is the single source of truth and recomputes positions after
every change. To evict the oldest non-system message, `llama_memory_seq_rm`
removes its span and `llama_memory_seq_add` shifts everything after it down by
the gap (RoPE positions corrected). The system prompt is never evicted. This is
the fast path, used whenever the cache supports shifting — true for Gemma 4 under
the default full-size SWA cache (verified: `can_shift == True`, K-shift on every
eviction, flat per-turn timings).

For caches that **cannot** shift (compact SWA, recurrent/Mamba state — where
dropping tokens loses the position information shifting needs), the wrapper falls
back to a **rebuild**: it drops the oldest messages from the bookkeeping, then
re-prefills the surviving conversation once via `reset` + a single `prefill`.
Correct, just not free. The strategy is chosen automatically from
`llama_memory_can_shift()` queried at construction, so a non-shiftable model
degrades gracefully instead of corrupting the cache.

**Template handling (correct by default):** the template is auto-detected from
the model's built-in chat template (`<|turn>` → Gemma 4, `<start_of_turn>` →
Gemma 2/3, `<|im_start|>` → ChatML), or set explicitly via a preset. At load
time the chosen turn-terminator is **validated** against the model's vocabulary —
if it does not tokenize to a single special token (the silent-mismatch trap
above), construction raises rather than degrading at runtime.

**Context management state:** general context size (`n_ctx`), a threshold
(fraction of `n_ctx`), per-message token length, and each message's cache span
after prefill. Eviction keeps the prefilled context under the threshold; an
oversized single message is still accepted as long as it fits `n_ctx`, with the
generation budget shrunk to whatever room remains.

**Net result on Gemma 4 / Pi 5:** both goals met — only new tokens are prefilled,
eviction is nearly free (in-place shift, flat timings), memory stays bounded. The
only thing traded away is recall of context older than the sliding window, which
is acceptable for a voice assistant.
