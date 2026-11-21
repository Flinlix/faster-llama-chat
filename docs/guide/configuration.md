# Configuration

`Config` holds every setting except what an add-on file gives and what one
request passes. It checks every value on construction, its type and its
range, and cannot be changed afterwards: a whole number is taken where a
fraction is asked for, but `True` is never taken for a number, nor a number
that is not finite.
`ChatWrapper.config` is the one the wrapper runs with: a sampling setting left
unset holds the value the model supplies, whether the wrapper loaded the model
or was given a cache context of your own.

**Model and sampling**

| field | default | meaning |
|------|------|------|
| `model_path` | `None` | the GGUF file; loading refuses without one |
| `temperature` | the model's | `<= 0` selects greedy decoding |
| `top_k` | the model's | `<= 0` disables |
| `top_p` | the model's | nucleus cutoff |
| `min_p` | the model's | drops tokens below `min_p` times the top token's probability; `0` disables |
| `repetition_penalty` | the model's | over `penalty_window_tokens`; `1.0` disables |
| `presence_penalty` | the model's | flat penalty on a token already in the window; `0` disables |
| `frequency_penalty` | the model's | penalty scaled by how often a token occurs in the window; `0` disables |
| `penalty_window_tokens` | the model's | how far back the penalties look, within one generation; at least `0` |
| `seed` | `None` | in `[0, 2**32 - 2]`, which every generation starts from; `None` draws a fresh one |

The eight marked "the model's" take, unless set, the model's own
recommendation, value by value: its [add-on file](add-on-files.md)'s first,
then what its GGUF states for a value the file leaves out, then llama.cpp's
default (`temperature` 0.8, `top_k` 40, `top_p` 0.95, `min_p` 0.05,
`repetition_penalty` 1, `presence_penalty` and `frequency_penalty` 0,
`penalty_window_tokens` 64). A penalty window the GGUF states as `-1`, which llama.cpp reads as
the whole context, is taken as `context_size`. Any other value the GGUF states
that `Config` would refuse stops the load with a `ValueError` naming the GGUF
as its source, unless `Config` or the add-on file replaces it.
`Sampling(...)` passed to one `reply` overrides any field of this table but
`model_path` for that request. `Sampling` refuses, where it is built, any
value `Config` refuses.

**Context management**

| field | default | meaning |
|------|------|------|
| `context_size` | `4096` | the hard limit in tokens |
| `evict_above_fraction` | `0.75` | the fraction of `context_size` above which the oldest exchanges are evicted, in `(0, 1]` |
| `max_history_exchanges` | `None` | a second eviction trigger: the most exchanges kept, at least `1`. An exchange is one user message and everything after it up to the next, the unit an eviction drops; what a history holds before its first user message counts as one more. The question of a `reply` may open one exchange more while its reply runs, and the oldest is dropped once the reply ends, so that no eviction for this limit delays a reply |
| `max_reply_tokens` | `None` | cap on the tokens of one reply, every generation of it together, a thought included; `None` sets none, so a reply may run until the context is full |
| `max_thought_tokens` | `None` | the most tokens one thought may take; a thought that reaches it is closed as the template closes one, and the answer follows; `None` sets no cap; `reply(max_thought_tokens=...)` sets it for one request |
| `min_reply_tokens` | `32` | the room a reply keeps: a `begin` that leaves less beside a short question is refused with `ContextOverflowError`, and a longer question is cut to leave it; never less than the shortest reply that holds an answer takes, which is all `0` keeps |

**Replies**

| field | default | meaning |
|------|------|------|
| `min_answer_tokens` | `1` | the end of the turn is not drawn until a reply holds this many tokens of answer, a thought left out, or a call, and a stop string counts only once the answer holds text, so a model that would end at once answers; `None` lets it end at any token |
| `answer_room_tokens` | `1024` | the tokens a reply keeps for its answer where the model thinks first: a thought that reaches the rest of the generation's room, at most half of it, is closed as the template closes one, so the model answers; `None` lets a thought run until the room is spent |
| `max_calls` | `10` | the ceiling on the calls one turn run against a registry may use, where a request gives none; in the last permitted round the model can no longer call, so it answers; `None` sets none |
| `tool_timeout_s` | `60.0` | how long a tool a registry runs may take before it is answered with a `tool_timeout` failure; `None` waits for it |
| `unread_reply_timeout_s` | `10.0` | how long an action waits for a reply nobody reads on before it ends that reply as closing its stream does; `None` waits until it is read, closed or collected |

**Backend and hardware**

| field | default | meaning |
|------|------|------|
| `gpu_layers` | `-1` | layers offloaded to the GPU: `-1` for all, else at least `0`; `0` uses no GPU at all, the processor alone running every step |
| `kv_cache_on_gpu` | `True` | each layer's KV cache where that layer runs, in VRAM for the layers `gpu_layers` offloads; `False` keeps all of it in host RAM |
| `compact_window_cache` | `False` | `True` gives each layer with a sliding window a cache of its window alone: much less memory, but a cache that cannot shift, so an eviction reads the history it keeps again; see [Memory or speed: the window cache](#memory-or-speed-the-window-cache). Changes nothing on a model without such layers |
| `decode_threads` | `None` | CPU threads for generating a reply, a token at a time, at least `1`; `None` uses one per physical core of the processors the process may run on, at most as many as its CPU quota allows |
| `prefill_threads` | `None` | CPU threads for prefill, the decode of tokens the cache is given (a question, or what is decoded again), at least `1`; `None` means as many as `decode_threads`, and so one per physical core where that is not set either |
| `max_batch_tokens` | `2048` | most tokens in one decode; a longer prefill is split |
| `micro_batch_tokens` | `None` | most tokens the engine computes at once, at most `max_batch_tokens`; `None` leaves the engine's default. A larger one can read a long prefill faster and takes more memory on the device |
| `flash_attention` | `None` | `True` or `False` forces flash attention on or off; `None` leaves it to llama.cpp, except under a quantized KV cache, which requires it and turns it on |
| `kv_cache_type` | `"f16"` | `"f16"`, `"q8_0"`, `"q5_1"`, `"q5_0"`, `"q4_1"`, `"q4_0"`; a quantized type requires flash attention, so is refused with `flash_attention=False` |
| `kept_conversation_bytes` | `0` | the most host RAM, in bytes, the caches of conversations a `begin` switched away from may take once kept, each with every position it holds, so that going back to one decodes only what changed since; the oldest goes first, and while one is saved it takes its own size beside them; only a cache that can shift or cut keeps any; `0` keeps none |
| `check_cache` | `False` | for finding faults: after the work after every reply and before every generation, compare the cache with a fresh begin of its conversation, and raise `InternalError` where they differ; each check renders and tokenizes the whole conversation |

## Memory or speed: the window cache

Some models attend over the whole conversation in a few layers and over a
sliding window of the newest tokens in the rest. With
`compact_window_cache=False`, the default, those window layers keep a cache of
the whole context anyway, so the cache can shift: an eviction removes the
oldest exchanges and moves the rest down in place, and costs almost nothing.
With `compact_window_cache=True` they keep only their window, which takes much
less memory on such a model, but the cache can no longer shift. An eviction
then drops the history well below the eviction threshold and reads the
conversation it keeps again from the start, every few turns rather than every
turn.

That reading happens in the work after a reply, while the reply is read or
spoken, so the wait before the next first word does not change. These do:

- An action that comes during that work waits for the rest of it. Shortening a
  reply right after it ended, as when the listener interrupts speech at once,
  is such an action.
- Shortening a reply by more than the window layers keep beyond their window
  (about one micro-batch) reads the whole conversation again within the call
  that shortens it.
- An eviction an action itself needs, because `max_history_exchanges` is
  reached or a question does not fit, reads the kept history again before the
  reply, as on any cache that cannot shift.
- Going back to a kept conversation reads it again from the start wherever
  the switch cuts back further than that, so `kept_conversation_bytes` saves
  little with the compact cache.
- A machine too slow to read the kept history again within the time an answer
  takes to read or speak turns the work after a reply into a pause the next
  question meets.

On the full-size cache too, the engine reuses the cells its window layers no
longer attend over when it needs room. After an eviction those layers may hold
only a little more than their window, so a reply shortened further back than
that leaves them a gap of usually a few dozen tokens. The library does not
read the conversation again for it, since that would take far longer on a slow
processor than the gap costs in the answer. The compact cache checks every cut
and reads the conversation again where its window falls short. Where one
action rewrites more than the compact cache keeps beyond its window, as where
a long thought is removed from the reply before it, every such turn reads the
conversation again, usually after the reply.

Choose the compact cache where memory is short and the machine reads the kept
history again in about a second, as a GPU does with a model that fits it; in a
spoken conversation, the time the listener takes to say the next question
covers the reading. Keep the full-size cache on a machine that reads slowly,
such as a small board's processor, where it costs little memory anyway. The
measurements behind this are in the decision record
[A compact cache for sliding-window layers](../decisions/0059-a-compact-cache-for-sliding-window-layers.md).

`ChatWrapper` also takes, outside `Config` because neither is a value to
tune: `context=` (a [cache context of your own](cache-context.md)) and
`on_load_progress=`, called with a fraction while the weights load, which must
not raise.

**Logging.** Everything llama.cpp writes, and what the library itself has to
say, goes to Python's `logging`, under the logger `llama_cpp_conversation` and
llama.cpp's lines under its child `llama_cpp_conversation.engine`, one record
per line: llama.cpp's errors as `ERROR`, its warnings as `WARNING`, its
load-time listings as `INFO` and `DEBUG`. With no logging configured, Python
shows warnings and errors on stderr, and nothing below them whatever level a
logger is set to; showing `INFO` and `DEBUG` lines takes a handler, such as the
one `logging.basicConfig()` puts on stderr. The library's own levels mean one
thing each: `INFO`, the conversation changed in a way the caller did not ask
for, such as history that went ("the 80 oldest history message(s) went so that
half of the 8192-token context stays free at a begin"), a call left out, a
result cut to fit, a reply taken back, or an open turn a question closed;
`DEBUG`, a slower path was taken, such as the cache decoded again; `WARNING`,
something to fix, such as an add-on file that is not used. Each `INFO` change
is also in the `Report` the action returns (`begin`, `inject`,
`inject_tool_result`) or in the `Reply`; why history went is one of
`EVICTION_CAUSES`, typed as `EvictionCause`. To show only errors, or
everything:

```python
import logging
logging.basicConfig()               # a handler on stderr that shows every level it is given
logging.getLogger("llama_cpp_conversation").setLevel(logging.ERROR)
logging.getLogger("llama_cpp_conversation.engine").setLevel(logging.DEBUG)   # every line llama.cpp writes
```

llama.cpp has one log output per process, so importing `llama_cpp_conversation`
routes it for the whole process, a model loaded by other code included.
