# 0059. A compact cache for sliding-window layers, chosen by the caller

Date: 2026-11-21

## Context

The context is created with the full-size cache for layers with a sliding
window (0014), so that the cache can shift and an eviction costs almost
nothing. On a model with many such layers that cache takes most of the
memory. A caller with a fast machine and little memory may prefer to read the
kept history again at an eviction instead.

## Decision

- `compact_window_cache` (default `False`) gives each window layer a cache of
  its window alone (llama.cpp's `swa_full` off). The cache then reports that
  it cannot shift, and the library takes the path it has for such caches: an
  eviction drops the history well below the threshold and decodes what it
  keeps again (0023), in the work after a reply (0027).
- An action that comes during that work waits for the rest of it; no call
  tells the library that a reply was heard.
- A cut checks what the window layers still hold. A compact cache passes the
  load's probe for a cut at its end, but a cut further back than those layers
  still reach leaves them without the positions the next token attends over,
  and the engine reports no error (found by review on 2026-11-21: after an
  eviction the window layers no longer held the start of the system prompt).
  So `remove_from` empties the cache where the oldest position the window
  layers hold is later than the cut less the window, and says so: the cache
  context protocol's `remove_from` may return where the cache now ends, and the
  library reads the conversation again from there. A context that returns
  nothing is cut as asked, as before. Record 0060 says why the full-size
  cache is not checked.
- The rule is one position stricter than the attention mask needs, since the
  next token at `pos` attends back to `pos - n_swa + 1`. llama.cpp's server
  reads again at `pos_min >= max(0, pos - n_swa)` in the source of binding
  0.3.23 and at `pos_min >= max(0, pos - n_swa + 2)` in that of 0.3.37; the
  rule here lies between them.
- A model whose layers also carry a state from token to token gets no check.
  For it the engine reports as the oldest position the larger of the
  attention's and the running state's, which is the newest token, so a check
  would empty the cache at every cut, and after every restore of a saved state
  (found by review on 2026-11-21). Such a cache takes only a cut back within
  the last micro-batch, which its running state keeps snapshots of, and the
  window layers keep at least that much beyond their window, so no cut it
  takes leaves them short.
- The configuration guide states the trade-off; the numbers stay here.

## Consequences

Measured on 2026-11-21 with a context of 20,480 tokens, 600-token questions
past a full context, the same settings for both forms:

| | RTX 3090, Gemma 4 26B A4B | i9-9900K, 4 threads, Gemma 4 E2B |
|---|---|---|
| cache memory, compact against full-size | 1.0 against 4.4 GiB | 132 against 360 MiB |
| reading the kept history again, every 13 turns | 1.0 s | 31 s |
| wait for the first word (median), compact against full-size | 0.155 against 0.153 s | 3.6 against 3.8 s |
| shortening a reply right after it ended on such a turn | waits 1.18 s (0.04 s full-size) | waits 30.6 s (0.26 s full-size) |

These were measured before the cut checked the window, so an eviction there
read the history it kept but not the system turn again; with the check it
reads that too. With the check, on the 26B model: the window layers held every
position from 0 after each of 5 trims; the trim took 1.0 s, the first word
came after 0.155 s (median), a shortening during a trim waited at most 1.19 s.
Without it, on the E2B model with window layers of 2,560 cells, more than
the history a trim keeps but less than the conversation before it, they lacked
the first 31 positions, the system turn, after each of 5 trims (measured
2026-11-21).
- A cut back to a point the window layers no longer reach reads the
  conversation again from the start, as when a reply is shortened by more than
  the window layers keep beyond their window: `pad(n_swa + n_ubatch, 256)`
  cells less the window, so one micro-batch or a little more. That reading
  happens within the call that shortens the reply.
- A conversation kept in host RAM is saved with the window it held, so going
  back to it usually reads it again from the start: the kept RAM buys little
  with the compact cache.

A Raspberry Pi 5 reads about 10.7 times slower than the processor above (its
published pp512 for Gemma 4 E2B against this processor's, measured), so the
reading again would take minutes there: the full-size cache, 230 MiB more at
this context, is the better choice on it. In a spoken conversation the
listener's next question takes longer to say than the GPU's reading again.

## Alternatives rejected

- A call that says a reply was heard, the reading again waiting for it: one
  more call for every caller, for a wait that on a fast machine falls inside
  the listener's next question anyway.
- Building the trimmed conversation in a second sequence beside the current
  one: about 40 % more cache cells and two sequences throughout the library,
  for the same gain.
- The compact cache by default: a slow machine would meet pauses of minutes.
