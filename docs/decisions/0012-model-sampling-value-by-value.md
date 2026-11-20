# 0012. A model's sampling is merged value by value, and an unstated value takes llama.cpp's default

Date: 2026-11-19

## Context

Sampling values come from three places: the caller, the model's add-on file
and the GGUF's `general.sampling.*` keys. Where the GGUF stated any value, the
add-on file was once ignored whole, and every value no source stated was
switched off, so a GGUF stating only a temperature lost the file's `top_k`
and `top_p`, and a model without `top_k` sorted the whole vocabulary for every
token.

## Decision

- Each sampling field left unset in `Config` takes, value by value: the add-on
  file's value, else the GGUF's, else llama.cpp's default (`temperature` 0.8,
  `top_k` 40, `top_p` 0.95, `min_p` 0.05, penalties off, window 64).
- The model's fields are temperature, `top_k`, `top_p`, `min_p`, the
  repetition, presence and frequency penalties, and `penalty_window_tokens`.
  Presence and frequency come only from an add-on file or the caller, since no
  GGUF key states them.
- A GGUF's window of -1 is the whole context; the loaded context hands it on
  as its `context_size`. `Config` refuses a negative window from the caller
  and from an add-on file.
- A value the model recommends is checked where it would be taken. A refused
  one raises a `ValueError` naming the model's recommendation as the source
  and how to override it; a value another source replaces cannot stop the
  load.
- A shipped file matched by its template also wins over a fine-tune's GGUF
  values; a file next to that GGUF, or the caller's values, override it.
- `Sampling` per request takes every sampling field of `Config`, each `None`
  to keep the wrapper's, and is checked by building a configuration.
- Every generation builds its own sampler chain from its effective settings
  and frees it at its end.
- Where every penalty only lowers a logit (repetition at least 1, presence and
  frequency at least 0) and `top_k` is on, the chain first keeps the
  `top_k + window` likeliest tokens (one plus the window under greedy
  decoding). The draws are the same; a test compares draws with and without
  the filter and fails for one a window too tight.

## Consequences

- The six shipped add-on files without `top_k` now sample with `top_k` 40 and
  `min_p` 0.05, where their makers state neither.
- Measured: sorting the whole vocabulary cost 1 to 2 ms per
  token; the penalty stage over a 262,000-token vocabulary went from 2.2 ms to
  0.18 ms per token.

## Alternatives rejected

- The GGUF winning whole: the add-on file is the source a person checked
  against the maker's published recommendation and cites, while a GGUF holds
  whatever its converter copied.
- Switching unstated values off: costly, and not what a maker who states
  nothing asked for.
- One shared sampler chain, reset per generation: three paths where one does.
