# 0020. A cache context is one protocol, checked once where it is adopted

Date: 2026-11-19

## Context

A caller may bring a cache context of its own. Nineteen presence checks were
spread across the code, the engine adapter held logic that names no engine,
and a context that answered too little or with other arguments failed at its
first call, far from its cause.

## Decision

- `CacheContext` in `cache_context.py` is the one exported protocol, in two
  halves: the model (tokenizer, tag list, `is_special`, `ends_generation`,
  `chat_template_text`, `eos_token_text`, `recommended_sampling`) and the cache
  (decodes, edits, saved states, `sampled_tokens`, `decode_generated`,
  `token_bytes`, `decode_blocks`, `save_whole_state`).
- It is checked once where the wrapper adopts it: each member's presence, each
  method bound to its protocol method's arguments (positionally and by name),
  the context's size, the template text present, and the tag list complete
  (0006). A context that can neither shift nor cut must save a state: a `None`
  or a `RuntimeError` from one `save_state` call refuses it. A method whose
  arguments cannot be read, such as one written in C, is taken as it is.
- The generation loop is engine-neutral (`generation.py`): UTF-8 decoding,
  holding back text, call recognition, cancelling, the token cap and the stop
  reasons. A context answers four steps: `sampled_tokens`, `decode_generated`,
  `ends_generation` and `token_bytes`. `sampled_tokens` is handed every value a
  generation draws with, so a context needs no configuration of its own.
- The model's tags, a text split at its tags and the first tag in a text are
  computed by `tokenizing.py` from `tokenize`, `token_text`, `is_special` and
  `special_tokens`, so the real tag logic runs in every test.
- `apply_eviction` takes the start and end of the stretch it removes.
- A context's token ids lie from -65,536 to 1,048,575: the search for a
  dropped stretch spells each id as one character, offset by 65,536, and
  Python's characters end at 1,114,111.
- The engine context probes its cache at load: a cut tried on the empty
  cache, and where it cannot cut, a state saved and restored; a build that
  cannot is refused. A cache that cannot shift is never asked to shift, not
  even at load, since some caches abort the process when asked.

## Consequences

- A context of one's own that does not fit fails at construction, naming what
  it lacks.
- The test stand-in only hands out scripted tokens, so the real call detection
  and tag logic run over every test template.

## Alternatives rejected

- Asking a context for its vocabulary size: the stand-ins number characters by
  code point, so a vocabulary walk would cost over a million calls per load.
- A public `Eviction` record of three integers: its third field only bounded
  the shift, and the engine shifts everything after the stretch when given no
  end.
- A `warmup` member and the export of `ModelReader`, `GenerationAccumulator`
  and `CallBoundary`: each served one implementation or suggested a second way
  to build a context.
