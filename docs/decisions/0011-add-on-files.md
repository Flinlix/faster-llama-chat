# 0011. What a model needs beyond its template comes from one small JSON add-on file

Date: 2026-11-15

## Context

Some facts a template cannot show: the sampling a model's maker recommends, a
call format a template never writes as the model writes it, a reasoning switch
or levels the renders cannot tell apart. A second way to state them in code
(`ChatWrapper(tool_calls=...)`) let a caller describe what no template writes.

## Decision

- A model's add-on file is the JSON file next to the GGUF with the model's
  name, where its `templates` list names this model's template fingerprint;
  otherwise the file shipped in `llama_chat/addons/` whose `templates` names
  it. Fingerprints are 64-digit lowercase hex. No two shipped files may name
  one template.
- A file next to the GGUF that is not JSON is refused at load, naming it. One
  that is JSON but no add-on file, or names another template, is not used: the
  shipped file applies, and a warning on `llama_chat.addon_files` names the
  file and the reason. A file that names the template is checked whole and
  refused where malformed: every value is type-checked by the code that checks
  `Config`, a null sampling value is refused, `call` must be an object with
  every field, each `reasoning_on` value text, a number or a boolean.
- A file fills in only what the template cannot tell, except a call format it
  names, which wins over the template's, tag-marked templates included; its
  closing is still checked against the template.
- The wrapper's constructor reads the add-on file, for every cache context,
  and hands the template its call format.
- Every shipped sampling value quotes its source in `source`: the model card's
  words, or the keys of the `generation_config.json` it cites, read online. The
  test template of that model states the values (`card_sampling`) and the
  quoted words (`card_quote`), so each value is checked against the words it
  was taken from.
- Shipped files are spelled as their test templates are; a shipped file is
  found by fingerprint, never by name.

## Consequences

- A model runs without a code change; what one model needs is in one file a
  person can read and cite.
- A model whose calls have tags but whose template does not show how it spells
  the arguments is refused tools until its add-on file can carry the
  spelling; that waits for such a model.
- Ministral 3 holds temperature 0.05, from its card's sentence "Use a
  temperature below 0.1", not the 0.15 its code examples set.

## Alternatives rejected

- Taking any file with the model's name: another program's settings beside
  the GGUF stopped the model from loading.
- An error for a JSON file that names another template: it may belong to
  something else; a warning shows without any logging configured.
- Keeping the lookup in the engine context: every context of one's own would
  repeat it, and without it lost tools and sampling.
- `ChatWrapper(tool_calls=...)` beside the add-on file: a second way to say
  one thing.
