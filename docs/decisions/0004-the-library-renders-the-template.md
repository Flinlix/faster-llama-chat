# 0004. The library renders the template; a cache context gives only its text

Date: 2026-11-20

## Context

A cache context had to render the chat template itself, and the bundled one
used the engine binding's Jinja formatter, while the library's own template
reader also went through the binding to list the names a template reads. A
context of one's own was told it only needed to render, yet the library needed
the binding all the same, and a template's own refusal reason was lost.

## Decision

- A context gives the template's text (`chat_template_text`) and the text of
  its end-of-sequence token (`eos_token_text`). `llama_chat/model_template.py`
  renders it in the environment chat templates are written for: Jinja's
  immutable sandbox, blocks trimmed, the `generation` block tag passed
  through, loop controls, and `tojson` writing non-ASCII as it is. These are
  the binding formatter's settings, so every render is unchanged. `jinja2` is
  a declared dependency.
- A template that is not valid template code is refused at load, naming the
  line and the cause.
- The names a template reads are parsed by the renderer's own environment.
- Template variables are passed as one mapping, `variables`. The names the
  renderer sets for every template are listed in `RENDERER_VARIABLES`
  (`messages`, `tools`, `add_generation_prompt`, the beginning and end tokens,
  `raise_exception`, `strftime_now`); every other name can be passed.
- The template's own reason for a refusal, given through `raise_exception`,
  reaches the caller as "(the template says: ...)". It is read by rendering
  again on the refusal path only.
- The sandbox's answer on whether a template may read an attribute depends
  only on the kind of object and the attribute's name in the Jinja release the
  library runs on (3.1), so it is remembered per kind and name. A test
  compares the remembered answers with Jinja's own over objects of every kind
  the sandbox treats apart, and fails on another release, which must be
  checked before it is admitted.
- Renders and their splits are remembered for one outermost action and the
  action after it, at most two actions' renders, except where the template
  reads the clock. The tokenizer memory starts over after 20,000 entries or
  before it would hold 4,000,000 characters (about 40 MB with their tokens); a
  longer text is tokenized without being kept. Each token's text and whether
  it is a tag are remembered, at most one entry per token of the vocabulary.

## Consequences

- A context of one's own needs no renderer, and tests render through the same
  code a loaded model runs.
- Measured on the stand-in: the remembered sandbox answers made renders 1.0
  to 2.0 times faster; remembering tag checks took a begin with 20 tools from
  121 ms to 45 ms. A 2 MB tool result left about 36 MB in the tokenizer memory
  before its character bound.
- Without a clear at each action the render memo grew by about seven entries
  a turn, each keyed by the whole conversation (2,122 entries after 300 turns
  against 10).

## Alternatives rejected

- Leaving the render in the context: the library needed the binding anyway,
  and the template's refusal reason could not reach the caller.
- Keying the render memo by message identity: identities are reused once a
  message is collected, so a stale render could come back.
