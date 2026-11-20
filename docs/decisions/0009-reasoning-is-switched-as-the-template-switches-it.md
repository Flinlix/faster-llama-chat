# 0009. Reasoning is switched by the template's shared variables or by the add-on file, never guessed

Date: 2026-11-20

## Context

Templates turn reasoning on and off through variables of their own choosing.
Searching every variable whose name held "think" or "reason", and trying
`"high"` as an effort, wrote two families' vocabulary into the core and was
needed by no test template. A template that reasons by default and turns
reasoning off only with `reasoning_effort: "none"` was read with an off equal
to its on. A past thought handed over in the wrong form was silently lost from
the cache.

## Decision

- The switch read from a template is `enable_thinking` where it changes the
  render, and otherwise a `reasoning_effort` whose `"none"` changes what
  follows the user turn: then `"none"` is off and the template's default is
  on. Any other switch is named in the add-on file's `reasoning_on`, the
  template variables with their values that turn reasoning on. They are laid
  over the switch wherever a render asks for reasoning on, the load probes
  included. A `reasoning_on` that writes the prompt as reasoning off does is
  refused at load.
- A prompt with reasoning on that already closes the thought means the model
  cannot reason; asking for reasoning is then refused, pointing to
  `reasoning_on`.
- `reasoning=True` needs a template that marks a thought and whose prompt with
  reasoning on leaves it open. `can_turn_off_reasoning` says whether the
  switch writes the prompt otherwise with reasoning off; where it does not,
  nothing is refused and the thought is still kept out of the answer.
- `begin` takes `reasoning` (`True` or `False`, off by default) and
  `reasoning_level` (`None` for the template's own way, or one of
  `reasoning_levels`). A level without `reasoning=True` is refused, naming
  `reasoning_level`. A level is given to the template as `reasoning_effort`.
  The levels are read at load: each effort text the template writes otherwise
  than an effort it does not know. `"none"` is never a level. Where a template
  writes any effort as given, its add-on file names the levels
  (`reasoning_levels`), and levels the template does not take are refused at
  load. A level that keeps a different generation prompt is not offered.
- A past reply's thought is handed to the template only where it keeps one,
  in the form it reads (`reasoning_content`, a `thinking` field or a content
  block), which the load check records. The conversation itself keeps it as
  `reasoning_content`.

## Consequences

- A template whose default does not reason yet writes something for `"none"`
  needs `reasoning_on` in its add-on file; the renders cannot tell it apart.
- `conversation()` gives `reasoning` and `reasoning_level` back.
- On Ministral 3 Reasoning a past thought was lost after the next question
  until the form was recorded; the template reads it only from a content
  block.

## Alternatives rejected

- Searching names holding "think" or "reason": no test template needed it, and
  it cost up to 30 renders and 7 ms per load on seven of them.
- `"high"` as the core's on: the vocabulary of two families, now in their
  add-on files.
- A short real reply at every load to see whether the default thinks: a few
  seconds per load.
- One parameter with a reserved word `"default"`: it reserved a word a level
  could hold. A marker object in its place: `conversation()` could not give it
  as plain JSON.
