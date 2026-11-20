# 0038. Reading a call back is its own module, and a call is readable only where all of its text is read

Date: 2026-11-19

## Context

The parser once read what it could of a call and dropped the rest, and guessed
each unmarked value's type from its spelling, so a misread argument reached the
tool unnoticed: `01234` became 1234, `1e5` became "100000.0".

## Decision

- Reading a call back from what the model wrote is `llama_chat.call_parsing`,
  apart from declaring tools (`tools.py`) and running them
  (`tool_registry.py`): an engine reached over a wire emits calls it cannot
  run, and reads them without a registry.
- A call is readable only where all of its text is read. Text after the last
  argument, an unclosed argument, a blank pair, a key written twice (in JSON
  too, nested ones included), a second copy of a call, a separator after the
  last argument and a call whose closing never came each make it unreadable;
  whitespace after the call is allowed. Arguments that are no JSON object make
  it unreadable. A quote marks a string only where a value starts, and the
  strict reading of a quoted value applies only where the template escapes its
  quotes.
- A value written without a type mark keeps its spelling until the call is
  read whole, and is then read against the declared schema of the tool it
  names: it stays text where some branch asks for a string and none for the
  literal it spells, or an `enum` lists the text; otherwise it is the literal.
  The types are collected through `anyOf`, `oneOf` and `$ref` inside the
  parameters' own schema, each reference followed once. A JSON call keeps each
  number as the text the model wrote and is read the same way. A value written
  like a list stays text where the schema asks for text alone; an empty value
  between tags is `{}` or `[]` where the schema asks for that alone.
- Arguments nested deeper than `MAX_ARGUMENT_DEPTH` (100), counted without
  recursion, and a JSON integer past Python's digit limit, make a call
  unreadable on every path. Only a `ValueError` counts as an unreadable call.
- Each entry of a batch is judged alone: an entry that is no call is an
  unreadable call of its own, answered in its place; a batch written as JSON is
  otherwise read whole, since reading half of it would drop what the model
  asked for.
- A generated call is held to what a history's call is held to.

## Consequences

- A misread argument never reaches a tool; the model is told the call could
  not be read and may write it again (0040).
- Reading a call takes time linear in the schema; 21 levels of doubled
  references had taken 11.8 s.
- `allOf` is not read; the schemas the registry writes use none of these
  keywords. A nested value written as a Python literal still takes the last of
  two equal keys.

## Alternatives rejected

- Converting values in the registry's readers: a call handed over to the
  caller runs through no reader.
- Answering a number where text is asked as a wrong argument: a postcode or a
  version is what the model meant far more often, and it costs a round.
- Turning the decoded number back into text: it changes the spelling (`3.10`
  to `3.1`).
