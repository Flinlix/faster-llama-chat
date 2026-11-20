# 0037. A call's format is recovered from the template's own renders, read in the one shape the template writes, and refused where it cannot be told

Date: 2026-11-19

## Context

Each family writes tool calls its own way: tags around a JSON object, XML-like
elements, `name{key:value}`, a JSON array of calls, text with no tag at all.
Code per family would break the rule that no model is special, and a reader
that tries several shapes guesses.

## Decision

- **Tags.** A tag is a control token or a token a fine-tune added
  (`LLAMA_TOKEN_ATTR_CONTROL` or `LLAMA_TOKEN_ATTR_USER_DEFINED`); fine-tunes
  often add their call tags as ordinary added tokens.
- **Opening and closing.** The opening is the first tag before the function
  name; what the template writes between it and the name the model writes
  too. The closing is the last tag between a first call's argument and a
  second call's opening; where an opening begins with a tag written once for
  all the calls of a turn, the second call is found by the shortest end of the
  opening that starts with a tag; where no tag stands between two calls, the
  closing is the first tag after the last argument. Where a call's opening tag
  also marks every reply, the call is recognized by the text it opens with and
  the text that ends it.
- **A plain-text opening** counts only at a reply's start or right after a tag,
  so a sentence that mentions it stays text, except on a template that writes a
  call right after a message's text (read at load), where it opens a call after
  any text but never inside a thought. One that begins with the words of a
  reply's own header is refused unless the add-on file gives the call, since a
  model may write its header in another order than its template.
- **A function named twice**, in the turn's header and the body: the call opens
  with what lies between.
- **The argument spelling** comes from four probes that differ only in their
  content (no argument, one string, one number, two strings) and three more
  for nested values (an object of one member, of two, a list of two strings).
  A type written between a key and its value is read from where a string and a
  number argument differ. Covered: JSON objects and arrays, `name{key:value}`,
  `name(k="v")`, attribute-style XML, `<function=name><parameter=k>v</parameter>`,
  `name<arg_key>k</arg_key><arg_value>v</arg_value>`.
- **One shape.** A template whose probe call is a JSON object or array around
  the name is read as JSON alone, by the keys it writes; any other is read by
  its delimiters alone. Arguments written as one JSON object after the name
  are read as that JSON (`arguments_json`).
- **Ids.** Probes render a call under two ids: an id between the name and the
  arguments is `id_open`, one under a JSON key `json_call_id_key`; an id that
  cannot be told from the delimiters, or stands anywhere else, refuses the call
  body. Whether the template escapes quotes inside a value is read at load
  (`escapes_quotes`).
- **Refused where it cannot be told:** a format that cannot tell where a value
  ends, a template that leaves a past call out once a question follows, and a
  tool name that reads back as another refuse tools at load. A format that
  cannot be recovered makes `can_call_tools` false, but the call is still
  detected, so its syntax never reaches the reply text.
- An add-on file's call format wins over all of this (0011).

## Consequences

The recovered delimiters on two real models:

| `CallBody` field | a delimited model | an element-shaped model |
|---|---|---|
| `name_close` | `{` | `>\n<` |
| `key_open` | (empty) | `parameter=` |
| `key_close` | `:` | `>\n` |
| `pair_sep` | `,` | `\n</parameter>\n<parameter=` |
| `body_close` | `}` | `\n</parameter>\n</function>\n` |
| `empty_close` | `}` | `/function>\n` |

A delimiter runs only as far as the probes agree. Two nested shapes stay lost
because the template writes them alike: an object whose only key is the items'
name reads as a list, and an empty object or list as an empty string. Open: the
separator between two calls is still found at a `<` inside the opening, which
on one test template depends on whether its vocabulary holds the opening as a
tag.

## Alternatives rejected

- Trying JSON first under the keys `name` or `tool_name` and `arguments` or
  `parameters`: one or two failed decodes per call on every other template
  (about 8 to 20 percent of reading a call) and two families' key names in the
  core.
- Falling back to delimiters where JSON arguments fail: it ran tools with
  values such as `Munich"}`.
- Accepting a plain-text opening anywhere: a sentence quoting it became a call.
- `ChatWrapper(tool_calls=...)`: replaced by the add-on file.
