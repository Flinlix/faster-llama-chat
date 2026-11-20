# 0008. A model's further template for tools writes every conversation begun with tools

Date: 2026-11-14

## Context

A GGUF can hold named chat templates beside `tokenizer.chat_template`, among
them `tokenizer.chat_template.tool_use` for a conversation that declares tools
(Hermes 2 Pro ships one). Only the default template was read, and what a model
can do was asked through four properties that answered for different
templates.

## Decision

- The engine context reads the template for tools too; the context members
  that hand out a template take a `template_name`, `None` for the default.
- `begin` decides the template once: `tools` given at all, an empty list
  included, writes the conversation with the template for tools; `tools=None`
  writes it with the default. `conversation()` gives `tools` back as `None` or
  as a list, possibly empty, so a new `begin` keeps the template.
- A template for tools is always given a list of tools, empty where none is
  declared, because such a template is written for one.
- A template for tools that fails the load check refuses only tools: the
  model loads with its default template, the cause is logged once as a
  warning, `capabilities(tools=True).can_call_tools` is false, and every
  declaration is refused with an `UnsupportedError` naming that cause. The
  default template must pass the check as before.
- `capabilities(tools=...)` returns one `Capabilities` record for a
  conversation begun with tools or without (`can_call_tools`, `can_reason`,
  `can_turn_off_reasoning`, `can_take_system_prompt`, `reasoning_levels`).
  Both records are settled at load and never change. `tools` has no default,
  since on a model with two templates the answers differ.

## Consequences

- Tools declared later need no switch of template; changing the tools is a
  `begin` with the conversation (0030).
- A history holding a tool turn begun without tools on such a model is
  refused with the advice to pass tools, an empty list included.
- Hermes 2 Pro could not load before: its template for tools loops over
  `tools` without a test for none.

## Alternatives rejected

- Switching templates when tools are declared later: it decodes the whole
  conversation again in the other format.
- Refusing the whole model where its template for tools cannot be read: its
  default template serves every conversation without tools.
- One argument per capability property (four calls for one question), and one
  answer for both templates (it would report a model whose template for tools
  reasons as unable).
