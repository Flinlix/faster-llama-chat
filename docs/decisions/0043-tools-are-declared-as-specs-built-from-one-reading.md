# 0043. A tool is declared as a checked spec, and its schema and argument check come from one reading of its signature

Date: 2026-11-19

## Context

The registry had several ways to register and to list tools, a bare decorator
that registered nothing, parameters that could never be passed, asynchronous
functions whose coroutine never ran, and declarations that reached Python's
own errors deep inside a render.

## Decision

- `@registry.tool` works bare, with the function's docstring as the
  description, or with a description string; `name=`, `description=` and
  `parameters=` override by keyword; `registry.tool(fn, description=...)`
  registers a function written elsewhere. It checks that it was given a
  function. A tool with no description is refused, since the model would read
  nothing about it.
- The annotations describe lists, dicts with text keys, optional values,
  `Literal` choices, enums, defaults and `Annotated` descriptions. One reading
  of each annotation builds both the schema and the check of the model's
  arguments. Arguments that do not fit are answered as `wrong_arguments`,
  naming the parameter, before the tool runs. A default no model can be given
  is left out.
- An explicit `parameters=` schema is copied, and `specs()` hands out copies.
- `begin(tools=...)` takes a list of `ToolSpec`, which `registry.specs()`
  gives; a registry passed as it is is a `TypeError` that says so. Two tools of
  one name in one declaration are refused. A tag in a tool's name, a
  parameter's name, a choice value or a title is refused (0005).
- `ToolSpec` checks itself where it is built: text keys, finite numbers, JSON
  values, at most `MAX_ARGUMENT_DEPTH` levels.
- Registry tools run on a thread of their own, bounded by `tool_timeout_s`; a
  tool's thread may not act on the wrapper. A tool that calls back into the
  wrapper is refused and answered as a tool that raised.
- `ToolRegistry.call` turns only an `Exception` into a result, so a stop such
  as Ctrl+C is never told to the model as a tool failure.

## Consequences

- One way to register and one way to declare.

## Alternatives rejected

- `register` beside the decorator, `ToolRegistry.__contains__` and
  `__iter__`, `declare_at_begin`: each a second way to do what one remaining
  call does.
- Keeping a schema by reference: a later change to the caller's dict changed
  the declaration.
- Stopping a timed-out tool: Python cannot stop a thread, so the tool runs on
  and what it returns is dropped.
