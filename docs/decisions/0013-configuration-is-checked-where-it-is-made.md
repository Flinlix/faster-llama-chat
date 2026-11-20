# 0013. The configuration is frozen and checked where it is made

Date: 2026-11-19

## Context

Several settings were not checked at all (a negative penalty window or thread
count, `gpu_layers` below -1, any value's type), so `temperature="hot"` failed
at the first reply. A list inside the frozen configuration could be changed
while the wrapper ran, and two settings did what a request argument does.

## Decision

- `Config` is frozen and holds every setting but the add-on files and what a
  request passes.
- `Config.__post_init__` checks every field's type against its own annotation,
  so the annotation is the one statement of what a field takes: an `int`
  field takes only an `int`, a `float` field an `int` or a finite `float`,
  never a `bool`, and `None` only where the annotation says so. Range checks
  follow. A setting refused at startup is an `InvalidSettingError` naming the
  field, a missing `model_path` included.
- A request's `sampling`, `max_reply_tokens` and `max_calls` are checked by
  the same code, as `InvalidSettingError`. `sampling`, `stop` and `cancel`
  are checked when `reply` is called, before the question is written.
- No setting holds a value that can change in place: `Config.stop_strings` is
  gone. `reply(stop=...)` takes one `str` as one stop text, or any iterable of
  `str`, copied when `reply` is called.
- `max_reply_tokens` is `None` by default, no cap. A cap counts every
  generation of one `reply` together under every policy; a reply that goes on
  with an open turn is a request of its own with a cap of its own.
- `ChatWrapper.config` is the configuration with the model's sampling merged
  in where the wrapper loads the model, and the configuration as passed with
  a context of one's own.
- The context's own size is checked where it is adopted.

## Consequences

- A deployment that could never work fails while the wrapper is built, not at
  its first request.
- A reasoning model no longer ends with an empty answer because a default cap
  of 1,024 counted its thought.

## Alternatives rejected

- A list of fields to check beside the annotations: it could fall out of step.
- Moving the hardware settings out of `Config`: ruled out by the owner.
- `Config.stop_strings`: a second way to do what `reply(stop=...)` does, and a
  mutable list.
- Reading a single `str` as a list of its letters, or refusing it: most chat
  interfaces take one string as one stop text.
