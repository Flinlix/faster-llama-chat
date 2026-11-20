# 0039. A request names what becomes of the model's calls; a withheld call is kept from being drawn

Date: 2026-11-17

## Context

A caller may run tools itself, let the library run them, or want no call at
all. Six entry points and a registry `begin` kept were several ways to say
this.

## Decision

- `reply(tool_call_policy=...)` takes a `ToolCallPolicy` (`"refuse"`,
  `"yield"`, `"withhold"`) or a `ToolRegistry`. `TOOL_CALL_POLICIES` is read
  from that type. `"refuse"` tells the model no tool can run; `"yield"` hands
  each call to the caller in the stream; a registry runs each call; a request
  whose calls are to run is handed the registry itself.
- The default `None` keeps the policy, a registry with what is left of its
  `max_calls`, of the reply that opened the turn a reply goes on with, chosen
  under the lock as the generation starts. A turn a history opened takes the
  first policy a reply gives it; any other reply takes `"refuse"`. The policy
  is held until the next turn starts or the conversation begins again.
- `"withhold"` keeps a call's opening token from being drawn, by a logit-bias
  stage in the generation's own sampler chain. Where a call opens with plain
  text, which no token can stand for, the generation ends where such a call
  opens, the opening's tokens leave the cache, and the call is reported
  `"unfinished"`.
- A round that recognizes no call also withholds the opening token: every
  round under `"refuse"` after a call, a registry's last round under
  `max_calls`, and a generation in a room too small for any call.
- A registry holding tools is checked before its exchange starts, as a
  declaration is: it is the same promise that the model can call them.

## Consequences

Measured on 8 real models with a weather tool declared and a question asking
for it: under `"withhold"`, Llama 3.2, Qwen 3.5 and Ministral 3 wrote the
call's markup as text anyway, and Gemma 4 26B-A4B repeated one phrase up to
the token limit; a question needing no tool was answered normally. Under
`"refuse"` Gemma 4 called, was told no tool can run, and made up a weather
report. So withholding suits a request that wants no tool.

## Alternatives rejected

- `"run"` against a registry `begin` kept: the library no longer holds the
  caller's functions between calls, so what runs is always what the request
  names.
- Taking `"refuse"` whenever no policy is given: a caller answering calls under
  `"yield"` had to repeat the policy at every step.
- Naming the parameter `calls` or `tool_calls`: it shared its name with
  `Reply.tool_calls`, the calls a reply made.
