# llama_chat: a KV-cache manager for llama.cpp

A Python chat engine over llama.cpp whose goal is to **prefill as little as
possible**. It keeps the KV cache alive across turns, decodes only what is new,
and when the context fills it *removes the oldest exchanges and shifts the
survivors down to close the gap*, reusing their KV instead of decoding them
again.

Every model whose GGUF ships a chat template is encoded exactly as that
template writes the conversation, without code particular to the model, or
refused: the template renders the conversation, and the cache is compared
with that render and decoded only where the two differ. The work for the next
question is done right after a reply, while the reply is read, so a question
waits only for its own tokens. Tool calls work on any model whose template
describes them: the tags and the argument spelling are recovered from the
template. What a template cannot tell, such as a model's recommended sampling,
comes from a small JSON add-on file.

- [Why](#why) and [how it differs from llama.cpp's context shift](#compared-with-llamacpps-context-shift)
- [Install](#install)
- [Quick start](#quick-start)
- [What to call, by task](#what-to-call-by-task)
- [Tool calling](#tool-calling)
- [Reasoning](#reasoning)
- [Keeping a conversation](#keeping-a-conversation)
- [Configuration](#configuration)
- [Add-on files](#add-on-files)
- [A model without a usable template](#a-model-without-a-usable-template)
- [Your own cache context](#your-own-cache-context)
- [Verify](#verify)

How each of these works, and why it was built that way, is in
[design_history.md](design_history.md).

## Why

1. **Minimum prefill:** only genuinely new tokens are processed; the cache from
   previous turns survives and is reused.
2. **Rolling eviction:** when the context fills, the oldest exchanges are
   dropped, so a chat runs indefinitely.

Trimming the prompt defeats the first goal: removing old messages from the
start shifts every later token's position and invalidates the whole cached
prefix, so every turn decodes in full. The library edits the cache in place
instead: it removes the oldest exchange's tokens and shifts the survivors'
positions down, so kept messages are never processed again.

### Compared with llama.cpp's context shift

Both use the same cache operations; the policy differs.

| | llama.cpp's server | llama_chat |
|------|------|------|
| what is removed | a fixed token count from `n_keep`, which can cut inside a message or a tag | whole exchanges, from a user turn to the next one, so the cache always holds turns the template writes |
| the system prompt | `--keep N`, a hand-counted offset | pinned, never evicted |
| when | when the context is full | above a threshold (`evict_above_fraction`), or above a count of exchanges (`max_history_exchanges`); a request that would not fit evicts first |
| caches that cannot shift | context shift off or unreliable | detected at load; such a cache is cut, restored to a saved state or decoded again instead, and an eviction there drops well below the threshold, so that this happens only every few turns |
| bookkeeping | none at message level | the conversation as the messages its template is given; the cache compared with the template's render of them |

## Install

llama-cpp-python is built **from source per machine**, so it uses that CPU's
own kernels or the CUDA backend.

```bash
./install.sh            # CUDA where nvidia-smi is present, else a native-CPU build
./install.sh --cpu      # force the native-CPU build
./install.sh --cuda     # force the Nvidia GPU build
./install.sh --dev      # add test dependencies (append to any of the above)
```

The same by hand:

```bash
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python "llama-cpp-python>=0.3.23"   # CPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python "llama-cpp-python>=0.3.23"     # Nvidia GPU
pip install -e ".[test]"
```

## Quick start

```python
from llama_chat import ChatWrapper, Config

chat = ChatWrapper(Config(model_path="models/your-model.gguf", context_size=8192,
                          gpu_layers=-1))

# begin: reset, and prefill the system prompt and as much recent history as fits.
# A template that writes a system prompt where later questions move it refuses
# one (can_take_system_prompt is False); pass "" there.
chat.begin("You are a helpful assistant.", [("user", "Hi"), ("assistant", "Hello!")])

# inject: prefill one message as context, without generating
chat.inject("Reference: the deadline is Friday.")

# reply: prefill only the new request, then stream the reply
for delta in chat.reply("What's the deadline?"):
    print(delta, end="", flush=True)

reply = chat.reply("Summarize the plan.").finish()    # or read it to its end at once
print(reply.text, reply.stop_reason)
```

A `ReplyStream` yields text deltas; its `reply` is the `Reply` once the stream
has ended of its own accord, and `finish()` reads it to its end and returns
that. Each token reaches the cache before its text is yielded, so closing a
stream early records exactly what the cache holds. An abandoned stream holds
the wrapper's lock until it is collected: read it to its end, close it, or use
it in a `with`. To stop a generation from another thread, pass
`cancel=threading.Event()` and set it; the reply ends within one token with
`stop_reason == "interrupted"`.

Actions that change the cache are serialized by one lock. `close()`, or
`with ChatWrapper(...) as chat:`, waits for a running generation, frees the
model and makes later actions raise. A reply holds the lock while it is read,
so an action on the thread that reads it, a tool the reply runs included,
would wait for itself; it is refused instead with a `RuntimeError` that names
the reply.

## What to call, by task

| doing what | call |
|------|------|
| starting, or starting again | `begin(system_prompt, messages=None, tools=None, *, reasoning=False)`: a reset. Prefills the pinned system prompt and the newest history that fits with it in half the context, from a user turn on, at least the newest exchange and no more than `max_history_exchanges` exchanges, and returns how many history items it kept. A history message is a `(role, text)` pair or a `{"role": ..., "content": ...}` dictionary, its role `"user"` or `"assistant"`, a reply's dictionary with its thought under `"reasoning_content"` where it has one; a turn that called tools is a `ToolTurn`. Any other form or role is refused, naming it. Older history is dropped without being rendered. A history that ends on a call keeps it owed a result. Refused where the system prompt and its declarations leave no room for a short question, where the newest exchange cannot fit, or a history with no user message, which nothing can be evicted from, cannot fit, and where the template writes a system prompt where later questions move it; a refused or failed `begin` leaves the conversation it would have replaced as it was |
| adding context | `inject(text, role="user")`: adds one message without a reply, and returns how many older messages were evicted for it. A user message is added as `reply(text)` adds its question, so `inject(text)` followed by `reply()` waits no longer than `reply(text)`; the eviction it needs may come after the reply instead, and is then counted in that reply's `n_evicted_messages`. An assistant message, or a user message once `max_history_exchanges` is reached, evicts first. An eviction drops up to the next user message, so the messages after the newest one stay; an assistant message with which they no longer fit under the threshold is refused. `role` is `"user"` or `"assistant"`: a system message is refused, since the system prompt is `begin`'s, and so is a tool result, which `inject_tool_result` writes |
| asking | `reply(text=None, *, tool_call_policy=None, max_calls=None, max_reply_tokens=None, stop=None, cancel=None, sampling=None)`. Without `text`, it answers the newest user message already written, or goes on with a turn a tool call left open. A conversation that cannot take the reply (none begun, a request while a reply is open, a result still owed, nothing to answer) is refused when `reply` is called, and so are a `text` that is not a `str` and a `max_reply_tokens` of 0 or less; the rest is checked when the stream is first read |
| answering a call yourself | `inject_tool_result(call, result)` for each of `pending_calls`, oldest first, then `reply()`, which keeps the turn's `tool_call_policy` |
| telling the model about more tools | `declare_tools(tools)` |
| shortening the newest reply to what was heard | `truncate_last_reply(keep_text)`. Refused where the template does not write the reply shortened to `keep_text`, as a template that writes no empty reply does not write it shortened to `""` |
| reading the conversation back | `conversation()`: everything `begin` takes, by its parameter names: `system_prompt`, `messages` (tool turns and the thoughts the template keeps included), `tools` (those `declare_tools` added included) and `reasoning`, so `begin(**chat.conversation())` begins again as it was. Safe from any thread, while a reply runs too: it reads all of it as the newest finished change left it |
| where it stands | `total_tokens`, `config`, `can_shift`, `can_call_tools`, `can_reason`, `can_turn_off_reasoning`, `can_take_system_prompt`, `turn_open`, `pending_calls`, `wait_idle(timeout_s)`. `total_tokens`, `turn_open` and `pending_calls` are safe from any thread, as `conversation()` is: each is read as the newest finished change left it, so a count read while a reply streams is the one from before that reply |
| ending | `close()` |

`stop_reason` on a `Reply` is one of `STOP_REASONS` (the type `StopReason`):

| | |
|------|------|
| `"end_of_turn"` | the model ended its turn |
| `"stop_string"` | one of the texts `stop=` gave matched the answer; the reply ends just before it |
| `"token_limit"` | `max_reply_tokens` ran out, or the context did |
| `"interrupted"` | `cancel` was set |
| `"tool_call"` | the reply stopped on a call, and the turn is open for its result |
| `"unreadable_tool_call"` | it stopped on a call whose body could not be read; the model is told, and this is the final reason only where calls are handed to you, or where the template has no result block |

`Reply` also carries `text`, `n_prefilled_tokens`, `n_prompt_tokens` (the
tokens the cache held when the reply's last generation drew its first, which
is all that generation was conditioned on), `n_generated_tokens`,
`n_evicted_messages`, `tool_calls`, `tool_limit_reached`,
`n_reasoning_tokens` and `thought`, the thought's text, empty where the model
thought nothing. A stream closed early has no `Reply`; what it generated
is recorded all the same.

**What fails at startup.** Whatever can be settled once is settled while the
wrapper is built, so a deployment that could never work says so then: every
`Config` field and the combinations no call could succeed under, that
`model_path` names a file, each cache edit tried once on the empty cache, that
a context of your own answers everything called on it, holds the
`context_size` asked for, and can shift its cache, cut its end or save a
state, that the chat template is valid template code (a
`ValueError` naming the line and the cause where it is not), that the
generation prompt and the closing after a reply are where the template writes them and the closing begins with one of the
model's tags, and whether tool calling works (`can_call_tools`). A setting
refused is an `InvalidSettingError`, a `ValueError` whose `field` names it.
What is left is what only a conversation can know: whether a request fits
(`ContextOverflowError`, a `ValueError`, for every message, history, declaration
or reply that does not), whether the model calls a tool that exists, and what
the tool does.

## Tool calling

```python
from llama_chat import ChatWrapper, Config, ToolCall, ToolRegistry

registry = ToolRegistry()

@registry.tool
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    return {"city": city, "celsius": 18}

chat = ChatWrapper(Config(model_path="model.gguf"))
chat.begin("You are a terse assistant.", tools=registry)
print(chat.reply("What is the weather in Munich?", tool_call_policy=registry).finish().text)
```

`tool_call_policy` says what becomes of a call the model makes, whether or not
tools were declared, since a model may write a call tag unasked. It is one of
`TOOL_CALL_POLICIES` (the type `ToolCallPolicy`) or a `ToolRegistry`. Left
out, it is the policy the open turn was started with, where the reply goes on
with one, and `"refuse"` otherwise; a reply that goes on after results thus
keeps its turn's policy unless it is given another:

| `tool_call_policy=` | what happens | what the stream yields |
|------|------|------|
| `"withhold"` | the model cannot write a call: the token that opens one is never drawn. A model that wanted to call may write something else entirely instead, so this suits a request that wants no tool. Where the template's call opens with plain text rather than one of the model's tags, no token can be withheld: a call the model writes is not recognized and stays in the reply's text | text |
| `"refuse"` (the default for a new turn) | nothing runs, not even a tool that was declared. The model is told no tool can be run and answers in words | text |
| `"yield"` | each call is handed to you after the text, and the turn stays open: answer each with `inject_tool_result(call, result)` in order, then call `reply()` again | text, then each `ToolCall` |
| a `ToolRegistry` | each call runs against it until the model answers. In the last round `max_calls` allows, a call the model writes is not recognized and stays text, so the turn always closes. `max_calls` is refused with any other `tool_call_policy=`, and where it is not a whole number of at least 0. A stream closed at a call handed over leaves that call unrun and owed, as under `"yield"` | text, and each `ToolCall` that runs, just before it runs |

```python
with chat.reply("What is the weather in Munich?", tool_call_policy="yield") as stream:
    for event in stream:
        if isinstance(event, ToolCall):
            chat.inject_tool_result(event, registry.call(event))
        else:
            print(event, end="", flush=True)
rest = chat.reply().finish()        # the turn stays open until this, and keeps "yield"
```

A call never appears in the reply text, though its tokens reach the cache,
since the model reads its own call back. Where the text after a call would
join the text before it without whitespace, a blank line parts them, whether
the call ran or was handed to you: the reply that goes on after a result then
starts with it. The turn keeps the tokens it was
generated as, so an action after which the template's render no longer shows
where that turn stands is refused with a `ValueError`, the conversation left
as it was. Every call is reported on
`Reply.tool_calls`, whatever became of it: one that was read with its
`ToolCall.status` `"complete"`, one that could not be read as `"unreadable"`,
and one the reply cap or a cancel cut short as `"unfinished"`, the last two
without a name, arguments or id, and never handed over in the stream. Where calls run, the `Reply` sums
every generation of the exchange, `n_prompt_tokens` aside, which is the last
generation's, and `tool_limit_reached` says whether a `max_calls` ceiling ran
out.

**Tools a model could not use are refused where you hand them over**, by
`begin(tools=...)`, `declare_tools` and a running `reply`, naming what
is missing: a template that describes a call, renders the declarations, has a
readable argument spelling, and a block for the result. `can_call_tools`
answers the same question up front. A declaration that names one tool twice is
refused there too, since the model could reach only one of them.

**Registering.** `@registry.tool` makes a function runnable and describes it:
its docstring is the description, and its annotations, postponed ones
included, are the parameters. `@registry.tool("...")` or `description=` gives
the description instead, `name=` the name the model calls, and `parameters=`
an explicit JSON Schema where the annotations cannot be described.
`registry.tool(fn, description="...")` is the same for a function written
elsewhere. Text, numbers, truth values, lists, dicts with text keys, optional
values (`str | None`), choices (`Literal["c", "f"]`) and enums are described,
with each default, and `Annotated[str, "the city"]` describes a parameter in
words. Before a tool runs, the model's arguments are checked against those
types, so arguments that do not fit are told to the model, which may try
again; a number where text is asked is taken as the text it was written as,
and an enum's value reaches the function as its member. A function that is
asynchronous, takes a parameter only by position, or has no description is
refused when it is registered. A registry lists its tools, so
`begin(tools=registry)` declares them; `registry.specs()` gives each tool's
`ToolSpec`, the declaration the model is told, and a `ToolSpec` of your own
declares a tool that this process does not run.

**When the model is told.** `begin(tools=...)` declares the specs it is given
when it writes the conversation, and `declare_tools` adds more later, skipping
those already declared; a tool to be declared later is left out of the list
`begin` is given. Either way the declaration goes where the template writes
declarations. Where that is the system turn, a later declaration decodes the
conversation after it again once. Neither keeps a registry: a request whose
calls are to run is handed one.

**A template of its own for tools.** A GGUF may ship, beside its default chat
template, one for a conversation that declares tools
(`tokenizer.chat_template.tool_use`). Where it does, both pass the check a
model meets at load, `can_call_tools` answers for that one, and a conversation
is written with it from the moment it declares tools: from `begin(tools=...)`
on, or from a `declare_tools`, which then decodes the conversation again once
in that template's form. A conversation that declares none is written with
the default template.

**Every failure is told to the model** in its own result block, and the reply
goes on: an unreadable call, a call under `"refuse"`, an unknown name,
arguments that do not fit, a tool that raised, a spent `max_calls`. Each says
that the call failed, why, and `try again` or `do not call again`, except the
answer to an unreadable call, which says only that the call failed and could
not be read. A result that a registered tool returns as a dict carrying
`error` counts as a failure too, and so does one that JSON cannot write. A
result handed to `inject_tool_result` is written as given, a dict carrying
`error` included, with nothing added to it; one that JSON cannot write is
refused there, and the call stays owed.
A call whose arguments are no object, such as a list or `null`, or nest
deeper than 100 levels of lists and objects, is unreadable, and its tool does
not run.

**Nothing is evicted while a turn is open**, since that would cut into the turn
being written, so a turn of many calls grows towards `context_size`. A result
too large for what is left is clipped to leave `min_reply_tokens`. Where the
result's wrapping alone leaves no room for the reply, `inject_tool_result`
raises `ContextOverflowError`, the call stays owed and nothing is written, and
the conversation must begin again; a reply that answers its calls itself,
under `"refuse"` or a registry, closes the turn instead on what the model
wrote, as the reply cap closes one, and ends with `"token_limit"`.

**A history with tool turns** goes into `begin` as `ToolTurn`s, each holding the
turn's steps in order: the model's text as `str`, its thought as
`Thought(text)`, each `ToolCall` and each `ToolResult(name, result)`. It is
written as the template writes a past turn.
A history ending on a result leaves that turn open, and `reply(tool_call_policy="yield")`
goes on with it. A turn that closed ends on the text of the reply that closed
it, in `conversation()` too, an empty `str` where the model wrote nothing after
its results, so that `begin(**chat.conversation())` begins it closed.

**Every call has an id**, `ToolCall.id`, that no other call of the
conversation has. A call handed over in the stream, in `Reply.tool_calls`, in
`pending_calls` and in `conversation()` carries it, so one call reads the
same in all of them. A `ToolCall` of a history keeps the id it carries, and
`begin` gives one to a call that carries none, so `begin(**chat.conversation())`
keeps every id; a history that gives two calls one id is refused. A call made
after a `begin` takes an id that none of the history's calls had, those that
did not fit included. `inject_tool_result` takes the call as it was handed
over, or one built by hand without an id that names the same function with
the same arguments.

## Reasoning

```python
chat.begin("You are a helpful assistant.", reasoning=True)
reply = chat.reply("Why is the sky blue?").finish()
reply.text                  # the answer alone
reply.thought               # what the model thought before it, without its tags
reply.n_reasoning_tokens    # about how many tokens the thought took
```

Reasoning is chosen per conversation, at `begin`, because a template may switch
it in the system turn. `can_reason` says whether `begin` accepts
`reasoning=True`, which it does where the template marks a thought. Reasoning
off is the template's own off: `can_turn_off_reasoning` says whether the
template has one. Where it has none, as on a template that always reasons, a
conversation begun without reasoning is written as the template writes it,
and the model may reason in every reply all the same. Where a template's
switch cannot be read from the template alone, the model's
[add-on file](#add-on-files) names the template variables that turn it on.
Every
stream yields the answer alone, with or without reasoning, and a header a
template writes for another reader is kept out of it too. Stop strings are
matched against the answer only. A finished reply is kept as the template
writes a past reply, without its thought where the template leaves one out,
and with it, handed over in the form the template reads one in, where the
template keeps it. Where that differs from what was generated, the cache is brought to that form
right after the reply, while it is read. A turn that called tools keeps the
tokens it was generated as. A thought the template keeps comes back from
`conversation()`: a reply's as `{"role": "assistant", "content": text,
"reasoning_content": thought}`, which `begin` takes as well, and one in a tool
turn as a `Thought` step before the text and calls it came before.

## Keeping a conversation

The cache holds only the recent window that fits; `conversation()` returns what
it holds as `begin` takes it, the system prompt, tool turns, the thoughts the
template keeps, the declared tools and the reasoning switch included, so a
conversation saved from it and begun again from it reads in as it was:

```python
saved = chat.conversation()
...
chat.begin(**saved)
```

A tool declared there is declared again, not registered: a request whose calls
are to run is still handed its `ToolRegistry`.

To keep everything, older turns included, keep each message as you send it
and each reply as you read it. A request whose generation fails before it says
anything is taken back, its question with it, so the conversation no longer
holds that question. So is a request whose reply produced nothing, where the
template writes no empty reply: one cancelled before its first word, one the
model ended at once, or one cancelled while a tool ran, which takes the turn's
calls and results back with it.

## Configuration

`Config` holds every setting except what an add-on file gives and what one
request passes. It checks every value on construction, its type and its
range, and cannot be changed afterwards: a whole number is taken where a
fraction is asked for, but `True` is never taken for a number, nor a number
that is not finite.
`ChatWrapper.config` is the one the wrapper runs with: a sampling setting left
unset holds the value the model supplies, whether the wrapper loaded the model
or was given a cache context of your own.

**Model and sampling**

| field | default | meaning |
|------|------|------|
| `model_path` | `None` | the GGUF file; loading refuses without one |
| `temperature` | the model's | `<= 0` selects greedy decoding |
| `top_k` | the model's | `<= 0` disables |
| `top_p` | the model's | nucleus cutoff |
| `min_p` | the model's | drops tokens below `min_p` times the top token's probability; `0` disables |
| `repetition_penalty` | the model's | over `penalty_window_tokens`; `1.0` disables |
| `presence_penalty` | the model's | flat penalty on a token already in the window; `0` disables |
| `frequency_penalty` | the model's | penalty scaled by how often a token occurs in the window; `0` disables |
| `penalty_window_tokens` | `64` | how far back the penalties look, within one generation; at least `0` |
| `seed` | `None` | in `[0, 2**32 - 2]`, which every generation starts from; `None` draws a fresh one |

The seven marked "the model's" take, unless set, the model's own
recommendation, value by value: its [add-on file](#add-on-files)'s first,
then what its GGUF states for a value the file leaves out, then switched off
(`temperature` 1, `top_k` 0, `top_p` 1, `min_p` 0, `repetition_penalty` 1,
`presence_penalty` and `frequency_penalty` 0). A model for which neither
states any value runs at llama.cpp's defaults. A value the GGUF states that
`Config` would refuse stops the load with a `ValueError` naming the GGUF as
its source, unless `Config` or the add-on file replaces it.
`Sampling(...)` passed to one `reply` overrides any field of this table but
`model_path` for that request. `Sampling` refuses, where it is built, any
value `Config` refuses.

**Context management**

| field | default | meaning |
|------|------|------|
| `context_size` | `4096` | the hard limit in tokens |
| `evict_above_fraction` | `0.75` | the fraction of `context_size` above which the oldest exchanges are evicted, in `(0, 1]` |
| `max_history_exchanges` | `None` | a second eviction trigger: the most exchanges kept, at least `1`. An exchange is one user message and everything after it up to the next, the unit an eviction drops; what a history holds before its first user message counts as one more |
| `max_reply_tokens` | `None` | cap on tokens per generation, a thought included; `None` sets none, so a reply may run until the context is full |
| `min_reply_tokens` | `32` | refuse a request with `ContextOverflowError` when fewer would be left for the reply; `0` disables |

**Backend and hardware**

| field | default | meaning |
|------|------|------|
| `gpu_layers` | `0` | layers offloaded to the GPU, at least `0`; `-1` for all |
| `kv_cache_on_gpu` | `True` | each layer's KV cache where that layer runs, in VRAM for the layers `gpu_layers` offloads; `False` keeps all of it in host RAM |
| `decode_threads` | `None` | CPU threads for decode, at least `1`; `None` lets llama.cpp decide |
| `prefill_threads` | `None` | CPU threads for prefill, at least `1`; `None` means as many as `decode_threads` |
| `max_batch_tokens` | `2048` | most tokens in one decode; a longer prefill is split |
| `flash_attention` | `None` | `True` or `False` forces flash attention on or off; `None` leaves it to llama.cpp, except under a quantized KV cache, which requires it and turns it on |
| `kv_cache_type` | `"f16"` | `"f16"`, `"q8_0"`, `"q5_1"`, `"q5_0"`, `"q4_1"`, `"q4_0"`; a quantized type requires flash attention, so is refused with `flash_attention=False` |

`ChatWrapper` also takes, outside `Config` because neither is a value to
tune: `context=` (a [cache context of your own](#your-own-cache-context)) and
`on_load_progress=`, called with a fraction while the weights load, which must
not raise.

**Logging.** Everything llama.cpp writes, and what the library itself has to
say, goes to Python's `logging`, under the logger `llama_chat` and llama.cpp's
lines under its child `llama_chat.engine`, one record per line: llama.cpp's
errors as `ERROR`, its warnings as `WARNING`, its load-time listings as `INFO`
and `DEBUG`. With no logging configured, Python shows warnings and errors on
stderr. To show only errors, or everything:

```python
import logging
logging.getLogger("llama_chat").setLevel(logging.ERROR)
logging.getLogger("llama_chat.engine").setLevel(logging.DEBUG)   # every line llama.cpp writes
```

llama.cpp has one log output per process, so importing `llama_chat` routes it
for the whole process, a model loaded by other code included.

## Add-on files

What a model needs beyond its chat template comes from a JSON add-on file,
never from code: the one next to the GGUF with its name and `.json`, else one
shipped in `llama_chat/addons/`, each used only where it names the model's
template by its SHA-256. A file next to the GGUF that names no template, or
another one, or is no JSON object, is not used, and a warning on the logger
`llama_chat` says why. It fills in only what the template cannot tell; what
the template states, the template decides. The file used is checked when the
model loads, each value's type included, and anything else in it is refused
with an error naming the file.

```json
{"templates": ["<sha256 of the chat template>"],
 "source": "where the values come from",
 "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
 "call": {"opening": "...", "name_close": "...", "closing": "..."},
 "reasoning_on": {"reasoning_effort": "high"}}
```

`sampling` sets any of the seven model sampling fields, `presence_penalty` and
`frequency_penalty` among them, and its values win over the GGUF's (see
[Configuration](#configuration)). Since a shipped file is matched by the
template, it also applies to a fine-tuned model that keeps the same template,
and its values win over those that fine-tune's GGUF states; a file next to the
GGUF, or values set in `Config`, take precedence for such a model. `call` says
how the model writes a call where its template shows none it can be read from:
the text a call opens with, the text after the function's name, and the text
that closes it; it is refused where the template never writes that closing.
`reasoning_on` names the template variables, with their values, that turn the
model's reasoning on, in place of what the library reads off the template; it
is for a template whose default does not reason although it writes something
of its own for the effort `"none"`, which the template alone reads as a default
that reasons. It is refused at load where it writes the prompt as reasoning off
does.

## A model without a usable template

A model that ships no chat template is refused when it loads: the template is
what every conversation is written with. For one whose template writes no
tool call it can be read from, its [add-on file](#add-on-files) says how the
model writes one. A reply's ending must begin with one of the model's own tags,
or construction raises: a model could never stop on plain text.

**Special tokens in message content.** A caller's text (a message, a thought, a
call's name and arguments) is tokenized with special-token parsing off, and the
template's tags with it on, so a literal tag typed into a message stays text
and can never forge a turn boundary. A conversation is refused where its
template writes anything derived from a text besides the text itself, such as
the text's length, or something it writes only when the text holds a certain
word. The plain text between two tags, the template's and the caller's alike,
is tokenized as one run, as a render of the whole conversation tokenizes it.

**A template that writes the date** follows the clock: a conversation that runs
past midnight is decoded again once, from the date on.

## Your own cache context

`ChatWrapper(context=...)` takes any object that answers the protocol
`CacheContext`: the model's tokenizer, its tags, its chat template and the
sampling it recommends, and the cache a conversation is decoded into, with the
type one of its members takes: `Sampling`, from `llama_chat`.
Nothing to subclass or register; the members are checked where the context is
adopted, so a missing one, or a method that does not take the arguments its
protocol method takes, positionally and by name, is a `TypeError` from the
constructor. A cache that
can neither shift (`can_shift`), cut its end (`can_cut_tail`) nor save a state
(`save_state` giving something other than `None`) is refused there with a
`ValueError`: it could go back to an earlier point only by decoding the whole
conversation again. The wrapper
reads the [add-on file](#add-on-files) itself, by `model_path` where one is set
and by the template `chat_template_text` gives, and merges the sampling, so a
context of your own gets both as one the wrapper loads does;
`recommended_sampling` gives what the model states for itself, by `Sampling`'s
field names, and each `sampled_tokens` call is handed every value it draws
with. What the template is read for comes from its
renders alone, and where the model's tags lie in a text from its tokenizer
alone: `special_tokens` lists the tags, `is_special` tells one among the tokens
of a text, `tokenize` with special-token parsing on gives each tag in a text as
its one token, and `token_text` spells that token as the text holds it. A
reply is generated by the library's own loop through four steps:
`sampled_tokens` draws, `decode_generated` decodes what the generation adds,
and `ends_generation` and `token_bytes` read a drawn token, so a call is told
apart and a reply ends the same on every context.
`render_with_model_template` and `chat_template_text` take a `template_name`:
`None` for the default template, else the name of a further one the model
ships, such as `TOOL_USE_TEMPLATE` from `llama_chat.cache_context`; a name the
model ships no template of answers `None`. A context whose `chat_template_text()`
answers `None` is refused where it is adopted, as a model that ships no chat
template is, whatever it renders.
Its token ids must lie from -65,536 to 1,048,575: an eviction searches the
cache's tokens for the stretch it dropped, and an id outside that range fails
that search. `tests/fake_context.py` is a worked one over a list of ints.

## Verify

```bash
python -m pytest -q                                  # no model needed
python examples/demo.py /path/to/model.gguf          # against a real model
```

The tests run every chat template in `tests/templates/` with a stand-in
tokenizer, one token per character and one per tag; each test file's docstring
says what it checks. The demo loads a real model with its own tokenizer and
prints how many tokens each action decodes; add `--gpu-layers -1` to run it on
the GPU.
