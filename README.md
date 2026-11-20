# faster-llama-chat: a KV-cache manager for llama.cpp

A Python chat library over llama.cpp whose goal is to **prefill as little as
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

| | llama.cpp's server | faster-llama-chat |
|------|------|------|
| what is removed | a fixed token count from `n_keep`, which can cut inside a message or a tag | whole exchanges, from a user turn to the next one, so the cache always holds turns the template writes |
| the system prompt | `--keep N`, a hand-counted offset | pinned, never evicted |
| when | when the context is full | above a threshold (`evict_above_fraction`), or above a count of exchanges (`max_history_exchanges`); a request that would not fit evicts first |
| caches that cannot shift | context shift off or unreliable | detected at load; such a cache is cut, restored to a saved state or decoded again instead, and an eviction there drops well below the threshold and below `max_history_exchanges`, so that this happens only every few turns |
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
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.23"   # CPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.23"     # Nvidia GPU
pip install -e .                    # or -e ".[test,dev]" with the test dependencies
git config core.hooksPath hooks     # with them: the checks before every push
```

It installs as `faster-llama-chat` and is imported as `llama_chat`.


## Quick start

```python
from llama_chat import ChatWrapper, Config

chat = ChatWrapper(Config(model_path="models/your-model.gguf", context_size=8192,
                          gpu_layers=-1))

# begin: prefill the system prompt and as much recent history as fits, as Chat
# Completions messages. A template that writes a system prompt where later
# questions move it refuses one (capabilities(tools=False).can_take_system_prompt
# is False); leave the system message out there.
chat.begin([{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}])

# inject: prefill one message as context, without generating
chat.inject("Reference: the deadline is Friday. What's the deadline?")

# reply: prefill only what is new, then stream the reply, which answers the
# injected message
for delta in chat.reply():
    print(delta, end="", flush=True)

reply = chat.reply("Summarize the plan.").finish()    # or read it to its end at once
print(reply.text, reply.stop_reason)
```

A `ReplyStream` yields text deltas; its `reply` is the `Reply` once the stream
has ended of its own accord, and `finish()` reads it to its end and returns
that. Each token reaches the cache before the stream yields the next text or
ends, so closing a stream early records exactly what the cache holds. Where the
model runs on the processor alone (`gpu_layers=0`, or a build without GPU
support), a token's text is yielded first and the token decoded right after,
so each word shows one decode step sooner. An abandoned stream holds
the wrapper's lock until it is collected: read it to its end, close it, or use
it in a `with`. To stop a generation from another thread, pass
`cancel=threading.Event()` and set it; the reply ends within one token with
`stop_reason == "interrupted"`. Set before the reply's first word, while its
question is decoded too, which stops within one part of the decode (512 tokens,
or `micro_batch_tokens` where that is more), it ends the request
with nothing changed: no question is added and no older exchange is dropped.
After a tool's result, set before the next first word, it leaves the turn
open with its results, for `reply()` to go on with. Set while registered
tools run, it runs no later call of that round: the calls that ran keep their
results, the rest stay owed. It also stops, within one such part, the decode the
reply before left for the next action, and the wait for a reply another
thread holds, which then ends at once with nothing changed. A tool that reads
the stream of the reply running it is refused with an `OutOfOrderError`, told
to the model as the tool's failure.

Actions that change the cache are serialized by one lock. `close()`, or
`with ChatWrapper(...) as chat:`, waits for a running generation, frees the
model and makes later actions raise, while `conversation()` and what says
where it stands still answer. A reply holds the lock while it is read.
An action on the thread that reads it ends that reply first, as closing its
stream does, with a warning logged, and then goes on. A `begin` refused by
the checks that read its arguments alone (the declarations, the message forms,
what the model can take, tags and tool names) ends nothing; any other refusal,
such as a history the template refuses, one that does not fit, or a
`truncate_last_reply` text that is no prefix of the reply, has ended it
already. A reply handed to
another thread is read by that thread from then on. An action from another
thread waits for the reply, and where nobody reads the reply on for
`unread_reply_timeout_s`, it ends that reply as closing its stream does (a call
already handed over stays owed, and a question then closes its turn) and goes
on.
A tool the reply runs may not act on the wrapper: such an action is refused
with an `OutOfOrderError`, which the model is told as the tool's failure, and
changes nothing, a `close()` included. A reply asked for and not yet read when
the wrapper closes raises that it is closed once it is read. A
tool runs on a thread of its own, and one that has not returned within
`tool_timeout_s` is answered with a `tool_timeout` failure, and what it returns
later is dropped; with `tool_timeout_s=None` it runs on the reply's thread.

## What to call, by task

| doing what | call |
|------|------|
| starting, starting again, or changing the tools or the reasoning | `begin(messages=None, tools=None, *, reasoning=False, reasoning_level=None, generated_turns=None)` |
| adding context | `inject(text, role="user")` |
| asking | `reply(text=None, *, tool_call_policy=None, max_calls=None, max_reply_tokens=None, max_thought_tokens=None, stop=None, cancel=None, sampling=None)` |
| answering a call yourself | `inject_tool_result(call_id, result)`, then `reply()` |
| shortening the newest reply to what was heard | `truncate_last_reply(keep_text)` |
| reading the conversation back | `conversation()` |
| where it stands | `total_tokens`, `config`, `can_shift`, `capabilities(tools=...)`, `turn_open`, `pending_calls`, `awaits_reply`, `generated_turns()`, `wait_idle(timeout_s)` |
| ending | `close()` |

**Starting, starting again, or changing the tools or the reasoning.** Takes the
conversation as Chat Completions messages: a system message first where there
is one, the pinned system prompt (an empty one is a system turn with no text,
which differs from none on a template that writes a default prompt where none
is given); questions; replies, a reply's thought under `"reasoning_content"`
and its calls under `"tool_calls"` with their arguments as a JSON string; and
each call's result with the role `"tool"` and its `"tool_call_id"` (see [Tool
calling](#tool-calling)). Anything else of that form, such as a list of content
parts, the `developer` role or a `name`, is refused, naming where it stands. Two
questions in a row are joined into one where the template writes no two in a
row, as a running conversation joins them, and a turn a call left open before
a later question is closed as that question closes it live, where the template
does not write the question after it as it stands. A newest reply that does
not fit the context with its closing, as one the full context stopped, goes as
it goes after the reply in a running conversation. It
compares the conversation with what the cache holds and decodes only from where
they differ, so beginning again with the same conversation and other tools or
another reasoning setting is how those change. It keeps the newest history that
fits with the system prompt in half the context, or under the eviction
threshold where the cache holds the conversation already, from a user turn on,
at least the newest exchange and no more than `max_history_exchanges`
exchanges; the newest exchange may take the room a reply leaves, its results cut
to fit as a running conversation cuts one. It returns a `Report` of what it kept
and left out. Older history is dropped without being rendered. A history that
ends on a call keeps it owed a result. Refused where the system prompt and its
declarations leave no room for a short question, where the newest exchange
cannot fit even with its results cut to nothing, or a history with no user
message, which nothing can be evicted from, cannot fit, and where the template
writes a system prompt where later questions move it; a refused or failed
`begin` leaves the conversation it would have replaced as it was.

**Adding context.** Adds one message without a reply, and returns a `Report` of
the older messages evicted for it. A user message is added as `reply(text)`
adds its question, so `inject(text)` followed by `reply()` waits no longer than
`reply(text)`; the eviction it needs may come after the reply instead, and is
then counted by the next request, as every eviction after a reply is. A
question after one no reply follows joins it where the template writes no two
in a row; where the two joined would not fit beside the system prompt, it takes
the unanswered one's place, which counts as evicted. An assistant message, or a
user message once `max_history_exchanges` is reached, evicts first. An eviction
drops up to the next user message, so the messages after the newest one stay;
an assistant message with which they no longer fit under the threshold is
refused. `role` is `"user"` or `"assistant"`: a system message is refused,
since the system prompt is `begin`'s, and so is a tool result, which
`inject_tool_result` writes.

**Asking.** Without `text`, it answers the newest user message already written,
or goes on with a turn a tool call left open. A conversation that cannot take
the reply (none begun, a result still owed, nothing to answer) is refused when `reply` is called, and so are a `text` that
is not a `str`, a `max_reply_tokens` of 0 or less, a `tool_call_policy` that is
none of `TOOL_CALL_POLICIES`, `None` or a `ToolRegistry`, a `stop` that is
neither a `str` nor an iterable of `str`, a `cancel` that is no event and a
`sampling` that is no `Sampling`; the rest is checked when the stream is first
read.

**Answering a call yourself.** Call `inject_tool_result` with the id of each of
`pending_calls`, oldest first, then `reply()`, which keeps the turn's
`tool_call_policy`.

**Shortening the newest reply to what was heard.** `keep_text` is a prefix of
the reply's text, or of the text its stream showed, which after a tool's result
may open with a blank line, and where one stream ran a call and went on, shows
the text before the call too; a cut inside that earlier text cuts the newest
reply to nothing, as the text a call followed stays. A cut that reads as either,
as the opening letters of both, is read as one of the newest reply. The newest exchange stays
after its reply until the next question comes, however long, while it fits the
context, so a long reply can still be cut. The reply that ends a turn that called tools keeps
its tokens as generated up to the cut; cut to nothing, its turn keeps them up
to its newest call. Refused where the template does not write the reply
shortened to `keep_text`, as a template that writes no empty reply does not
write it shortened to `""`.

**After a reply.** Its stream ends once the reply is recorded. What follows it,
the evictions over the limits and the decode ahead of the next question, runs
right after on a thread of the library's own, holding the wrapper as any
action does; the next action, or `wait_idle`, waits for it, or does it first
where that thread has not begun. An action that waits stops a long decode
there between its parts: what that work did stands, and the rest of the
decode is done first by that action. What that work evicted and decoded is counted by the next request: in
its `Reply`, or in the next `Report`; a `begin` leaves it to the conversation
it replaces.

**Reading the conversation back.** Everything `begin` takes, by its parameter
names: `messages` (the Chat Completions messages `begin` takes, the system
message first, with `"reasoning_content"` where the template keeps a reply's
thought, a reply that only calls with `null` content, and each call's
arguments as compact JSON), `tools` (`None` where the conversation declares
none), `reasoning`, `reasoning_level` and `generated_turns` (each turn the model generated with a
call, as its tokens and its own messages, for the same model; see [Keeping a
conversation](#keeping-a-conversation)), so `begin(**chat.conversation())`
begins again as it was. Whatever `begin` takes without leaving anything out,
this gives back unchanged but for the spelling of arguments, an empty reply
text written as said, and two questions in a row joined where the template
writes no two. Safe from any thread, while a reply runs too: it reads
all of it as the newest finished change left it.

**Where it stands.** `awaits_reply` says whether `reply()` without text would
answer something: a question, or a turn whose results are all in.
`capabilities(tools=False)` and `capabilities(tools=True)` each give a
`Capabilities` record of what `begin` accepts for a conversation begun without
tools or with them: `can_call_tools`, `can_reason`, `can_turn_off_reasoning`,
`can_take_system_prompt` and `reasoning_levels`. Both are settled when the model loads, never
change, and are safe from any thread. `total_tokens`, `turn_open` and
`pending_calls` are safe from any thread, as `conversation()` is: each is read
as the newest finished change left it, so a count read while a reply streams is
the one from before its generation, its question included, or, while a tool
the reply runs is running, the one from the newest step of the reply that
finished: its call, or the results written before it.

`stop_reason` on a `Reply` is one of `STOP_REASONS` (the type `StopReason`):

| | |
|------|------|
| `"end_of_turn"` | the model ended its turn |
| `"stop_string"` | one of the texts `stop=` gave (any iterable of them, or a single `str`) matched the answer; the reply ends just before it |
| `"token_limit"` | `max_reply_tokens` ran out, or the context did |
| `"interrupted"` | `cancel` was set |
| `"tool_call"` | the reply stopped on a call, and the turn is open for its result |
| `"unreadable_tool_call"` | it stopped on a call whose body could not be read, and the model could not be told: the template has no result block, or the context has no room left for one. Where it can be told, the reply goes on and ends for another reason; under `"yield"` the model may write another call as long as there is room, and after three unreadable calls in a row it is told not to call again |

`Reply` also carries `text`, `n_prefilled_tokens`, `n_prompt_tokens` (the
tokens the cache held when the reply's last generation drew its first, which
is all that generation was conditioned on), `n_generated_tokens`,
`n_evicted_messages`, `tool_calls`, `tool_limit_reached`, `tool_results_cut`
(the calls whose result was cut to fit, see "Tool calling"),
`n_reasoning_tokens`, `thought`, the thought's text, empty where the model
thought nothing, `evicted_for`, why history went for it, one of
`EVICTION_CAUSES`, `taken_back`, whether the reply was taken back with its
request because the template writes neither it nor an empty reply, and
`kept_text`, the answer as kept where the template keeps less of it than was
shown. A stream closed early keeps the newest generation's `Reply`; what it
generated is recorded all the same.

A `Report`, which `begin`, `inject` and `inject_tool_result` return, holds
`n_evicted_messages` and `evicted_for` as a `Reply` does; for a `begin`,
`n_kept_messages` (the history messages it kept), `n_left_out_calls` (calls
that could not be read, left out with their results), `n_turns_restored` (the
turns it restored as generated) and `generated_turns_ignored` (the generated
turns given were another model's); `tool_results_cut`; `turn_closed`, whether
the action closed a turn a call left open; `question_joined`, whether a
question was joined to one no reply followed; and `taken_back`, whether a
reply was taken back with its request, as one a result found no room for is
where nothing of it is left that the template writes.

A count of messages, as in `n_evicted_messages` or what `begin`, `inject` and
`inject_tool_result` return, counts the messages `conversation()["messages"]`
holds after the system message: a question, a reply and each call's result are
one each. It is what the action itself evicted, and for the first action after
a reply, what went after that reply to make room for it.

**What fails at startup.** Whatever can be settled once is settled while the
wrapper is built, so a deployment that could never work says so then: every
`Config` field and the combinations no call could succeed under, that
`model_path` names a file, each cache edit tried once on the empty cache, that
a context of your own answers everything called on it, holds the
`context_size` asked for, and can shift its cache, cut its end or save a
state, that the chat template is valid template code (a
`ValueError` naming the line and the cause where it is not), that the
generation prompt and the closing after a reply are where the template writes them and the closing begins with one of the
model's tags, or, where the template writes nothing after a reply, that the
tag opening the next turn ends a generation for the model, and whether tool calling works (`capabilities(tools=True).can_call_tools`). A setting
refused is an `InvalidSettingError`, a `ValueError` whose `field` names it.
What the model cannot do (call tools, reason, keep a system prompt in place)
is refused where it is asked for with an `UnsupportedError`, a `ValueError`
whose `field` names the parameter that asked for it.

**What an error tells you.** Once the wrapper is built, every public action
raises only the library's own errors, each naming the argument and the cause:
a refusal, an `ArgumentTypeError`, an `EngineError` or an `InternalError`
(`LIBRARY_ERRORS` in `llama_chat.errors`). Anything else raised inside an
action is raised as an `InternalError` that names the action and the cause.
Every refusal is raised before the action changes anything. Each is a `RefusalError`, a `ValueError` whose `field` names
the argument or setting it is about, where it is about one:

| Refusal | What it means |
|---|---|
| `InvalidSettingError` | a setting of `Config`, `Sampling` or a request holds a value it refuses |
| `InvalidArgumentError` | an argument of a call holds a value it refuses |
| `UnsupportedError` | the model cannot do what was asked: call tools, reason, keep a system prompt |
| `TemplateRefusedError` | the model's template does not write what was asked |
| `OutOfOrderError` | the action needs another one first, which the message names |
| `ContextOverflowError` | it does not fit; `needed_tokens` and `available_tokens` say by how much |

A setting of the wrong type, of `Config`, of `Sampling` or a request's own
(`max_reply_tokens`, `max_calls`), is an `InvalidSettingError` naming it, as
a value out of its range is. Any other argument of the wrong type, or one an
action does not take, is an `ArgumentTypeError`, a `TypeError` that names the
argument. A text holding a
lone surrogate (half of a UTF-16 pair, which no model can read) is refused with
an `InvalidArgumentError` naming its argument; a tool result holding one is
told to the model as an `unwritable_result` failure. A `ToolSpec` is checked
where it is built: text keys only, a JSON value every model can be given, no
`NaN`, nested at most 100 levels. An `EngineError`, a
`RuntimeError`, says the engine failed while it decoded, edited or generated;
the next action brings the cache back first. Where it failed while the results
of the tools a reply ran were written, those results are kept and written by
the next `reply()` or `inject_tool_result()` (a result handed over for the
same call takes the place of the kept one), so no tool that ran is told to the
model as not run. An `InternalError`, a
`RuntimeError` too, is a fault of this library, found before it could spoil the
conversation: a loop that made no progress, or, with `check_cache`, a cache that
does not read as the render. After a loop that made no progress the
conversation stays usable, as after any failed action; a fault the check finds
in the cache is raised again by every reply until the conversation begins
again. Either is a fault to report.
What is left is what only a conversation can know: whether a request fits
(`ContextOverflowError`, a `ValueError`, for every message, history, declaration
or reply that does not), whether the model calls a tool that exists, and what
the tool does. A question never fails to fit: one too long for the room the
system prompt leaves under the eviction threshold keeps as much of its end as
fits, possibly none of it, `QUESTION_CUT_MARK` in place of its beginning, and
a warning is logged. A question that waits for its reply is cut again when the
reply starts where the system prompt has grown since, as a date the template
writes may overnight; it keeps one mark. `begin` refuses a system prompt and
declarations that would leave no room for a short question and the room a
reply keeps (`min_reply_tokens`, at least the shortest reply that holds an
answer), measured on the day of the year whose date the template writes
longest, so no conversation it accepts is one that cannot answer.

## Tool calling

```python
from llama_chat import ChatWrapper, Config, ToolCall, ToolRegistry

registry = ToolRegistry()

@registry.tool
def get_weather(city: str) -> dict:
    """Current weather for a city."""
    return {"city": city, "celsius": 18}

chat = ChatWrapper(Config(model_path="model.gguf"))
chat.begin([{"role": "system", "content": "You are a terse assistant."}], tools=registry.specs())
print(chat.reply("What is the weather in Munich?", tool_call_policy=registry).finish().text)
```

`tool_call_policy` says what becomes of a call the model makes, whether or not
tools were declared, since a model may write a call tag unasked. It is one of
`TOOL_CALL_POLICIES` (the type `ToolCallPolicy`) or a `ToolRegistry`. Left
out, it is the policy the open turn was started with, where the reply goes on
with one when it runs, and `"refuse"` otherwise; a reply that goes on after
results thus keeps its turn's policy unless it is given another:

| `tool_call_policy=` | what happens | what the stream yields |
|------|------|------|
| `"withhold"` | the model cannot write a call: the token that opens one is never drawn. A model that wanted to call may write something else entirely instead, so this suits a request that wants no tool. Where the template's call opens with plain text rather than one of the model's tags, no token can be withheld: the reply ends where such a call opens, what the model wrote before it kept, so a reply that opens with a call is empty, and the call is reported `"unfinished"` | text |
| `"refuse"` (the default for a new turn) | nothing runs, not even a tool that was declared. The model is told no tool can be run and answers in words, the token that opens a call no longer drawn, as under `"withhold"` | text |
| `"yield"` | each call is handed to you after the text, and the turn stays open: answer each with `inject_tool_result(call.id, result)` in order, then call `reply()` again. A call the library cannot read is answered with a failure and the reply goes on by itself; the model may write another call as long as there is room, and after three unreadable calls in a row it is told not to call again and answers in words | text, then each `ToolCall` |
| a `ToolRegistry` | each call runs against it until the model answers. In the last round `max_calls` allows, the token that opens a call is never drawn, as under `"withhold"`, so the turn always closes. A call that spends the last of `max_reply_tokens` is not run, since nothing would read its result: the turn closes as the cap closes one, and the call is reported `"unfinished"`. `max_calls` is refused with any other `tool_call_policy=`, and where it is not a whole number of at least 0. A stream closed at a call handed over leaves that call and the rest of its round unrun and owed, as under `"yield"`, and the results of the calls before it that ran written. `max_calls` counts every call handed over, a turn opened without one takes `Config.max_calls`, and a reply that goes on with the turn without a `max_calls` of its own keeps what is left of it | text, and each `ToolCall` that runs, just before it runs |

```python
with chat.reply("What is the weather in Munich?", tool_call_policy="yield") as stream:
    for event in stream:
        if isinstance(event, ToolCall):
            chat.inject_tool_result(event.id, registry.call(event))
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
and one the reply cap or a cancel cut short, the room cut at its plain
opening, or whose plain opening ended a reply under `"withhold"`, or one not
run because its turn closed for want of room, as `"unfinished"`. A call whose
tool ran but whose turn closed for want of room before its result was written,
so that the model never read what the tool returned, is `"result_dropped"`.
The status is of the type `ToolCallStatus`. Neither an unreadable nor an
unfinished call is handed over in the stream or listed in `pending_calls`. An
unfinished call has no name, arguments or id, except one read whole that was
not run for want of room, which keeps them. An unreadable call has no name or arguments; where the reply stopped on
it, it carries in `ToolCall.text` what the model wrote between the call's
opening and its closing, and as an entry of a batch it carries its id.
`conversation()` leaves it out of its messages with its result, the reply it
stood in going on in the next reply of its turn, and two questions this leaves
in a row joined where the template writes no two in a row; the turn's own
messages in `generated_turns` keep it, so beginning again on the same model
restores the turn as it was generated, the call and the failure the model was
told included. A `ToolCall` may be kept in a set or used as a
dict key, and equals another only where every field does. Where calls run, the `Reply` sums
every generation of the exchange, `n_prompt_tokens` aside, which is the last
generation's, and `tool_limit_reached` says whether a `max_calls` ceiling ran
out.

**Tools a model could not use are refused where you hand them over**, by
`begin(tools=...)` and a running `reply`, naming what
is missing: a template that describes a call, renders the declarations, has a
readable argument spelling, a block for the result, and keeps a past call
written once a later question follows it, with an
`UnsupportedError` whose `field` names the parameter the tools came in.
`capabilities(tools=True).can_call_tools` answers the same question up
front. A declaration that names one tool twice is
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
again. A value that a call format writes without marking its type is read by
the type declared for it, so text keeps its spelling (`01234` stays
`"01234"`); a number where text is asked is taken as the text it was written
as, and an enum's value reaches the function as its member. A function that is
asynchronous, takes a parameter only by position, or has no description is
refused when it is registered. `registry.specs()` gives each tool's
`ToolSpec`, the declaration the model is told, so `begin(tools=registry.specs())`
declares them, and a `ToolSpec` of your own declares a tool that this process
does not run.

**When the model is told.** `begin(tools=...)` declares the specs it is given
when it writes the conversation; beginning again with the conversation and
another list declares that one instead. The declaration goes where the
template writes declarations. Where that is the system turn, a later
declaration decodes the conversation after it again once. No registry is kept:
a request whose calls are to run is handed one.

**A template of its own for tools.** A GGUF may ship, beside its default chat
template, one for a conversation that declares tools
(`tokenizer.chat_template.tool_use`). Where it does, both meet the check a
model meets at load, `capabilities(tools=True)` answers for that one and
`capabilities(tools=False)` for the default, and a conversation is written
with it when `begin` is given `tools`, even an empty list; with `tools=None` it
is written with the default template. Where the template for tools fails that
check, the model loads with its default template alone: a warning names the
cause, `capabilities(tools=True)` says it calls no tools, and declaring a tool
is refused with that cause. Beginning again with the conversation moves it
from one to the other, its messages kept.

**Every failure is told to the model** in its own result block, and the reply
goes on: an unreadable call, a call under `"refuse"`, an unknown name,
arguments that do not fit, a tool that raised, a spent `max_calls`. Each says
that the call failed, why, and `try again` where a call the model writes next
is read, else `do not call again`, as under `"refuse"`, once `max_calls` is
spent, or after the third call in a row that could not be read, under every
policy. A result that a registered tool returns as a dict carrying
`error` counts as a failure too, and so does one that JSON cannot write, `NaN`
and infinity included. A result handed to `inject_tool_result` is told the
same way, a dict carrying `error` included; one that JSON cannot write is
answered with the same `unwritable_result` failure, and a warning on the
logger `llama_chat` says so. A call is read only where all of its text is
read: text left after it, a blank pair or a key written twice makes it
unreadable. Each call of a batch is judged on its own, so one entry that is no
call is told so in its place while the others are answered.
A call whose arguments are no object, such as a list or `null`, or nest
deeper than 100 levels of lists and objects, is unreadable, and its tool does
not run.

**A turn grows towards `context_size`** as it calls, since an eviction would
cut into the turn being written. A result too long for what is left, with
the room a reply keeps free, first evicts the oldest exchanges, as a question
does, the open turn kept whole: its question, its calls and its results; on a
cache that cannot shift it evicts well below that, as every eviction there
does. `inject_tool_result` returns a `Report` of the history messages that went, and under a
registry they count in the reply's `n_evicted_messages`. A result still too
long is cut, its tags kept, and ends with a note the model reads, such as
`(cut: 3100 of 20000 characters kept)`; the next reply lists its call in
`Reply.tool_results_cut`. Where not even a result cut to nothing leaves room
for the reply, the turn closes on what the model wrote, as the reply cap
closes one: a reply that answers its calls itself, under `"refuse"` or a
registry, ends with `"token_limit"`, each call not yet answered reported
`"result_dropped"` where its tool ran and `"unfinished"` where it did not, and
after `inject_tool_result`
`turn_open` is false, the calls not yet answered are dropped, and the next
question is asked as after any reply.

**A history with tool turns** goes into `begin` as Chat Completions messages:
a reply holding its calls, `{"role": "assistant", "content": text or None,
"tool_calls": [{"id": id, "type": "function", "function": {"name": name,
"arguments": "{...}"}}]}`, then each result, `{"role": "tool",
"tool_call_id": id, "content": text}`, in the order of the calls. Every call
needs its id. A call whose arguments are no JSON object, `""` aside, which is
none, or that names no function, could not be read: it is left out with its result, said once at
`INFO`, and a later result for it is refused naming that cause. A result out
of order, one that answers no call, a reply between a call and its result, or
a result that is no text, is refused naming it. It is written as the template
writes a past turn. A result handed to `inject_tool_result` as a value rather
than text is written as JSON at once, and `conversation()` gives back that
text, cut with its note where it was cut to fit.
A history ending on a result leaves that turn open, and `reply(tool_call_policy="yield")`
goes on with it. No reply started such a turn, so the first reply that goes on
with it and is given a policy gives the turn that policy, which a later
`reply()` keeps. A turn that closed ends on the text of the reply that closed
it, in `conversation()` too, an empty `str` where the model wrote nothing after
its results, so that `begin(**chat.conversation())` begins it closed. A turn
followed by a question while it still owes a result is written as given,
nothing added to it, and `conversation()` hands it back in its place, before
that question.

**Every call has an id**, `ToolCall.id`, that no other call of the
conversation has. A call handed over in the stream, in `Reply.tool_calls`, in
`pending_calls` and in `conversation()` carries it, so one call reads the
same in all of them. A call of a history keeps the id it carries, so
`begin(**chat.conversation())` keeps every id; a history that gives two calls
one id is refused. A call made after a `begin` takes an id that none of the
history's calls had, those that did not fit or were left out included, and
that no call before it in the same `ChatWrapper` had, evicted ones included.
`inject_tool_result` takes the id of the call it answers.

## Reasoning

```python
chat.begin([{"role": "system", "content": "You are a helpful assistant."}], reasoning=True)
reply = chat.reply("Why is the sky blue?").finish()
reply.text                  # the answer alone
reply.thought               # what the model thought before it, without its tags
reply.n_reasoning_tokens    # about how many tokens the thought took
```

Reasoning is chosen per conversation, at `begin`, and changed by beginning
again with the conversation, because a template may switch it in the system
turn. `reasoning=False`, the default, asks for none; `reasoning=True` for the
template's own way of reasoning; and `reasoning=True` with an effort level,
such as `reasoning_level="low"`, for that level, where the template knows it.
A level without `reasoning=True` is refused, naming `reasoning_level`.
`capabilities(tools=...).can_reason` says whether `begin` accepts
`reasoning=True`, which it does where the template marks a thought and its
prompt with reasoning on leaves that thought open, and `reasoning_levels` lists
the levels `reasoning_level` takes. They are the levels the template tells apart,
read from the template when the model loads, each one writing the prompt
reasoning on writes; where the template writes any level it is given as it is,
the levels are those the model's [add-on file](#add-on-files) names. A level
the template does not know is refused, naming `reasoning_level`. Reasoning
off is the template's own off: `can_turn_off_reasoning` in the same record says whether the
template has one. Where it has none, as on a template that always reasons, a
conversation begun without reasoning is written as the template writes it,
and the model may reason in every reply all the same. Where a template's
switch cannot be read from the template alone, the model's
[add-on file](#add-on-files) names the template variables that turn it on.
Every
stream yields the answer alone, with or without reasoning, and a header a
template writes for another reader is kept out of it too. Once the answer has
begun, a thought opens only where the model writes its own thought tag, not
where the answer spells the tag's letters, and any other tag of the model's
that it writes inside its answer is left out of the stream. Stop strings are
matched against the answer only. A finished reply is kept as the template
writes a past reply, without its thought where the template leaves one out,
and with it, handed over in the form the template reads one in, where the
template keeps it. An answer that holds the thought tag's text is kept as
the part of it the template writes as text, and as an answer with no text
where that is nothing. Where that differs from what was generated, the cache is brought to that form
right after the reply, while it is read. A turn that called tools keeps the
tokens it was generated as. A thought the template keeps comes back from
`conversation()`: a reply's as `{"role": "assistant", "content": text,
"reasoning_content": thought}`, which `begin` takes as well, a reply that
called tools included.

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

**Exact restore, and another model.** The messages are in no model's own
format: a call is its name and arguments, a thought plain text. A turn the
model generated with a call is kept in the cache as the tokens it generated,
which no template writes back exactly, so `conversation()` also gives each
such turn as `generated_turns`: its tokens, and its own messages as the
conversation holds them, a thought and a call the model wrote unreadably
included; `generated_turns()` gives them alone. Given back to `begin` on the
same model, each turn is found by its Chat Completions messages and the
question before them, a thought or an empty text aside, wherever it now stands
among them, so generated turns kept beside messages you add to or drop from
still serve them; it is put back as the conversation held it and restored
token for token. On another model, or without them, every turn is written as
that model's template writes it from the messages alone. `generated_turns` is
JSON-ready and only meaningful to the model that wrote it; reasoning on or off
and the declared tools carry over to another model, its other settings come
from it. It is trusted input: its tokens go into the cache as they stand, the
model's own tags included, and nothing checks them against the messages'
text, so a `generated_turns` someone edited can put any turn, a system turn
included, in front of the model. Give back only what `conversation()` gave,
kept where nobody else can change it; where a saved conversation passes
through hands you do not trust, begin again from its messages alone.

To keep everything, older turns included, keep each message as you send it
and each reply as you read it. A request whose generation fails before it says
anything changes nothing: its question is taken back, and the older exchanges
it pushed out come back. So does a request whose reply the template refuses to
write, and one whose reply produced nothing, where the template writes no
empty reply: one the model ended at once, or one closed for lack of room after
its calls, which takes the turn's calls and results back with it. Where the
template writes an empty reply and two questions in a row, a reply to a
question that has no answer, such as one that only thought, is taken back
alone: its question stays, the next
question follows it, and its thought is not kept.

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
| `penalty_window_tokens` | the model's | how far back the penalties look, within one generation; at least `0` |
| `seed` | `None` | in `[0, 2**32 - 2]`, which every generation starts from; `None` draws a fresh one |

The eight marked "the model's" take, unless set, the model's own
recommendation, value by value: its [add-on file](#add-on-files)'s first,
then what its GGUF states for a value the file leaves out, then llama.cpp's
default (`temperature` 0.8, `top_k` 40, `top_p` 0.95, `min_p` 0.05,
`repetition_penalty` 1, `presence_penalty` and `frequency_penalty` 0,
`penalty_window_tokens` 64). A penalty window the GGUF states as `-1`, which llama.cpp reads as
the whole context, is taken as `context_size`. Any other value the GGUF states
that `Config` would refuse stops the load with a `ValueError` naming the GGUF
as its source, unless `Config` or the add-on file replaces it.
`Sampling(...)` passed to one `reply` overrides any field of this table but
`model_path` for that request. `Sampling` refuses, where it is built, any
value `Config` refuses.

**Context management**

| field | default | meaning |
|------|------|------|
| `context_size` | `4096` | the hard limit in tokens |
| `evict_above_fraction` | `0.75` | the fraction of `context_size` above which the oldest exchanges are evicted, in `(0, 1]` |
| `max_history_exchanges` | `None` | a second eviction trigger: the most exchanges kept, at least `1`. An exchange is one user message and everything after it up to the next, the unit an eviction drops; what a history holds before its first user message counts as one more. The question of a `reply` may open one exchange more while its reply runs, and the oldest is dropped once the reply ends, so that no eviction for this limit delays a reply |
| `max_reply_tokens` | `None` | cap on the tokens of one reply, every generation of it together, a thought included; `None` sets none, so a reply may run until the context is full |
| `max_thought_tokens` | `None` | the most tokens one thought may take; a thought that reaches it is closed as the template closes one, and the answer follows; `None` sets no cap; `reply(max_thought_tokens=...)` sets it for one request |
| `min_reply_tokens` | `32` | the room a reply keeps: a `begin` that leaves less beside a short question is refused with `ContextOverflowError`, and a longer question is cut to leave it; never less than the shortest reply that holds an answer takes, which is all `0` keeps |

**Replies**

| field | default | meaning |
|------|------|------|
| `min_answer_tokens` | `1` | the end of the turn is not drawn until a reply holds this many tokens of answer, a thought left out, or a call, and a stop string counts only once the answer holds text, so a model that would end at once answers; `None` lets it end at any token |
| `answer_room_tokens` | `1024` | the tokens a reply keeps for its answer where the model thinks first: a thought that reaches the rest of the generation's room, at most half of it, is closed as the template closes one, so the model answers; `None` lets a thought run until the room is spent |
| `max_calls` | `10` | the ceiling on the calls one turn run against a registry may use, where a request gives none; in the last permitted round the model can no longer call, so it answers; `None` sets none |
| `tool_timeout_s` | `60.0` | how long a tool a registry runs may take before it is answered with a `tool_timeout` failure; `None` waits for it |
| `unread_reply_timeout_s` | `10.0` | how long an action waits for a reply nobody reads on before it ends that reply as closing its stream does; `None` waits until it is read, closed or collected |

**Backend and hardware**

| field | default | meaning |
|------|------|------|
| `gpu_layers` | `-1` | layers offloaded to the GPU: `-1` for all, else at least `0`; `0` uses no GPU at all, the processor alone running every step |
| `kv_cache_on_gpu` | `True` | each layer's KV cache where that layer runs, in VRAM for the layers `gpu_layers` offloads; `False` keeps all of it in host RAM |
| `decode_threads` | `None` | CPU threads for generating a reply, a token at a time, at least `1`; `None` uses one per physical core of the processors the process may run on, at most as many as its CPU quota allows |
| `prefill_threads` | `None` | CPU threads for prefill, the decode of tokens the cache is given (a question, or what is decoded again), at least `1`; `None` means as many as `decode_threads`, and so one per physical core where that is not set either |
| `max_batch_tokens` | `2048` | most tokens in one decode; a longer prefill is split |
| `micro_batch_tokens` | `None` | most tokens the engine computes at once, at most `max_batch_tokens`; `None` leaves the engine's default. A larger one can read a long prefill faster and takes more memory on the device |
| `flash_attention` | `None` | `True` or `False` forces flash attention on or off; `None` leaves it to llama.cpp, except under a quantized KV cache, which requires it and turns it on |
| `kv_cache_type` | `"f16"` | `"f16"`, `"q8_0"`, `"q5_1"`, `"q5_0"`, `"q4_1"`, `"q4_0"`; a quantized type requires flash attention, so is refused with `flash_attention=False` |
| `kept_conversation_bytes` | `0` | the most host RAM, in bytes, the caches of conversations a `begin` switched away from may take once kept, so that going back to one decodes only what changed since; the oldest goes first, and while one is saved it takes its own size beside them; only a cache that can shift or cut keeps any; `0` keeps none |
| `check_cache` | `False` | for finding faults: after the work after every reply and before every generation, compare the cache with a fresh begin of its conversation, and raise `InternalError` where they differ; each check renders and tokenizes the whole conversation |

`ChatWrapper` also takes, outside `Config` because neither is a value to
tune: `context=` (a [cache context of your own](#your-own-cache-context)) and
`on_load_progress=`, called with a fraction while the weights load, which must
not raise.

**Logging.** Everything llama.cpp writes, and what the library itself has to
say, goes to Python's `logging`, under the logger `llama_chat` and llama.cpp's
lines under its child `llama_chat.engine`, one record per line: llama.cpp's
errors as `ERROR`, its warnings as `WARNING`, its load-time listings as `INFO`
and `DEBUG`. With no logging configured, Python shows warnings and errors on
stderr, and nothing below them whatever level a logger is set to; showing
`INFO` and `DEBUG` lines takes a handler, such as the one `logging.basicConfig()`
puts on stderr. The library's own levels mean one thing each: `INFO`, the
conversation changed in a way the caller did not ask for, such as history that
went ("the 80 oldest history message(s) went so that half of the 8192-token
context stays free at a begin"), a call left out, a result cut to fit, a reply
taken back, or an open turn a question closed; `DEBUG`, a slower
path was taken, such as the cache decoded again; `WARNING`, something to fix,
such as an add-on file that is not used. Each `INFO` change is also in the
`Report` the action returns (`begin`, `inject`, `inject_tool_result`) or in
the `Reply`; why history went is one of
`EVICTION_CAUSES`, typed as `EvictionCause`. To show only errors, or everything:

```python
import logging
logging.basicConfig()               # a handler on stderr that shows every level it is given
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
`llama_chat` says why; one that is not JSON at all is refused when the model
loads, naming the file. It fills in only what the template cannot tell; what
the template states, the template decides, except a call format the file
names, which wins. The file used is checked when the
model loads, each value's type included and a null value refused, and
anything else in it is refused with an error naming the file.

```json
{"templates": ["<sha256 of the chat template>"],
 "source": "where the values come from",
 "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
 "call": {"opening": "...", "name_close": "...", "closing": "..."},
 "reasoning_on": {"reasoning_effort": "high"},
 "reasoning_levels": ["low", "medium", "high"]}
```

`sampling` sets any of the eight model sampling fields, `presence_penalty`,
`frequency_penalty` and `penalty_window_tokens` among them, and its values win over the GGUF's (see
[Configuration](#configuration)). Since a shipped file is matched by the
template, it also applies to a fine-tuned model that keeps the same template,
and its values win over those that fine-tune's GGUF states; a file next to the
GGUF, or values set in `Config`, take precedence for such a model. `call` says
how the model writes a call: the text a call opens with, the text after the
function's name, and the text that closes it. It is taken over whatever the
template shows of a call, and refused where the template never writes that
closing.
`reasoning_on` names the template variables, with their values, that turn the
model's reasoning on, in place of what the library reads off the template. The
library reads only the switches templates share: `enable_thinking`, and a
`reasoning_effort` whose `"none"` turns reasoning off, the template's default
then being on. Any other switch is named here, such as an effort whose
`"high"` turns reasoning on, and so is the case of a template whose default
does not reason although it writes something of its own for `"none"`. A
template whose prompt with reasoning on closes the thought at once is read as
unable to reason. `reasoning_on` is refused at load where it writes the prompt
as reasoning off does.
`reasoning_levels` names the effort levels `begin(reasoning_level=...)` takes, for a
template that writes any `reasoning_effort` it is given into the prompt as it
is, so that the template itself tells no level apart. Each one is checked when
the model loads, as a level read from a template is; a file that names a level
the template does not take as one, or names levels for a template that tells
its own apart, is refused at load with the cause.

## A model without a usable template

A model that ships no chat template is refused when it loads: the template is
what every conversation is written with. For one whose template writes no
tool call it can be read from, its [add-on file](#add-on-files) says how the
model writes one. The closing after a reply must begin with one of the model's own tags,
or construction raises: a model could never stop on plain text. A template
that writes nothing after a reply is used only where the tag that opens the
next turn ends a generation for the model, since the reply ends where the
model draws that tag.

### Special tokens in message content

A caller's text (a message, a thought, a
call's name and arguments, and each description of a declared tool) is
tokenized with special-token parsing off, and the template's tags with it on,
so a literal tag typed into a message or a tool's description stays text and
can never forge a turn boundary. The rest of a declaration (a tool's name, a
parameter's name, a choice value, a title) is what a template reads to decide
what it writes, so it is written as it stands: `begin` refuses one that holds
a tag with an `InvalidArgumentError` naming `tools`. A conversation is refused where its
template writes anything derived from a text besides the text itself, such as
the text's length, or something it writes only when the text holds a certain
word. The plain text between two tags, the template's and the caller's alike,
is tokenized as one run, as a render of the whole conversation tokenizes it.

### A template that writes the date

A template that writes the date follows the clock: a conversation that runs
past midnight is decoded again once, from the date on.

## Your own cache context

`ChatWrapper(context=...)` takes any object that answers the protocol
`CacheContext`: the model's tokenizer, its tags, its chat template and the
sampling it recommends, and the cache a conversation is decoded into, with the
type one of its members takes: `DrawnSampling`, every sampling value set, from
`llama_chat`.
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
field names other than `seed`, and each `sampled_tokens` call is handed every value it draws
with. A context gives its chat template as text
(`chat_template_text`) and the text of its end-of-sequence token
(`eos_token_text`), which a template may write as `eos_token`; the library
renders the template itself, in the sandboxed template environment chat
templates are written for, so a context renders nothing. Where the model's
tags lie in a text comes from its tokenizer alone: `special_tokens` lists the tags, `is_special` tells one among the tokens
of a text, `tokenize` with special-token parsing on gives each tag in a text as
its one token, and `token_text` spells that token as the text holds it. A
reply is generated by the library's own loop through four steps:
`sampled_tokens` draws, `decode_generated` decodes what the generation adds,
and `ends_generation` and `token_bytes` read a drawn token, so a call is told
apart and a reply ends the same on every context; a call's closing tag is
decoded with `prefill` asking for no logits, since a result follows it, not a
draw. `decode_blocks` says
whether `decode_generated` returns only once the decode is done; where it
does, each token's text is yielded before the token is decoded.
`save_whole_state` and `restore_whole_state` keep and put back everything the
cache holds, for `kept_conversation_bytes`; a context that cannot may raise
`RuntimeError` from the first, and no conversation is then kept.
`chat_template_text` takes a `template_name`: `None` for the default template,
else the name of a further one the model ships, such as `TOOL_USE_TEMPLATE`
from `llama_chat`; a name the model ships no template of answers `None`. A
context whose `chat_template_text()` answers `None` is refused where it is
adopted, as a model that ships no chat template is. `model_fingerprint` names
the model its tokens are valid for: `generated_turns` restore turns as
generated only on a context whose fingerprint is the one they were saved on.
Its token ids must lie from -65,536 to 1,048,575: an eviction searches the
cache's tokens for the stretch it dropped, and an id outside that range fails
that search. `llama_chat.testing` ships a worked one, `FakeContext` in
`llama_chat/testing/fake_context.py`, over a list of ints and a scripted model,
with the small invented templates it renders in
`llama_chat/testing/invented_templates.py`, for tests of code that drives a
wrapper without loading a model.

## Verify

```bash
python -m pytest -q                                  # no model needed
python -m pytest -q -n auto                          # the same, on every processor core
python -m pytest -q -m slow                          # many new random sequences of actions
python -m pytest -q -m gpu                           # real model files, named as below
python examples/demo.py /path/to/model.gguf          # against a real model
```

The suite also runs the linter and the type checker, and checks the writing
rules of the prose and that every file the contract names exists. Every test
has a time limit; past it the run stops and prints where each thread stood.
`LLAMA_CHAT_CHECK_EVERY_ACTION=1` runs the suite with `check_cache` on in every
wrapper, the tests that count the library's work left out.

The random sequences are drawn by Hypothesis: a failing one is cut down to the
shortest that still fails and printed step by step, with what replays it. The
suite draws the same ones every run; `-m slow` draws new ones,
`LLAMA_CHAT_SLOW_EXAMPLES` of them, 1500 by default. Nothing of either is kept
on disk by Hypothesis; a sequence once found failing is kept in
`tests/found_sequences/` instead, as what it drew and its steps, and replayed
every run.

The tests on real model files read their paths from `LLAMA_CHAT_TEST_MODELS`,
separated as the system separates paths, and offload `LLAMA_CHAT_TEST_GPU_LAYERS`
layers, all by default. Each runs a long conversation with `check_cache` on and
bounds how much the process grows; `LLAMA_CHAT_TEST_GPU_MEMORY_COMMAND`, a shell
command that prints the MiB the process `{pid}` holds on the GPU, bounds GPU
memory too; the design history, "Faults that show themselves", holds one.

`./install.sh --dev` switches on `hooks/pre-push`, which refuses a push the
suite and a shorter slow run do not pass. Mutation testing,
`python -m tests.mutation_run` and then `mutmut results`, changes the code on
purpose and lists each change no test noticed; it runs for hours, and writes its
copies and results to `mutants/`. A later run tests again what changed since: a
changed library function, and whatever a test edited, added or removed since
reaches, all of it where code the tests share changed. The tests that
read the source text or trace functions by name are left out of it, since they
would read the tool's rewritten copy (`[tool.mutmut]` in `pyproject.toml`).

The tests run every chat template in `tests/templates/` with a stand-in
tokenizer, one token per character and one per tag, and again with one that
merges a space with the letter or digit after it; each test file's docstring
says what it checks. The demo loads a real model with its own tokenizer and
prints how many tokens each action decodes; add `--gpu-layers -1` to run it on
the GPU.
