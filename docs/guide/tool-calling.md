# Tool calling

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
