# What to call, by task

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
calling](tool-calling.md)). Anything else of that form, such as a list of content
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
where that thread has not begun. A question `reply` asks before that thread
begins, on a cache that cannot shift, is answered first where the evictions
would decode the history again, no tools are declared, and the question and a
reply of its cap (`max_reply_tokens`) fit beside the history as it is; the
work then follows its reply, and the question after that waits for it. An
action that waits stops a long decode
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
conversation](keeping-a-conversation.md)), so `begin(**chat.conversation())`
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
raises only the library's own errors, each naming the argument and the cause: a
refusal, an `ArgumentTypeError`, an `EngineError` or an `InternalError`
(`LIBRARY_ERRORS` in `llama_cpp_conversation.errors`). Anything else raised
inside an action is raised as an `InternalError` that names the action and the
cause. Every refusal is raised before the action changes anything. Each is a
`RefusalError`, a `ValueError` whose `field` names the argument or setting it
is about, where it is about one:

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
