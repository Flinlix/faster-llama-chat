# Design history

Why faster-llama-chat is built the way it is: the mechanism behind what the
README promises, the measurements, and the decisions with the alternatives
they ruled out. Newest entries last.

## 2026-11-07: The mechanism, moved here from the README

The README now says what a caller can rely on. How the library achieves it is
recorded here, as the README described it before the structure review below.
Where the two disagree, the review's entry holds: eviction removes whole
exchanges, the six reply entry points are one `reply()`, and a later
declaration joins the system turn.

### How eviction works

Superseded in part on 2026-11-08: the message table is gone, and what the cache
should hold is the template's render, "Every text placed by a marker render".
The triggers and the cache operations below still hold.

Eviction has two triggers, and they are OR'd: the cache is trimmed when it
crosses `evict_above_fraction × context_size`, and again when it holds more than
`max_history_messages` non-system messages. Whichever binds first evicts, so a
verbose conversation is bounded by its tokens and a long one by its message
count. The second is off by default.

A pure-Python `MessageTable` is the single source of truth: the cache for
sequence 0 is a contiguous run `[0, total)`, each message owns a `[start, stop)`
slice, and the table recomputes every position after any structural change. To
evict the oldest non-system message at `[a, b)`:

```
llama_memory_seq_rm(seq, a, b)               # drop those tokens
llama_memory_seq_add(seq, b, total, -(b-a))  # shift survivors down to close the gap
```

`seq_add` also fixes the RoPE positions, so the survivors stay coherent without
re-prefilling them. It does not do that work where you call it: `seq_add` only
marks the cells as shifted, and the RoPE K-shift over the whole cache is built
into the graph of the **next** `llama_decode` (the shift is expensive only for a
quantized cache). That cost is therefore paid by whatever decodes next, which is
why `reply(calls="refuse")` decodes the assistant turn's closing tag
*after* the post-reply eviction rather than before it: a decode the turn
performs anyway then carries the shift, instead of the next generation's
prefill.

This reuse is *lossy* (the survivors' cached K/V were computed
while attending to the now-removed tokens), which is the standard StreamingLLM
trade-off: fast, with slight quality drift. The system prompt is never evicted.

The strategy is chosen automatically at construction from
`llama_memory_can_shift()` and a cut tried on the empty cache. A cache that
refuses the cut holds a running state and is never shifted, whatever it
reports: a hybrid cache reports the shift of its attention part, and takes
every removal that stops short of its end without its running state losing
those tokens.

- **shift** (default, the fast path): the in-place edit above, applied
  incrementally as messages are dropped. Used everywhere it can be, including
  sliding-window models under a full-size cache.
- **rebuild** (caches that cannot shift: compact sliding-window, recurrent or
  Mamba state, where dropping tokens loses the position information shifting
  needs):
  drop the oldest messages from the bookkeeping, then re-prefill the surviving
  conversation once. Correct, and it costs one prefill of the survivors; a
  non-shiftable model degrades gracefully instead of corrupting the cache. Such
  a cache may still take a cut at its end, which a load-time probe settles;
  then the newest reply is shortened by a cut. One that cannot, a recurrent or
  windowed one, has the part of its state a cut cannot reach saved whenever
  the cache comes to rest, the last two kept in RAM, one from before the
  newest reply and one from after it. That part's size is the model's, not
  the conversation's. A reply
  kept in another form than it was generated in, see the README's "Reasoning",
  a cut and a removed reply then restore the earlier state and decode only
  what follows it. Where the build saves no state, or the newest reply lies
  before both, the conversation is rebuilt instead. A cache that cannot shift
  is never asked to shift, not even at load.

`_backend.py` resolves the cache symbols defensively across llama.cpp API
churn: the modern `llama_memory_*` handle API when present, falling back to the
older `llama_kv_self_*` / `llama_kv_cache_*` names.

**Memory and quality settings.** On a GPU, `flash_attention=True` with
`kv_cache_type="q8_0"` roughly halves the KV cache's memory at a small cost in
output quality; `kv_cache_on_gpu=False` frees VRAM at the cost of speed.
Decode is often faster with fewer threads than prefill, because filling the
pool oversubscribes llama.cpp's spin-waiting workers; measure on the host.

### How the template is read

Superseded in part on 2026-11-08: the fragments are read from the template's
renders alone and only for what a generation reads; the check that they
round-trip is replaced by the narrower one in "Every text placed by a marker
render".

**The fragments are derived from the model automatically.** At construction the
wrapper reads the model's own chat template from the GGUF
(`tokenizer.chat_template`) and recovers the per-role fragments from it. It works by rendering
short probes through the model's template with sentinel content and splitting the
render on the sentinels to recover each role's `prefix`/`suffix` (the inverse of
the validation probe below), plus whitespace-padded probes to detect
`trim_content`, `reply_strip` and `system_strip`. The user and assistant tags are read from a
conversation of two user turns, because a first turn may carry text the
template writes once at the start, such as a default system block, and a newest
reply may carry what only the newest one does. A past reply may close with a
tag that only a following turn shows; the next user turn then opens with the
longest end of what follows the reply that a first user turn's opening also
ends with. What a template writes once after the last message of a
conversation without a generation prompt, such as an end-of-text token, belongs
to no turn and is left out of every probe. Every probe renders with
reasoning off. A conversation begun with an empty system prompt
has no system message, and opens as the template opens one without, with any
tools declared where the template declares them then. A system prompt is
refused where the template has no system turn of its own and writes one into
the first user turn, because a prompt kept for the whole conversation cannot
be. A system message added later, by `inject(..., "system")` or
`declare_unknown_tools`, takes the tags the template gives a later system turn,
and is refused where it gives none. The `system` role has no portable ground truth, because some models
have no distinct system role (some reject it, others fold it into the first user
turn). So when the system probe does not survive, the system fragment falls back
to the user tags, the same convention llama.cpp uses.

The load-time checks make a mismatch fail loudly instead of degrading silently:

- **At construction** the turn-terminator (`assistant_suffix`) is validated
  against the model's vocabulary: if it does not tokenize into the model's own
  tags, which is the silent mismatch described above, construction raises. A
  reply ends at the first of those tags, so a model whose closing takes several
  writes none of them into its answer. Inside a thought whose closing begins
  with that tag, the tag is the closing's and the reply goes on.
- **At construction** the fragments are checked against the model's own chat
  template read from the GGUF: a conversation of a system prompt, a past reply
  and a newest user turn, followed by the generation prompt, is rendered both
  ways, in each reasoning mode the model has, and must tokenize identically.
  The content is padded with whitespace, so a wrong `trim_content` is caught
  too, and the fragment side is tokenized exactly as real prefill is (tags
  special-on, content special-off). The system turn is left out where the
  template has none of its own. The check is skipped for models that ship no
  chat template.

### How reasoning is read

A template that turns reasoning on through the variable chat templates share
for it, `enable_thinking`, and that marks a thought, is read for both: what the
system turn, its opening and its closing, and the generation prompt are with
reasoning on and off, and the tags around a thought. Where `enable_thinking`
changes nothing, `reasoning_effort` is the switch if its setting changes what
follows the user turn: `"high"` turns reasoning on, and the template's own
default stands for off. The tags around a thought are read from a reply that
holds one; where a template writes no thought into a reply, from the empty
thought that only the generation prompt without reasoning ends in; and where it
removes one from a past reply instead, from the model's own tags: the closing
tag is the one after which a past reply keeps only its answer, the opening tag
the one of the same name without the slash. `ChatWrapper.can_reason` says
whether that succeeded.

### How a call's format is recovered: what is recovered, and what happens when it cannot be

The call tags are found by probing the model's template, then resolving each
one to the single token it stands for. A tag counts as such if it is either a
control token or **a token added by a fine-tune**. Both are needed: a fine-tune
commonly adds the tags around a tool call as ordinary added tokens, for which
`llama_vocab_is_control` returns false, so a check that accepts only control
tokens turns tool calling off without saying so.

A call's opening tag is the first tag before the name, and what the template
writes between that tag and the name is fixed text the model writes too; what
it writes before the tag belongs to the reply. Its closing tag is read
from a turn that makes two calls: the last tag between the first call's
argument and the second call's opening. The first tag after the arguments is
not enough, because a template may close each argument with a tag of its own,
`<arg_value>v</arg_value></tool_call>`. Where the opening begins with a tag
written once for all the calls of a turn, `<tool_calls><tool_call>`, the second
call is found by the shortest end of the opening that starts with a tag. Where a
template writes no tag between two calls, the closing tag is the first after
the last argument, which may be the token that ends the turn. A turn that calls
may open without the content tag a reply in words opens with; the call then
begins after the tags the two share.

Where the tag a call opens with also marks every reply, as in a template whose
few tags open and close the message, the answer and the call alike, a call is
recognized by its texts instead: the text it opens with, tags and plain text
together, and the text that ends it, which a call with an argument and one
without both go on with. Such a reply then ends where its closing parts from a
call, once that much of the closing is generated; those tokens leave the cache
again and the closing is written whole.

A call may also open with plain text and no tag at all, such as
`<function name="` in a template whose vocabulary holds no tag for it. Such an
opening counts only at the start of a reply or right after a tag, so a reply
that mentions it in a sentence goes on as text. Where a template writes no two
calls in one turn, the text that ends a call is what a call with an argument and
one without both end with. A plain-text opening that begins with the words of a
reply's own opening, such as a role name in its header, is refused: a model may
write its header in another order than its template does, so the template
cannot tell how a call it generates begins. Where a template names the function
twice, once in the turn's header and again in the call's body, the call opens
with what lies between the two; the header, like any header naming a reader
other than the answer's own, is never shown. Where a model ends its turn inside
such a call, the call ends there, closed as the template closes one. A result
the template writes between quotes is escaped as a JSON string, as the template
escapes it.

Where the template cannot tell how a model writes a call, because the model
writes it in another form than its template shows a past one in, the model's
add-on file says so, under `call`: the `opening` a generated call starts with,
the `name_close` after its name, and the `closing` that ends it, which the
template must write after a call too. The arguments after `name_close` are read
as JSON.

A template that declares tools somewhere other than the system turn, such as in
the first user turn, may have a variable of its own that moves them there. A
variable the template reads qualifies where setting it changes nothing in a
conversation without tools and puts the declarations into the system turn; it is
found when the model loads and passed with every render.

The argument spelling is recovered from four probes that differ only in their
content, so each delimiter is found by comparing the results. A body that is
JSON is read directly instead, because it already carries the function name and
the arguments, under `name` or `tool_name` and `arguments` or `parameters`. Together the two cover JSON objects and arrays,
`name{key:value}` with quoted strings, `name(k="v")`, attribute-style XML,
`<function=name><parameter=k>v</parameter>`, and keys and values in tags of
their own right after the name, `name<arg_key>k</arg_key><arg_value>v</arg_value>`,
including nested objects, lists, and separators inside quoted values. A type
written between a key and its value, `key="k" type="string">v`, is recovered
from where the render of a string argument and that of a number differ. A value
of the string type stands as its text, and any other is read as JSON. A value
closed by a tag that names its own key, `<city>Berlin</city>`, is read up to that
tag, and arguments written after the name as one JSON object are read as JSON.
Where such a template writes a nested object as elements named by their keys,
and a list as elements of one fixed name, `<where><city>Bern</city></where>` and
`<days><item>1</item><item>2</item></days>`, three more probes recover both
forms: an object with one member, one with two, and a list of two strings,
each compared with the string argument. These values are read by depth, so a
member may carry the key of the value around it.

What the recovered delimiters look like on two real models, since the field
names alone are hard to picture:

| `CallBody` field | a delimited model | an element-shaped model |
|------|------|------|
| `name_close` | `{` | `>\n<` |
| `key_open` | (empty) | `parameter=` |
| `key_close` | `:` | `>\n` |
| `pair_sep` | `,` | `\n</parameter>\n<parameter=` |
| `body_close` | `}` | `\n</parameter>\n</function>\n` |
| `empty_close` | `}` | `/function>\n` |

A delimiter runs only as far as the probes agree, which is why the element
column splits `<` off its `<parameter=`: the no-argument render shares that
character, so it lands in `name_close` rather than in what follows.

Where the format cannot be recovered, `can_call_tools` is `False`. The call is
still detected, so its syntax never reaches the reply text, and the model is
told that its call could not be read, so it can write it again or answer in
words instead.
Nothing is guessed at. For such a model, pass
the spelling in yourself with
`ChatWrapper(tool_calls=ToolCallFragments(call_open=..., call_close=...,
body=CallBody(...)))`, which makes `can_call_tools` `True` again, the same way
`fragments=` covers a template that cannot be read.

**Every call is answered, readable or not.** A result block has to name the
function it answers, and here that is exactly what could not be read, so the
block names `unknown` and its message says what happened. Writing nothing would
leave a call with no result in the history, which no template produces and the
model would read back on every later turn.

A model that writes a batch of calls as a JSON array has all of them read, and
each owes a result. The results are written as the template writes the results
of calls made together: one after another, with what closes the block, and the
generation prompt, only after the last. Any other shape ends at its closing
tag, so generation stops after one call either way.

### What a failed call is told

The instruction is `try again` where another call is still allowed and
`do not call again` where none is. It says only that, because what to do
instead needs no saying: a model that cannot call a tool can only answer.
Which one it gets is decided after every call in the round has been settled,
because the last of them is what settles the budget, so the same failure ends
in `try again` while a call remains and in `do not call again` once a ceiling
is spent. Telling a model to retry where the call tags are about to be withheld
would ask for the one thing it cannot do.

An unreadable call is retryable like any other, and spends the budget like any
other, so under a `max_calls` ceiling a model that cannot write a readable one
gives up after that many attempts rather than looping. Under
`reply(calls="refuse")` the retry is skipped, since a second call could only be
refused as well. It is the one failure whose
result carries no instruction: it is written while the reply is being recorded,
before the round settles the budget, so there is nothing true to say yet.

## 2026-11-07: Structure review: what changed and why

Five fresh reviewers read the library, each on one part (modules, public API,
the template approach, the engine and cache layer, tests and documentation).
The user chose to build all of it except two items.

- **Eviction removes whole exchanges.** Single-message eviction could leave a
  conversation opening on a reply: 6 of the 50 test templates refuse that shape
  and 4 render it differently from what the cache held, so long conversations
  broke rule 1. The survivors now open on a user turn, and a restart keeps its
  history from a user turn on. Consequence: under a tiny limit an exchange can
  leave whole, the newest reply included.
- **A request evicts before it is refused.** The room check used to count
  against the cache as it rested, so a request was refused although evicting
  would have made room.
- **The message callback runs before anything changes,** or after everything
  has landed, so a failing store cannot leave the cache and the table apart.
- **Declaring tools later writes them where the template writes them.** The
  old message of its own in a later system turn is a shape no template writes.
  The user kept the ability; now the system turn is rendered with all tools and
  the conversation after it decoded again once, or the declarations join the
  newest user turn where the template puts them there. Rejected: removing the
  call (the reviewers' proposal).
- **A resumed turn adds nothing after a result.** Gemma 4's generation prompt
  holds an empty thought that its template does not write after a tool result;
  the wrapper used to add it. Measured on the real model (4 tool questions,
  greedy): identical answers with and without it; without it the model writes
  the empty thought itself, 4 tokens more per resume. Following the template
  wins.
- **The template is resolved again when its render changes,** as a template
  writing today's date does after midnight. Checked at each `begin` with four
  renders, which cost about a millisecond.
- **One boundary for a cache context,** two protocols in `cache_context.py`,
  checked once where a context is adopted. 19 presence checks went: a test
  double without a renderer used to skip the load check rule 1 relies on.
- **One routine for the plain text between tags** (`tokenizing.py`) replaced
  three copies. Verified on eight real models (Gemma 3, Llama 3.2, Qwen 3.5,
  Ministral 3, MiniCPM 5, Gemma 4, gpt-oss, the R1 distill): memory equals the
  template's render after a chat and after a history with a tool turn.
- **Generation takes one call boundary,** and stop strings are matched in one
  place, on the answer, after the thought and header filters.
- **Cutting the cache back is one object** (`cache_sync.py`): shift, cut,
  restore a saved state, or rebuild, chosen by what the cache allows.
- **One `reply()` instead of six entry points,** the call policy explicit:
  `"refuse"`, `"yield"`, `"run"` or a registry. Without text it answers the
  newest user message or goes on with an open turn.
- **`history()` round-trips,** tool turns included, so `begin(system,
  history())` reads the conversation in as the template writes it. The
  persistent wrapper, which stored text only and mirrored 25 members by hand,
  went with it.
- **`Config` is frozen,** and holds every setting except the add-on files and
  what a request passes. The reviewers proposed moving the hardware settings
  out; the user ruled that out.
- **Not done:** reading nested values in MiniMax M3.1's own call elements (a
  further template probe is needed); moving the tool loop into a module of
  its own (it needs a public option to withhold call tokens that nothing else
  wants); splitting the cache context into separate tokenizer and cache
  objects; separating the template's load-time facts from its per-request
  renders beyond the call format.
- **Considered, not chosen: rendering the whole conversation each turn.** One
  render per turn diffed against the cache would replace about 1000 lines of
  template reading with about 300 and make rule 1 hold by construction, at a
  few milliseconds per turn; decoding would stay incremental. The user ruled
  out rendering the whole conversation each turn (123C) and asked for more
  information before deciding.

## 2026-11-07: Rendering the whole conversation each turn, measured

The structure review's estimate of a few milliseconds per turn does not hold
for Gemma 4, so the user kept today's approach. Rendering the conversation in
memory through the template before each answer was timed at a full 8k-token
context on an Intel i9-9900K. The setup was Gemma 4's own template (as the
26B-A4B GGUF ships it), four tools declared and the Gemma 4 tokenizer, taking
the median of 15 runs:

| Conversation | Messages | Render | Tokenize |
|---|---|---|---|
| about 12 words per message | 394 | 234 ms | 25 ms |
| about 30 words per message | 178 | 51 ms | 25 ms |
| about 90 words per message | 64 | 9 ms | 26 ms |

The render grows faster than the number of messages: twice the messages take
four to five times as long. So short voice exchanges are the worst case.
Comparing the two token lists took under half a millisecond.

On a Raspberry Pi 5 with Gemma 4 E2B, estimated by scaling these times by 2 to
3, every turn would wait 0.5 to 0.8 s longer for short exchanges, 0.15 to
0.23 s for medium ones and 0.07 to 0.10 s for long ones. The Geekbench 6
single-core scores, about 770 to 960 for the Pi 5 against about 1650 for the
i9-9900K, give a factor of 1.7 to 2.2, and plain Python widens the gap. Today
the first word comes about 1.2 s after a turn, assuming 30 new tokens at the
24 tokens/s one report gives for llama.cpp Q4_0 on four threads. Against that,
the render adds 6 to 65 %. Comparing the text first and tokenizing only its
changed end would remove the tokenizing share. The render itself is what the
approach consists of. The estimate assumes E2B ships the 26B model's template
and tokenizer, which was not checked.

## 2026-11-07: Nested values in a model's own elements, and tests by module

- **Nested values written as elements are read back.** A template may write an
  object argument as elements named by its keys and a list as elements of one
  fixed name. Three more probes at load recover both forms, and the values are
  read by depth, so a member may carry the key of the value around it. Two
  shapes stay lost because the template writes them alike: an object whose only
  key is the items' own name reads as a list, and an empty object or list reads
  as an empty string. A null is still left out by the template itself.
- **The tests are sorted by the module they test** instead of by the review
  round that added them. The helpers that several files build on are in
  `tests/wrappers.py`. The move kept every test: the collected test names
  before and after are the same list.

## 2026-11-07: The template as the program

The user asked whether a model could work every time, a future one included,
and chose a prototype (182A) of the answer: the template renders the whole
conversation, and the cache is kept equal to that render by decoding only what
differs. It is `llama_chat/rendered_chat.py`, beside `ChatWrapper`, which it
replaces nothing of. It reopens the wording of 123C ("we will not render the
whole conversation each turn"), which the user knew when choosing it. It never
decodes the whole conversation again on a cache that shifts.

How it works:

- After each reply, while it is read or spoken, the conversation is rendered
  with a stand-in where the next question goes. The cache is compared with
  that render token by token, and decoded up to the run of text the question
  joins.
- A question is put in the stand-in's place where the template writes plain
  questions exactly so. That is checked at `begin` with three questions. Only
  its own tokens and the generation prompt are decoded before the reply.
  Other questions, such as ones with surrounding spaces or a tag, are rendered
  when they come.
- When the conversation renders longer than `evict_above_fraction`, the
  oldest exchanges are dropped from the list. The stretch they held is found
  where the cache goes on as the new render does, removed and shifted past.
  Everything from the first token that still differs is decoded again.

Measured:

- All 52 test templates: the cache equals the render before every reply, over
  conversations, evictions on caches that shift and on ones that cannot,
  restarts from a history, a past tool turn and a later declaration (357
  tests). Splicing a plain question was exact on all 52. Evicting removed a
  stretch from the middle on 51. On the last, the only exchange left was the
  newest, which was cut from the end.
- 8 real models (Gemma 3 1B, Llama 3.2 3B, Qwen 3.5 9B, Ministral 3 8B,
  MiniCPM 5 2B, Gemma 4 26B-A4B, gpt-oss 20B, R1 distill 8B, Q4 or MXFP4,
  RTX 3090): the cache equals the tokenized render before each of 12 replies,
  at 1,024 and at 512 tokens of context, and with a past tool turn on the 7
  that call tools. Every cut after a reply lay at the newest reply, which the
  template writes without the empty thought it was generated with.
- Gemma 4 26B-A4B at 8k, memory full, every turn evicting, median of 7 turns:
  the wait before the first word was 199 ms against 194 ms for `ChatWrapper`
  with short messages, and 200 against 201 ms with medium ones. The work after
  each reply was 1.27 s against 0.42 s with short messages, and 0.63 against
  0.41 s with medium ones. The prototype renders three to five times there.

Not done in the prototype, so not comparable yet:

- A cache that can neither shift nor be cut at its end is decoded again whole
  at every cut (Qwen 3.5: 12 times in 12 turns). `CacheSync`'s saved states
  would serve it, if they took the tokens the cache should hold instead of the
  message table.
- A turn that called tools is written as the template writes it, not kept as
  generated, which rule 1 asks for.
- No refusal at load of a template that rewrites older history every turn.
- `inject`, `snapshot`, the capability properties, `wait_idle`, the failure
  results of the tool loop and the reply's prefill and eviction counts are
  missing.

A reviewer proposed a third design: render a window of constant size, the
system message and the last two exchanges with the stand-in, and compare its
end with the cache's end. It measured that window's text identical to the full
render from the second-newest reply on, for all 52 templates. That would make
the render cost per turn independent of the conversation's length. Not
measured: tool loops and reasoning inside the window.

## 2026-11-07: Second structure review: what changed and why

Five fresh reviewers read the library again; the user chose the fixes, the
tests, the tidying, two restructurings and a public way to forbid calls.

- **One place decodes into the cache.** `CacheSync` now holds the cache's end
  and every decode, generation, removal and cut, and takes the tokens the cache
  should hold as a callable instead of the message table, so the prototype uses
  the same saved states. A decode that fails part-way marks the cache unknown,
  and the next action decodes it again whole from the table: before, it left
  the cache and the table apart until the next `begin`.
- **Every "does not fit" is one error.** `ContextOverflowError` is now a
  `ValueError` and is raised for a request, an injected message, a history, a
  system prompt and a declaration that do not fit, not only for room to reply.
  A server could not tell a request too long from one it cannot serve.
- **A restart that would keep nothing is refused.** The newest exchange, from
  its user message on, must fit under the token limit; before, a short reply
  after a question too long to keep emptied the conversation without a word.
  The message limit may still keep nothing, as it does between turns.
- **The context's own size is checked where it is adopted.** `context_size` is
  part of the cache-context protocol, and a `Config` asking for more is refused
  at construction instead of breaking a reply halfway.
- **A running exchange holds the lock throughout**, tools running included, and
  the choice between resuming and starting a request is made under it. Another
  thread's action used to slip in between rounds. A tool must therefore not
  call back into the wrapper.
- **Withholding calls.** `calls="withhold"` keeps the call's opening token
  from ever being drawn, with a logit-bias stage in a sampler chain of the
  generation's own. A call that opens with plain text cannot be kept from the
  model that way. A new request also never passed the call setting on before:
  only a resumed turn did. Measured on 8 real models with a weather tool
  declared and the question asking to use it: Llama 3.2, Qwen 3.5 and
  Ministral 3 wrote the call's markup as text anyway, and Gemma 4 26B-A4B
  wrote one repeated phrase up to the token limit. A question that needs no
  tool was answered normally. `"refuse"` on the same question let Gemma 4
  call, told it no tool can be run, and it then made up a weather report. So
  withholding is offered for a request that wants no tool, and the last round
  under `max_calls` keeps leaving a call unrecognized rather than withheld.
- **What a template does to a message's text is learned at load.** One render
  with every tag of the model and the characters an escaping filter changes,
  each between markers, finds what the template drops, escapes or replaces in
  content, and content is written that way. Found by the new token-level test:
  one template drops its thought tags from a message and escapes its turn tags,
  so a question holding one was encoded unlike the render.
- **`history()` comes with `system_prompt`**, so a server begins again from the
  conversation's own history and keeps its tool turns as the template writes
  them, instead of from a text-only copy.
- **The tests try to break the rules.** The conversation test now runs with
  reasoning on in every template that reasons, with a live call on every tool
  template (the call keeps its generated tokens, the result and the turn after
  read as the template writes them), with a tokenizer stand-in that keeps a
  tag typed into a message as text, and with checks that no eviction rebuilds
  the cache and no cut reaches before the newest reply. Every test on the
  model-free cache now ends with that cache read against the template's render
  where nothing is kept otherwise. The check for model names splits names at
  `_`, digits and capitals; it caught the engine's name in an attribute and a
  docstring outside the engine's module.
- **Removed:** the free `@tool` decorator and `ToolRegistry.spec`, so a tool is
  declared either as a `ToolSpec` or by registering it; `CacheSync.rested_at`;
  the add-on path of `with_model_sampling` that only tests used; the doubled
  cut and clear in the load-time cache check. `Backend.can_edit_cache` stays:
  the install script reads it.
- **Moved:** `Reply`, `ReplyStream` and `ContextOverflowError` to `reply.py`,
  and `GenerationAccumulator` into the cache-context protocol it belongs to.
  The message table enforces that it opens with the system message, and where
  kept history may open is decided by one function.
- **The prototype's gaps are closed.** It keeps a tool turn as it was
  generated, by holding that stretch of the cache where the template writes the
  call otherwise; it goes back to a saved state on a cache that cannot be cut,
  instead of decoding everything again; and it refuses at `begin` a template
  that writes an older exchange otherwise once another follows.

## 2026-11-07: Rendering windows

The user chose to try the window design (194A) in the prototype, so that the
work after a reply does not grow with the conversation.

- After a reply, only the system message, the newest two exchanges and the
  stand-in question are rendered. From the first tag after the older
  exchange's reply on, that render is the whole render. The tag is found in the
  cache from where the newest question's run was decoded: the stretch between
  them is checked token for token, and everything before it is kept.
- When the oldest exchange is dropped, only the system message and the oldest
  two kept are rendered, up to the first tag after the first one's reply. The
  stretch the dropped exchange held is found and removed as before; what
  follows that tag stays as it is in the cache. Exchanges are dropped before
  the decode ahead, not after it: the engine shifts the cache at its next
  decode, and after it that is the next question's, which made the wait
  before the first word 500 ms instead of 200 ms on Gemma 4 26B-A4B at 8k.
  Only where the window's own older exchange must go too is it dropped after
  the decode.
- At `begin`, and when tools are declared later, five probe exchanges check
  both properties against whole renders, with the conversation's system
  message, tools and reasoning: the window's end, the history before the tag
  as the prompt for the newest reply wrote it, and the oldest two kept after a
  drop, with everything after them unchanged. A template that fails is
  rendered whole after every reply, as before.
- A whole render is still made after the first reply, which follows no spliced
  question, after a question rendered as it came, where either window holds a
  tool turn, where a reply is empty so nothing anchors the tag, and wherever a
  check on the cache fails. Tool turns are left whole because the probes hold
  none.

Tests: all 52 test templates pass the probes. Over a conversation that drops
exchanges, on caches that shift and on ones that cannot, the cache equals the
render before every reply, and every reply after the first is settled in
windows on the 47 templates whose scripted replies have text. On the other
five, the template opens a thought for the reply, the scripted reply is all
thought, and its text is empty.

Measured on 8 real models (Gemma 3 1B, Llama 3.2 3B, Qwen 3.5 9B, Ministral
3 8B, MiniCPM 5 2B, Gemma 4 26B-A4B, gpt-oss 20B, R1 distill 8B), 12 replies
at 512 tokens of context: the cache equals the tokenized render before every
reply, and with a past tool turn on the 7 that call tools. After the first
reply every reply was settled in windows on 6; gpt-oss and the R1 distill
spent the 40 tokens of each reply on thought, so their replies were empty and
were rendered whole. Qwen 3.5, whose cache can neither shift nor be cut, was
decoded again whole 2 times, at evictions, as before.

Gemma 4 26B-A4B at 8k, memory full, every turn evicting, median of 7 turns,
against `ChatWrapper`:

| messages | wait before the first word | work after each reply |
|---|---|---|
| short | 198 ms against 197 ms | 415 ms against 418 ms (whole renders: 763 ms) |
| medium | 200 ms against 201 ms | 411 ms against 413 ms (whole renders: 513 ms) |

What remains after a reply is the shift after the eviction and decoding the
newest reply again as the template writes it, which `ChatWrapper` does too.

## 2026-11-08: The render is the conversation's encoding

The user chose to make the prototype the replacement (195A), under the name
`ChatWrapper` with the same calls (198A), to refuse a model that ships no chat
template (199A), and to refuse a message that does not fit instead of cutting it
(200A).

What changed:

- `ChatWrapper` keeps its calls, its refusals and its tool loop, and holds the
  conversation as the messages a template is given (`llama_chat/rendered_cache.py`).
  The cache is the template's render of them, decoded only where it differs
  from what the cache holds, with the windows of "Rendering windows". The
  message table, the per-role building of each message, the moving
  declarations, the join of two user turns, the later system turn, the
  result wrapping and the past tool turn built from pieces are gone: the
  template writes each of them where it writes them.
- The pieces are still read from the template at load, for what a generation
  holds: a thought, the end of a reply, a call and its format, and what a
  template does to message text. They are no longer handed in;
  `ChatWrapper(fragments=...)` is removed, and a model without a template is
  refused at load. `oversize_policy` is removed with the cutting it chose.
- A turn that called tools keeps the tokens it was generated as, from its
  generation prompt through its calls, results and final reply. Each kept turn
  records where it sits in the cache, moves with every removal before it, and
  is dropped by a cut into it. Where the template writes the turn otherwise,
  the cache goes on after it from where the template's version ends: after as
  many call closings as the template writes for the turn's calls, or at the
  closing it writes after the turn's last reply. Matching on the tokens after
  the turn failed where rounds repeat, as nine identical calls do.
- A call's tags are tokenized as the single tokens a generated call holds, so
  a history read in is token for token the live turn it came from. A call's
  arguments that hold a tag's characters are the caller's own text, never
  read as tags, as message contents are.
- A reply's thought is given to the template as `reasoning_content` only where
  the template keeps a past reply's thought. Granite writes a past reply
  otherwise when it is handed one, even though it drops it.
- A reply without an answer is taken out again where the template writes two
  user turns in a row, as before, and the limits are applied after it.
- The limits count the conversation as the template renders it, without the
  stand-in question. Counting the stand-in made a conversation seem longer
  than it is and dropped exchanges that fitted. The window path counts the
  same way, from a render of the window without the stand-in.
- Exchanges are dropped before the decode ahead, so the shift lands on that
  decode and not on the next question's. Where the window's own older
  exchange must go too, or what would be decoded ahead does not fit the
  context, the rest is rendered whole; the conversation is then down to one
  exchange.
- `Reply.n_prefilled_tokens` counts every token decoded for a request, the
  decode ahead after the reply and a return to a saved state included.
  `snapshot()` rows hold a message's role and text; the cache positions they
  held came from the table.

Tests that read the table were moved to what the cache reads as. The check
every test on the model-free cache ends with now accepts the cache as the
template's render of the conversation, with its generation prompt, without it,
or decoded ahead up to where the next question goes. The fake template writes
several rounds of calls in one turn, as real templates do.

## 2026-11-08: Every text placed by a marker render, and the review's fixes

The structure review of the replacement found nineteen faults it brought in,
four the old code had, eight holes in the tests and a set of restructures. The
user chose to fix all of them (201A, 202A), close every test hole and merge the
two template-wide test files (203A), give the turn state one owner (204A),
replace the searches with a marker render and split the engine file (205A),
narrow the load check and stop a cache context from handing in template pieces
(206A), make the public calls one way each (207A), rename the tool flag
(208A), let a conversation follow the clock (209B), drop well below the limit
where the cache cannot shift (210A), and commit without pushing (211A); the
language-model image is not rebuilt yet (212B).

### Every text placed by a marker render

The cache must hold the render with each text of the caller's own (a content,
a thought, a call's name and string arguments) tokenized as plain text, and
each kept tool turn as it was generated. Both need to know where a text sits
in the render. The replacement searched for each content with `find`, which
took the first match: a message whose whole text was a stretch of template
text, such as `<|im_end|>`, made the template's own tag plain text and the
typed one a real turn end, on 49 of 52 templates. Kept turns were placed by
counting closings, which found nothing on templates without a reply closing
(GLM 4.7, GLM 5.3, Flint) and lost the turn.

Now each render is made twice: as it is, and with a marker (`QQ<n>MQQ`,
letters and digits, which trimming, escaping and quoting leave whole) in place
of every text (`llama_chat/split_render.py`). The marked render says where each
text goes; the real render must be the marked one with each marker replaced by
its text as the template writes it: the text itself, as a content of its role
is written (its strip and escapes), trimmed, or inside a JSON string. Where none
of those fits, the text is what lies before the template's next text, placed
only where exactly one place leaves the rest of the render matching; MiniCPM 5
wraps an argument holding a tag in CDATA this way, and the text inside stays
the caller's. A template that writes anything else of a text, its length or a
branch on a word in it outside the text's place, makes the two renders differ
elsewhere, and the conversation is refused there, rather than encoded with a
guess. Empty replies carry no marker, because a template may write a reply
only where it has text; empty user and tool contents carry one.

A call's end is anchored after its last string argument, else after its name
where the template first writes it: Gemma 4 and gpt-oss write the name again
in the result block.

Two renders per render cost one more template render each; the renders of one
action are kept and reused within it.

### The target, built without the cache

What the cache should hold is now built from the render alone: the render
tokenized, each kept turn's tokens in place of the template's version of it.
A kept turn is placed by its messages: from the first tag after the text of
the message before it (the question), to the end of its last call's closing
after that call's own text, or to the end of its final reply's text, or, for a
reply without text, to where the template's closing after a reply begins. A
reply that ends a resumed turn with neither text nor call has nothing to place
its end by; the turn stays kept up to its call, and the template writes the
rest. Because the target no longer depends on the cache, the limits count what
the cache holds, a kept turn longer than the template's version included, and
a rebuild decodes kept turns as generated.

The cache is then compared with the target token by token. A stretch the
target no longer holds is still found by a search over the cache's tokens
(`_dropped_stretch_end`), now tried after a drop and wherever the declarations
move with the newest question (Mistral Small 3.2, Ministral), whose old block is
removed in place instead of decoding the exchange after it again. A wrong match
costs decoding only: what differs after it is decoded again, and the target is
not taken from the cache. Taking the stretch from a recorded map of the cache
instead was weighed and left out: it needs every cache edit to keep the map in
step, for a search that is a decoding shortcut, not a placement.

### One owner of the turn, of the tokens, and of a failed action

- The conversation alone owns whether a reply is open and which calls are owed
  a result (`RenderedCache.reply_open`, `owed_calls`); the wrapper keeps no
  copy. A history that ends on a call restores it as owed.
- `CacheSync` owns the token list the cache should hold. A cache that shifts
  or cuts takes each change at once; one that cannot is brought up once before
  the next generation or rest, from the newest saved state that still holds a
  beginning of the tokens, else by decoding them again. So one turn that drops
  twice is brought back once, nothing is decoded into a cache whose decode
  failed, and a cut the cache refuses leaves it unknown and repaired by the
  next action. A generation that follows a decode without logits decodes the
  last token again with them.
- Every action saves the conversation's state and the tokens first and puts
  both back where it raises; the cache is brought back to them by the next
  action. A refused or failed `begin`, a question whose decode fails, a result
  that leaves no room for the reply: each leaves the conversation as it was.
  A generation that fails keeps what it said, or, where it said nothing, the
  question its own request asked is taken back; its cache is left to the next
  action, so a second engine error cannot replace the first.
- The history and the system prompt are published as a copy after each
  change, so a reader on another thread never sees one half-built.
- The wrapper uses the conversation's calls only: it no longer reaches into
  the cache layer, renders through `ResolvedTemplate.render` alone, and takes
  a reply's thought from `ThoughtFilter`, which collects it as it filters.

### The clock, and a cache that cannot shift

A template that writes the date follows the clock (209B). Each action renders
how the conversation opens and compares it with the last; where it moved, the
render made ahead and the window are set aside and the conversation is
rendered whole, so the cache is decoded again once from the date on. Freezing
the date per conversation was the other option, exact without the decode, but
the model would read yesterday's date until the next `begin`. The template is
no longer read again at `begin` for this, since nothing it is read for holds
the date.

Where the cache cannot shift, an eviction drops the oldest exchanges until the
conversation holds at most `evict_to_fraction_without_shift` (0.5 by default)
of the threshold, so the cache is decoded again from a saved state, or whole,
only every few turns rather than every turn once full (210A). Refusing such
models was the other option; it would have kept the rule's wording and lost
Qwen 3.5.

### What a template is read for, and refused for

The load check no longer rebuilds a conversation from per-role pieces and
compares it with the render, since the cache is the render. It checks what
generation reads from the pieces: that the generation prompt ends the render
of a question and that the closing after a reply follows a past reply's text,
in each reasoning mode. `can_reason` now means the template marks a thought;
the switch is the template's wherever it writes it, so a model whose switch
changes more than its system turn reasons as its template does. The pieces
are always read from the template's own render: `ModelReader.extract_fragments`
is gone, and `TemplateFragments` is no longer exported. The `Reasoning` fields
for the system turn and a past reply's opening, `TemplateFormatter.parts`,
`fragment` and `tokenize_message` served only the old check and are removed.

The check that older history stays written now refuses a template that refuses
its probe, instead of passing it.

### The public calls

- `snapshot()` is removed; `history()` and `system_prompt` are safe from any
  thread, and a caller's record keeps tool calls and results.
- `can_take_system_prompt` says whether `begin` takes a system prompt.
- `inject(text)` puts a question where the render made ahead left room for
  it, as `reply(text)` does, generation prompt included.
- `awaiting_tool_result` is removed: `reply_open` is the one state property.
- `STOP_REASONS` and `StopReason` list the stop reasons; a refused setting is
  an `InvalidSettingError` whose `field` names it.
- `ToolRegistry.register(in_system_prompt=)` is `declare_at_begin=`, and
  `specs_for_system_prompt()` is `specs_declared_at_begin()`: the template, not
  the flag, decides where a declaration goes.
- `calls="run"` where `begin` kept no registry, and `max_calls` with a policy
  that runs nothing, are refused instead of ignored.

### Faults the old code had

- Llama 3.2 writes `<|python_tag|>` before a call its template opens with
  plain text. A tag right before such a call is now held back while a call may
  still follow, and is the call's where one does, so it is never shown.
- A tool result JSON cannot write is an error result where a registry runs the
  tool, and refused where it is handed to `inject_tool_result`.
- The README promised `"interrupted"` for a stream closed early, which has no
  reply to carry it.

### Tests

The two template-wide files are one, `tests/test_conversations.py`. Its checks
take the expected render from what the test sent, not from the library's
record; bound what one reply decodes by about four exchanges; drive a template
of the file's own whose windows differ from its whole render; space a live
call unlike the template writes it, on every tool template, add-on ones
included; put a typed tag into the system prompt, a past message, an injected
one, a tool result and a call argument and compare tokens with an expectation
worked out in the test; and give every template replies with text, a thought
closed where the generation prompt opens one. Measured against two deliberate
breakages: recording every reply as empty failed 90 of the render checks, and
switching off the removal in place failed the eviction check on all 53
templates. `tests/conftest.py` now also covers the real-template contexts and
checks the cache around each kept turn, and that no kept turn was lost.
`tests/test_failures.py` drives each refused or failed action. The faults were
found by the probes of the review; their tests were written after the fixes,
not run against the code before them.

### A fresh review of the change, and its fixes

A reviewer who had not seen the work read the diff and probed it with fault
injection over every template. What it found, and what changed:

- **A question decoded at a stale position.** The render made ahead noted
  where a spliced question's run began, and a later question that was not
  spliced, a failed generation or a rolled-back settle left that note in
  place. The window check compared only the template's text before the
  question, which is the same before every question, so the cache could lose a
  whole exchange that the conversation still held (48 of 52 templates after a
  generation that failed part-way). The note now holds the question it was
  taken for, is cleared by every new question and every failed generation,
  and the window checks that the cache still holds what the splice decoded
  there. (A render's tokens of the question cannot be compared instead: where
  the template's text between a question and its reply holds no tag, as in
  Mistral's, the question shares its plain run with the reply.)
- **Declarations that move with each question, on a cache that cannot cut.**
  Mistral Small 3.2 and Ministral 2410 declare tools before the newest
  question. On a cache that can neither shift nor cut, both saved states lay
  past the block the next move cuts back to, so every reply decoded the whole
  conversation again, against invariant 2. Where declarations move, each sync
  now saves a state where it cuts, which is where the next move cuts back to.
  A reply then decodes the previous exchange and the declarations again, about
  one exchange more than on a cache that shifts. The cost is one saved state
  more per action on such a template. Pinning a state after the system turn,
  so that an eviction on such a cache also skips decoding the system turn
  again, is offered as a decision, not built.
- **A failed generation took back a message it had not added.** Only the
  question the request itself asked is taken back now, the tokens put back as
  they were before it, and `on_message` is not told of a reply that was never
  recorded.
- **A turn nobody could close.** Where the decode of the error result for an
  unreadable call failed, the turn stayed open and owed a call the caller was
  never handed. It now closes as a reply without the call before the error is
  raised.
- **A kept reply without text placed at zero width.** Its end was searched
  from the first tag after the question, which found the question's own
  closing where the two closings are the same text, so the cache held the
  reply's opening twice. Its end is now the last closing before the next text
  of the caller's own. The window and the oldest-exchange renders also refuse
  an exchange that holds a kept turn, which a reply whose call was dropped
  still does. `tests/conftest.py` now requires every kept turn to stand in for
  template text of its own.
- **Exchanges that repeat word for word.** Removing the oldest exchange
  searched for where the cache goes on as the new head render does; where two
  exchanges read the same, the search found nothing to remove and the loop
  dropped everything down to two exchanges. The stretch is now located by
  known positions: before the drop, the cache up to the tag after the second
  oldest exchange's reply is checked against a render of the oldest two; after
  it, the render of the oldest two up to the tag after the first one's reply
  replaces it, which the window check guarantees reads as the whole render.
- **A refused action cost a cache that cannot cut a whole decode.** Putting
  back unchanged tokens dropped the note that the last decode left logits, so
  the next generation decoded the last token again, which on such a cache
  means going back to an older state or decoding everything (503 tokens
  instead of 15 in the probe). Unchanged tokens now change nothing.
- **Shared argument dicts.** `history()`, an owed call and a yielded call each
  hand the caller its own copy.
- **Kept turns by object, not by `id()`.** A removed turn's position could have
  been read for a new turn given the same id.
- **Shortening a kept reply** is refused like shortening one that holds a
  call: its tokens are kept as generated.
- **A thought's closing tag held back as a call's.** Only a tag that is not
  the thought's own is held back where a plain-text call may follow it.

Found while fixing the last one: with calls turned off on a template whose
call opens as plain text and whose reply can end in text, the generation loop
called `startswith(None)` on the first special token. A plain-text opening is
now only looked for where there is one.

Not a defect: while a call is open, Mistral Small 3.2's cache keeps the
declarations before the question, where its template's render of a
conversation ending on a call writes none; the render with the result written
puts them back, and that is what the cache serves next.

Not changed: `history()` and `system_prompt` are two reads, so a reader on
another thread can pair them across a `begin`. One call returning both would
reopen the decision above to remove `snapshot()`; it is offered as a decision.

### Measured before this round's changes

On Gemma 4 26B on the GPU, about 7,350 tokens kept in an 8k context, the
replacement as it stood before this entry: the wait for the first word was
196 ms for an 8-word question (the old code 195 ms) and 202 ms for a 20-word
one (202 ms), and the work after each reply 416 ms (406 ms). Plain chat on 8
real models, 12 replies each in a small context, held the template's render
before every reply. This round's changes are not measured on a real model yet:
the language-model image still runs the code before them (212B).


## 2026-11-08: Measured through the stack, an injected question made fast, one read of the conversation, a state kept before the oldest question

### Measured through the stack (213A)

The language-model image was rebuilt on the committed library and driven
through the running stack's Chat Completions endpoint: 36 questions of about
280 tokens each into Gemma 4 26B's 8k context, so that the conversation
evicted, then a tool call, its result and a question after it.

| library | first word, first turns | first word, context full |
|---|---|---|
| before the replacement | 90 to 93 ms | 99 ms |
| as committed | 92 to 93 ms | 182 to 184 ms |
| with the fix below | 91 to 93 ms | 97 to 98 ms |

Replies, the call and the answers after it read the same in all three. On
Llama 3.2 3B, after a past tool turn, the next call shows no
`<|python_tag|>` and raises nothing (N1, O1); with calls run it answers from
the result. Its odd reply `"}"` to "Thanks." follows the template, which writes
a tool result as a quoted JSON string.

### An injected question waits only for itself

The server injects each question as it arrives and then asks for a reply
without text. That path was slow in two ways, both measured on the GPU with
the stack's settings (224 ms against 84 ms for `reply(text)` once full):

- An injected question was spliced into the render made ahead only below
  `evict_above_tokens`, so once the conversation was full it was rendered
  whole. It is now spliced within the room left for a reply, as a question
  asked with `reply(text)` is, and the eviction follows its reply. Only a full
  `max_history_messages` still evicts first, since its count is held after
  every action (`tests/test_wrapper.py`).
- `reply()` after it rendered the whole conversation again, though the splice
  had decoded the generation prompt already. It now goes straight to the
  generation where the newest message is the question spliced there.

The test that a question waits only for its own tokens had missed both: its
conversation was short enough that a whole render stayed under its bound, and
it read only the first renders of the request. It now runs three exchanges
first, counts every render up to the reply's start, and drives the injected
path too; a second test puts an injected question over the threshold.

### One read of the system prompt and the history (215A)

`conversation()` returns both from the same published copy, so a reader on
another thread never pairs the system prompt of one `begin` with the history
of another. `system_prompt` and `history()` stay for a caller that needs one.

### A state kept where the oldest question begins (216A)

On a cache that can neither shift nor cut, an eviction removes a stretch after
the system turn, and both states the cache rested in lie past it, so it was
emptied and decoded again from the start, system prompt and declarations
included. `CacheSync.pin` now keeps a third state where the oldest question's
run begins, read from a render of two exchanges so that nothing that moves
with the newest question, such as Mistral's declarations, stands before it.
The state goes once anything before it changes. Measured on the model-free
templates with tools and a long system prompt over 16 replies: Gemma 4 from
5,213 tokens decoded to 1,694, Qwen 3.5 from 3,059 to 1,762, Llama 3.2 from
1,739 to 766. The cost is one saved state more, about the size of the cache
up to the oldest question.

Found while testing it: where the system turn takes most of
`evict_to_fraction_without_shift` of the threshold, an eviction on such a
cache drops the whole history. Measuring the fraction from the room after the
system turn instead is offered as a decision.

## 2026-11-08: Where an eviction stops on a cache that cannot shift (218A)

`evict_to_fraction_without_shift` was a fraction of the whole threshold. Where
the system prompt and its declarations took more than that fraction, an
eviction on such a cache dropped the whole history: on the model-free
templates that render whole, a system turn of about two thirds of the
threshold left no exchange after the first eviction. The fraction is now of the
room after the system turn: an eviction stops at the system turn plus that
fraction of what is left, so some history is kept however long the system
turn. Where declarations move with the newest question, as Mistral's do, they
are not counted in the system turn, so a declaration block larger than half
that room can still take the history with it.

Found while doing so: where a template refuses a system message alone, the
system turn was measured as what a system message adds to a render of a
question. Mistral Small 3.2 writes a long system prompt of its own where none
is given, so a shorter one given measured negative (about -2,000 tokens with
tools declared), and in the test of the kept state an eviction dropped all
history. The measure is now at
least the tokens before the oldest question, the same count the state kept on
such a cache is saved at.

## 2026-11-08: Measured on a cache that cannot shift

Qwen 3.5 9B (Q4_K_M) on an RTX 3090, an 8,192-token context, the default
eviction settings, six tools declared. A conversation of 70 questions of about
185 tokens each, with replies capped at 40 tokens, was run on three versions:
before the state kept before the oldest question (254080c), with it
(ce18915), and with the eviction target measured from the room after the
system turn (780c5b7). It was run once with a system turn of 987 tokens and
once with one of 3,517 tokens.

| System turn | Version | Questions kept after an eviction | Evictions | Tokens decoded in total |
|---|---|---|---|---|
| 987 | before the kept state | 10 | 3 | 34,677 |
| 987 | with the kept state | 10 | 3 | 31,716 |
| 987 | new eviction target | 13 | 4 | 35,780 |
| 3,517 | before the kept state | 2 | 5 | 45,039 |
| 3,517 | with the kept state | 2 | 5 | 27,463 |
| 3,517 | new eviction target | 6 | 8 | 34,222 |

The median wait for the first word was 55 to 57 ms in every run, and 56 to
58 ms on every turn that evicted: before the first word only the question's
185 tokens are decoded, and the history kept by an eviction is decoded after the
previous reply. The kept state cuts the total by 9% and 39%. The new target
keeps more history, so evictions come sooner and the total rises by 13% and
25% over the kept state alone. That cost falls after a reply, never before the
first word, so the default `evict_to_fraction_without_shift` of 0.5 stays; 0.4
would, by calculation, bring the short system turn back to the old total and
history. Whether a question asked while that decode is still running waits
for it was not measured.

## 2026-11-08: Bugs found by the rethink (224A)

**The system turn's length (B1).** Two measures were kept: the render of the
system message alone, for the limits, and the tokens before the oldest
question's run, for the saved state. On templates that write their
declarations and settings only once a question follows, the first measure
missed them: mistral-large-4 with one tool counted 26 tokens where the
conversation opens with 256 or more. `begin` then accepted a system turn with
declarations of 1,260 tokens against a 600-token threshold, and the first
question was refused; on a cache that cannot shift, an eviction aimed too low.
Now:

- `_opening_length` is the tokens two renders share that differ only in the
  oldest question. It places the saved state. Over every test template it is
  at least either old measure. A saved state is only used where it still
  matches the cache, so an opening measured one token too long costs a decode,
  never a wrong cache.
- `_with_a_short_question` is what the conversation takes with no history and
  a one-character question. It is what no eviction makes smaller, wherever the
  template writes declarations. `begin` and a later declaration refuse a system
  turn that leaves no room for it, and the eviction target where the cache
  cannot shift is measured from it. That also closes the case 218A left open:
  where declarations move with the newest question (ministral-2410), the
  history is no longer dropped whole.

**Beginning from a long history (B2).** `begin` dropped one exchange at a time
and rendered the whole conversation after each: 800 messages took 716 renders
and 5.4 s without a model. It now keeps the newest exchanges that fit with the
system turn in half the context, the user's choice, and renders only those:
twice as many exchanges each time until they no longer fit, then the count
between is halved. The same history takes 40 renders and 0.03 s. Half rather
than the eviction threshold leaves room for some turns before the first
eviction. The newest exchange is kept even where it alone takes more than
half, and `max_history_messages` still applies after. A history refused by the
template is now refused only for what would be kept.

**Smaller ones.**

- B3: the last round under `max_calls` does not withhold the call tokens; a
  call written there is not recognized and stays text, as the second
  structure review decided. The texts said otherwise and now say this; the
  internal switch is `recognizes_calls`.
- B4: `ChatWrapper.config` was the configuration as given, while the context
  it loads runs with the model's own sampling filled in. It is now that
  configuration where the wrapper loads the model.
- B5: a call that cannot be read counts as no call; any other fault while
  reading one was also swallowed and reported as an unreadable call. Only
  `ValueError` and `RecursionError` are now. A fault that escapes leaves the
  cache consistent and the question without its reply; the test that required
  the turn to close on any fault is replaced by one that requires the fault to
  be raised and the conversation to stay whole.
- B6: a `ToolTurn` whose result answers no call made before it in the turn is
  refused when it is built, where it used to be written with an invented call
  id.

## 2026-11-08: A system prompt is kept wherever no later question moves it (228A)

Supersedes the refusal in "How the template is read" of a template that writes
the system prompt into the first user turn. Whether a template has a system
turn of its own was read from extracted fragments: a system prompt was refused
wherever its tags equalled the user tags. That refused Gemma 3, which writes
the system text at the head of the first user turn. After an eviction the
render puts it at the head of the new oldest question, so it never moves
relative to the start of the conversation, and the cache, which is the render,
holds it correctly.

The test is now a render probe: two renders that differ only in the oldest
question, and the system prompt is kept where its text lies in the opening
they share. Ministral 2410 writes it before the newest question, so it moves
every turn and stays refused; every other test template answers as before.
`system_prefix`, `system_suffix` and their extraction went, as nothing else
read them.

Measured before the change, with the refusal patched off: Gemma 3 1B (Q4_K_M)
on the CPU, context 2,048 (threshold 1,536), a 16-word system prompt, 30
questions of about 45 tokens and replies capped at 30 tokens. The cache can
shift; ten evictions each decoded 64 tokens up to the next question, the same
as the median turn without an eviction. The dropped stretch is found and
shifted at the new seam between the system text and the next question, so the
real tokenizer keeps the memory rolling. Without a model, all of Gemma 3's
conversation tests now run with a system prompt and pass.

## 2026-11-09: Restructured inside the library (226A)

The rethink found the library doing several things in two places, and the
engine adapter holding logic that names no engine. No public name changed.
What a caller can notice is named below: a few error messages and one
warning, and the members the two cache-context protocols ask for (I1, I9).

**The generation loop is engine-neutral (I1).** The loop that generates a
reply moved out of the engine adapter into `llama_chat/generation.py`: the
UTF-8 decoding, holding back text that may start a call or a reply's end,
recognizing calls by token or by text, cancelling, the token cap and the stop
reasons. A cache context answers four steps instead of `generate`:
`sampled_tokens`, `decode_generated`, `ends_generation` and `token_bytes`.
The test stand-ins only hand out scripted tokens, so the real call detection
runs over every test template. Three call-budget counts and one Llama 3.2
script changed, because the old stand-ins skipped detection: a call that opens
with plain text counts only at a reply's start or after a tag.
`decode_generated` stays separate from `prefill`, so that the stand-ins'
prefill counts keep checking the wrapper's own decode accounting.

**Tags found from the tokenizer's answers (I9).** The model reader protocol
no longer asks for the model's tag list, for a text split at its tags, or for
the first tag in a text. `llama_chat/tokenizing.py` computes all three from
`tokenize`, `token_text`, `is_special` and `special_tokens`, with the
algorithm the engine adapter ran before, so the real tag logic runs in every
test. The three template stand-ins became one, and the fake model writes each
tag as one token. `is_special` stays beside `special_tokens` because the
template stand-ins hold every tag-shaped text, such as argument names written
as tags, which no list made ahead of time contains. Asking for a vocabulary
size was rejected: the stand-ins number characters by code point, so a
vocabulary walk would cost over a million calls per load.

**One sampler per generation (I10).** Every generation builds its sampler
chain from its effective settings and frees it when it ends; the shared chain
built at load and reset for generations without overrides was removed. A new
chain starts its penalty window and its seed exactly as a reset one did, so
one path replaces three, at the cost of building a few stages per reply. A
warning about an incompatible penalties signature now appears at the first
generation rather than at load.

**Add-on files in one module (I11).** The add-on file and the model's
sampling share one module, since both settle what a model needs beyond its
template and are read together at load. The template renderer is a plain
function of the engine adapter, because it resolves no C symbol, so rendering
a template no longer builds the whole adapter.

**Reading a call (I3).** Reading a generated call is one call,
`ResolvedTemplate.read_calls`, which puts back the lead-in an opening made of
text holds and parses with the model's call body. The wrapper and the tests
used to rebuild that step from five separate properties; those are removed,
as is `tool_tokens`, and every reader of how a call is marked, the rendering
included, goes through `call_boundary()`. The call-format probes (the tags around a call, the
separator and closing between calls, a name written twice, the argument
spelling) moved from `template_extract.py` into `call_format.py`, which only
the call resolution uses. The probe helpers both need are public names in
`template_extract` (`USER_SENTINEL`, `ASSISTANT_SENTINEL`, `TOOL_SENTINEL`,
`as_probes_read`, `Render`), so the dependency runs one way and no module
imports a sibling's private name.

**No formatter (I5).** `TemplateFormatter` is removed. `ResolvedTemplate`
gives the generation prompt and the answer opening for a reasoning mode and
refuses reasoning a model cannot do (`ensure_can_reason`, as
`ensure_tools_usable`, so `begin` refuses before it resets anything).
`RenderedCache` alone holds whether the conversation reasons, and
`RenderedCache.generate` bounds a reply by the room left beside the closing,
which is tokenized once at load instead of on every generation.

**Fewer probes (I8).** The fragments' three whitespace settings
(`trim_content`, `reply_strip`, `system_strip`) and the padded probes that
measured them are removed. They fed only the content form the marker split
tries, and the split already tries each text stripped on both sides, at either
end and of newlines; it now applies the learned rewrites to each of those
forms too, so a template that both strips and escapes is still matched. The
rewrites stay: without them the typed-tag test on flint fails. The
template-wide splice probe (three plain questions rendered at begin and at
each later declaration) is removed as well. Each question is checked by
`question_splices` on its own text before it is spliced, and windows need the
newest question spliced, so the probe added no guarantee; it only kept a
template that failed one probe question from splicing any question. Windows
rest on `windows_are_exact` alone. All 52 test templates passed the probes,
so a new test with an escaping template covers questions that cannot be
spliced.

**The conversation's state (I6).** The state a failed action puts back is one
record, `_ConversationState`, which `_action` copies field by field and
restores whole, so a field added to the conversation is rolled back unless it
is deliberately put outside the record (the current generation, the action's
render memo and the published copy are). The calls owed a result are no
longer stored: they are read from the messages, from the newest calling turn
and the results after it, so the owed list cannot drift from the
conversation. A full run first checked that the stored list matched that
reading on all 79,891 reads. A call made live is now reported as a copy, as
one from a history given to begin already was. The records made ahead (the
render for the next question, the splice note and the question an ask added)
each carry the conversation and the cache version they were made for, and are
used only while both are unchanged, or, for the window, while the
conversation continues from theirs; no action clears them. The cache length
alone was rejected, because two different targets of one conversation can
have the same length; `CacheSync.version` changes with every change to the
token list, checked by a random test of 3000 operations with injected
failures. After a failed action that changed the cache, the next question is
rendered whole once.

**Renders and the cache, each in one place (I7).** Bringing the cache to a
target is `CacheSync.hold`, so the comparison with the target, the search for
a dropped stretch, the cut and the decode live with the tokens they change;
the conversation only says when a stretch may have been dropped and when to
rest at the cut. The prefix is compared on token lists, measured faster than
converting both to text (15 ms against 24 ms at 128k tokens); the text form is
built only for the stretch search. A conversation's renders, their splits and
their tokens as the cache holds them are `ConversationRendering` in
`rendering.py`, which reads the declarations, the reasoning switch and the
kept turns from the conversation at every use. A target is returned with the
position of each kept turn in it (`Target`), and the conversation records
those positions only where it brings the cache to that target, so measuring a
length no longer changes them.

**One generation path in the wrapper (I2).** One generation function, run with
the lock held, chooses between going on with the open turn and a new request.
The tool loop drives it directly and reads the calls from its `Reply`, instead
of collecting them back out of a stream that had appended them. The wrapper's
own check that no call is owed a result went, since `RenderedCache.generate`
makes the same check before anything is generated. The "yield" path still
releases the lock before it hands the calls over, because a caller answers
each call while reading the stream.

**A reply's text as one object (I4).** `ReplyText` feeds each piece through
the thought filter, the header filter and the stop strings in that order, and
settles what is held back from the stop reason. The wrapper no longer
assembles the three filters or carries their flags, and the filters are
private to `reply_text`. `reply_header` and `begins_in_thought` stay public:
the tests use the first for two cases the object cannot show (a thought tag
or the answer opening always matches first there), and the second to script
how a model writes.

**Small items (I12).** A cache that cannot shift is not asked even for a shift
by nothing at load, because some caches abort the process when asked to shift.
A sampler chain whose stage fails to join is freed before the error
propagates. A history refused at begin is explained by the part that would be
kept, not by messages that would be dropped, and a question that does not fit
after dropping gets the message of `_fitting_prompt`. Every load-time probe
reads a render that raises as a refusal, in one function (`refusing`),
because a cache context of one's own may raise where the bundled one returns
None, and a probe decides an optional capability; a render during a
conversation still raises, and a raising context at load gets the probe's own
refusal, for example "the model cannot be used", instead of its raw
exception. One common-prefix function serves texts and token lists, comparing
slices by halving; the common suffix is the same function on reversed
sequences. Probe and live messages are built by the same helpers, so their
shapes agree by construction. A reply header is read only where both openings
leave a reader between their shared ends.
