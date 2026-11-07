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
