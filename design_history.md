# Design history

Why faster-llama-chat is built the way it is: the mechanism behind what the
README promises, the measurements, and the decisions with the alternatives
they ruled out. "Current design" says how the library works today. The dated
entries follow, newest last; an entry whose decision a later one revised names
that later entry first.

## Current design

Two rules govern the library. Every model whose GGUF ships a chat template is
encoded exactly as that template renders the conversation, or refused. A model
needs no code change, and no model is supported by decoding the whole
conversation again each turn.

- **The cache holds the template's render** of the conversation, which is kept
  as the messages a template is given. Each action decodes only from the first
  token that differs; no message is assembled from pieces of the template. See
  "The render is the conversation's encoding"; the design it replaced is "The
  mechanism, moved here from the README".
- **Each text of the caller's is placed by a second render** with a marker in
  its place, and tokenized with special-token parsing off, so a typed tag stays
  text. See "Every text placed by a marker render".
- **A turn that called tools keeps the tokens it was generated as,** in place
  of the template's version. See "The target, built without the cache".
- **The next question is prepared after each reply,** while the reply is read,
  so a question decodes only its own tokens; windows keep that work from
  growing with the conversation. See "The template as the program", "Rendering
  windows" and "An injected question waits only for itself".
- **Eviction drops whole exchanges,** never the system turn. A cache that
  shifts keeps the survivors' KV. One that cannot is cut, restored from a saved
  state or decoded again, and drops far enough that this happens only every few
  turns. See "How eviction works", "Eviction removes whole exchanges", "A state
  kept where the oldest question begins", "Where an eviction stops on a cache
  that cannot shift" and "Bugs found by the rethink".
- **One owner for each kind of state:** `RenderedCache` the conversation,
  `CacheSync` the cache's tokens and every decode, `ConversationRendering` the
  renders. A failed action puts the state back. See "One owner of the turn, of
  the tokens, and of a failed action" and "Restructured inside the library".
- **The template is read at load only for what a generation needs:** the
  generation prompt, the closing after a reply, a thought's tags, the call
  format, and what the template does to message text. See "What a template is
  read for, and refused for", "How reasoning is read" and "How a call's format
  is recovered".
- **An add-on file gives what a template cannot tell,** and the public
  interface has one way to do each thing. See "One way to do each thing in the
  public interface".

## 2026-11-07: The mechanism, moved here from the README

Superseded in large part by "The render is the conversation's encoding". This
is how the library worked while each message was built from pieces read out of
the template. Where it disagrees with the structure review below, the review
holds.

### How eviction works

Superseded in part by "The render is the conversation's encoding": the message
table is gone, and the target is the template's render. Revised by "The
rethink's remaining decisions": the survivors' shift runs to the cache's end
(`-1`) instead of to `total`. Revised by "The conversation and its surface"
(270.5): a cache context that can neither shift, cut its end nor save a state
is refused where it is adopted, so such a context is no longer decoded again.
The triggers, the cache operations and the choice between them still hold.

Two triggers, OR'd: more than `evict_above_fraction × context_size` tokens, and
more than `max_history_messages` non-system messages (off by default), so a
verbose conversation is bounded by its tokens and a long one by its count. A
`MessageTable` gave each message a `[start, stop)` slice of sequence 0, and
evicting `[a, b)` was:

```
llama_memory_seq_rm(seq, a, b)               # drop those tokens
llama_memory_seq_add(seq, b, total, -(b-a))  # shift survivors down to close the gap
```

`seq_add` fixes the RoPE positions, so the survivors are not decoded again, but
only marks the cells: the K-shift is built into the next `llama_decode`, and is
expensive only for a quantized cache. So an assistant turn's closing tag is
decoded after the eviction that follows the reply, and a decode the turn makes
anyway carries the shift instead of the next prefill. The reuse is lossy, the
StreamingLLM trade-off: the survivors' K/V were computed while attending to the
removed tokens, so quality drifts slightly. The system prompt is never evicted.

The strategy is chosen at construction from `llama_memory_can_shift()` and a
cut tried on the empty cache. A cache that refuses the cut holds a running
state and is never shifted, whatever it reports: a hybrid cache reports its
attention part's shift, and takes every removal short of its end without its
running state losing those tokens.

- **shift** (the default): the edit above, wherever the cache allows it,
  sliding-window models under a full-size cache included.
- **rebuild**, for compact sliding-window, recurrent or Mamba state, where
  dropping tokens loses the positions a shift needs: the survivors are decoded
  once, so the model degrades instead of corrupting the cache. A cache that can
  be cut at its end (a load-time probe) has its newest reply shortened by a
  cut. One that cannot has the state a cut cannot reach saved whenever it
  rests, the last two in RAM, from before and after the newest reply, each the
  size of the model's state, not of the conversation's; a reply kept in another
  form than generated, a cut and a removed reply then go back to the earlier
  one and decode only what follows. Where the cache context saves no state, or
  the newest reply lies before both, the conversation is decoded again. Such a
  cache is never asked to shift, not even at load.

`_backend.py` requires llama-cpp-python 0.3.23 or later and uses the
`llama_memory_*` calls; see "The oldest llama-cpp-python supported is 0.3.23".

**Memory and quality settings.** On a GPU, `flash_attention=True` with
`kv_cache_type="q8_0"` roughly halves the KV cache's memory at a small cost in
quality; `kv_cache_on_gpu=False` frees VRAM at a cost in speed. Decode is often
faster with fewer threads than prefill, because filling the pool
oversubscribes llama.cpp's spin-waiting workers; measure on the host.

### How the template is read

Superseded by "What a template is read for, and refused for"; the
system-prompt refusal by "A system prompt is kept wherever no later question
moves it"; the whitespace settings by "Restructured inside the library".

Each role's prefix and suffix were recovered by rendering probes with sentinel
content and splitting the render on the sentinels. Still true of the reading:
the user and assistant tags come from two user turns, because a first turn may
carry text written once and a newest reply what only the newest carries; a
past reply may close with a tag that only a following turn shows, and the next
user turn's tag then starts where the role tags' shared opening last occurs, or
is the longest end of what follows the reply that a first user turn's opening
also ends with; what a template writes once after the last message, such as an
end-of-text token, belongs to no turn; every probe renders with reasoning off
(2026-11-11: true of the probes for the turn tags only; the probes for a
thought, and the load check of the generation prompt and of the closing after
a reply, render in each reasoning mode);
an empty system prompt is no system message, and the conversation opens as the
template opens one without, tools declared where it declares them then.
Replaced since: a system
prompt was refused where the template writes it into the first user turn; a
later system message took a later system turn's tags; a system probe that did
not survive fell back to the user tags, as llama.cpp does, because some models
reject a system role and others fold it into the first user turn.

The closing after a reply must tokenize into the model's own tags, or
construction raises. A reply ends at the first of them, so a model whose
closing takes several tags writes none of them into its answer; inside a
thought whose closing begins with that tag, the tag is the closing's and the
reply goes on. That check still holds. A second check, that a
conversation built from the pieces tokenizes as the template's render of it in
each reasoning mode, went with the pieces.

### How reasoning is read

Revised by "What a template is read for, and refused for": `can_reason` means
the template marks a thought, and the switch is the template's. Revised by
"The rethink's remaining decisions": `reasoning_effort: "none"` is the
switch's off wherever it changes what follows the user turn.

`enable_thinking` is the switch where it changes the render; otherwise
`reasoning_effort` is, if it changes what follows the user turn, `"high"` for on
and the template's default for off. The thought tags come from a reply that
holds one; else from the empty thought that only the generation prompt without
reasoning ends in; else, where a past reply loses its thought, from the model's
tags: the closing is the tag after which a past reply keeps only its answer,
the opening the tag of the same name without the slash.

### How a call's format is recovered

Revised by "One way to do each thing in the public interface":
`ChatWrapper(tool_calls=...)` is removed, and the add-on file is the one way to
say how a model writes a call.

- **A tag is a control token or a token a fine-tune added.** Fine-tunes often
  add their call tags as ordinary added tokens, for which
  `llama_vocab_is_control` is false, so accepting only control tokens would turn
  tool calling off without saying so.
- **The opening** is the first tag before the function name. What the
  template writes between it and the name the model writes too; what it writes
  before the tag belongs to the reply. A turn that calls may lack the content
  tag a reply in words opens with; the call then begins after the tags the two
  share.
- **The closing** is the last tag between a first call's argument and a second
  call's opening, because a template may close each argument with a tag of its
  own, `<arg_value>v</arg_value></tool_call>`. Where the opening begins with a
  tag written once for all the calls of a turn, `<tool_calls><tool_call>`, the
  second call is found by the shortest end of the opening that starts with a
  tag. Where no tag stands between two calls, the closing is the first tag
  after the last argument, which may be the token that ends the turn.
- **Where a call's opening tag also marks every reply,** the call is recognized
  by the text it opens with, tags and plain text together, and by the text that
  ends it: the longest text, from a tag's start to a tag's end, found both
  after the first of two calls and in the end that a call with and without an
  argument share, or, where a template writes no two calls in one turn, the
  longest such part of that shared end alone. A reply ends once its
  closing has gone far enough to differ from a call's; those tokens leave the
  cache again, and the closing is written whole.
- **A plain-text opening,** such as `<function name="`, counts only at a
  reply's start or right after a tag, so a sentence that mentions it stays
  text. One that begins with the words of a reply's own opening, such as a role
  name in a header, is refused: a model may write its header in another order
  than its template, so the template cannot tell how a generated call begins.
- **A function named twice,** in the turn's header and the call's body: the call
  opens with what lies between, and the header, like any header naming another
  reader, is never shown. A turn ended inside such a call closes it as the
  template does. A result the template quotes is escaped as JSON.
- **The add-on file,** where a model writes a call unlike its template's past
  call, gives the `opening`, the `name_close` after the name and the `closing`,
  which the template must write after a call too; the arguments are JSON.
- **Declarations outside the system turn** move into it through a variable the
  template reads, where setting it changes nothing without tools; it is found
  at load and passed with every render.
- **The argument spelling** comes from four probes that differ only in their
  content. A template that writes its call as one JSON object is known at
  load, with the keys it writes (see "Calls are read in the one shape their
  template writes", 2026-11-13); its calls are read as JSON alone. Covered:
  JSON objects and arrays,
  `name{key:value}`, `name(k="v")`, attribute-style XML,
  `<function=name><parameter=k>v</parameter>`,
  `name<arg_key>k</arg_key><arg_value>v</arg_value>`, with nested objects, lists
  and separators inside quoted values.
- **A type written between a key and its value,** `key="k" type="string">v`, is
  recovered from where the renders of a string and a number argument differ. A
  value of the string type stands as its text, and any other is read as JSON
  or as a Python literal.
- **A value closed by a tag that names its own key,** `<city>Berlin</city>`, is
  read up to that tag, and arguments written after the name as one JSON object
  are read as JSON. Nested values written as elements are in "Nested values in
  a model's own elements, and tests by module".

The recovered delimiters on two real models:

| `CallBody` field | a delimited model | an element-shaped model |
|------|------|------|
| `name_close` | `{` | `>\n<` |
| `key_open` | (empty) | `parameter=` |
| `key_close` | `:` | `>\n` |
| `pair_sep` | `,` | `\n</parameter>\n<parameter=` |
| `body_close` | `}` | `\n</parameter>\n</function>\n` |
| `empty_close` | `}` | `/function>\n` |

A delimiter runs only as far as the probes agree: the element column splits
`<` off `<parameter=` because the no-argument render shares that character.

- **A format that cannot be recovered** makes `can_call_tools` `False`, but the
  call is still detected, so its syntax never reaches the reply text, and the
  model is told it could not be read. Nothing is guessed.
- **Every call is answered,** an unreadable one by a result naming `unknown`:
  writing nothing would leave a history no template produces, which the model
  would read back on every later turn.
- **A JSON array of calls** is read whole, each call owing a result, written as
  the template writes calls made together. Any other shape ends at its closing
  tag, one call per generation.

### What a failed call is told

`try again` while another call is allowed, `do not call again` once none is,
and nothing more: a model that cannot call a tool can only answer. The choice is
made after the whole round, since its last call settles the budget; a retry
asked for where the call tags are about to be withheld would be impossible. An
unreadable call spends the budget like any other, so under `max_calls` a model
gives up instead of looping; under `calls="refuse"` the retry is skipped, as it
could only be refused. Its result alone carries no instruction, because it is
written before the round settles the budget.

## 2026-11-07: Structure review: what changed and why

Revised by "The rethink's remaining decisions": `"run"` is no longer a call
policy.

Five fresh reviewers each read one part: modules, public API, the template
approach, the engine and cache layer, tests and documentation. The user chose
all of it except the two items marked rejected.

- **Eviction removes whole exchanges.** Single-message eviction could open a
  conversation on a reply, which 6 of the 50 test templates refuse and 4 render
  otherwise than the cache held, so a long conversation was no longer encoded
  as its template renders it. The survivors and a restart's history open on a
  user turn; under a tiny limit the newest reply can go with its exchange.
- **A request evicts before it is refused,** instead of being refused where an
  eviction would have made room.
- **Declaring tools later writes them where the template writes them:** the
  system turn with the conversation after it decoded again once, or the newest
  user turn. A later system turn of its own is a shape no template writes.
  Rejected: removing the call, as the reviewers proposed; the user kept it.
- **A resumed turn adds nothing after a result.** Gemma 4's generation prompt
  holds an empty thought its template does not write after a tool result.
  Measured on the real model, 4 tool questions, greedy: identical answers; left
  out, the model writes it itself, 4 tokens more per resume.
- **One boundary for a cache context:** two protocols in `cache_context.py`,
  checked once where a context is adopted; 19 presence checks went, one of
  which let a test double without a renderer skip the load check.
- **One routine for the plain text between tags** (`tokenizing.py`) replaced
  three copies. Verified on eight real models (Gemma 3, Llama 3.2, Qwen 3.5,
  Ministral 3, MiniCPM 5, Gemma 4, gpt-oss, the R1 distill): the cache equals
  the template's render after a chat and after a history with a tool turn.
- **One object cuts the cache back** (`cache_sync.py`), by shift, cut, saved
  state or rebuild; one call boundary per generation; stop strings matched once,
  on the answer, after the thought and header filters.
- **One `reply()` instead of six entry points,** the call policy explicit:
  `"refuse"`, `"yield"`, `"run"` or a registry.
- **`Config` is frozen** and holds every setting but the add-on files and what a
  request passes. Rejected: moving the hardware settings out, as the reviewers
  proposed; the user ruled it out.
- **The message callback runs before anything changes, or after everything
  has landed,** so a failing store cannot leave the cache and the table apart.
  Superseded by "One way to do each thing in the public interface", which
  removes the callback.
- **`history()` round-trips,** tool turns included, replacing a persistent
  wrapper that stored text only and mirrored 25 members by hand. Superseded by
  "One way to do each thing in the public interface": `conversation()` is the
  one read.
- **The template is resolved again when its render changes,** as one writing
  the date does after midnight, checked at each `begin` with four renders of
  about a millisecond. Superseded by "The clock, and a cache that cannot
  shift".
- **Not done:** nested values in MiniMax M3.1's own call elements, done in
  "Nested values in a model's own elements, and tests by module"; a module of
  its own for the tool loop, which would need a
  public option to withhold call tokens that nothing else wants; separate
  tokenizer and cache objects; separating the template's load-time facts from
  its per-request renders beyond the call format.
- **Considered, not chosen: rendering the whole conversation each turn.** It
  would replace about 1000 lines of template reading with about 300 and make
  encoding as the template renders hold by construction, at a few milliseconds
  per turn, decoding staying incremental. The user ruled it out and asked for
  more information. Revised by "The template as the program".

## 2026-11-07: Rendering the whole conversation each turn, measured

Answered by "Rendering windows", which renders a window of constant size.

The estimate of a few milliseconds per turn does not hold for Gemma 4, so the
approach of the time stayed. Rendering the conversation through the template
was timed at a full 8k-token context on an Intel i9-9900K, with Gemma 4's own
template (as the 26B-A4B GGUF ships it), four tools declared and the Gemma 4
tokenizer, median of 15 runs:

| Conversation | Messages | Render | Tokenize |
|---|---|---|---|
| about 12 words per message | 394 | 234 ms | 25 ms |
| about 30 words per message | 178 | 51 ms | 25 ms |
| about 90 words per message | 64 | 9 ms | 26 ms |

Twice the messages take four to five times as long, so short voice exchanges
are the worst case. Comparing the two token lists took under half a
millisecond.

On a Raspberry Pi 5 with Gemma 4 E2B, scaling by 2 to 3, a turn would wait 0.5
to 0.8 s longer for short exchanges, 0.15 to 0.23 s for medium and 0.07 to
0.10 s for long ones. Geekbench 6 single-core scores, about 770 to 960 for the
Pi 5 against about 1650 for the i9-9900K, give 1.7 to 2.2, which plain Python
widens. Against a first word about 1.2 s after a turn (30 new tokens at the 24
tokens/s one report gives for llama.cpp Q4_0 on four threads), that adds 6 to
65 %. Tokenizing only the changed end would remove the tokenizing share; the
render is the approach itself. E2B was assumed to ship the 26B model's
template and tokenizer, which was not checked.

## 2026-11-07: Nested values in a model's own elements, and tests by module

- **Nested values written as elements are read back:** an object as elements
  named by its keys, `<where><city>Bern</city></where>`, a list as elements of
  one name, `<days><item>1</item><item>2</item></days>`. Three more probes at
  load (an object with one member, one with two, a list of two strings) recover
  both, read by depth, so a member may carry the key of the value around it.
  Two shapes stay lost because the template writes them alike: an object whose
  only key is the items' name reads as a list, and an empty object or list as
  an empty string. A null is left out by the template itself.
- **The tests are sorted by the module they test,** not by review round, shared
  helpers in `tests/wrappers.py`; the collected test names stayed the same list.

## 2026-11-07: The template as the program

A prototype, made the library by "The render is the conversation's encoding".

The user asked whether a model could work every time, a future one included,
and chose a prototype (`llama_chat/rendered_chat.py`, beside `ChatWrapper`):
the template renders the whole conversation, and the cache is kept equal to
that render by decoding only what differs. It reopened the review's ruling
against rendering each turn, knowingly, and never decodes the whole
conversation again on a cache that shifts. After each reply, while it is read,
the conversation is rendered with a stand-in where the next question goes and
decoded up to it. A question then takes the stand-in's place where the template
writes plain questions exactly so, checked at `begin` with three questions (a
check removed by "Restructured inside the library"), and only its tokens and
the generation prompt are decoded before the reply; other questions, such as
ones with surrounding spaces or a tag, are rendered when they come. An eviction
finds the dropped exchanges' stretch where the cache goes on as the new render
does, and removes it.

Measured:

- All 52 test templates (357 tests): the cache equals the render before every
  reply, over conversations, evictions on caches that shift and that cannot,
  restarts from a history, a past tool turn and a later declaration. Splicing a
  question was exact on all 52; evicting removed a stretch from the middle on
  51, and on the last the only exchange left was the newest, cut from the end.
- 8 real models (Gemma 3 1B, Llama 3.2 3B, Qwen 3.5 9B, Ministral 3 8B,
  MiniCPM 5 2B, Gemma 4 26B-A4B, gpt-oss 20B, R1 distill 8B; Q4 or MXFP4; RTX
  3090): the cache equals the tokenized render before each of 12 replies, at
  1,024 and at 512 tokens of context, and with a past tool turn on the 7 that
  call tools. Every cut after a reply lay at the newest reply, which the
  template writes without the empty thought it was generated with.
- Gemma 4 26B-A4B at 8k, memory full, every turn evicting, median of 7 turns:
  the wait before the first word was 199 ms against 194 ms for `ChatWrapper`
  with short messages and 200 against 201 ms with medium ones; the work after
  each reply 1.27 s against 0.42 s and 0.63 against 0.41 s, since the
  prototype renders three to five times there.

Missing then, closed by the second structure review: a cache that can neither
shift nor be cut at its end was decoded again whole at every cut (Qwen 3.5: 12
times in 12 turns); a tool turn was not kept as generated; a template that
rewrites older history was not refused; `inject`, `snapshot`, the capability
properties, `wait_idle`, the tool loop's failure results and the reply's counts
were absent.

A reviewer proposed rendering a window of constant size instead, the system
message and the last two exchanges with the stand-in. Its text was identical to
the full render from the second-newest reply on, for all 52 templates, so the
cost would not grow with the conversation; tool loops and reasoning inside the
window were not measured. Built in "Rendering windows".

## 2026-11-07: Second structure review: what changed and why

Five fresh reviewers read the library again; the user chose the fixes, the
tests, the tidying, two restructurings and a public way to forbid calls.

- **One place decodes into the cache.** `CacheSync` holds every decode,
  generation, removal and cut, and takes the tokens the cache should hold as a
  callable, so the prototype shares its saved states. A decode that fails
  part-way marks the cache unknown and the next action decodes it again whole,
  where the cache and the table used to stay apart until the next `begin`.
- **Every "does not fit" is one error,** `ContextOverflowError`, a `ValueError`,
  for a request, a message, a history, a system prompt or a declaration, so a
  server can tell a request too long from one it cannot serve.
- **A restart that would keep nothing is refused:** the newest exchange must
  fit, where a short reply after a question too long to keep used to empty the
  conversation without a word. The message limit may still keep nothing.
- **The context's own size is checked where it is adopted,** so a `Config`
  asking for more fails at construction, not halfway through a reply.
- **A running exchange holds the lock throughout,** tools included; another
  thread's action used to slip in between rounds. A tool must therefore not
  call back into the wrapper.
- **Withholding calls.** `calls="withhold"` keeps a call's opening token from
  being drawn, by a logit-bias stage in the generation's own sampler chain; a
  call that opens with plain text cannot be kept out that way. A new request
  had never passed the call setting on, only a resumed turn. Measured on 8
  real models with a weather tool declared and the question asking for it:
  Llama 3.2, Qwen 3.5 and Ministral 3 wrote the call's markup as text anyway,
  and Gemma 4 26B-A4B repeated one phrase up to the token limit; a question
  needing no tool was answered normally. `"refuse"` let Gemma 4 call and told
  it no tool can be run, and it made up a weather report. So withholding suits a
  request that wants no tool, and the last round under `max_calls` leaves a
  call unrecognized rather than withheld. Revised by "Review fixes of the
  streaming, the thought filter and the tool loop" (316.9): that round, and
  every round under `"refuse"`, also withholds the call's opening token.
- **What a template does to a message's text is learned at load:** one render
  with every tag and every character an escaping filter changes, each between
  markers, shows what the template drops, escapes or replaces. The new
  token-level test had found a template that drops its thought tags from a
  message and escapes its turn tags.
- **The tests try to break the rules:** reasoning on in every template that
  reasons, a live call on every tool template, a tokenizer stand-in that keeps a
  typed tag as text, checks that no eviction rebuilds the cache and no cut
  reaches before the newest reply, and every model-free test ending with the
  cache read against the render. The model-name check splits names at `_`,
  digits and capitals, and caught the engine's name outside its module.
- **Removed:** the free `@tool` decorator, `ToolRegistry.spec`,
  `CacheSync.rested_at`, the add-on path of `with_model_sampling` that only
  tests used, and the doubled cut and clear in the load-time check.
  `Backend.can_edit_cache` stays, because the install script reads it.
- **Moved:** `Reply`, `ReplyStream` and `ContextOverflowError` to `reply.py`;
  `GenerationAccumulator` into the cache-context protocol, which the generation
  loop left again in "Restructured inside the library". One function decides
  where kept history may open.
- **The prototype's gaps are closed:** a tool turn is kept as generated where
  the template writes the call otherwise, a cache that cannot be cut returns to
  a saved state, and `begin` refuses a template that writes an older exchange
  otherwise once another follows.
- **Superseded since:** `history()` with `system_prompt`, so that a server
  begins again from the conversation's own history and not a text-only copy,
  is now `conversation()`.

## 2026-11-07: Rendering windows

The user chose the reviewer's window design, so that the work after a reply
does not grow with the conversation.

- After a reply only the system message, the newest two exchanges and the
  stand-in are rendered; from the first tag after the older exchange's reply
  on, that render is the whole render. The tag is found in the cache from where
  the newest question's run was decoded, that question being the one spliced
  there; the stretch from the tag to the run, and what the splice decoded, are
  checked token for token, and everything before the tag is kept.
- When the oldest exchange goes, only the system message and the oldest two
  kept are rendered. Exchanges are dropped before the decode ahead, because the
  engine shifts at its next decode, which after the decode ahead is the next
  question's: the wait for the first word was 500 ms instead of 200 ms on
  Gemma 4 26B-A4B at 8k.
- At `begin` and at a later declaration, five probe exchanges are checked
  against whole renders, with the conversation's system message, tools and
  reasoning: the window of the newest two from the first tag after the older
  one's reply; the history before that tag as the prompt for the newest reply
  wrote it; and, after the oldest exchange is dropped, the oldest two kept up
  to the first tag after the first one's reply, with everything after that tag
  as before. A template that fails is rendered whole after every reply. A whole
  render is also made after the first reply, after a
  question rendered as it came, where a window holds a tool turn (the probes
  hold none), after an empty reply, and wherever a check on the cache fails.

Tests: all 52 test templates pass the probes; the cache equals the render
before every reply, and every reply after the first is settled in windows on
the 47 templates whose scripted replies have text (on the other five the
scripted reply is all thought). On the 8 real models, 12 replies at 512 tokens:
the cache equals the tokenized render before every reply, with a past tool turn
on the 7 that call tools; replies after the first were settled in windows on 6,
while gpt-oss and the R1 distill spent each reply's 40 tokens on thought and
were rendered whole. Qwen 3.5, whose cache can neither shift nor be cut, was
decoded again whole 2 times, at evictions.

Gemma 4 26B-A4B at 8k, memory full, every turn evicting, median of 7 turns,
against `ChatWrapper`:

| messages | wait before the first word | work after each reply |
|---|---|---|
| short | 198 ms against 197 ms | 415 ms against 418 ms (whole renders: 763 ms) |
| medium | 200 ms against 201 ms | 411 ms against 413 ms (whole renders: 513 ms) |

What remains after a reply is the shift after the eviction and decoding the
newest reply again as the template writes it, which `ChatWrapper` does too.

## 2026-11-08: The render is the conversation's encoding

The user chose to make the prototype the replacement, as `ChatWrapper` with the
same calls; to refuse a model that ships no chat template; and to refuse a
message that does not fit instead of cutting it.

- The conversation is the messages a template is given
  (`llama_chat/rendered_cache.py`), and the cache is their render, decoded only
  where it differs, with the windows above. Gone, because the template writes
  each where it writes it: the message table, the per-role building of
  messages, the moving declarations, the join of two user turns, the later
  system turn, the result wrapping and the past tool turn built from pieces.
- Pieces are read at load only for what a generation holds: a thought, a
  reply's end, a call and its format, and what the template does to text.
  `ChatWrapper(fragments=...)` and `oversize_policy` are removed.
- A turn that called tools keeps the tokens it was generated as, from its
  generation prompt to its final reply. It moves with every removal before it,
  and goes when an eviction drops its messages.
  Matching on the tokens after the turn failed where rounds repeat, as nine
  identical calls do; how the turn is placed is in "The target, built without
  the cache".
- A call's tags are the single tokens a generated call holds, so a history read
  in is token for token the live turn; its arguments are never read as tags.
- A reply's thought goes to the template as `reasoning_content` only where the
  template keeps a past thought: Granite writes a past reply otherwise when it
  is handed one, though it drops it.
- The limits count the render without the stand-in question, which made a
  conversation seem longer and dropped exchanges that fitted. Where the
  window's own older exchange must go too, or the decode ahead does not fit,
  the rest is rendered whole. A reply without an answer is taken out again
  where the template writes two user turns in a row.
- `Reply.n_prefilled_tokens` counts every token decoded for a request, the
  decode ahead and a return to a saved state included.

The check that ends every test on the model-free cache accepts the render with
its generation prompt, without it, or decoded ahead up to the next question.
The fake template writes several rounds of calls in one turn, as real
templates do.

## 2026-11-08: Every text placed by a marker render, and the review's fixes

Revised by "The rethink's remaining decisions": `declare_at_begin=`,
`specs_declared_at_begin()` and `calls="run"` are gone, and the parameter is
`tool_call_policy`.

The structure review of the replacement found nineteen faults it brought in,
four the old code had, eight holes in the tests and a set of restructures. The
user chose to fix them all; close every test hole and merge the two
template-wide test files; give the turn state one owner; replace the searches
with a marker render and split the engine file; narrow the load check and stop
a cache context from handing in template pieces; make the public calls one way
each; rename the tool flag; let a conversation follow the clock; drop well
below the limit where the cache cannot shift; and commit without pushing. The
language-model image was rebuilt later, in "Measured through the stack".

### Every text placed by a marker render

The cache must know where each text of the caller's (a content, a thought, a
call's name and string arguments) sits in the render. Searching with `find`
took the first match: a message whose text was template text, such as
`<|im_end|>`, made the template's tag plain text and the typed one a real turn
end, on 49 of 52 templates. Counting closings to place kept turns found nothing
where a template has no reply closing (GLM 4.7, GLM 5.3, Flint).

Now each render is made a second time with a marker (`QQ<n>MQQ`, letters and
digits that trimming, escaping and quoting leave whole) in place of every text
(`llama_chat/split_render.py`). The real render must be the marked one with
each marker replaced by its text as the template writes it: as a content of its
role, trimmed, or inside a JSON string. Failing those, the text is what lies
before the template's next text, where exactly one place leaves the rest
matching; MiniCPM 5 wraps an argument holding a tag in CDATA this way. A
template that writes anything else of a text makes the renders differ, and the
conversation is refused rather than encoded by a guess. Empty replies carry no
marker, since a template may write a reply only where it has text; empty user
and tool contents carry one. A call's end is anchored after its last string
argument, else after its first name, because Gemma 4 and gpt-oss write the name
again in the result block. The cost is one more template render per render,
reused within an action.

### The target, built without the cache

What the cache should hold is built from the render alone, each kept turn's
tokens in place of the template's version: from the first tag after the
question's text to the end of the turn's last call closing or final reply
text. A resumed turn that ends with neither stays kept up to its call. The
limits therefore count what the cache holds, and a rebuild decodes kept turns
as generated.

The cache is compared with the target token by token. A stretch the target no
longer holds is found by searching the cache (`_dropped_stretch_end`), after a
drop and where declarations move with the newest question (Mistral Small 3.2,
Ministral), whose old block is then removed in place. A wrong match costs
decoding only, since the target is never taken from the cache. Rejected: a
recorded map of the cache, which every cache edit would have to keep in step,
for a search that is a decoding shortcut, not a placement.

### One owner of the turn, of the tokens, and of a failed action

- The conversation alone owns whether a reply is open and which calls are owed;
  a history ending on a call restores it as owed.
- `CacheSync` owns the tokens the cache should hold. A cache that shifts or cuts
  takes each change at once; one that cannot is brought up once before the next
  generation or rest, from the newest saved state that still matches, else by
  decoding again, so a turn that drops twice is repaired once. A refused cut
  leaves it unknown until the next action, and a generation after a decode
  without logits decodes the last token again with them.
- Every action saves the conversation's state and the tokens and puts both back
  where it raises. A generation that fails keeps what it said, or takes back
  its own question where it said nothing, and leaves its cache to the next
  action, so a second engine error cannot replace the first.
- The history and the system prompt are published as a copy after each change,
  so another thread never reads one half-built.
- The wrapper calls only the conversation, renders through
  `ResolvedTemplate.render`, and takes a reply's thought from `ThoughtFilter`.

### The clock, and a cache that cannot shift

A template that writes the date follows the clock: where the conversation's
opening renders otherwise than at the last action, the conversation is rendered
whole once and decoded again from the date on. Rejected: freezing the date per
conversation, which is exact without the decode, but has the model read
yesterday's date until the next `begin`. Nothing the template is read for holds
the date, so it is no longer read again at each `begin`.

Where the cache cannot shift, an eviction drops to at most
`evict_to_fraction_without_shift` (0.5 by default) of the threshold, so the
cache is decoded again only every few turns. Rejected: refusing such models,
which would have kept the wording of the rule against decoding the whole
conversation each turn and lost Qwen 3.5. What the fraction is taken of was
revised by "Where an eviction stops on a cache that cannot shift".

### What a template is read for, and refused for

The load check no longer compares a conversation built from pieces with the
render. It checks what a generation reads: that the generation prompt ends the
render of a question and that the closing after a reply follows a past reply's
text, in each reasoning mode. `can_reason` means the template marks a thought;
the switch is the template's wherever it writes it. The pieces are read from
the template's own render only, so `ModelReader.extract_fragments`, the export
of `TemplateFragments` and what served only the old check are removed. The
check that older history stays written refuses a template that refuses its
probe, instead of passing it.

### The public calls

Revised in part by "One way to do each thing in the public interface", which
leaves `conversation()` as the one read and adds `owed_calls`.

- `snapshot()` and `awaiting_tool_result` are removed. `history()` and
  `system_prompt` are safe from any thread, and a caller's record keeps tool
  calls and results.
- `can_take_system_prompt` says whether `begin` takes a system prompt, and
  `inject(text)` splices a question as `reply(text)` does.
- `STOP_REASONS`, `StopReason` and `InvalidSettingError`, whose `field` names
  the setting, are public.
- `declare_at_begin=` replaces `in_system_prompt=`, and
  `specs_declared_at_begin()` replaces `specs_for_system_prompt()`: the
  template, not the flag, decides where a declaration goes.
- `calls="run"` without a registry, and `max_calls` where no call runs, are
  refused instead of ignored.

### Faults the old code had

- Llama 3.2 writes `<|python_tag|>` before a call its template opens with plain
  text; such a tag is held back while a call may follow, and is never shown
  (2026-11-11: no longer true: where the text after it cannot begin a call,
  it is shown as the reply's text).
- A tool result JSON cannot write is an error result, and refused by
  `inject_tool_result`.
- The README promised `"interrupted"` for a stream closed early, which has no
  reply to carry it.

### Tests

The two template-wide files became `tests/test_conversations.py`. It takes the
expected render from what the test sent, bounds what one reply decodes by about
four exchanges, drives a template whose windows differ from its whole render,
spaces a live call unlike its template, and puts a typed tag into every kind of
text. Against two deliberate breakages: recording every reply as empty failed
90 of the render checks, and switching off the removal in place failed the
eviction check on all 53 templates. `tests/conftest.py` checks the cache
around each kept turn, the real-template contexts included, and
`tests/test_failures.py` drives each refused or failed action. The tests for
the faults were written after the fixes, not run against the code before
them.

### A fresh review of the change, and its fixes

A reviewer who had not seen the work probed the diff with fault injection over
every template. Each finding and its fix:

- **A question decoded at a stale position:** a note of where a spliced
  question began survived what should clear it, so the cache could lose an
  exchange the conversation held (48 of 52 templates after a generation that
  failed part-way). The note now names its question and the window checks what
  the splice decoded; comparing the question's tokens fails where no tag stands
  between a question and its reply, as in Mistral's. Later replaced by records
  tied to a version ("Restructured inside the library").
- **Declarations that move with each question, on a cache that cannot cut:**
  on Mistral Small 3.2 and Ministral 2410 every reply decoded the whole
  conversation again, which the rule against decoding the whole conversation
  each turn forbids. Each sync there now saves a state
  where it cuts, so a reply decodes about one exchange more than on a cache that
  shifts, for one saved state more per action. A state pinned after the system
  turn was offered, and built in "A state kept where the oldest question
  begins".
- **A failed generation took back a message it had not added;** now only its
  own question, and `on_message` is not told of a reply never recorded.
- **A turn nobody could close,** where the error result for an unreadable call
  failed to decode: it now closes as a reply without the call before raising.
- **A kept reply without text placed at zero width,** on the question's own
  closing where both closings are the same text. Its end is the last closing
  before the caller's next text. The window and oldest-exchange renders refuse
  an exchange that holds a kept turn, and `tests/conftest.py` requires every
  kept turn to stand in for template text of its own.
- **Exchanges that repeat word for word** made the search for the dropped
  stretch find nothing, and everything down to two exchanges went. The stretch
  is now located by known positions: before the drop, the cache up to the tag
  after the second oldest exchange's reply is checked token for token against a
  render of the oldest two; after it, the render of the oldest two kept, up to
  the tag after the first one's reply, replaces that stretch, which the window
  check guarantees reads as the whole render.
- **A refused action cost a cache that cannot cut a whole decode** (503 tokens
  instead of 15 in the probe): putting back unchanged tokens dropped the note
  that the last decode left logits. Unchanged tokens now change nothing.
- **Shared argument dicts:** `history()`, an owed call and a yielded call each
  hand the caller its own copy.
- **Kept turns by object, not by `id()`,** so a removed turn's position is never
  read for a new one.
- **Shortening a kept reply** is refused like shortening one that holds a call.
- **A thought's closing tag held back as a call's:** only a tag that is not the
  thought's own is held back before a plain-text call. With it, a
  `startswith(None)` in the generation loop, reached with calls off, is gone.

Not a defect: while a call is open, Mistral Small 3.2's cache keeps the
declarations before the question, which the render with the result puts back.
Not changed then: `history()` and `system_prompt` as two reads that another
thread can pair across a `begin`; one read of both is built in "One read of the
system prompt and the history".

### Measured before this round's changes

Gemma 4 26B on the GPU, about 7,350 tokens kept in an 8k context: the wait for
the first word was 196 ms for an 8-word question (the old code 195 ms) and
202 ms for a 20-word one (202 ms), and the work after each reply 416 ms
(406 ms). Plain chat on 8 real models, 12 replies each in a small context, held
the template's render before every reply.

## 2026-11-08: Measured through the stack, an injected question made fast, one read of the conversation, a state kept before the oldest question

### Measured through the stack

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
Llama 3.2 3B, after a past tool turn, the next call shows no `<|python_tag|>`
and raises nothing, and with calls run it answers from the result. Its odd
reply `"}"` to "Thanks." follows the template, which writes a tool result as a
quoted JSON string.

### An injected question waits only for itself

The server injects each question, then asks for a reply without text. That path
was slow twice over, measured on the GPU with the stack's settings (224 ms
against 84 ms for `reply(text)` once full):

- An injected question was spliced only below `evict_above_tokens`, so a full
  conversation was rendered whole. It is now spliced within the room left for a
  reply, as `reply(text)` splices, and the eviction follows its reply; only a
  full `max_history_messages` evicts first, since its count holds after every
  action.
- `reply()` then rendered everything again, though the splice had decoded the
  generation prompt; it now goes straight to the generation.

The test that a question waits only for its own tokens missed both, being
short enough to stay under its bound when rendered whole. It now runs three
exchanges first, counts every render, and drives the injected path; a second
test puts an injected question over the threshold.

### One read of the system prompt and the history

Superseded by "One way to do each thing in the public interface", which leaves
`conversation()` as the only read.

`conversation()` returns both from one published copy, so another thread never
pairs the system prompt of one `begin` with the history of another.

### A state kept where the oldest question begins

On a cache that can neither shift nor cut, an eviction removes a stretch after
the system turn, past which both resting states lie, so the cache was decoded
again from the start, system prompt and declarations included. `CacheSync.pin`
keeps a third state where the oldest question begins, read from a render of two
exchanges so that nothing that moves with the newest question stands before it;
the state goes once anything before it changes. On the model-free templates
with tools and a long system prompt over 16 replies: Gemma 4 from 5,213 tokens
decoded to 1,694, Qwen 3.5 from 3,059 to 1,762, Llama 3.2 from 1,739 to 766.
The cost is one saved state more, about the size of the cache up to the oldest
question.

## 2026-11-08: Where an eviction stops on a cache that cannot shift

Revises "The clock, and a cache that cannot shift"; how the system turn is
measured was revised again by "Bugs found by the rethink".

`evict_to_fraction_without_shift` was a fraction of the whole threshold, so a
long system turn took the history with it: on the model-free templates that
render whole, one of about two thirds of the threshold left no exchange after
the first eviction. It is now a fraction of the room after the system turn.
Declarations that move with the newest question are not counted in the system
turn, so a block larger than half that room can still take the history.

Also found: where a template refuses a system message alone, the system turn
was measured as what a system message adds to a question's render. Mistral
Small 3.2 writes a long system prompt of its own where none is given, so a
shorter one measured negative (about -2,000 tokens with tools declared). The
measure is now at least the tokens before the oldest question.

## 2026-11-08: Measured on a cache that cannot shift

Qwen 3.5 9B (Q4_K_M) on an RTX 3090, an 8,192-token context, the default
eviction settings, six tools declared. A conversation of 70 questions of about
185 tokens each, with replies capped at 40 tokens, was run on three versions:
before the state kept before the oldest question (254080c), with it (ce18915),
and with the eviction target measured from the room after the system turn
(780c5b7), once with a system turn of 987 tokens and once with one of 3,517.

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
185 tokens are decoded, and the history an eviction keeps is decoded after the
previous reply. The kept state cuts the total by 9% and 39%. The new target
keeps more history, so evictions come sooner and the total rises by 13% and 25%
over the kept state alone. That cost falls after a reply, never before the
first word, so the default of 0.5 stays; 0.4 would, by calculation, bring the
short system turn back to the old total and history. Whether a question asked
while that decode still runs waits for it was not measured.

## 2026-11-08: Bugs found by the rethink

**The system turn's length.** The render of the system message alone, used for
the limits, missed declarations and settings that a template writes only once
a question follows: mistral-large-4 with one tool counted 26 tokens where the
conversation opens with 256 or more, so `begin` accepted a system turn of
1,260 tokens against a 600-token threshold and the first question was
refused, and an eviction on a cache that cannot shift aimed too low. Now two
measures serve:

- `_opening_length`, the tokens two renders share that differ only in the
  oldest question, places the saved state. Over every test template it is at
  least either old measure, and a saved state is used only where it still
  matches the cache, so a measure one token too long costs a decode, never a
  wrong cache.
- `_with_a_short_question`, what the conversation takes with no history and a
  one-character question, is what no eviction makes smaller. `begin` and a later
  declaration refuse a system turn that leaves no room for it, and the eviction
  target on a cache that cannot shift is measured from it. That closes the case
  "Where an eviction stops on a cache that cannot shift" left open: where
  declarations move with the newest question (ministral-2410), the history is
  no longer dropped whole.

**Beginning from a long history.** `begin` dropped one exchange at a time,
rendering everything after each: 800 messages took 716 renders and 5.4 s
without a model. At the user's choice it now keeps the newest exchanges that
fit with the system turn in half the context, found by doubling and then
halving the count, rendering only those: 40 renders and 0.03 s. Half rather
than the eviction threshold leaves room for some turns before the first
eviction. The newest exchange is kept even where it alone takes more than half,
`max_history_messages` applies after, and a history the template refuses is
refused only for what would be kept.

**Smaller ones.**

- The last round under `max_calls` leaves a call unrecognized, as text, rather
  than withheld, as the second structure review decided; the texts said
  otherwise. The internal switch is `recognizes_calls`.
- `ChatWrapper.config` was the configuration as given, while the context it
  loads runs with the model's own sampling filled in; where the wrapper loads
  the model, it is now that configuration.
- Any fault while reading a call was reported as an unreadable call; only
  `ValueError` and `RecursionError` are now. A fault that escapes leaves the
  cache consistent and the question without its reply, and the test requires
  it to be raised instead of the turn to close.
- A `ToolTurn` whose result answers no earlier call of the turn is refused when
  it is built, instead of being written with an invented call id.

## 2026-11-08: The oldest llama-cpp-python supported is 0.3.23

The adapter looked every C symbol up under each name llama.cpp had given it
since 0.3.0: the `llama_kv_self_*` and `llama_kv_cache_*` cache edits beside
`llama_memory_*`, `llama_load_model_from_file`, the `llama_token_*` vocabulary
calls, a missing vocabulary handle, the boolean `flash_attn` beside
`flash_attn_type`, and builds without saved states. Checked against each
release's `llama_cpp.py`: 0.3.10 adds the memory API, 0.3.23 the
`llama_state_seq_*_ext` calls with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` that
the saved states rest on, and every other name the adapter calls is present
from 0.3.9 on; the machines this runs on use 0.3.36. The floor is now 0.3.23
in pyproject.toml, install.sh, the README and the server's image, every symbol
is required under its one current name, and the fallbacks went with
`Backend.can_edit_cache` and `Backend.can_save_state`, which could no longer
answer anything but yes. install.sh now checks the build by constructing the
adapter, which raises where a symbol is missing. The lookup under several
names stays for the next rename.

The penalties call still differs inside the supported range: 0.3.23 to 0.3.34
declare four parameters, 0.3.35 on five, the vocabulary size first, so
`_penalties_take_vocab_size` stays.

## 2026-11-08: A system prompt is kept wherever no later question moves it

Supersedes the refusal in "How the template is read" of a template that writes
the system prompt into the first user turn. That refusal, made wherever the
system tags equalled the user tags, refused Gemma 3, which writes the system
text at the head of the first user turn. After an eviction the render puts it
at the head of the new oldest question, so it never moves relative to the
start, and the cache, being the render, holds it correctly.

The test is now a render probe: two renders that differ only in the oldest
question, the system prompt kept where its text lies in the opening they share.
Ministral 2410 writes it before the newest question, so it moves every turn and
stays refused; every other test template answers as before. `system_prefix`,
`system_suffix` and their extraction went, as nothing else read them.

Measured before the change, the refusal patched off: Gemma 3 1B (Q4_K_M) on the
CPU, context 2,048 (threshold 1,536), a 16-word system prompt, 30 questions of
about 45 tokens and replies capped at 30 tokens. The cache can shift; ten
evictions each decoded 64 tokens up to the next question, the same as the
median turn without an eviction: the dropped stretch is found and shifted at
the seam between the system text and the next question. Without a model, all
of Gemma 3's conversation tests run with a system prompt and pass.

## 2026-11-09: Restructured inside the library

The rethink found the library doing several things in two places, and the
engine adapter holding logic that names no engine. No public name changed.
What a caller can notice is named below: a few error messages and one
warning, and the members the two cache-context protocols ask for (the first
two items below).

**The generation loop is engine-neutral.** The loop that generates a
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

**Tags found from the tokenizer's answers.** The model reader protocol
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

**One sampler per generation.** Every generation builds its sampler
chain from its effective settings and frees it when it ends; the shared chain
built at load and reset for generations without overrides was removed. A new
chain starts its penalty window and its seed exactly as a reset one did, so
one path replaces three, at the cost of building a few stages per reply. A
warning about an incompatible penalties signature now appears at the first
generation rather than at load.

**Add-on files in one module.** The add-on file and the model's
sampling share one module, since both settle what a model needs beyond its
template and are read together at load. The template renderer is a plain
function of the engine adapter, because it resolves no C symbol, so rendering
a template no longer builds the whole adapter.

**Reading a call.** Reading a generated call is one call,
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

**No formatter.** `TemplateFormatter` is removed. `ResolvedTemplate`
gives the generation prompt and the answer opening for a reasoning mode and
refuses reasoning a model cannot do (`ensure_can_reason`, as
`ensure_tools_usable`, so `begin` refuses before it resets anything).
`RenderedCache` alone holds whether the conversation reasons, and
`RenderedCache.generate` bounds a reply by the room left beside the closing,
which is tokenized once at load instead of on every generation.

**Fewer probes.** The fragments' three whitespace settings
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

**The conversation's state.** The state a failed action puts back is one
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
conversation continues from theirs; no action clears them (2026-11-11: no
longer true: `begin` clears all three, and a generation clears the record of
the question its request asked). The cache length
alone was rejected, because two different targets of one conversation can
have the same length; `CacheSync.version` changes with every change to the
token list, checked by a random test of 3000 operations with injected
failures. After a failed action that changed the cache, the next question is
rendered whole once.

**Renders and the cache, each in one place.** Bringing the cache to a
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

**One generation path in the wrapper.** One generation function, run with
the lock held, chooses between going on with the open turn and a new request.
The tool loop drives it directly and reads the calls from its `Reply`, instead
of collecting them back out of a stream that had appended them. The wrapper's
own check that no call is owed a result went, since `RenderedCache.generate`
makes the same check before anything is generated. The "yield" path still
releases the lock before it hands the calls over, because a caller answers
each call while reading the stream.

**A reply's text as one object.** `ReplyText` feeds each piece through
the thought filter, the header filter and the stop strings in that order, and
settles what is held back from the stop reason. The wrapper no longer
assembles the three filters or carries their flags, and the filters are
private to `reply_text`. `reply_header` and `begins_in_thought` stay public:
the tests use the first for two cases the object cannot show (a thought tag
or the answer opening always matches first there), and the second to script
how a model writes.

**Small items.** A cache that cannot shift is not asked even for a shift
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

## 2026-11-09: One way to do each thing in the public interface

Revised by "The rethink's remaining decisions": the four smaller removals that
the "Kept" paragraph leaves to the user's answer were made, and so was the
removal of `calls="run"` and of the registry `begin` keeps.

The rethink found several second ways of doing one thing in the public
interface. Each change below removes one; the service changed to match where
it used one. Four of them revise a recorded decision, named at each.

**One way to say how a call is written.** This revises "How a call's
format is recovered", which kept `ChatWrapper(tool_calls=...)` as a way out
where the template cannot be read. The add-on file is now the one way to say
how a model writes a call, as the rule that a model works without a code
change states. A caller can no longer give tags
a template never writes, nor the argument spelling of a call that has tags of
its own; the add-on file cannot describe that spelling, so such a model would
now be refused tools. All 41 test templates that write calls still call tools
without it. `ToolCallFragments` and `CallBody` are internal again, and
`ToolCallFragments.body` is removed.

**One way to read the conversation.** This reopens "One read of the system
prompt and the history", which kept `system_prompt` and
`history()` beside `conversation()` for a caller that needs only one. All
three read the same published copy, and two separate reads could pair the
system prompt of one `begin` with the history of another. `conversation()` is
now the only read, so a caller cannot make that mistake.

**No message callback.** This revises the first structure review ("The
message callback runs before anything changes, or after everything has
landed"). `ChatWrapper(on_message=...)` is removed: a caller already has every
message it sends and every reply it reads, and `conversation()` reads back
what the conversation holds, tool turns included. The hooks that ran it inside
`ask`, `add` and `record_reply` went with it; a caller that wants to refuse a
message does so before calling.

**A request's sampling, checked once.** A per-request `Sampling` held its
own copy of the configuration's `top_p` check and applied itself through
public members (`overrides`, `applied_to`). It is now checked by building a
configuration from its settings, so a request is refused by exactly the rules
the configuration is held to; a test pins that. `overrides` and `applied_to`
are removed rather than made private, so that no module reaches into another's
private name; one internal function, `with_request_sampling`, checks and
applies a request. Checking against a default configuration is sound because
the five sampling fields have only checks of their own, none tied to another
setting.

**A tool result answers the call itself.** `inject_tool_result` takes
only the `ToolCall` it answers; the name alone is no longer accepted, because
results are owed in order and the call handed over already says which one is
answered. The check compares the whole call, and a refusal shows both calls,
so one with the right name and other arguments explains itself.

**`declare_tools`.** `declare_unknown_tools` is renamed `declare_tools`.
The old name described how the method filters (it skips tools already
declared) rather than what a caller gets; the filtering stays and is stated in
the docstring.

**The calls still owed a result.** This revises "The public calls", which
kept `reply_open` as the one state property. `ChatWrapper.owed_calls` lists
the calls still waiting for a result, published with the conversation after
each change, so a reader on any thread sees them as the newest finished change
left them. A locked read was rejected: it would block a thread that reads it
while its own stream holds the lock. The service keeps only its own call ids
and pairs the newest ids with the owed calls; its copy of the owed calls is
gone, and with it a defect in reading back a batch of calls only partly
answered (see its design notes, 2026-11-09).

**One cache type spelling.** `kv_cache_type` defaults to `"f16"`, and
`None` is no longer accepted. The documentation already said `None` meant the
engine's default, which is f16, so one setting had two spellings. The setting
now always holds the type the cache is built with, and the context always
passes it to the engine.

**A wrong state refused at once.** This revises the contract that
`reply()` is lazy and checks nothing until the first iteration. Holding or
briefly taking the lock at call time was rejected: an unread stream never runs
its `finally`, and a thread whose own stream is paused holds the lock and
would deadlock on its next `reply()`. Instead the conversation publishes, with
each change, whether it was begun, whether a reply is open, the owed calls and
the newest role, and `reply()` refuses against that copy at once, without the
lock. The same rule functions check the live conversation again at the first
read, so a concurrent change can at most move a refusal to the read, as
before; a wrong state is never accepted.

**Kept.** `max_history_messages`, which the rethink proposed to remove as a
second eviction trigger, stays at the user's word. Removing `calls="run"` and
the registry `begin` keeps, and the smaller removals the rethink proposed, wait
for the user's answer to an explanation of each: the eviction record's third
field and the public `Eviction` type; `CacheContext.warmup` and the wrapper's
warmed flag; `ToolRegistry.__contains__`; and `result_text` in
`tools.__all__`. A fifth, the unused defaults of the cache context's
`generate`, went with that member when the generation loop left the engine
adapter.

## 2026-11-09: The README keeps to what a caller relies on

The README's passages on how the library works inside were removed. Each was
already recorded in the entry named here:

| Removed from the README | Recorded in |
|---|---|
| the render with a stand-in where the next question goes, made after each reply | "The template as the program", "Rendering windows" |
| an injected question put where the next question was prepared for | "An injected question waits only for itself" |
| `can_reason` as "the thought tags were recovered", the reasoning switch being the template's | "How reasoning is read", "What a template is read for, and refused for" |
| declarations that move with each question, where a template declares tools before the newest user turn, their old stretch removed in place | "Declaring tools later writes them where the template writes them" under the first structure review, "The target, built without the cache" |
| each text placed by a second render with a marker in its place | "Every text placed by a marker render" |
| what each test file checks | each test file's docstring |
| the table of modules and their roles | each module's docstring |

The Verify section now says only how to run the tests and the demo.

## 2026-11-09: A test suite that tries to break the rules, not to repeat them

The rethink found the tests leaning on a hand-written renderer, on copies of
names the code defines, on the library's own readings of a template, and on
its private fields. Each change below removes one of those. The suite went from
2,779 tests in 285 s to 2,786 in 55 s, with no check dropped.

**One stand-in, rendering real templates.** The two model-free stand-ins
became one, `FakeContext` in `tests/fake_context.py`, with one tokenizer, one
cache, one record of what was asked of it and one scripted model. The invented
test format is a real Jinja text rendered by the renderer a model is loaded
with: each of the 14 small templates in `tests/invented_templates.py` is that
text with its switches set at the top. They replace the hand renderer and the
five subclasses that rewrote its renders with regular expressions, so no test
can pass on render behaviour only a hand-written renderer had. Before the hand
renderer was deleted, 20,881 random renders were shown to match it. The
invented reasoning switch is `<think_on>`, since real templates write
placeholders such as `<function-name>`, so a hyphen cannot mark a tag. The
test templates' helpers are in `tests/chat_templates.py`, because the README
names `tests/fake_context.py` as the worked example of a cache context.
Generated token ids start above every character's code point.

**Expected renders from what a template states.** The expected render in the
template-wide tests takes a template's variables from what its fixture
states (a `render_variables` field in each), not from what the library reads
off the template, so a wrong reading fails a test instead of passing it.
Against a copy of the library that reads no reasoning switch, the old
expectations let 122 of 130 reasoning tests pass, the new ones fail 51.
Stating the variables showed one such case: cohere-2 reasons by default and
switches off with `reasoning_effort: "none"`, which the switch rule in "How
reasoning is read" never tries, so its reasoning-off prompt is the template's
reasoning-on prompt. Its 18 cases are expected failures in strict mode, so
they turn into failures once the library tries that value; the fix waits for
a decision.

**The automatic check, tested.** The check `tests/conftest.py` runs after
every test used to repair the cache first, which hid a cache an action had left
behind its tokens. It now repairs only where a failed decode, edit or
generation marked the cache unknown, which is what the next action would do;
over the whole suite every wrapper ends with the cache current except two
after a failed generation. It requires the template's version of a kept turn
to begin and end where the turn does, and compares everything around it
exactly. `tests/test_cache_check.py` breaks the cache one token at a time and
shows the check catches each break; the opening token, which spells no text,
is covered by the token comparison alone. With the check tested, 78
end-of-test checks that repeated it were removed.

**One property for a failed action.** The rule that a failed action leaves
everything as it was is tested as one property: every public action, on three
kinds of cache, fails at each of its cache operations in turn. A reply is the
documented exception: it keeps what it produced before the failure, pinned as
the text the stream showed and then the calls and results it made, never a
lone question. The property replaced ten single cases. A second, roomy size
reached no further library function, so only the size at which actions evict
is kept.

**Nothing copied from the code.** The probe markers are read from the
library's source (the hand list had missed seven of sixteen), the call probe
texts are recorded from what the probes ask a template to render, and the
cache types come from the configuration. Renderers that copied a real format
were replaced by that format's template; their shape round trip was dropped,
since the real-template test covers every one of those shapes with a superset
of its arguments.

**Where each check lives.** `tests/test_real_templates.py` holds what a load
checks; the whole-conversation tests on real templates live together in
`tests/test_conversations.py`, all fixtures read through one loader. Test
contexts compiled their template again for each conversation, and compiling
costs ten times a load's own checks (40 ms against 4 ms); compiling each
template once took the suite from 285 s to 76 s. The stand-in also remembers
up to 200 renders per template, except for templates that read the clock,
since a Jinja render costs about 130 µs where the hand renderer took 4 µs; that
took it to 55 s, with peak memory from 158 MB to 177 MB. Hand-picked templates
became selections by property only where the property holds on its own, as
with the thought test over every template that reasons.

**No private reads.** Tests read only public members, except in
`tests/internals.py`, where each fact no public member gives has one function
returning plain values, so an internal change that keeps the behaviour changes
only that module. The engine adapter's own tests (`tests/test_backend.py`,
`tests/test_generation.py`) still read the adapter's internals.

**Duplicates and names.** Tests were deleted as duplicates only where another
test makes every one of their checks on the same code path. Three generation
tests had become empty when stop strings left the generation loop; they now
give the loop a closing text to hold back, and fail where the release at a
call or the final flush is removed. Test names that spoke of old settings use
the current ones.

The contract's verification text now lists, for each rule, one line per test file and what that file tries to break.

## 2026-11-10: The rethink's remaining decisions

The user decided the items the rethink left open. Each change below is a
commit of its own.

**Reasoning off where a template reasons by default.** This revises "How
reasoning is read". The switch tried `enable_thinking`, then
`reasoning_effort` with `"high"` for on and the template's default for off.
A template that reasons by default and turns reasoning off only with
`reasoning_effort: "none"` renders `"high"` and its default alike, so its
switch had off equal to on, and a reply asked without reasoning was generated
after the reasoning-on prompt. Where neither of the first two changes what
follows the user turn, the switch now tries the default for on and `"none"`
for off. The 18 cases of cohere-2 that were expected failures pass, and no
other test template's switch changed.

**The eviction record goes.** A cache context's `apply_eviction` takes the
start and end of the stretch it removes, and the public `Eviction` type is
removed. Its third field, the cache's length before the removal, only bounded
the shift of what follows, and the engine shifts everything after the stretch
when given no end; a stand-in knows its own length. A type for two integers
was one more public name for a context's author to learn.

**The engine context warms itself.** Revised by "Load time and thread
defaults" (316.1): the engine context decodes a batch of two tokens and then
one, not its full batch. `CacheContext.warmup` and the wrapper's
flag that called it once at the first `begin` are removed. The engine context
decodes its full batch and its single token at the end of loading, so loading
takes one warm-up longer and the first `begin` that much less. Warming is a
property of the engine, not of a conversation: a stand-in needs none, and a
context of one's own warms itself where it needs to, so the protocol no longer
asks every context for a member only one of them uses.

**No membership test on the registry.** `ToolRegistry.__contains__` is
removed: only tests asked whether a name was registered, and a caller reads
that from `specs()`, which the tests now do.

**`result_text` is internal.** It leaves `tools.__all__`. It writes a tool's
result as the text a template is given, which only the library does; the
modules that use it still import it by name, since no module imports a
sibling's private name.

**One reading of the plain generation prompt.** The fragments' generation
prompt is empty where it is only the assistant tag, so four places read "the
generation prompt, or else the tag". That reading is now one property,
`TemplateFragments.generation_prompt_without_reasoning`, which all four use.
No behaviour changed.

**Two exports fewer.** `GenerationAccumulator` and `CallBoundary` are no
longer exported from `llama_chat`. They were exported for a cache context that
ran its own generation; since the generation loop moved inside the library
("Restructured inside the library"), a context only answers the four steps
the loop asks for, and no caller builds or reads either type.

**README corrections.** The Configuration text said every unset sampling
setting in `ChatWrapper.config` holds the model's value; that holds only where
the wrapper loads the model itself, and with a cache context of one's own
`config` is the configuration as passed, as its docstring says. The task table
now names `declare_tools`'s parameter `tools`, and shows that `reply`'s `text`
defaults to `None`.

**Pointers to the document that holds the text.** Three comments pointed to a
README section that does not hold what they named: the eviction ordering a
wrapper test pins, the shapes of `CallBody`'s fields, and what counts as a
tag. They now name the design-history sections "How eviction works" and "How
a call's format is recovered".

**The call policies by name.** `CALL_POLICIES`, beside `STOP_REASONS` and
exported the same way, lists the values `reply(calls=...)` takes besides a
registry. The check in `reply` and its error message read it, and so do the
tests that went through the policies by hand; one of them had left out
`"withhold"` and `"run"`, which now hand their sampling over under test too.

**A call whose argument spelling the template does not show.** A model whose
calls have tags of their own but whose template does not show how it spells
the arguments is refused tools, since `ChatWrapper(tool_calls=...)` went ("One
way to do each thing in the public interface"). The user decided to wait until
such a model appears, and then let its add-on file carry the spelling.

**`"none"` is the off wherever it changes the prompt.** The switch above
still took `"high"` and the default first. A template that reasons by
default, writes something more for `"high"`, and stops reasoning only for
`"none"` got the default as its off, and the default reasons. Where
`enable_thinking` changes nothing, the off is now `"none"` wherever it changes
what follows the user turn compared with the default, else the default; the
on is `"high"` wherever it differs there from that off, else the default.
Where on and off are the same, there is no switch. Of the 52 test templates,
only cohere-2's on changed, from the default to `"high"`, which it renders
alike, so no render changed.

The review then found that this picks `"high"` as the on even where the
template's own default already reasons, so such a template would reason harder
than its default when reasoning is asked for. At the user's choice, where
`"none"` is the off the template's default is the on, since that default is
what the template writes when it reasons; where the default is the off, the on
is `"high"` as before. Cohere-2's on is the default again. A template whose
default does not reason and that still writes something of its own for
`"none"` would get its default, so no reasoning, when reasoning is asked for.
No test template has that shape, and the renders alone cannot tell it from a
default that reasons, so it waits until such a template appears.

**Four more pointers to the design history.** Four comments pointed to the
README's "Tool calling", which holds none of what they named: how a call's
format is recovered, how nested values are read, what counts as one of the
model's own tags, and why reading a call and running one are separate. The
first three now name "How a call's format is recovered" and "Nested values in
a model's own elements, and tests by module". No section records the reason
for the fourth; its comment names "Restructured inside the library", which
records how a generated call is read.

**The tokens a reply was generated from.** `Reply.n_prompt_tokens` is the
cache's length when the reply's last generation drew its first token: what
that generation was conditioned on, its generation prompt included.
`n_prefilled_tokens` counts only what the request decoded, so a caller could
not tell how long a prompt the model read. It is read where a generation
starts, after the question or the generation prompt is decoded, so an
injected question decoded with its generation prompt and a turn resumed after
a result report the cache as the model sees it. Where calls run, it is the
last generation's, not a sum: a sum would count the conversation once per
generation, which is no prompt any generation read.

**Template variables as one mapping.** The render member of a cache context
took the template's variables as loose keywords beside its own
`generation_prompt` and `reasoning`, so a template variable of either name
could never reach the template, and the search for the variable that moves
declarations into the system turn had to leave both names out. The member,
the bundled renderer, the test stand-in, the reasoning switch's render and the
probes' render now take the variables as one mapping, `variables`. Every name
a template reads can be passed except those the renderer sets for every
template, which `RENDERER_VARIABLES` in `cache_context.py` lists: `messages`,
`tools`, `add_generation_prompt`, the beginning and end tokens,
`raise_exception` and `strftime_now`. These are the names the bundled
renderer sets itself; `functions`, `function_call` and `tool_choice`, which it
also takes, pass through to the template when given, so they stay
searchable. No test template's reasoning switch or declaration variable
changed.

**One way to run tool calls.** `reply(calls="run")`, which ran calls against
the registry `begin` kept, is removed, and so is that kept registry: a request
whose calls are to run is handed the registry itself. `begin(tools=...)` and
`declare_tools` take only a list of `ToolSpec`, which `registry.specs()` gives,
declare it and keep nothing. `ToolRegistry`'s `declare_at_begin` and
`specs_declared_at_begin` are removed: a tool to be declared later is left out
of the list `begin` is given, and passed to `declare_tools` then. Each was a
second way to do what the remaining call does. The library no longer holds
the caller's functions from one call to the next, so what runs is always what
the request names, and the error for `"run"` where `begin` was given no
registry went with the case it reported.
At the user's request, `reply`'s parameter `calls` is renamed `tool_calls`,
and its error messages and the README's table name it so. At the user's next
answer it became `tool_call_policy`, so that it no longer shares its name with
`Reply.tool_calls`, the calls a reply made, and `CALL_POLICIES` became
`TOOL_CALL_POLICIES` with it.

## 2026-11-11: Settings, add-on files and logging

The user's decisions on the settings, the add-on files and the engine's log
output, each under the ID it was decided by.

**270.25: the message for a missing engine package.** It named neither the
oldest release supported nor how to build one, and its commands installed the
prebuilt wheel, which install.sh avoids. It now names 0.3.23, the floor
pyproject.toml and install.sh require, and the repository's `./install.sh`
with the build by hand beside it.

**261.4: Ministral 3 at 0.05.** Its model card says: "Use a temperature below
0.1 for daily-driver and production environments; higher temperatures may be
explored for creative use cases." The 0.15 the add-on file held is what the
card's code examples set, which contradicts that sentence. The file now holds
0.05, below the bound with room on both sides, and quotes the sentence as its
source. Verified on the CPU with Ministral-3-8B-Instruct-2512 Q4_K_M: the
wrapper ran at temperature 0.05 and answered "Paris".

**266.4: no stop texts in the configuration.** `Config.stop_strings` did what
`reply(stop=...)` does, for every reply, so there were two ways to say one
thing; it was also a list inside a frozen configuration, which a caller could
change while the wrapper ran. It is removed. `reply` copies the `stop` it is
given when it is called, so the texts a reply stops at are those it was asked
with, even though the generation starts at the first read.

**266.3: how far an eviction drops without a shift is fixed.**
`evict_to_fraction_without_shift` could be chosen well only by measuring the
decode it trades against the history it keeps, as "Measured on a cache that
cannot shift" did, so no caller could set it better than its default. It is
now `EVICT_TO_FRACTION_WITHOUT_SHIFT` in `rendered_cache.py`, at the 0.5 that
measurement kept. A higher value keeps more history and decodes the cache
again more often; that decode falls after a reply, never before the first
word. The constraint it must meet stays beside it in the code: in (0, 1].

**265.5: no reply cap by default.** `max_reply_tokens` defaulted to 1,024, and
the cap counts a thought's tokens too, so a reasoning model that thought
longer than that ended with an empty answer and `"token_limit"`. The default
is now `None`, no cap: a reply runs until the model ends it or the context is
full, and the context bound is the one no generation can cross anyway. A
caller who wants a bound sets one, in the configuration or per request.

**270.15, the configuration part: a missing model and a cap of nothing.** A
`model_path` that was unset or named no file raised a bare `ValueError` or
`FileNotFoundError`, although the README promises an `InvalidSettingError`
naming the setting for every setting refused at startup; it now raises that,
with `field == "model_path"`. A `max_reply_tokens` of 0 or less passed to
`reply` was taken as a cap of nothing and produced an empty reply; it is now
refused when `reply` is called, by the same check `Config` runs
(`ensure_reply_cap`), so both say the same.

**270.21: the second trigger counts exchanges.** At the user's decision,
`max_history_messages` became `max_history_exchanges`. An eviction drops
whole exchanges, so a limit counted in messages could never be met exactly: a
limit of 3 kept 2, and a limit of 1 kept nothing between turns, since the
newest reply left with its question. An exchange is now the unit the limit
counts: one user message and everything after it up to the next, as
`exchange_starts` and `_drop_oldest_exchange` already took it, with what a
history holds before its first user message counted as one more, since an
eviction drops that whole too (`messages.exchange_count`). A limit of 1 keeps
the newest exchange; below 1 is refused, and `None` is still no limit. `begin`
honours it as every other action does, keeping the newest exchanges of a
history and at least the newest; tests pin that for histories that open on a
user message, on another role, and that end on a question. The rest of the
rule is as before: at rest after each action the limit holds, `inject` evicts
before a user message that would open one exchange too many, and while a reply
runs its question may open one more until the reply settles; a turn that
called tools outgrows it until the turn closes. `n_evicted_messages` and the
counts `inject` and `declare_tools` return still count history items.

**266.5: the KV cache goes where the model is.** `kv_cache_on_gpu` defaulted
to `False`, so the README's quick start, with `gpu_layers=-1`, put the model
on the graphics card and its cache in host RAM, where every decode reads it
across the bus. It now defaults to `True`, which is llama.cpp's own
`offload_kqv` default: the engine places each layer's cache on the device
that layer runs on, so with `gpu_layers=0` the whole cache stays in host RAM
and with a partial offload each part follows its layer. `False` still forces
all of it into host RAM, to free VRAM. That placement is llama.cpp's
(`llama_kv_cache` takes each layer's buffer type from the layer's device when
offloading is on); it was not measured here, since the GPU was in use by
another process.

**266.6: flash attention is the engine's choice by default.** It was forced on,
although llama.cpp has an automatic mode (`LLAMA_FLASH_ATTN_TYPE_AUTO`) that
turns it on where the device and the model support it and off where they do
not, which a fixed `True` cannot do. `flash_attention` now defaults to `None`,
that automatic mode, except under a quantized `kv_cache_type`: a quantized V
cache works only with flash attention, so the automatic mode turning it off
would fail the load, and it is forced on there. `True` and `False` still
force it; `False` with a quantized cache is refused as before.

**266.7: the engine's log output through Python's `logging`.** At the user's
choice (B), `log_level` and `LOG_LEVELS` are removed. `log_level` set the
level of llama-cpp-python's own logger, whose callback prints straight to
stderr past every Python handler, and no level of it showed real errors while
hiding the load-time listings. Measured on this machine's build (0.3.36),
recording the engine's raw log calls while gemma-3-1b loaded on the CPU and a
corrupt file failed to load: the engine numbers its levels as ggml.h now does,
1 debug, 2 info, 3 warn, 4 error and 5 a continuation of the line before, and
nothing came at 0. The binding maps them by an older numbering, 1 to `INFO`, 2
to `WARNING`, 3 to `ERROR` and 4 to `DEBUG`, so the engine's errors ("error
loading model", "invalid magic characters") were filed as `DEBUG`, its
warnings (the full-size sliding-window cache, a control-looking token) as
`ERROR`, and its 194 lines of metadata listing as `WARNING`. That is why
"error" showed what looked like banners and "none" was the only way to hide
them; the inference server's note "llama.cpp's banners can only be silenced
entirely" read the same effect as the banners being filed at level 0.

`_backend.py` now installs its own callback through `llama_log_set` when it is
imported, held at module level since the engine keeps only a raw pointer, and
turns each line into a record of the logger `llama_chat.engine`: debug to
`DEBUG`, info and a line without a level to `INFO`, warn to `WARNING`, error to
`ERROR`. A continuation goes on with the open line at its level, a call of
several lines becomes a record per line, and a line left open ends at the next
call with a level of its own; blank lines are dropped. Measured on the same
load: 1,922 `DEBUG` records, 195 `INFO`, 2 `WARNING` and 1 `ERROR` (the CUDA
initialization, which fails with no device visible); unconfigured, Python
shows only the last three, and with the logger at `ERROR` only the one error.
It is process-wide by nature: the engine has one log sink, so importing the
package routes the engine's output for the whole process, a model loaded by
other code included. Rejected: keeping a level setting beside it, which would
be a second way to do what `logging` does, and patching the binding's map,
which would still print past Python's handlers.

**266.8: every setting checked where the configuration is made.** Several
settings were not checked at all: a negative penalty window or thread count, a
`gpu_layers` below -1, and the type of any value, so `temperature="hot"` was
taken and failed at the first reply, and `True` passed for a number.
`Config.__post_init__` now first checks every field's type against its own
annotation (`config.ensure_types`), so the annotation is the one statement of
what a field takes and no list of fields can fall out of step with it: an
`int` field takes only an `int`, a `float` field an `int` or a finite `float`,
never a `bool`, and a field that may be `None` says so in its annotation. The
range checks follow: `penalty_window_tokens` at least 0, `gpu_layers` at least
-1, each thread count `None` or at least 1. `Sampling` is checked by the same
code, through the configuration it is applied to, and the cap `reply` takes by
`ensure_reply_cap`, which checks its type too. The list-valued setting the
item named, `stop_strings`, went with 266.4, so no setting holds a value that
can be changed in place.

**268.7: every sampling value per request.** `Sampling` took five values;
top-k, min-p, the repetition penalty and the penalty window could be set only
for the whole wrapper, so a caller wanting one request greedy-ish and the next
creative had to rebuild the wrapper, and with it the cache. It now takes every
sampling field of `Config`, each `None` to keep the wrapper's, and is checked
by the same code. No field was left out: each of them shapes only the chain a
generation draws from, which is built for each generation anyway.

**270.20: a construction that fails frees the model it loaded.** The wrapper
closed the context it had loaded where checking the template failed, but built
the conversation's cache state after that guard, so a failure there left the
model and its context allocated, in VRAM where it was offloaded, with nothing
holding them that could free them. Every step after the load is now inside the
guard. A context of your own is still left open where construction fails, as
documented, since the caller holds it and may use it again.

**265.6: the add-on file and the model's sampling are the wrapper's.** The
engine's context read the add-on file and merged the sampling while it
loaded, and built each generation's sampler from its own configuration, so a
cache context of your own got neither: no add-on file, so a model whose calls
only its file describes could not call tools, and no recommended sampling.
Both now happen in `ChatWrapper.__init__`, for every context. The protocol
lost `addon_call` and gained `recommended_sampling`, the values the model
states for itself by `Sampling`'s names; the engine's context reads them from
the GGUF's `general.sampling.*` keys, which now live in `context.py`, the one
file that knows a GGUF. The wrapper finds the add-on file from `model_path`
and the template text, merges the sampling into its configuration, and hands
the template the file's call format (`ResolvedTemplate(context, addon_call)`).
`sampled_tokens` is handed every value a generation draws with
(`config.generation_sampling`), a request's where it sets one, so a context
never needs a configuration of its own for sampling and `KVContext.config` is
gone. Rejected: keeping the add-on file in the context and copying it into a
context of your own, which would make every implementation repeat the lookup.

**265.3: the model's sampling, merged value by value.** Where the GGUF stated
any sampling value, its add-on file was ignored whole, and every value the
GGUF left out was switched off, so a GGUF stating only a temperature lost the
file's top-k and top-p. The user decided the new rule: value by value, the
add-on file first, its values winning, then the GGUF's for a value the file
leaves out, then off. The file wins because it is the one source a person
checked against the makers' published recommendation and cites, while a
GGUF's values are whatever its converter copied, often a generation config
written for another runtime. A model for which neither states any value still
runs at llama.cpp's defaults, as before; whether such a model should run with
every value off instead was not part of the decision and is left open. (277.2)
The README says that a shipped file, matched by its template, also wins over
the values a fine-tune that keeps the template states in its own GGUF; a file
next to that GGUF, or the caller's own values, are how to override it. Of the
GGUFs on the machine this was built on, none both states sampling values and
matches a shipped file, so the change alters no model's sampling there.

**265.2: presence and frequency penalties in add-on files.** An add-on file
could carry only the five sampling values a GGUF can state, although makers
publish penalties too: Qwen 3.5's model card recommends `presence_penalty=1.5`
for every mode but precise coding. `presence_penalty` and `frequency_penalty`
are now "the model's" fields like the other five: unset by default, taken from
the add-on file where it gives them, else off (0). No GGUF key states either,
so they come only from an add-on file or the caller. The Qwen 3.5 file now
holds 1.5, cited from the card's Best Practices for thinking mode on general
tasks, the mode whose other values it already held. Verified on the CPU with
Qwen3.5-9B Q4_K_M: the wrapper ran with presence penalty 1.5 and the file's
other values and answered.

**265.4: a file next to the GGUF is used only where it names the template.**
Any file with the model's name and `.json` was taken as its add-on file, even
one written for another template, and an unrelated file of that name (another
tool's settings, say) stopped the model from loading. It is now used only
where its `templates` names this model's template, as a shipped file must.
Otherwise it is not used, the shipped file for the template applies, and a
warning on `llama_chat.addon_files` names the file and the reason: not JSON,
not an object, no list of templates, other templates, or a model without a
template to name. A warning rather than an error, since such a file may be
another program's; a warning rather than information, since a person who
wrote one with a wrong fingerprint needs to see it, and Python shows warnings
without any logging configured. A file that does name the template is checked
whole, and refused where malformed, as before. Run on the CPU with gemma-3-1b and a
file of another program's settings beside it: the warning named the file and
the reason, and the model loaded with its shipped file's values and answered.

**265.7: every value checked at load, naming its source.** An add-on file's
values were checked only for range, so `"temperature": "hot"` loaded and
failed at the first reply, and `"templates": "7f0e..."` as text was read
letter by letter, matching any template whose fingerprint held those letters.
A value the GGUF recommended that the settings refuse stopped the load with an
`InvalidSettingError` naming a setting the caller never set. Now an add-on
file's sampling values are checked for type by the same code as `Config`'s
(266.8), its `templates` must be a list of 64-digit lowercase hex
fingerprints and its `source` text, each refusal naming the file. A value
the model recommends itself is checked where it would be taken, that is
where neither the caller nor the add-on file replaces it, and refused with a
`ValueError` that names the model's own recommendation as its source and says
how to override it; a value another source replaces cannot stop the load. A
name it recommends that is no model sampling field is refused too, for a
cache context of your own. A test now pins that no two shipped files name one
template, since the first in file order would win and the other be dead.

## 2026-11-11: Templates, reasoning and the load check

The user decided these items; each is a commit of its own.

**264.2: The names a template reads, parsed as it renders.** The search for
the variable that moves declarations into the system turn read the names a
template reads with a plain `jinja2.Environment`. The renderer's environment
knows tags and extensions a plain one does not, such as `break`, `continue`
and `{% generation %}`, so the plain parse failed on 7 of the 52 test
templates and returned no names at all. The names are now parsed by the
renderer's own environment, so every template the renderer can render has its
names read. No test template's declaration variable changed.

**264.3: The reasoning switch among the names a template reads.** This
revises "How reasoning is read". The switch was looked for only in
`enable_thinking` and `reasoning_effort`; templates also switch reasoning
with `thinking`, `thinking_mode`, `thinking_option`, `reasoning_strength` or
`reasoning`, and the test templates that do worked only because they read one
of the two as well. The two are still tried first, by their rules as before.
After them comes each other name the template reads that holds "think" or
"reason": first as a flag, `True` for on and `False` for off, then by the
effort's rule, `"none"` for off where it changes what follows the user turn.
A discovered name counts only where it changes what follows the user turn.
Both limits keep out variables that change the prompt for another purpose:
over the 52 test templates, every name read was tried as a flag and as an
effort, and `low_effort`, `medium_effort` and `tool_choice` (`"none"`) each
change what follows the user turn without being a reasoning switch. No test
template's switch changed. A caller's variable of the switch's name now
replaces the switch's value instead of making the render fail, since a
discovered switch may share its name with a variable a caller passes.

A switch whose values are words of the template's own, such as
`thinking_option` (`"on"` and `"off"`) or `thinking_mode` (`"thinking"` and
`"chat"`), is found by neither rule. The words could be read from the
comparisons in the template's source, but nothing in a render tells which of
them is off where the template's default does not reason; this waits for the
user's decision.

**261.5: Whether reasoning can be turned off.** Reasoning off is the
template's own off, and some templates have none: of the 37 test templates
that mark a thought, 12 write the prompt the same with reasoning on and off,
among them those of DeepSeek R1, gpt-oss, Muse Glimmer and Ministral 3
Reasoning. `begin(reasoning=False)` was accepted on them without a word,
though the model may reason in every reply. `can_turn_off_reasoning` now says
so: it is true where the switch writes the generation prompt with reasoning
off otherwise than with it on, and where the template marks no thought at
all, since then nothing is known to turn off. Nothing is refused: such a
conversation is written as the template writes it, and the thought is kept out
of the answer as before. Read as: the property is the switch's effect on the
prompt, the one place the renders show whether an off exists.

**261.3: The answer's opening after an empty thought.** Measured on North
Mini Code: with reasoning off, every answer began with a stray
`<|START_TEXT|>`, in 8 of 8 real replies, and the tag was kept with each past
reply, so the model read it twice from then on. Its generation prompt without
reasoning ends in an empty thought, `<|START_THINKING|><|END_THINKING|>`,
where a past reply's tag holds `<|START_TEXT|>` instead; the answer's opening
was recognised only where the prompt is the start of the tag, as on a
template that routes replies through channels. Where the prompt, past what it
shares with the tag, is a thought opened and closed with nothing in it, and
the tag's rest completes that thought's closing, the tag's rest is now the
answer's opening, which the model writes and the answer leaves out. The
condition asks for the closing in the prompt, so an opened thought, as the
prompt with reasoning on writes it, is no such case. Of the 52 test
templates, only cohere-2's opening without reasoning changed.

**264.1: A past thought in the form the template reads.** This broke
invariant 1, measured on Ministral 3 Reasoning: a past reply's thought was
handed to every template as `reasoning_content`, while the load check found
the thought through any of three forms, that field, a `thinking` field or a
content block of its own. That template reads a thought only from a content
block, so the check said it keeps a past thought, the reply was recorded with
its thought, and the render left the thought out: it was lost from the cache
after the next question. The load check now records the form a past reply
keeps its thought in, trying the form the thought's tags were read from
first, and the template's render is the one place that hands a recorded
thought over in that form. The conversation itself keeps the thought under
`reasoning_content`, so its content stays one text everywhere else. A
template that prints a content block as text rather than reading it is not
taken to keep the thought: the check now looks for each block printed, not
only the whole list, which kept hunyuan-4, lfm-2.5 and lfm-3 from being read
as keeping thoughts in blocks they only print. Of the 52 test templates, only
mistral-3-reasoning's form changed. The tests' own check of the cache renders
each reply's thought in each of the three forms, written out in the tests,
and accepts the cache where it matches one of them.

**264.5: A thought's tags read from the model's own tags.** Three places
that read a thought took a tag to be text in angle brackets: the empty
thought a prompt without reasoning ends in was matched by a pattern of two
`<...>` texts, the opening of a thought a template removes from a past reply
was the closing's text with `</` replaced by `<`, and the last tag before a
thought was the text from the run's last `<`. A model marking its thought as
`◁think▷` slipped through all three. Each now reads the model's own tags: the
empty thought is two of them at the prompt's end, the removed thought's
opening is the tag that reads as its closing without the first slash, so
`[/THINK]` and `◁/think▷` count as well, and the last tag is the last of them
in the text. Over the 52 test templates nothing read changed. The stand-in
tokenizer of the tests listed only the tags a template spells in its source;
it now also lists those the template builds from parts and writes, as
hunyuan-3 and hunyuan-4 build their thought tags, since a real vocabulary
holds them too.

A fourth place, the separator between two calls of one turn, finds each
call's opening at a `<` inside the opening a template writes. It was left as
it is: on deepseek-v4.1 the call opens with `<｜DSML｜ invoke`, which no tag of
the stand-in vocabulary starts, and reading the model's tags there instead
lost the separator and made the call's closing text run to the end of the
turn, `</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>` in place of
`</｜DSML｜ invoke>`. Whether that model's vocabulary holds `<｜DSML｜` as a tag
decides which reading is right; this waits for the user's decision.

**270.15: A template that is not template code is refused at load.** A chat
template the renderer cannot compile, such as one with a block never closed
or a filter the renderer lacks, failed with the renderer's own parser error,
which is no `ValueError` and named no refusal. The renderer's construction now
turns that error into a `ValueError` that says the template is not valid
template code and names the line and the cause, at load, as every other
template that cannot be used is refused.

**270.18: A kept turn that cannot be placed refuses the action.** A turn that
called tools is held as the tokens it was generated as, in place of the
template's version of it. Where a render held such a turn but could not place
it, the turn was left out of the placement and the template's own version of
it stood in the cache without notice, which breaks invariant 1. The target of
a render now raises a `ValueError` for a kept turn among its messages that it
cannot place, so the action that rendered it fails and leaves the
conversation as it was. A kept turn whose messages the render does not hold,
such as one outside a window, is no such case. Only a change to how the
template writes the conversation can make a placed turn unplaceable, such as a
declaration, so that is the action that is refused. The test stands the
unplaceable render in by replacing the placement, since none of the 52 test
templates writes a kept turn where it cannot be placed. Seen while testing:
answering a call that could not be read catches every `ValueError` as "no
room, or no result block" and then records the reply without its call; a
placement refusal there would take that path too.

**270.19: The checks of what renders allow run wherever their inputs change.**
Two checks read the template's renders with the conversation's system turn,
declarations and clock: that an older exchange stays written as it was once
another follows, and whether windows of the conversation stand for its whole
render. `begin` ran both; `declare_tools` ran only the second, though the
declarations it adds are among the first's inputs; and an action that found
the clock moved, so that the template writes another date, only pinned the
new opening. All three now run both checks through one method, so a
template that rewrites older exchanges only with tools declared, or only on a
later date, is refused there rather than kept by a check made for other
inputs. The test records that both checks run at each of the three, since none
of the 52 test templates rewrites an older exchange under either change.

**270.22: Saving and restoring a state is tried at load.** A cache that
cannot be cut at its end, such as one with a running state, is brought back
to an earlier point from a state saved there, and decodes the whole
conversation again where none can be saved. Loading tried the cut and the
shift but not the save and the restore, so a build that cannot do either was
found only at the first reply that needed it, and then decoded everything
again without a word. Where the cache cannot be cut, loading now decodes one
token, saves the state, decodes two more, and goes back to the saved state
through the same restore a conversation uses, then clears the cache; a build
that cannot is refused there with a `RuntimeError`, as one missing the shift
is. Measured on Qwen 3.5 9B on the CPU, whose cache cannot be cut: the probe
passes at load and the model answers as before.

**270.23: A call's texts take the tolerance a thought's tags take.** A
thought's tags are matched without the whitespace around them, which a model
does not always reproduce; the texts a call opens and closes with, where the
call has no tag of its own, had to match exactly as the template writes them,
so a model that left out a line break before its closing ran the call to the
end of the reply. Both texts are now matched without their surrounding
whitespace. The call's text is read without whitespace at its ends, and the
reading puts back the opening's own text after its last tag as before, so a
call written exactly as the template writes it reads as it did. Whitespace
right before an opening that begins with whitespace stays the call's, as it
was while the whole opening had to match. The closing a call is given where
the reply ends inside it is still the template's whole closing.

**264.4: A template of its own for tools.** Revised by "The user's answers to
the round's open questions" (301.9B): a `declare_tools` that would switch
templates is refused, and `begin` given any `tools` selects the template for
tools. At the user's choice (B). A GGUF
can hold further named chat templates beside `tokenizer.chat_template`, among
them `tokenizer.chat_template.tool_use` for a conversation that declares
tools; only the default one was read. The engine context now reads that one
too, and the two members of a cache context that hand out the template, its
render and its text, take a `template_name`, `None` for the default one, so
that a context of one's own can offer the same. Where the model ships a
template for tools, it is recovered and checked at load exactly as the
default one is, and a model whose template for tools fails that check is
refused. A conversation is written with that template from the moment it
declares tools, and with the default one while it declares none; the choice
is made in one place, from the conversation's declarations, and everything
that renders or generates reads the template from there. `can_call_tools`
answers for the template a conversation with tools is written with. A
`declare_tools` that switches templates is the render changing, as any
declaration in the system turn changes it: the cache is compared with the new
render and decoded again from the first token that differs, which for two
templates of different formats is the whole conversation, once. A turn kept as
generated under the default template that the template for tools cannot place
refuses the declaration, as 270.18 refuses any such action. None of the
models in the local cache ships a template for tools, so the tests run an
invented pair: a default template that knows no tools, and the invented tool
template in another format.

Open, for the user: whether a `declare_tools` that switches templates should
decode the conversation again once in the new form, as built, or be refused
in favour of a `begin` with the tools; which template `can_reason`,
`can_turn_off_reasoning` and `can_take_system_prompt` answer for, now the
current conversation's (before any `begin`, the default one), while `begin`
checks the template its own `tools` select; and which template a history
holding tool turns but declaring no tools is written with, now the default
one, which refuses it where only the template for tools can call tools.

## 2026-11-11: Two state names in plain words

At the user's decision, `owed_calls` (the calls still waiting for a result) is
renamed `pending_calls`, and `reply_open` (a tool call left the newest turn
open) is renamed `turn_open`. Both needed a definition before a line using them
could be read. The entries above keep the old names, as they were written.

## 2026-11-11: An add-on file may name what turns reasoning on

The reasoning switch is read off the template's text: where the effort
`"none"` changes what follows the user turn, `"none"` is off and the template's
default is on (the user's 248.4A). A template whose default does not reason,
yet writes something of its own for `"none"`, is then read with a default that
reasons, and `reasoning=True` gives an answer without a thought, with no error.
The text cannot tell this case apart; only a run of the model can. None of the
test templates has this shape.

At the user's decision (289.5B), an add-on file may hold `reasoning_on`: the
template variables, with their values, that turn reasoning on. They are laid
over the switch the template shows wherever a render asks for reasoning on, the
load check's probes included, so the thought tags are read from the prompt the
model will really get. A `reasoning_on` that writes the prompt as reasoning off
does is refused at load, naming the add-on file's key. It applies to both of a
model's templates, as `call` does. Rejected: waiting until a model shows the
case (289.5A), and a short real reply at every load to see whether the default
thinks (289.5C), which costs every load a few seconds. The test for it uses an
invented template, since no local model needs it; no shipped add-on file sets
it.

## 2026-11-11: The conversation and its surface

The user's decisions on the conversation, its history, its cache and the
public interface, each under the ID it was decided by.

**268.12: `ModelReader` is no longer exported.** It is the part of
`CacheContext` that reading a template needs, and the wrapper never takes a
context that answers it alone, so exporting it suggested a second way to
build a context of one's own. It stays in `cache_context.py`, where the
template's readers name it as the type they take, and the README names
`CacheContext` as the one protocol.

**262.2: a context of one's own must report its template text.** The template
was read from a context's renders alone, so a context whose
`chat_template_text()` answered `None` was accepted as long as it rendered.
The text is what an add-on file names a template by, and what the
declaration variables are read from, so such a context ran without either.
Reading the template now refuses it first, with the same `ValueError` as a
model that ships no chat template.

**270.5: a context of one's own must be able to go back.** A cache is brought
back to an earlier point by a shift, a cut at its end, or a saved state; one
that can do none of the three decodes the whole conversation again after
every reply that changes what it holds, which invariant 2 refuses. The engine's
own context already tries saving and restoring a state at load where its
cache cannot be cut (270.22 under "Templates, reasoning and the load check").
A context of one's own is now checked where the wrapper adopts it: where it
can neither shift nor cut, `save_state` is called once on it, and a `None`
or a `RuntimeError` refuses it with a `ValueError`. The engine's own context
is left to its load check, so the save is not tried twice. A context that
stops saving later is still decoded again whole, as before.

**268.1: the forms and roles of a history message.** A history item that was
not a tool turn was unpacked as `role, text = item`, so the usual dictionary
`{"role": "user", "content": "Hello"}` unpacked into its two keys and was
stored as the message `("role", "content")` without an error, and a role the
template happened to accept, such as `"human"`, went into the cache. `begin`
now takes a `(role, text)` pair, as a tuple or a list, or a dictionary
holding `role` and `content` and nothing else, each with the role `"user"` or
`"assistant"`. Any other form is a `TypeError`, and another key or role a
`ValueError` naming it; a system message points to `begin`'s first argument,
and a tool message to `ToolTurn`. `ensure_role` in `messages.py` is the one
check of a role, so that `inject` can share it.

**268.6: no system message is injected.** `inject(text, role="system")` wrote
the message into the cache, but `conversation()` reads the system prompt from
the first message only and left the later one out, and the count of evicted
messages counted the message itself as one dropped. A system prompt kept for
the whole conversation is `begin`'s first argument, so `inject` now refuses
the role with a `ValueError` that says so, before anything changes. It checks
the role with the same `ensure_role` as a history, so a tool message points
to `inject_tool_result` (one part of 270.15, which failed with a `KeyError`
before) and an unknown role is named. The template's refusal text for a later
system message, which no action can reach now, is removed.

**270.15, the remaining parts: refused where it enters, naming its cause.**
Three inputs failed far from where they entered, with an error that named
something else. A message that is not text, given to `begin` as the system
prompt or in the history, to `inject` or to `reply`, reached the template and
came back as "the template refuses to write this conversation"; it is now a
`TypeError` naming the message, raised before anything changes, by
`ensure_text` in `messages.py`. A context of one's own whose methods take
other arguments than the protocol gives them passed the check at adoption,
which looked at names only, and failed at the first call; adoption now binds
each method to the arguments of its protocol method, once positionally with
the keyword-only ones by name and once all by name, and a method that takes
neither is a `TypeError` naming it with the arguments it is given. A method
whose arguments cannot be read, such as one written in C, is taken as it is.
A negative `max_calls` was taken as a ceiling already spent; `reply` now
refuses with a `ValueError` a `max_calls` that is not a whole number of at
least 0, `True` included. `inject(role="tool")` is refused with 268.6.

**268.4: the reply hands back its thought.** A reasoning model's thought was
kept out of the stream and out of `Reply.text`, and only its approximate
length came back, as `n_reasoning_tokens`. `Reply.thought` now holds its
text, as the stream's filter collected it: without its tags and the
whitespace around them, whether or not the template keeps a past reply's
thought. A request that took several generations, around tool calls, joins
each generation's thought in order, with a blank line between two.
The stream still yields the answer alone, so a caller who shows only the
answer changes nothing.

**268.11: `conversation()` holds everything `begin` takes.** It returned the
system prompt and the history, so a caller beginning again passed the tools
and the reasoning switch by hand, and every thought the template keeps was
lost: a past reply's thought was held in the conversation's messages but left
out of the history. `conversation()` now returns a dictionary keyed by
`begin`'s parameter names, `system_prompt`, `messages`, `tools` and
`reasoning`, so `begin(**chat.conversation())` begins again in one step;
`tools` holds every declaration, those `declare_tools` added later included,
which `begin` then writes from the start, where the template writes
declarations anyway. A dictionary was chosen over a record type of its own
because it unpacks into `begin` as it is and adds no name to the interface. A
thought the template keeps comes back in the form that the field already
uses: a reply as `{"role": "assistant", "content": ..., "reasoning_content":
...}`, which `begin` now takes too (a question with that key is refused), and
a tool turn's thought as a `Thought` step, a new public type in `tools.py`,
before the text and the calls of the message it belongs to. A reply without a
thought stays a `(role, text)` pair. Sixteen of the test templates write a
past tool turn's thought, so a tool turn needs it as much as a plain reply.
The template still decides what is written: the invented template that keeps
thoughts leaves out the thought of a reply that closes a tool turn, which the
live turn held as generated, so a conversation begun again holds that turn as
the template writes it, as any past tool turn.

**270.14: the token count and the open turn are read from the finished copy.**
`total_tokens` and `turn_open` read the live state, which an action in
progress may still put back: a `begin` that failed after emptying the cache
showed 0 tokens and an open turn to another thread for a moment, although the
conversation it left standing held neither. Both now read the copy published
after each finished change, as `conversation()` and `pending_calls` do, so
the copy carries the token count. A count read while a reply streams is the
one from before that reply, until the reply is recorded. The library's own
readers under the lock still read the live state.

**270.1: a cut to a reply the template does not write is refused.** Three
of the test templates (mistral-3, devstral-2, mistral-large-4) refuse an
assistant message with empty text. `truncate_last_reply("")` on them recorded
the empty reply, found that no render of the conversation could be made, and
left the cache holding the old text; every later question was then refused
until a new `begin`. The shortened conversation is now checked before
anything is decoded: where the template writes it neither as it is nor with a
question after it, the cut is refused with a `ValueError` that asks to keep
more of the text, and the conversation and the cache stay as they were. The
same check, `_writes`, serves 270.3.

**270.3: a reply that produced nothing the template cannot write takes its
question back.** On the same three templates, a reply that produced nothing
(cancelled before its first word, ended by the model at once, or cancelled
while a tool ran) was recorded as an empty assistant message, which no render
of the conversation could hold, so the reply raised and every later question
was refused until `begin`. A request whose generation fails before it says
anything was already taken back whole; such a reply now is too: where the
template writes the conversation with the empty reply neither as it is nor
with a question after it, the newest question and everything after it are
taken out again, a tool turn's calls and results and the turn kept as
generated included, and the cache is brought to the render of what is left.
A tool that ran has run all the same; its result leaves the conversation with
the question. Where no user message precedes the reply, as after a history
that opens on a tool turn, everything after the system prompt is taken back,
there being no question to stop at. Where the template does write an empty
reply, nothing changes: such a reply is kept, or, on a template that writes
two questions in a row, taken out alone so that its question waits for the
next. Both removals are now one method, `_remove_from`.

**270.2: a question not prepared ahead passes its dropped stretch on.** A
question the stand-in cannot stand for, such as one ending in a space, is
added to the conversation and rendered whole. Once `max_history_exchanges`
was reached, adding it dropped the oldest exchange, but the cache was then
brought to the new render without being told that a stretch had gone, so it
was cut where the dropped exchange began and everything after decoded again
(87 of 124 tokens instead of 34, as measured for the decision). Every other
drop already passed this on, so that the stretch is found and removed and the
rest shifted; this one now does too. On the plain invented template the test
asks the fourth question of a limit of two: the cache is shifted once, cut
never, and 22 tokens are decoded where 44 were.

**270.4: what no question follows is refused once it no longer fits.** An
eviction drops the oldest exchange up to the next user message, so the
messages after the newest user message, or every message of a history that
holds none, can never be dropped while no question follows them. A `begin`
with a history of replies alone, or replies injected one after another,
therefore grew the cache past the context size (630 tokens in 500, and 1,320
in 500, as measured for the decision). As settled with the user, replies
before a later question are still evicted first; only where the conversation
still renders longer than the eviction threshold after every exchange that
can go was dropped is the action refused, with a `ContextOverflowError` that
says no user message follows what is left, and the conversation and the
cache stay as they were. The check, `_ensure_within`, runs where `begin` has
dropped what it can and where `inject` has dropped for a message it does not
splice; a question `inject` splices is a user message, which opens an
exchange of its own. The threshold is the bound, as for a message checked
alone, so that a reply has room under the context size.

**270.17: every refusal for want of room says what to change.** Three of the
`ContextOverflowError` messages gave the cause, how many tokens were needed
and how many fit, but not what to do, while the others name a remedy. A
question too long beside room for its reply now asks to shorten it or the
system prompt, or to raise `context_size` or lower `min_reply_tokens`; an
injected message too long beside the system prompt asks to shorten it or the
system prompt, or to raise `context_size` or `evict_above_fraction`; and a
conversation that an eviction leaves without room for the reply, which a
threshold close to the context size allows, asks to lower
`evict_above_fraction` or `min_reply_tokens`, or to raise `context_size`.

**270.24: texts that no longer held.** Three sentences of this history
described the code as it no longer is, and each now carries a dated note
where it stands: not every probe renders with reasoning off, only those for
the turn tags; `begin` and a generation clear the records made ahead, which
"no action clears them" denied; and a tag held back while a call may follow
is shown as reply text where none follows. The README left four things out,
each now stated and pinned by a test. `"withhold"` keeps a call's opening tag
from being drawn, so a call that opens with plain text, as one model's
template writes it, cannot be withheld and stays in the reply's text. The
answer to an unreadable call says that it failed and could not be read, and
gives no instruction to try again or not. A result handed to
`inject_tool_result` is written as given, a dict carrying `error` included,
without the sentence a registry's failure gets. And a context of one's own
must keep its token ids from -65,536 to 1,048,575: the search for a dropped
stretch spells each id as one character, offset by 65,536, and Python's
characters end at 1,114,111.

## 2026-11-11: Tools and tool calls

The user's decisions on declaring tools and on the calls a model makes, each
under the ID it was decided by.

**268.8: a type for the tool call policies.** The policies a request names
were a plain tuple of strings, so a type checker could not tell a misspelled
policy from a real one. They are now the type `ToolCallPolicy`, as the stop
reasons are `StopReason`, and `TOOL_CALL_POLICIES` is read from that type, so
the list and the type cannot disagree. `reply` takes the type.

**270.11: one name, one tool.** A declaration that held two tools of one name
declared both, and the model could reach only one of them, without a way to
tell which. `begin` and `declare_tools` now refuse such a declaration with a
`ValueError` naming the tool, before anything is written. A tool that the
conversation already declares is still skipped by `declare_tools`, as before.

**267.3: the tool decorator in full.** `@registry.tool` now works bare, with
the function's docstring as the description; a string given to it is still
the description, as `@registry.tool("...")`, and `name=`, `description=` and
`parameters=` override by keyword. `register` is folded into it:
`registry.tool(fn, description="...")` registers a function written
elsewhere, so there is one way to register. The annotations describe lists,
dicts with text keys, optional values, `Literal` choices, enums, defaults and
`Annotated` descriptions, and the same reading of each annotation builds both
the schema and the check of the model's argument, so the two cannot disagree.
Arguments that do not fit are answered as `wrong_arguments`, naming the
parameter and what it must be, before the tool runs; the model is told and
may try again. A number where text is asked is taken as its text, since a
call format that marks no strings reads `42` as a number, and refusing it
would tell a model to write what its format cannot. A registry lists its
tools, so `begin(tools=registry)` declares them. Four defects of the old code
are fixed: a bare `@registry.tool` registered nothing and replaced the
function with the inner decorator; a parameter taken only by position was
declared, but every call passes arguments by name, so the tool could never
run; an asynchronous function registered, and every call answered the model
with "unwritable result" while the coroutine never ran; and a parameter that
cannot be described, on a callable without a `__name__` such as a partial,
raised `AttributeError` instead of the documented `TypeError`. A fifth was
found on the way: an explicit `parameters=` schema was kept by reference, so
changing the caller's dict later changed the declaration; it is copied now,
and `specs()` hands out copies. A tool with no description, neither a
docstring nor `description=`, is refused, since the model would read nothing
about it.

**270.9: arguments that are no object, and arguments nested too deeply.** A
call whose arguments were a JSON list or `null` was read as a call with no
arguments, and its tool ran with none of what the model gave. Such a call is
now unreadable, and the model is told so. Very deep nesting was read on one
path and not on another: the JSON reader followed thousands of levels, while
copying the arguments into the conversation and rendering them did not, so a
call 600 levels deep raised `RecursionError` out of the reply, measured with
the test stand-in. The reading now refuses arguments that nest deeper than
`MAX_ARGUMENT_DEPTH`, 100 levels, on every path, and a reader that runs out
of recursion refuses the call as unreadable too, so the wrapper treats only a
`ValueError` as an unreadable call.

**268.5: an id for every call.** The ids a conversation's calls are written
with were numbered from the count of calls the conversation held. After a
`begin` that dropped older exchanges the count was smaller than the highest
number kept, so the next call took an id that a kept call already had, and a
template that pairs a result with its call by id paired it with the wrong
one. `ToolCall` now carries its `id`, and every place a call is handed out
carries it: the stream, `Reply.tool_calls`, `pending_calls` and
`conversation()`. A call of a history keeps its id; one without takes a new
one, and the next call of the conversation is numbered past the highest id
the history holds, counted before the oldest exchanges are dropped, so no
call takes the id of one that did not fit. A history that gives one id to two
calls is refused. The service kept its own list of ids for this
reason and can read the id from what the library returns instead.
`inject_tool_result` still takes a call built by hand without an id, matched
by its name and arguments, as before; a call that carries an id must carry
the owed call's.

**270.6: a turn that ends inside a call closed by a tag.** Where a call
closes with text, a turn that ended inside the call had the call closed as the
template closes one, and the call was read. Where it closes with a tag of its
own, the same turn ended as `"end_of_turn"`, and the call was dropped without
a word, its tokens left in the cache. The closing tag is now decoded after the
call's last token, and the call is read and answered as any other, so both
kinds of closing end the same.

**270.12: a held tag on a reply that was cut.** Where a call opens with
plain text, a tag of the model's own is held back while a call may follow it,
since the tag is then the call's. A reply that the cap or a cancel cut right
after such a tag released it into the answer. It is now dropped there, with
whatever followed it, as the thought filter drops the start of a thought's
tag on a cut reply. A reply that ends of its own accord after the tag keeps
it, as before.

**270.13: an action on the thread whose reply is being read.** Revised by
"Review fixes of the streaming, the thought filter and the tool loop"
(316.28): the thread that takes each step of a stream is noted as the holder,
so a stream resumed on another thread is inside the check. A reply holds
the wrapper's lock while it is read, so an action on the same thread, such as
a tool that the reply runs calling back into the wrapper, or an `inject` made
between two deltas of the stream, waited for the lock it already held and
hung for ever. The wrapper now records which thread holds the lock, and every
action, `wait_idle` and `close` included, refuses on that thread with a
`RuntimeError` naming the open reply. A tool that does so is answered as a
tool that raised, so the model is told, and the reply ends. Another thread
still waits for the reply, as before. A stream resumed on another thread than
the one that began it is outside this check, as the stream is to be driven
from one thread at a time.

**270.10: one separator around a call.** A reply that runs its calls parts
the text before a call from the text after it with a blank line where the two
would join without whitespace. Under `"yield"` the text after the call comes
from the next `reply`, which knew nothing of the text before, so the two
joined ("Ich sehe nach.Es ist neun."). A reply that goes on with an open turn
now reads what the model wrote in that turn so far and applies the same rule,
so the stream and `Reply.text` start with the blank line where it is needed.
The history keeps each text as the model wrote it, as before.

**268.2: a reply that goes on keeps its turn's policy.** `reply` took
`"refuse"` whenever no policy was given, so a caller answering calls under
`"yield"` who went on with `reply()` had the model's next call refused, and
had to repeat the policy at every step. The policy now defaults to `None`: a
reply that goes on with an open turn keeps the policy the reply that started
the turn was given, and any other reply takes `"refuse"`, as before. A policy
given to a reply that goes on holds for that reply alone. A turn that a
history leaves open was started by no reply, so going on with it takes
`"refuse"` unless a policy is given. The wrapper holds that policy, a
registry included, until the next turn starts or the conversation begins
again.

**270.8: no room left to answer a call inside a running reply.** Where a
reply answers its calls itself, under `"refuse"` or a registry, and the
context had no room left for a result's wrapping, the reply raised
`ContextOverflowError` and left the turn open, a call owed and no reply able
to answer it, so only a new `begin` recovered. The turn now closes as the reply
cap closes one: the reply is kept as the model wrote it, without the calls
that could not be answered, results already written for them are taken out
again, the cache is brought to the template's render, and the reply ends with
`"token_limit"`, the context having run out. This is the path a call that
could not be read already took where its error had no room.
`inject_tool_result` still raises, since the caller holds the call there and
decides what follows.

**270.7: every call reported.** The README promised that every call is
reported on `Reply.tool_calls`, but a call that could not be read, and one the
reply cap or a cancel cut short, were missing from it. `ToolCall` now carries
a `status`, the type `ToolCallStatus`: `"complete"` for a call that was read,
`"unreadable"` and `"unfinished"` for the other two, which are reported with
no name, arguments or id, since none could be read, and are never handed over
in the stream, since nothing answers them. The history keeps an unreadable
call as before, under the name a result is written with where none was read.
The related find is fixed with it: answering an unreadable call took every
`ValueError` for "no room, or no result block" and recorded the reply without
its call, so a refusal to place a kept turn took that quiet path too. The
template's result block is now checked before the answer is written, and only
`ContextOverflowError` counts as no room; any other refusal is raised.

**268.9: a call the library runs is handed over before it runs.** Where a
registry runs the calls, the stream held only text, so an app could not show
what was being looked up while a tool ran. Each call that runs is now handed
over in the stream as its `ToolCall` just before it runs, so the stream holds
calls besides text in that mode too, as it does under `"yield"`. A call that
does not run, refused for a spent `max_calls`, is not handed over, since
nothing is being looked up for it; it is reported on `Reply.tool_calls` as
before. A stream closed at a call handed over leaves that call unrun and owed
a result, as under `"yield"`, and the reply that goes on after its result
keeps the registry (268.2). Verified on the CPU with MiniCPM5-2B Q4_K_M and a
tool registered by the bare decorator: the call was handed over before the
tool ran, and the stream, `Reply.tool_calls` and `conversation()` carried the
same id; the exchange took 1.9 s.

## 2026-11-11: Review fixes

**R2: a tool turn closed by an empty reply begins again closed.** Where the
model ended at once after a result, or the reply was kept without the calls it
could not have answered, the turn closed on a reply with no text.
`conversation()` left that empty text out, so the turn it handed back ended on
its result, and `begin(**chat.conversation())` began it open: the round trip
compared equal while `turn_open` differed. The reply that closes a tool turn is
now always the turn's last step, an empty `str` where it wrote nothing, and
`begin` already read a trailing text as the closing reply.

**R3: no id is given twice within one wrapper.** `begin` numbered new calls
from the ids of the history it was given. After every tool turn had been
evicted, `begin(**chat.conversation())` held no call, so numbering began at
the first id again and a new call took the id of one the caller had already
seen. The count of ids written now only goes up within one `ChatWrapper`, and
a call of the history given without an id is numbered from it too. A failed
`begin` puts the count back with the rest of the conversation.

**R11: `begin` counts the items it kept as they were given.** A reply given as
an item of its own after a `ToolTurn` that ends on a result is the reply that
closes that turn, and `conversation()` hands it back inside the turn, the one
shape a closed tool turn has. `begin` counted what it kept in that regrouped
shape, so it returned one item fewer than it kept. It now counts the newest
items of the history as given whose messages were kept. The regrouping stays,
since one turn read back in two shapes would make the round trip ambiguous; the
README states it.

**R10: a question sees the same history however it is written.** At
`max_history_exchanges`, a question the render made ahead stood in for was
spliced in and answered with one exchange more in view, as 270.21 settled:
the eviction waits until the reply ends, so that it does not delay the reply.
A question that could not be spliced, such as one with a leading space, was
added with the eviction first, so the model saw one exchange fewer depending
on how the question was spaced. Both paths now follow 270.21: the question is
added with only the token room enforced, and the limit is restored when the
reply settles, where the dropped stretch is shifted out as 270.2 requires.
The review proposed the other direction, evicting before every question at the
limit. That was not taken: it contradicts 270.21, and at the limit it would
skip the question decoded ahead on every turn and put the eviction, and on a
cache that cannot shift the decode after it, before each reply. `inject`
still evicts first, since no reply follows it to restore the limit, as the
README states.

**R12: the logging example shows what it says.** The README's example set
`llama_chat.engine` to `DEBUG` to show every line llama.cpp writes, but with no
handler configured Python falls back to a handler that shows only warnings and
errors, so no `INFO` or `DEBUG` line appeared. The example now adds
`logging.basicConfig()`. It leaves the root logger at `WARNING`, so other
libraries' debug output stays hidden, while the records `llama_chat.engine`
passes at its own level reach the handler. A test runs the example in a fresh
interpreter and reads a debug line back from stderr.

**R7: the policy a turn keeps is noted under the lock.** `reply` read whether
it went on with an open turn from the live state, without the lock, and noted
its policy as the turn's when it was called, before the reply ran. Two replies
asked for before either ran left the policy of the one called last, so a turn
opened under `"yield"` could go on under the other reply's registry and run a
call nobody was handed. `reply` now reads the published state, as its other
checks before the stream do, and the policy is noted when the generation
starts, under the lock, and only where that reply opens the turn. A reply
given no policy still reads the kept one when it is called; that read, like
every check before the stream, is advisory, and the reply is checked again
under the lock.

**R8: a turn going on keeps what is left of its `max_calls`.** A stream closed
at a call a registry handed over leaves that call owed (268.9), and the reply
that goes on after its result keeps the registry. It did not keep the turn's
`max_calls`: it went on with no ceiling, and the count began again. The turn
now keeps its ceiling and what is left of it beside its policy, so a reply
going on without a `max_calls` of its own goes on with both, and the error
for a spent ceiling names the limit the turn was given. A call counts against
the ceiling once it is handed over, before it runs, since a stream closed
there leaves it to the caller, who answers it. A turn going on counts as one
that has called, so `tool_limit_reached` is reported when its ceiling is spent.

**R9: a stream closed inside a round writes what already ran.** A registry
settles every call of a round before it writes any result, because what a
failure tells the model depends on the budget the last call leaves. A stream
closed at the second call of a round therefore dropped the first call's
result, though the tool had run: both calls stayed owed, and a caller that
answered them by running the tools ran the first one twice. Closing the
stream at a call now writes the results of the calls of the round that ran
before it, each failure worded for the budget left at that point, and leaves
that call and the rest of the round owed, as 268.9 leaves a single call.
Writing them is the same `add_result` a caller's `inject_tool_result` makes,
so the cache stays the template's render. Where no room is left for a result,
it is not written and the call stays owed, since the caller's own result
would find no room either; that is the one case where a call that ran is
still owed.

**R14: the constructor names the refusals of the add-on file.** Reading the
add-on file and the sampling the GGUF recommends moved into the constructor,
which then raised three `ValueError`s its docstring left out: an add-on file
that names the model's template but is malformed, a `reasoning_on` that writes
the prompt as reasoning off does, and a recommended sampling value the
settings refuse where nothing replaces it. The docstring now names them, and
a test pins that each one reaches the caller of the constructor. A file next
to the model that is not JSON, or names another template, is still not used
and only warned about, since it may belong to something else.

## 2026-11-11: Tests that name no model

At the user's decision (263.4A), the tests name no model, family or vendor,
and write no tag of a real template. Before, about twenty places in the tests
named one, and some checked what was true of one named template only. The
name check read only the library's modules, so none of this was caught.

**What changed.** A test written for one named template now chooses its
templates by what they do, and runs on every template that does it: writing
the date, a system prompt that a later question moves, a reply the template
cannot write empty, a reply that ends at a text where a call parts from its
closing, an opening the model writes before its answer, a call that opens with
plain text, and a shipped add-on file that says how a call is written or sets
sampling. What a test expects of one template stands in that template's own
file in `tests/templates/`, beside `marks_thought`, `can_reason` and
`calls_tools`: `turn_end_token`, `turn_end_text`, `answer_opening`,
`call_opens_as_text`, `writes_empty_reply`, `can_take_system_prompt`,
`date_format`, `lost_arguments`, the fragments and call body extraction must
find, and the sampling a model's card recommends with the card's words. A key
a file leaves out holds what most templates do; `tests/chat_templates.py`
lists those values. Where a key selects the templates a test runs on,
`tests/test_real_templates.py` checks the key against what the template does,
on every template: `date_format` stands exactly where the template reads the
clock, and its render holds the date in that format; `writes_empty_reply` is
false exactly where the template refuses an empty reply. A check of a typed
value, written for one template, became a property: wherever a template
writes the text `"42"` otherwise than the number `42`, the text reads back as
text. Tag literals of real templates in tests became invented tags; a test
that needs a template's own tag reads it from that template's file or render.

**What counts as a model file.** A test template and what its file says of
it, in `tests/templates/`, and an add-on file in `llama_chat/addons/`. The
publisher and chat format names the check needs beyond a file's source moved
from `tests/test_neutral.py` into the template files (`publisher`,
`chat_format`, and `publisher_is_a_word` for a publisher whose name is also an
everyday word), so the check names none itself.

**What the widened check exempts.** It reads every file of the project except
hidden folders, `__pycache__`, `*.egg-info` and files that are not text, and
exempts:

- the model files;
- `design_history.md`;
- the README's install section, from `## Install` to the next section;
- `install.sh` and `pyproject.toml`, which install the engine and name the
  maker of the GPU its build is for.

In the engine's module `llama_chat/_backend.py` and its tests
`tests/test_backend.py`, the engine's own function and constant names, those
that begin with the first word of the engine module's name, are allowed. The
library's and the engine's own names are never taken for a model. The tag rule
is the one the library's modules had: no tag longer than three characters that
a test template writes. A rule for one family's tags alone was rejected: a chat
format's tags are shared by several families and identify the format as its
name does. Besides the parts of a name split at `_`, before capitals and
between letters and digits, the check now also tries the parts split at `_`
and before capitals alone, so that a vendor whose name holds digits is found
inside an identifier.

**What was kept narrower.** The test that a reply kept as generated is not
shortened searches over context sizes for one that leaves the error result no
room; it ran on one real template and now runs on an invented template that
calls tools, since the search on every template that calls tools would
lengthen the suite for no other gain. The date tests begin without a system
prompt, since some templates write the date only then; the old test, on a
template that writes it either way, began with one. The fragments and call
bodies extraction must find are stated for the templates that had them
before; the other templates are checked through whole renders in
`tests/test_conversations.py`, and values computed by the code under test
would add no check of their own.

## 2026-11-12: The user's answers to the round's open questions

**300.9A: the penalty window is the model's too.** A GGUF may state how far
back the repetition penalty looks, as `general.sampling.penalty_last_n`, and
only the other sampling values were read. `penalty_window_tokens` is now the
eighth model sampling field: left unset in `Config`, it takes the add-on
file's value, else the GGUF's, else 64, llama.cpp's own default, whether or
not another field is stated, since a shorter window is no more "off" than a
longer one. An add-on file may set it as it sets the other seven, so it went
into the one list of model sampling fields, which every check of an add-on
file, of a GGUF's recommendation and of the merge reads. A GGUF value is read
as a whole number, as `top_k` is. llama.cpp reads a window of -1 as the whole
context; `Config` refuses a negative window, so a GGUF that states -1 stops
the load naming the GGUF, as any value the settings refuse does, unless
`Config` or the add-on file sets the window.

**301.7A: a reply closed by nothing ends at the next turn's tag.** A template
may write nothing after a reply, so that the next turn's tag follows the
reply's text directly. The load check refused a closing that opens with plain
text but let an empty one pass, although a model on such a template stops only
where the engine ends a generation on the tag that opens the next turn. Such a
template is now refused at load where that tag does not end a generation, or
where the next turn opens with no tag, naming the tag. Whether a token ends a
generation is a fact of the model's vocabulary, so `ends_generation` moved from
the cache's half of the cache context protocol to the model's half, which the
template is resolved through; a context answers the same members as before.
Measured on GLM-4.7 Flash with llama.cpp: `<|user|>`, `<|observation|>` and
`<|endoftext|>` end a generation, `<|assistant|>` does not. The stand-in
context ends a generation on the fixture's end-of-sequence tag and on the tags
its `ends_generation` lists, which the GLM-4.7, GLM-5.3 and Flint templates
state as `<|user|>` and `<|observation|>`. For GLM-5.3 and Flint this is not
measured: their GGUFs are not in the local cache, and the value assumes the
converter marks the same tags as for GLM-4.7, whose chat format they share.

**301.9B: the template for tools is chosen at `begin`.** Under 264.4 a
`declare_tools` into a conversation begun without tools moved it to the
model's template for tools and decoded it again in that form. At the user's
choice that switch is refused now, with an error that says to begin again
with the tools. In its place `begin` decides the template once: `tools` given
at all, a registry or a list, even an empty one, writes the conversation with
the template for tools, so that tools declared later need no switch, and
`tools=None` writes it with the default template. Read so because an empty
registry is still a promise of tools, and a list of specs is the same
promise in another form. The conversation keeps which template it was begun
with beside its declarations; a declaration on a model that ships no template
for tools marks it too, so that whatever declares tools is written with the
template a `begin` with them would choose. `conversation()` hands `tools`
back as `None` for a conversation written with the default template, and as
a list, possibly empty, for one written with the template for tools, since an
empty list given to `begin` now selects the template for tools and the round
trip must keep the template. A `begin(tools=[])` on a model that cannot call
tools is accepted as before; the declarations that follow are refused there.

**302.6B: `conversation()` hands every message back as a dictionary.** It
gave a message as a `(role, text)` pair and a reply that kept a thought as a
dictionary, so a caller reading the history back handled two shapes for one
kind of item, and a reply changed shape when its thought was kept. At the
user's choice, to simplify, every message now comes back as
`{"role": ..., "content": ...}`, with `"reasoning_content"` where a reply
keeps its thought; a tool turn stays a `ToolTurn`. `begin` still takes both
pairs and dictionaries (268.1A), so a history written by hand stays short,
and `begin(**chat.conversation())` begins as before, since `begin` reads a
dictionary as it read the pair.

**302.7A: the answer to an unreadable call says whether to try again.** It
said only that the call failed and could not be read (270.24), while every
other failure ends on `try again` or `do not call again`. At the user's
choice it now ends on the same instruction, decided by the same rule: `try
again` where a call the model writes next is read, which is under `"yield"`
and under a registry with a call left after the one the unreadable call
spends, and `do not call again` under `"refuse"` and at a registry's last
call, where the next call would stay text. The user's words were "try
again"; it was read as the rule the other failures follow, since telling a
model to try again where its next call cannot be read asks for what it
cannot do ("The mechanism, moved here from the README"). Saying `try again` in every case was
the other reading. The wrapper words the answer, as it words every other
failure, and hands it to the conversation, which writes it.

**303.6A: an unreadable call carries what the model wrote.** A call reported
as `"unreadable"` on `Reply.tool_calls` had no name, arguments or id, and
nothing else, so a caller could not see what the model had tried to write.
`ToolCall` now has a `text` field, empty for every other call, which holds
for an unreadable one what the model generated between the call's opening and
its closing, as the parser was given it. A field of its own was chosen over
putting the text into `arguments` or `name`, which would read as a call that
was understood. It takes part in equality as the other fields do: nothing
compares an unreadable call with a call built by hand, since none is handed
over or answered. The conversation's history still keeps an unreadable call
without its text, under the name a result is written with.

**307.1A: the demo runs on a template that refuses two user turns in a
row.** The demo injected a user message and then asked a question, which a
template that writes no two user turns in a row refuses, so it stopped at its
first reply on such a model. The library refuses it rightly, so the example
changed: its first reply now answers the injected message, and the questions
follow that reply. Run on the CPU with Gemma 3 1B, which refused before, and
with Llama 3.2 3B: both finish, and each reply decodes only its own new
tokens.

**307.2A: the stand-in keeps its token ids where a context may have them.** A
context of one's own must hand out ids from -65,536 to 1,048,575 (270.24),
but the stand-in, asked to generate tokens that spell no character's own
(`tokens_spell_text=False`), numbered them from just past the last code
point, outside that range. The search for a dropped stretch could then have
failed on the stand-in where it never fails on a context that keeps the rule.
Those ids now count up from -65,536, the lowest id allowed, towards the tags,
which count down from -3; the stand-in asserts that the two never meet. A
test collects every id the stand-in hands out in a conversation, those it
draws, those the cache holds and its tags, and checks each against the range
the search can spell, which it reads from the library rather than restating.

## 2026-11-12: Review fixes

**S4: a model that states only its penalty window samples at llama.cpp's
defaults.** Whether an unstated field is switched off or takes llama.cpp's
default was decided by whether any field was stated, the window included. A
GGUF or add-on file that stated only `penalty_window_tokens` therefore ran at
temperature 1 with `top_k`, `top_p` and `min_p` off, though it recommended no
sampler at all. The window says how far back the penalties look, not how a
token is drawn, and 300.9A already gives it the same fallback in both cases,
so the choice between the two fallbacks now reads every field but the window.

**S5: a GGUF's penalty window of -1 is the whole context.** llama.cpp reads
`penalty_last_n = -1` as the whole context, and `Config` refuses a negative
window, so under 300.9A such a GGUF stopped the load unless `Config` or the
add-on file set the window. The value is a valid statement of the model's, not
a malformed one, so the loaded model's context now reads it as its own
`context_size` before it is handed on, and `Config` keeps refusing a negative
window from the caller and from an add-on file, where `-1` has no meaning of
its own. Only `-1` is read so, since llama.cpp gives no other negative value a
meaning. A context of one's own still gives its window as a length that
`Config` accepts.

**S6: a reply given no policy takes the policy of the turn open when it
runs.** R7 moved the noting of a turn's policy under the lock, but a reply
given no policy still chose its own when it was called. One asked for before
another reply opened a turn under `"yield"`, and run after that turn opened,
went on with it under `"refuse"`, the default for a new turn, and the call it
met was answered with an error instead of handed over. `reply` now keeps the
policy as given, `None` included, and the reply chooses it under the lock as
it starts. The choice made when `reply` is called stays, as an advisory check
of `max_calls`, which is checked again under the lock against the policy then
chosen.

**S8: a reply that fails before it generates leaves its turn's budget.** A
reply going on with a turn and given a `max_calls` of its own replaced the
turn's budget before it ran, so where it then failed, for instance because a
result was owed by the time it ran, the turn went on with the new ceiling,
against principle 5. The policy and the budget a reply leaves its turn are now
noted only once its request is written and its generation has started, before
anything it produces is recorded, so a reply that fails before that changes
neither. Putting the old values back on failure was the other way; it was not
taken, since every path that fails would have to restore them, while one
point where the generation starts covers all of them.

**S7: a turn the history opened keeps the first policy a reply gives it.** A
turn's policy was noted only by the reply that opened it, and a turn that
`begin` left open was opened by no reply, so it never had one. Going on with
it under `"yield"` and then calling `reply()` after the results therefore
went on under `"refuse"`, against the README's promise that `reply()` keeps
the turn's policy, and the model's next call was refused instead of handed
over. The first reply that goes on with such a turn and is given a policy now
notes it as the turn's, as if it had opened the turn; a reply given none
leaves the turn without one, so the next still goes on under `"refuse"`.

**S12: the test whether a call is read after an unreadable one reads as a
statement.** The local function that tells a reply's generations whether a
call written after an unreadable one is still read was named
`after_unreadable`, which names a moment, not what is true. It is now
`may_call_after_unreadable`, after the `may_call_again` it is handed to.

**S11: every publisher a source names counts as a vendor.** The name check
took a source's text before its first slash as the publisher. A source that
opens with prose, as the add-on file that measured how a call is written does,
gave a vendor of two words that no single word can match, and a second
publisher named later, as a source that also names the copy it was read
from, was not picked up. Every name directly before a slash is now taken. A
test reads the sources again by splitting them at spaces and punctuation, a
second construction of the same set, and checks that each name found is a
vendor and that no vendor holds a space.

**S9: the name check reads compound words, and the tag check every tag a
template file states.** The check split a word at `_` and before capitals,
so a name joined by `-` or `.`, as in a model's repository name, a file name
such as one ending in `.gguf`, or a compound word, was read whole and matched
no family. Its parts are now split at `-` and `.` as well, and a vendor whose
own name holds such a separator is found wherever it stands between
separators inside a word. Splitting so found two names that are not a
model's: the project's own distribution name, which joins the engine's name
with `-` and is now among the library's own names, and the name of a GPU
maker's tool in the engine module's install message, which now says "where
the GPU driver is found"; the install script and the README's install section
keep it, as the files that install the engine. The tag check read only the
tag shapes `TAG` knows, and two end-of-sequence tags that test template files
state had a shape it did not, so a test could have written them unnoticed;
one of them is also written by its template, which the stand-in therefore
spelled as plain characters. `TAG` now reads both shapes and the `◁…▷` form
some templates mark a thought with, a test checks that every tag a template
file states has a shape `TAG` reads, and those tags count among the tags no
other file may write. The test of a thought marked by tags of that form used
a real template's tag and widened `TAG` for itself; it now uses an invented
tag of the same shape, which the wider `TAG` reads without a fixture.

**S10: a family, publisher or chat format whose name is an everyday word
means the word inside a longer name.** Only a publisher could be marked as an
everyday word, so the check refused a compound such as a spark plug or a
flint stone, written as one identifier, because the family of the same name
was found among its parts. A test template's file now lists, under
`everyday_words`, which of its own names are also everyday words, in place of
`publisher_is_a_word`, and a test checks that each word listed is a family,
publisher or chat format the check reads from that file or from the add-on
file that names its template, so the list cannot exempt anything else. Marked
so: spark, granite, dolphin, muse, flint, north, ling, phi and inkling among
the families, harmony among the chat formats, and meta among the publishers.
Laguna and Hermes were not marked, being names rather than English words.
The review read the failing whole word `meta` as the exemption applied out of
order; it was the rule, and it now holds for every everyday word alike: the
whole word is still refused, since a word standing alone, as "Granite" or
"Meta" in a sentence, is how prose names a model or its maker, and so is the
word followed by a version, as `granite-4` or `spark-x2`, while as a part of
a longer name it passes. Since the parts of a model's own compound name, such
as one made of two everyday words, would now pass, every model file's own
name is read as a name too, found whole or between separators inside a word.

**T2: a test template's file holds only the keys the tests know.** A key a
file leaves out holds the value most templates have, so a misspelled key was
read as left out, and its value silently fell back to that default. A test now
checks that each file states every key a file must and no key outside the
ones the tests and the stand-in context read. The tag keys the name check
reads lost `bos_token`, which no file states and nothing reads.

**T5: a reply's opening is tested with a tag that holds a colon again.** When
the tests stopped writing real templates' tags, the test that whitespace after
a channel's header is not answer lost its only tag with a colon, a shape some
templates' tags have and `TAG` reads. It now runs with an invented tag of
that shape beside the plain one.

**T1: the date tests run with a system prompt too.** Since the tests named no
model, the date tests began without a system prompt, the case in which every
template that reads the clock writes the date. A template that writes the
date inside the system turn beside the caller's prompt, at midnight, was no
longer run. A test template's file now states
`writes_date_beside_system_prompt` where its template writes the date with a
system prompt as well, which `tests/test_real_templates.py` checks against
the template's render, and the date tests run each such template a second
time with a system prompt. The refused restart on a template that writes the
date runs with and without one on every dated template that takes one, since
it asserts no date.

**T4: the opening of a call that opens with plain text is stated, not
computed.** The tests of a call that opens with plain text took the opening
they expected from the library's own reading of the template, so an opening
the extraction found wrongly would have been expected wrongly too. Each such
test template's file now states the opening under `call_opening_text`, which
replaces the flag `call_opens_as_text`; `tests/test_real_templates.py` checks
that the extraction finds that opening, and the tests that quote an opening
or look for it in the cache read it from the file.

**T3: the card sampling is stated wherever the add-on file's source quotes
it.** A shipped add-on file's sampling was checked against the card's values
only for two templates; for the others the tests read the values from the
file under test, so a wrong value `Config` accepts went unnoticed. The values
are stated in a test template's file only where the add-on file's own
`source` gives them in words, so that the test compares two texts that can be
read side by side: that is now also the reasoning model of the family whose
instruct model was already stated. The other add-on files that set sampling
name a model card or a `generation_config.json` without quoting it, and
neither is in the repository or the local model cache, so their values were
not restated: the ones for the R1 distill of Qwen3 8B, Gemma 3 1B, Llama 3.2
3B, Muse Glimmer 30B, Nemotron 3.5 Transcribe, North Mini Code, Qwen2.5
Coder 7B, Qwen3 Coder 30B and Qwen3 8B. Each needs its card's words quoted in
its add-on file's `source` before its values can be checked.

## 2026-11-12: Model cards quoted in the add-on files

The nine shipped add-on files that set sampling without quoting their source
now quote it, and the test template each one names states the values under
`card_sampling` and the quoted words under `card_quote`, so that every shipped
sampling value is checked against the words it was taken from. Each source was
read online on 2026-11-12: the model card where the file cites a card, and the
`generation_config.json` where it cites that file, whose sampling keys and
values are quoted as they stand in it. A card is quoted as its page reads,
without the Markdown escapes and the math markup of its raw text. The
`generation_config.json` of google/gemma-3-1b-it is gated, so it was read from
its copy at unsloth/gemma-3-1b-it, as the Llama 3.2 file already was from its
copy. Every value read matched the value the file holds, so none was changed.
Two of the quotes are narrower than a general recommendation. The card of the
R1 distill of Qwen3 8B states its temperature and top-p for the benchmarks that
sample, and the temperature of 0.6 alone for its own web and application
environments. The card of Nemotron 3.5 Transcribe states temperature 0 without
reasoning for transcription, translation and language identification, and the
values the file holds for audio understanding and text reasoning, where "top-k
disabled" is the `top_k` of 0 that the file sets.

## 2026-11-12: Load time and thread defaults

### The warm-up decodes two tokens and then one

316.1 (P1). Supersedes, in "The engine context warms itself" (2026-11-10), the
sentence that the context decodes its full batch at the end of loading. The
context now decodes a batch of two tokens and then a single token, as
llama.cpp's own tools do, and clears the cache again.

The full batch of `max_batch_tokens` was meant to move the engine's first
preparation of a large batch from the first `begin` into the load. Measured:

- Graphics card, the user's production settings (Gemma 4 26B A4B Q4_K_M, all
  layers on the card, batch 512, flash attention, q8_0 cache), median of 3
  cold runs:

  | Warm-up | Load | of which warm-up | First `begin` | First word |
  |---|---|---|---|---|
  | Full batch | 3.88 s | 0.68 s | 0.017 s | 0.036 s |
  | 2 tokens, then 1 | 3.21 s | 0.03 s | 0.59 s | 0.036 s |
  | None | 3.18 s | | 0.59 s | 0.045 s |

  The first word after a 1,346 token question took 0.244 s in all three, and
  one decode step 7.9 ms in all three. On the card the full batch only moves a
  one-time 0.6 s from the first `begin` into the load; the total is the same.
- Processor (i9-9900K, 8 cores, 16 threads, `CUDA_VISIBLE_DEVICES=""`, 6
  threads, context 4096, default batch 2048, 3 runs each, range):

  | Model | Warm-up | Load | of which warm-up | First `begin` | First word |
  |---|---|---|---|---|---|
  | gemma-3-1b Q4_K_M | Full batch | 9.30-9.49 s | 8.28-8.46 s | 0.063-0.066 s | 0.074-0.079 s |
  | gemma-3-1b Q4_K_M | 2 tokens, then 1 | 1.03-1.05 s | 0.055-0.059 s | 0.063-0.067 s | 0.072-0.078 s |
  | gemma-3-1b Q4_K_M | None | 0.98-1.00 s | | 0.062-0.067 s | 0.074-0.079 s |
  | Llama-3.2-3B Q4_K_M | Full batch | 21.98-22.67 s | 20.36-20.53 s | 0.361-0.364 s | 0.341-0.347 s |
  | Llama-3.2-3B Q4_K_M | 2 tokens, then 1 | 1.73-1.80 s | 0.141-0.146 s | 0.361-0.364 s | 0.341-0.347 s |
  | Llama-3.2-3B Q4_K_M | None | 1.61-1.68 s | | 0.362-0.366 s | 0.341-0.352 s |

  At batch 512 the review measured the full batch at 2.0 s (gemma-3-1b) and
  4.6 s (Llama-3.2-3B). On the processor the full batch buys nothing: the
  first `begin` and the first word are the same with it, with the small
  warm-up and without any. On a Raspberry Pi 5 it would cost an estimated
  45 s per load.

So the full batch cost 8 to 20 s of every processor load at the defaults and
saved nothing in total on the card. The small warm-up costs 0.03 to 0.15 s
and keeps the engine's first decode out of the first word. Rejected: no
warm-up at all, which is as fast on the processor but left the card's first
word 9 ms slower; and a warm-up sized by a setting, which no caller asked for.

### The engine keeps its own micro-batch

316.24 (P7). `max_batch_tokens` sets only the batch, the most tokens handed to
the engine in one decode call. The micro-batch, the most tokens the engine
computes at once and the size its compute buffer is reserved for, was set to
the same value and is now left at llama.cpp's default.

Measured on the processor (i9-9900K, `CUDA_VISIBLE_DEVICES=""`, 8 threads,
context 4096, batch 2048, a 1,031 to 1,055 token history, 3 interleaved pairs
while other processes kept the load average at 5 to 8):

| Model | Micro-batch | Compute buffer | Prefill | Decode |
|---|---|---|---|---|
| gemma-3-1b Q4_K_M | 2048 | 2,125 MiB | 301.9, 327.4, 217.5 tok/s | 45.4, 45.0, 43.4 tok/s |
| gemma-3-1b Q4_K_M | default (512) | 531 MiB | 328.0, 322.0, 213.3 tok/s | 45.8, 45.8, 40.5 tok/s |
| Llama-3.2-3B Q4_K_M | 2048 | 1,162 MiB | 139.5, 136.1, 103.1 tok/s | 18.3, 18.2, 16.3 tok/s |
| Llama-3.2-3B Q4_K_M | default (512) | 291 MiB | 138.9, 114.1, 104.0 tok/s | 18.3, 17.3, 15.4 tok/s |

Each pair ran back to back, so the pairs compare and the rows do not: the
speed is the same within the noise of the shared machine, and the review had
measured the same on a quieter one (140.2 to 140.6 against 141.7 tok/s on
Llama-3.2-3B). The buffer is a quarter of the size, 0.9 to 1.6 GB less RAM at
the default batch. On the graphics card, runs at batch 2048 did not start
beside the user's other GPU processes (cause not measured), which is
consistent with the larger buffer; a larger micro-batch may speed prefill on a
card, which is not measured. Rejected: a setting of its own for the
micro-batch, which no caller asked for.

### Unset thread counts follow the processors

316.8 (P3), the user's decision 317.4A. A `decode_threads` left `None` used to
let llama.cpp decide, which is 4 threads whatever the machine, and a
`prefill_threads` left `None` took as many. Now a `decode_threads` left `None`
is one thread per physical core of the processors the process may run on, and
a `prefill_threads` left `None` is as many as `decode_threads` where that is
set, else every thread the process may run on. Each one set still wins. The
counts are read once at load: the processors from the process's affinity
where the system reports one, else all of them; the physical cores from each
processor's package and core as Linux describes them under
`/sys/devices/system/cpu`, and where that cannot be read, each thread counts
as a core. The standard library alone reads them, so no dependency is added.
A `prefill_threads` left `None` beside a `decode_threads` that is set keeps
following it, as before: a caller who limits the threads limits both.

The note under "Memory and quality settings", that filling the pool
oversubscribes llama.cpp's spin-waiting workers, matches what the measurement
below shows for prefill on every thread while the machine is busy.

Measured on the processor (i9-9900K, 8 cores, 16 threads,
`CUDA_VISIBLE_DEVICES=""`, context 4096, batch 512, a 1,031 to 1,055 token
history, then 64 generated tokens, greedy), interleaved runs while other
processes of the session put a load average of 1 at the start and 9 to 14 at
the end on the machine:

| Model | Threads (decode, prefill) | Prefill, tok/s, per run | Decode, tok/s, per run |
|---|---|---|---|
| gemma-3-1b Q4_K_M | 4, 4 (the old default) | 173.8, 172.4, 173.4 | 37.8, 37.0, 38.6 |
| gemma-3-1b Q4_K_M | 8, 16 (the new default) | 335.6, 56.0, 51.8 | 45.1, 44.0, 43.5 |
| gemma-3-1b Q4_K_M | 8, 8 | 327.1, 209.3, 213.8 | 45.5, 41.5, 43.4 |
| Llama-3.2-3B Q4_K_M | 4, 4 (the old default) | 73.9, 72.9, 73.4 | 14.0, 13.8, 13.8 |
| Llama-3.2-3B Q4_K_M | 8, 16 (the new default) | 134.8, 49.8, 51.1 | 17.0, 16.8, 16.0 |
| Llama-3.2-3B Q4_K_M | 8, 8 | 105.0, 104.0, 133.1 | 16.4, 16.9, 16.7 |

A further run of 16 against 8 prefill threads, decode on 8, 3 interleaved
pairs at a load average of 9 to 15: gemma-3-1b 54.5, 329.2, 50.6 tok/s on 16
threads against 213.6, 208.6, 215.7 on 8; Llama-3.2-3B 51.2, 51.7, 70.4 tok/s
on 16 threads against 136.6, 102.7, 132.3 on 8.

- Decode on the physical cores is 1.15 to 1.2 times as fast as on 4 threads,
  under every load.
- Prefill on every thread is 1.8 to 1.9 times as fast as on 4 threads while
  the machine is otherwise idle, and about as fast as on the physical cores
  then (the review measured 140.2 to 140.6 tok/s on 8 threads for
  Llama-3.2-3B). While other processes keep the processor busy, it falls to
  50 to 70 tok/s in 9 of 10 such runs, 0.3 times the speed of 4 threads
  for gemma-3-1b and 0.7 to 1.0 times for Llama-3.2-3B
  (cause not measured; it is consistent with the oversubscription above). On
  the physical cores it stays at 1.2 to 1.9 times the speed of 4 threads
  under every load.
- The fall reaches the load too: the first decodes of more than one token,
  the cache-edit checks at load, run on the prefill threads. At a load
  average of 7 to 9, 3 runs each, a load took 1.0 to 8.6 s for gemma-3-1b and
  1.7 to 11.1 s for Llama-3.2-3B with prefill on every thread, against 1.0 s
  and 1.7 to 1.8 s on 8 threads and 1.0 s and 1.8 s on 4; the checks alone
  took up to 6.9 s instead of 0.03 to 0.1 s.
- Decided by the user afterwards (337.4A): prefill takes the physical cores
  too where it is not set. See "Prefill on the physical cores" under
  "2026-11-12: The user's decisions after the second review".

### What a template writes in content is read in one pass

316.23 (P4). `extract_content_rewrites` renders every tag text between
numbered marks in one user message. It then tested each mark's presence and
split the whole render twice per text, so its work grew with the number of
tags times the render's length. It now finds the marks in order, each search
starting where the last mark ended, and reads each text between two of them.
A render that loses a mark, or puts them out of order, still reads as no
rewrites. On gemma-3-1b, Llama-3.2-3B and MiniCPM5-2B the result is
the same as before (none of them rewrites a tag); the tests pin a template
that escapes a character and one that drops a tag.

Measured on the processor (i9-9900K, `CUDA_VISIBLE_DEVICES=""`, 8 threads,
batch 512, 3 runs each):

| Model | Tag texts | Reading the rewrites, before | after | Load, before | after |
|---|---|---|---|---|---|
| gemma-3-1b Q4_K_M | 6,419 | 0.395-0.406 s | 0.0034-0.0035 s | 0.99-1.01 s | 0.60-0.61 s |
| Llama-3.2-3B Q4_K_M | 261 | 0.0008-0.0009 s | 0.0002 s | 1.71-1.76 s | 1.73-1.74 s |

With 20,000 texts on a render that only wraps the message, the old reading
took 4.2 s and the new one well under the test's bound of 0.5 s.

### The vocabulary is scanned for its tags in one call to the backend

316.23 (P8). The context read the model's tags by asking the backend, token by
token, whether each is one, so every token cost two Python calls around the
one call into llama.cpp. The backend now answers for the whole vocabulary in
one pass, with the engine call bound once; `is_special` and the scan read the
same attributes. llama.cpp offers no call that reads the attributes of many
tokens at once, so the call per token stays and bounds the gain.

Measured on the processor (i9-9900K, `CUDA_VISIBLE_DEVICES=""`, 3 runs each,
at load): gemma-3-1b (262,144 tokens, 6,414 tags) 0.096 to 0.100 s before and
0.080 to 0.081 s after; Llama-3.2-3B (128,256 tokens, 256 tags) 0.048 to
0.060 s before and 0.038 to 0.039 s after.

## 2026-11-12: Review fixes of the streaming, the thought filter and the tool loop

**316.2 (P2): a reply holds back only text that could begin a call or its end.**
The generation held back the last `held_chars - 1` characters of every reply,
`held_chars` being the longest of the call's opening, its closing and the text
that ends a reply, on every template that defines one of these texts: 6 of the
52 test templates, from 9 characters (llama-3.2) to 62 (kimi-k3). It now holds
back only the longest end of the text that is the start of the opening or of
the reply's ending, and releases it with the token that shows it is neither.
Where the opening is plain text, it is held only where a call may open, at the
reply's start or right after a tag, since one anywhere else is a mention and
already passed as text. Where the opening begins with whitespace, the
whitespace before it is held with it, as it is the opening's. The closing no
longer counts, as it is searched for only inside a call, where nothing is
shown. Measured on Llama-3.2-3B Q4_K_M on the processor (i9-9900K, 6 threads,
context 4096, batch 512, greedy, `"refuse"`), six short questions, two runs
each with `holdback2.py`: before, 2 to 10 tokens were sampled before the first
delta, a median of 0.18 s (0.052 to 0.522 s), and "2 + 2 = 4" and "Blue" came
only at the end; after, the first delta came after the first sampled token on
every question, 0.051 to 0.053 s, which is the one decode before a token's text
is yielded (P6, not changed here).

**316.22 (P5): stop strings hold back only text that could begin one.** The
answer held back its last `max(len(stop)) - 1` characters whenever stop strings
were given. It now holds back only the longest end of the answer that is the
start of a stop string, the same test the generation applies to a call's
opening since 316.2, and both read it from one function in `reply_text.py`.
Measured on gemma-3-1b-it Q4_K_M on the processor (same setup as 316.2) with
`stop=["<|im_end|>", "User:"]`, six questions, two runs each: before, 2 to 10
tokens were sampled before the first delta, a median of 0.11 s (0.044 to
0.224 s); after, the first delta came after the first sampled token on every
question, 0.022 to 0.025 s.

**316.3 (I1): whether the turn's end closes a thought is read from every
token generated.** Where a thought's closing begins with the tag that also
ends the turn (2 of the test templates), the test whether that tag ends the
reply read the thought filter, which is fed only the text the generation has
already shown. While the generation still held back the thought's opening,
the filter did not know a thought had begun, so the tag ended the reply: on
gpt-oss, a thought under about 35 characters lost the whole answer, and a
call written after it. 316.2 shortened the hold so far that the existing
templates no longer meet the case, but the test still depended on what the
stream had shown. It now feeds a thought filter of its own with every token
generated before the one it tests, so no hold before the answer's filter can
change where a reply ends. The test stand-in had generated every character
of a scripted reply as plain text, never a tag as its token, which is why the
suites missed the fault; `FakeContext(writes_tags=True)` now generates a
scripted reply's tags as their tokens, as a model writes them, and the
reasoning tests run a short thought, and a call after one, in that mode on
every template that reasons.

**316.9 (I4): a round that recognizes no call cannot open one with a tag.**
Under `"refuse"`, and in the last round a registry's `max_calls` allows, the
reply after the results was generated with calls unrecognized, so a call the
model wrote again stayed in the answer as raw text, `<call>fn:clock{}</call>`
on the invented template, was streamed to the caller and stored as the reply's
text, against principle 7. The real context renders tag tokens as text, so
this reached real callers too. Such a round now also withholds the token that
opens a call, as `"withhold"` does, which the failure the model is told
already assumed ("do not call again"). Where a call opens with plain text no
token can be withheld, and a call written there still stays text; ending the
reply at such an opening was proposed by the review and not built here.

**316.29, first half (C9): a reply that goes on with a turn under `"refuse"`
or a registry is parted from the turn's text too.** 270.10 decided that every
reply going on with an open turn starts with a blank line where it would
otherwise join the text before the call without whitespace. Only the
`"yield"` and `"withhold"` path read the turn's text for that; the path that
settles calls itself began from an empty text, so "Moment." followed by a call
under `"yield"`, a result, and `reply(tool_call_policy="refuse")` gave
"Done." where `"yield"` gave "\n\nDone.". Both paths now begin from what the
model wrote in the open turn.

**316.26 (I10), decided in 324.3A: under `"yield"` a call that cannot be read
is written again in the same reply.** A reply under `"yield"` whose call could
not be read ended there with `"unreadable_tool_call"`: the failure was written
for the model, nothing was handed over and nothing was owed, yet the turn
stayed open, so `reply("next")` was refused and the caller had to know to call
`reply()` once more (against principle 6, a turn waits only for results). The
user chose that the library go on by itself, as it does under a registry. The
reply now generates again after such a call, reading calls as before, until
the model makes a call that can be handed over or answers. Under a registry
`max_calls` bounds the retries; under `"yield"` there is no call budget, so
the generations of the reply share one `max_reply_tokens` instead, each
generation getting what the earlier ones left, and the last one, cut by the
cap, closes the turn with `"token_limit"`. The other options were to close the
turn as `"refuse"` does, which takes from the model the chance to correct its
call, or to hand the unreadable call over as an event, which leaves every
caller a case to handle; the user chose neither. A server that called
`reply()` again on `"unreadable_tool_call"` no longer meets that reason under
`"yield"`.

**316.16 (C4): a value written without a type mark is read by its declared
type.** Call formats that write values bare (qwen-3-coder, minicpm-5, glm-4.7
and 20 more of the test templates) left the parser to guess each value's type
from its spelling, so a text argument changed before the tool ran or the
caller saw it: `01234` became 1234 and then the text "1234", `1e5` became
"100000.0", `nan` became "NaN", and `True` and `None` were refused as
`wrong_arguments`. The parser now keeps such a value's spelling until the call
is read whole, and then reads it against the declared parameter schema of the
tool it names: where the schema asks for text and not for the literal the
value spells, it stays text as written; otherwise, and where the tool is not
declared, it is the literal, as before. A schema's `enum` that holds the
spelling as text keeps it as text. The rule is applied once, at the end of
`parse_calls`, so a `ToolCall` handed over under `"yield"` carries the same
values a registered tool receives. Converting in the registry's readers
instead was not taken, since a call handed over runs through no reader. A
nested value the format writes as JSON or as a Python literal marks its own
strings and is read as written. The registry still takes a number where text
is asked as the text it was written as, for a JSON call that writes `42` for a
text parameter.

**316.17 (C5, C6): a request's options are checked before its question is
written.** `sampling`, `stop` and `cancel` were first used after `reply` had
written the question, so one of the wrong kind, such as
`sampling={"temperature": 0.1}` or `stop=[1]`, raised there and left the
question without a reply: the next question made two user turns in a row, or
was refused where the template refuses that, against principle 5. `reply` now
checks all three when it is called, as it already checked `text` and
`max_reply_tokens`, and builds the sampling the generation draws with there,
so the combination with the config is settled before anything is written as
well. A bare `str` for `stop` is refused: it was read as a list of its
letters, so `stop="END"` ended a reply at its first "E", "N" or "D". Reading
it as one stop text instead was not taken, since the signature asks for a
list and one way to give it is kept (design history, "One way to do each
thing in the public interface").

**316.18 (C7, K2): an add-on file's null sampling value is refused at load.**
`"sampling": {"temperature": null}` passed the load check, since
`Config(temperature=None)` is the unset value and valid. The wrapper's config
then kept `None` as the model's temperature, which the sampler compares with
a number at the first reply, and the null counted as a stated value, so the
fields the file left out fell to the values that switch them off instead of
llama.cpp's defaults. A null in an add-on file states nothing, and principle 2
asks that it fail at load, so `read_addon_file` now refuses it, naming the
file and the field; a field the file means to leave to the model is left out.

**316.20 (K1), decided in 322.3A: a result JSON cannot write reaches the model
as a failure, wherever it comes from.** `result_text` wrote results with
`json.dumps`, which writes `NaN` and `Infinity`, so a tool returning
`{"t": float("nan")}` handed the model text that is not JSON, while
`_is_json` in the same file used `allow_nan=False`: two rules for one
question. A result JSON cannot write was answered with `unwritable_result`
where a registered tool returned it, but refused with a `ValueError` where the
caller handed it to `inject_tool_result`. The user chose one behaviour for
both: `result_text` now refuses `NaN` and infinity too, the registry and
`inject_tool_result` both answer such a result with the `unwritable_result`
failure, and `inject_tool_result` no longer raises for it but logs a warning
on the `llama_chat` logger, so a caller still learns of it. The failure
handed over by the caller is explained as the registry's is, telling the
model to try again where the turn goes on under `"yield"`, and not to call
again otherwise. The conversion happens in the wrapper, before the result
reaches the conversation, so the conversation keeps refusing an unwritable
result of its own callers.

**316.31 (C12), decided by the user: a `ToolCall` hashes by its name and
id.** `ToolCall` is a frozen dataclass holding its arguments as a dict, so the
hash the dataclass generated from every field raised "unhashable type:
'dict'", and a caller could not put calls in a set or use them as keys. It now
hashes by `(name, id)`, which two equal calls always share, and equality stays
the comparison of every field, so two calls of one name and id with other
arguments are unequal and only share a hash bucket. Setting `__hash__` to
`None`, which says plainly that a call cannot be hashed, was the other option;
the user chose hashing.

**316.28 (I12, I13): the thread that reads a reply is noted at every step,
and the capabilities are read from the finished copy.** The wrapper noted the
thread holding it once, when a reply's stream was first read. A stream started
on one thread, still alive, and read on by another left the first noted, so a
tool the reply ran on the second thread and that called back into the wrapper
waited for the lock its own reply held, for ever, instead of being refused.
The stream now notes the thread that takes each step as the holder while the
reply holds the lock, and that thread's actions are refused. The holder is now
the thread object rather than its number: a thread that has ended can pass
its number to one started later, which was then refused as the reader though
it never read the reply, where it should wait. `can_reason`,
`can_turn_off_reasoning` and `can_take_system_prompt` read the template of the
conversation's live state without the lock, so during a `begin(tools=...)`
that later failed they answered for the template for tools, which the
conversation never came to be written with (principle 8, readers see a
finished copy). They now read the template the newest publication names,
through a `published_template` the conversation gives, as `turn_open` and
`pending_calls` already read theirs.

## 2026-11-12: Review fixes to the cache and the conversation

**316.30 (C11): the test stand-in reads a closing bracket tag as one tag.**
The stand-in's tokenizer read `[INST]` as one tag but `[/INST]`,
`[/TOOL_RESULTS]`, `[/AVAILABLE_TOOLS]` and `[/SYSTEM_PROMPT]` as letters,
though the vocabularies of the models that write them hold each as one
control token. Every template of that family was therefore tested with a
tokenization no model has: a turn that went on after its result could not be
placed, since no tag followed the result's text before the generation prompt.
The tag pattern now reads `\[/?[A-Z_]+\]`. Nothing in the suite failed once it
did; the paths it opened are covered by the tests of the fixes below. This
fix was made first, before the one for 316.6, whose test on these templates
needs it.

**316.6 (C1): one rule for where a kept turn begins.** A reply that went on
after a result kept its tokens from the end of the last text of the caller's
own in the render with the generation prompt, while the render placed the kept
turn from the first tag after the newest message's content. The two differ
where the result is empty, since an empty text marks no place of its own, and
where the template writes the calling message's text after its results. A
turn begun again from `conversation()` after an empty result then held the
call's closing and the result twice in the cache, on 39 of the 41 test
templates that call tools; on the template that writes the text after the
results, the result's closing was lost. Both places now ask the same
function, `turn_start`, which is the first tag after the content of the
message before the turn. `tests/test_conversations.py` begins a turn again
after its result, empty or not, with text before the call and without, on
every template that calls tools, and checks the cache after the reply and
after the next question.

**316.7 (C2): a turn closed for lack of room after several calls no longer
blocks the conversation.** Where no room was left to answer the newest calls,
the turn closed on a reply without text, kept as generated from its first
call on. The end of such a reply was searched for only up to the first text of
the caller's own after the turn's start, which lies inside the turn when it
holds earlier calls and results, so the turn could not be placed and every
later action was refused until a new `begin`. The closing is now searched for
from the end of the message before the reply, its call or its content. Where
a kept turn still cannot be placed, or would stand for none of the template's
text, it is held as the template writes it instead, so that no action is
blocked by it; the cache is then brought to the template's render. Where the
template writes nothing of a reply left without text, as some templates
refuse an empty reply, the turn is taken back with its question, as an empty
reply already is, since a conversation the template refuses cannot be held.
The end-of-test check accepts a kept turn that ends on a reply without text
where the template's version of it ends at the closing after a reply.
`tests/test_conversations.py` fills the context after two calls of a turn
begun after a result, on every template that calls tools, and checks that
the turn closes with `"token_limit"` and the next question is answered;
before the fix it failed on 18 of the 41 templates.

**316.15 (C3): a calling message's text is written once.** A turn kept as
generated holds the text the model wrote before its call, and the template was
also given that text as the calling message's content. A template that writes
a calling message's text after its results, as the one the user runs in
production does, then wrote it a second time after the result, outside the
kept tokens, and closed the turn after it, so the reply after the result
followed a closed turn with no header of its own. The invented template that
writes call blocks before content did the same. The template is now given each
calling message inside a kept turn without its text, which the turn's tokens
already hold; the history `conversation()` returns keeps the text, so a
restart writes it where the template writes it. The end-of-test check renders
those messages the same way. `tests/test_conversations.py` has the model write
text before its call on every template whose call opens with a tag, and checks
that the text is held once before the reply after the result and before the
next question, and that the cache before that reply ends as it does where the
model wrote no text before its call.

**316.29, second half (C10): a stop string after a result leaves nothing hidden
in the model's memory.** A reply that went on with a turn after its result
extended the turn's kept tokens to everything it generated, so where a stop
string ended the answer, the cache still held the stop string and what
followed it, text the caller never saw; a plain reply is written by the
template up to the answer. Such a reply now leaves the turn kept up to its
call, as a reply without text already does, and the template writes the
result and the answer after it, so the cache holds the answer as the caller
read it. Cutting the kept tokens back to where the stop string began was the
other way; it was not taken, since a token may hold both the end of the answer
and the start of the stop string, and the template's own render of the answer
needs no such cut. `record_reply` is told by the wrapper whether a stop string
ended the answer. `tests/test_conversations.py` ends a reply after a result
with a stop string on every template that calls tools and checks that neither
the cache nor the next question's prompt holds the hidden text.

**316.4 (I2): a limit on exchanges drops well below it on a cache that cannot
shift.** Where only `max_history_exchanges` bound, an eviction dropped one
exchange, the fewest that brought the count back to the limit, so on a cache
that cannot shift every reply from the fifth on decoded the whole
conversation again: 591 tokens for a 565-token cache, where a cache that
shifts decoded 75. Such an eviction now leaves `EVICT_TO_FRACTION_WITHOUT_SHIFT`
of the limit, at least one exchange, as the token threshold already did, so
that with a limit of 4 the cache is decoded again every third reply: 337
tokens then, 75 on the two replies between (measured with the review's probe
on the invented plain template, a cache that can neither shift nor cut). A
limit of 1 still drops one exchange per reply, which no bound can avoid.
`tests/test_conversations.py` runs sixteen replies under a limit of four on
every template, on a cache that shifts and on one that cannot, and counts the
replies that decode more than an exchange or two.

**316.5 (I3): a question that repeats a message is rendered with the whole
conversation before it is decoded.** A question was put in the stand-in's
place where the template writes it there after one probe exchange. One test
template writes its declarations before every user message that equals the
newest one, so a question asked twice made the template write the earlier
one otherwise. That shows only in a render of the whole conversation, which
the splice skips: the reply streamed from a cache that was not the render,
and only settling it then refused, after the answer had been read; with a
call, the call could never be answered. A question whose text equals that of
a message already in the conversation is now never spliced, so the whole
conversation is rendered and split before the question is decoded, and a
conversation the template cannot write is refused before anything streams.
Rendering the whole conversation before every spliced question was the
literal reading of the fix; it was not taken, since the probes at load
already cover what a template does with a question's text alone, equality
with an earlier message is what this template reads, and a whole render per
question would undo the rendering windows ("Rendering windows"), which
`test_a_question_waits_only_for_its_own_tokens` holds to. A template that
reads a question's text against earlier ones in some other way, such as by
length, would still pass unseen. `tests/test_conversations.py` asks the same
question twice on every template, with tools where the template calls them,
and checks that the second reply follows the render or is refused before it
starts, the conversation unchanged.

**316.11 (I6): a begin that ends on a question rests before the generation
prompt.** A `begin` whose history ends on a user message or a result decoded
the generation prompt with the rest and only then rested, so a cache that can
neither shift nor cut held saved states only at its end and at its opening.
Where the next action wrote what follows the newest message otherwise, a
second question, or a past reply the template writes otherwise than it was
generated, the cache went back to the opening: 645 of 670 tokens decoded again
on the first reply after such a `begin` (measured with the review's probe on
the qwen-3 test template). The cache now rests where the template text the
reply follows begins, then decodes that text, and the same probe decodes 36
tokens. Where the template writes no such text, the cache rests once, after
decoding with the logits the reply needs: resting with no logits there made
the reply decode its last token again, which dropped the state just saved.
The text after the newest message's own text was chosen over the first tag of
the reply's opening, since a template may open a reply with no tag.
`tests/test_conversations.py` begins with twelve exchanges and a question on a
cache that can neither shift nor cut, on every template, then answers it and
asks another, or asks another at once, and checks that no third of the cache
is decoded again.

**316.12 (I7): a reply the template refuses to write is taken back with its
question.** A reply whose text the template refuses was recorded, and only
settling the cache then raised, after the text had streamed; the cache was
left unsettled and every later question and injection was refused, since the
template refused the conversation they would have joined. Such a reply is now
taken back with its question, as a reply that produced nothing the template
writes already was, and the conversation goes on from where it was before the
question. Whether the template writes the reply is read from a render of the
system message and the newest exchange, which the template's refusals turn on,
so that no reply renders the whole conversation for it. `tests/test_wrapper.py`
has the model reply with a text the invented refusing template refuses, and
checks that the conversation is as before and the next question is answered.

**316.19 (C8), as decided in 321.5B: a call without a result in the middle of
a history stays where it was given.** `begin` writes a `ToolTurn` that a later
question follows while it still owes a result as given, with nothing added to
it, as the user decided. `conversation()` read the history back by collecting a
tool turn's steps until the next reply, across the question in between, so the
unanswered call came back after the newest question, joined to the reply that
followed it, and a restart from that gave a different conversation. A user
message now ends the tool turn before it, so that the turn comes back in its
place, ending on its call or its result. `tests/test_messages.py` reads such
histories back, with a call owed a result and with a result no reply
followed, and `tests/test_history.py` begins again from `conversation()` after
a reply and gets the same conversation.

**Invariant 3, as decided in 329.3A: the test that beginning again gives the
same conversation.** `tests/test_conversations.py` now begins again from
`conversation()` at every step of a conversation on every test template, in
each reasoning mode: after the first `begin`, after a reply, with a live call
owed its result, with the turn open after its result, after the reply that
closed it, after a later declaration, after a `begin` whose history holds a
call left without a result before a later question, after the reply to that
question, and after each of fifteen more questions, which evict the oldest
exchanges. Each time the system prompt, the declarations and the reasoning
setting come back the same, the messages are those saved or the newest part
of them that a `begin` keeps, each unchanged and in its place, and
`conversation()` then returns the same again. The steps up to the evictions
begin again on the same wrapper, so that what follows them runs on a
conversation begun again; the evictions begin again on a second wrapper, as a
`begin` keeps only half the context and would otherwise keep any eviction from
happening. A template that writes no call without its result refuses the
history with the unanswered call, and the test checks that the refusal leaves
the conversation as it was. Once the fixes for 316.6 and 316.19 were in, the
test found nothing more; without the fix for 316.19 it fails on 39 of the 90
cases.

**316.14 and 316.27 (I9, I11), as decided in 320.5A: a result too long for
the room left evicts, then is cut with a note, then closes the turn.** A
result too long for what was left of the context was cut silently: the model
read cut-off text as if it were whole (417 of 2410 characters in the review's
probe), and `inject_tool_result` and the reply said nothing. Where not even an
empty result fitted, `inject_tool_result` raised and left the call owed, so
under `"yield"` every later action was refused until a new `begin`. Now, in
this order: the oldest exchanges are evicted first, exactly as for a new
question (`_drop_while_over` with the room for the reply as its limit and
`max_history_exchanges` left out), the turn the result answers kept whole, and
on a cache that cannot shift down to well below the room, as every eviction
there goes; a result still too long is cut to the longest start that fits
with `RESULT_CUT_NOTE` after it, which tells the model how many of how many
characters it got, and its call is listed in the new `Reply.tool_results_cut`
of the reply that follows (named after `tool_calls` and `tool_limit_reached`;
a list of calls rather than a count, so that the caller knows which result
was cut); and where not even a result cut to nothing fits, the turn closes as
`close_unanswered` closes it under a registry, under `"yield"` too.
`inject_tool_result` returns how many history items were evicted for the
result, as `inject` does, and under a registry they count in the reply's
`n_evicted_messages`. Under `"yield"`, a turn closed this way is not reported
by raising: closing it is the action's outcome, and raising after changing the
conversation would break principle 5, so `inject_tool_result` returns and
`turn_open` is false; a raise that leaves the call owed was the old behaviour
this replaces. The failure test of every action now lets a failed reply lose
the oldest exchanges that a result of it evicted, as a question's eviction is
not put back either. Tests in `tests/test_tool_loop.py` check the eviction on
a cache that shifts and one that cannot, the cut and its note under `"yield"`
and under a registry, and the closed turn under `"yield"`;
`tests/test_failures.py` checks that the conversation goes on after it.

**316.10 (I5): a tag typed into a tool's description stays text.** The marker
render marked only the texts of messages, so a declaration went into the
cache tokenized with the template's tags read, and a tag typed into a tool's
description was decoded as the model's own token on 41 of the 52 test
templates (the review's probe, with the end-of-sequence tag). Tool
descriptions often come from third parties. The marker render now also marks
every `description` of a declaration, at any depth of its parameters, so that
a description is the caller's own text and is tokenized as text, as a message
is. Only descriptions are marked: a template reads a declaration's names,
types and enum values to decide what it writes, as one writes a parameter
differently by its type, and a marker in their place would change the render
around them; they are names a caller chooses, not prose taken from elsewhere.
The render as it is shares its memo with every other render, so a split
renders no more than before. `tests/test_conversations.py` types the tag into
a tool's description and a parameter's description on every template that
calls tools, beside the texts it already covered, and checks every token of
the cache against a tokenization worked out in the test.

**316.13 (I8): the end-of-test check compares tokens.** The check every test
on the stand-in cache ends with compared what the cache spells with the
template's render, so a cache that held every template tag as its letters,
or a tag typed into a message as the model's own token, read the same and
passed. It now spells both sides so that a tag reads otherwise than its
letters: the cache from its tokens, each of the model's tags set apart, and
the render from a marker render worked out in the test, the template's text
tokenized with its tags read and each text of the caller's as plain text (a
content, a thought, a call's name and string arguments, a declaration's
descriptions). Everything around a kept turn is compared so; the bounds of a
kept turn are still found by its characters, since the stand-in writes some
of a call's tags as text, as a model may. The marker render was
`_expected_tokens` of `tests/test_conversations.py`, which now uses the same
function. `tests/test_cache_check.py` shows the check failing on a cache that
reads the same with its tags held as letters, in a plain conversation, one
that reasons and one with a kept turn, and on one with a typed tag held as the
model's. The check renders each candidate twice now, which made the suite
take about 134 seconds instead of 120 on the development machine.

## 2026-11-12: Reasons moved out of the code

316.32 (K21). Comments and docstrings that gave the reason for a line now
leave it here, and the code keeps what it does and a pointer where one helps.

- **Everything is settled while the wrapper is built,** so that a deployment
  that could never work fails then, not at its first request.
- **The wrapper's lock is held for a whole action, a generation included.**
  One cache sequence serves every generation, so two actions whose decodes
  interleave would corrupt it.
- **A registry holding tools is checked before its exchange starts,** as a
  declaration is: a registry is the same promise that the model can call
  them, and refusing it after the model has asked for a tool would leave a
  call nobody can answer.
- **The warm-up's reasons** are in "The warm-up decodes two tokens and then
  one"; **the fraction an eviction without a shift drops to** is argued in
  266.3, under "Settings, add-on files and logging".
- **Reading a call back is a module of its own,** apart from declaring and
  running tools: an engine reached over a wire emits calls it cannot run, and
  reads them without a registry.
- **A batch of calls written as JSON is read whole or not at all.** Reading
  half of it would drop what the model asked for without a word.

## 2026-11-12: While a call is owed, the cache ends with its turn

A test that ended with a call under `"yield"` still owed a result failed the
end-of-test check on 26 of the 41 test templates that call tools (49 of 74
cases, counting each reasoning mode), with reasoning on and off alike, after a
thought or without one. The cache held the conversation up to the last token
the model generated for its call; the template's render of a conversation
that ends on that call closes the turn after it, such as with `<|im_end|>`,
and on mistral-small-3.2 it also leaves out the declarations before the
question, which the template writes again once a result follows.

This is a gap in the check, not in the library. While a call is owed, the
library decodes nothing after the turn: the closing, each result and the
generation prompt are decoded together once the last result is in. Nothing
reads the cache in between, since a reply is refused while a result is owed.
Measured on every test template that calls tools, in each reasoning mode, 74
cases: the result's arrival only appended to the cache, with no cut and no
restore. So the cache is the render of the conversation as its results
complete it, up to the open turn. The check now compares an owed call's cache
with that render, a stand-in result in place of each owed one, and accepts
the rest of the render as not decoded where the cache ends with the newest
kept turn. Decoding the closing as soon as the call is made was rejected: on
a template that writes the results inside the turn, or writes earlier text
otherwise once a result follows, it would have to be taken back when the
result arrives.

Not changed here, and revised by "Second round of review fixes (final review
F1 to F11)": `begin` with a history that ends on an owed call renders the
conversation as it stands. On qwen-3, qwen-3-hermes and mistral-small-3.2 the
result then takes back 50 to 66 tokens, by a cut or by restoring a saved
state, and decodes them again.

## 2026-11-12: Second round of review fixes (final review F1 to F11)

### A reply's tokens cost the same at its end as at its start

The hold code that keeps back what could still begin a call's opening asked,
on every token of a template whose call opens with plain text (llama-3.2,
minicpm-5, deepseek-v4.1), which token the text before the held part ended
in, by scanning every earlier token. On the fake context, 8,000 generated
tokens took 1.70 s instead of 0.03 s, the last tenth of them 0.37 s. Three
further costs grew with the reply on every template: the decoded text, the
accumulator's text and the answer behind the stop strings were each
extended with `+=` on an attribute, which CPython copies whole every time
(80,000 appends of five characters took 2.3 s, 40,000 took 0.43 s), and a
call that closes with text joined all of its pieces on every token.

Now the text buffer keeps only what is not shown yet, and tracks where the
text ends without its trailing whitespace, so the hold code reads a bounded
tail; the token a position falls in is found by binary search, and only
where an opening may begin; the accumulated texts are lists joined when
read; and a call that closes with text keeps only the end its closing may
begin in. After the change, 80,000 tokens take 0.28 s to 0.40 s on every
template, the last tenth about as long as the first. The test counts the
library's bytecode steps per token, which does not depend on the machine:
on every test template, in each reasoning mode and under each policy, the
last eighth of a 400 token reply costs no more than 1.1 times the second
eighth. Before the change it failed on the three templates above, in each
of their reasoning modes and under each policy.

### An empty description is not marked

The split render marks each description so that a tag typed into it stays
text. gemma-4 and gpt-oss leave out a `description` that is empty, so the
marker for an empty one was written where the empty text was not, and
`begin` refused the declarations as writing a message's text otherwise than
in its place; before that marking, an empty description was accepted. Only a
description with text is marked now: an empty one holds no tag to keep as
text. The test declares a tool and a parameter with empty descriptions on
every test template that calls tools; it failed on those two.

### A cut result leaves room for the results still owed

A result is cut to the room left, measured with the generation prompt. Where
the model called twice and the first result was long, cutting it filled all
of that room; the second result then did not fit even cut to nothing, so the
turn closed and both results were dropped, and the model saw neither (fake
context, 400 tokens, a first result of 5,000 characters and a second of 10).
Now each result still owed after the one being written is measured with it,
as the note a result cut to nothing carries, of a result as long as the one
being written; the eviction before the cut measures the same. That stand-in
is the smallest form a later result can take, short of being empty, so a
later result whose note needs more digits than the stand-in's still does not
fit, and the turn then closes as before. Reserving an empty result instead
was rejected: a later result that does not fit whole is written with its
note, so the room for an empty one would not take it.

A turn closed for want of room also clears the list of cut results, which
the next reply reported although the results it names were dropped with the
turn.

### When a reply ends on a call that could not be read

The README said a reply ends with `"unreadable_tool_call"` only where the
template has no result block. Since a call that cannot be read is answered
under `"yield"` as well, a reply also ends so where the context has no room
left for that answer: on the fake context, with every call unreadable and no
reply cap, 15 failed calls and 390 tokens at a context of 2,048, 59 calls and
1,534 tokens at 8,192, after which the question was taken back as well. The
README row and the docstrings now say this. Whether the tries under
`"yield"` get a limit of their own was then the user's open decision; it was
decided afterwards (337.5A), see "At most three unreadable calls in a row".

### The reply cap covers the whole reply under every policy

316.26 made the generations of one reply under `"yield"` share one
`max_reply_tokens`; under `"refuse"` and a registry each generation still had
a cap of its own. A caller that reads the cap as one for the reply, as a
server's `max_tokens` is read, got more: a refused call and the answer after
it gave 24 tokens for a cap of 12. Now every policy counts the generations of
one `reply` together, each getting what the earlier ones left, and the one the
cap cuts closes the turn with `"token_limit"`, as under `"yield"`. A reply
that goes on with an open turn (`reply()` without text, after the results
under `"yield"`) is a request of its own and has a cap of its own. The
`Config` and `reply` docstrings and the README say this in one sentence. The
README's row for `"yield"` also lost "as under a registry": under a registry
`max_calls` bounds the tries after a call that cannot be read, under
`"yield"` only the room left does.

### Thread counts without a topology, and under a CPU quota

Hardening of "Unset thread counts follow the processors"; which threads
prefill takes by default was then the user's open decision, decided
afterwards (337.4A), see "Prefill on the physical cores".

Where `/sys/devices/system/cpu` does not say which core a thread runs on
(macOS, Windows, a container without `/sys`), every thread used to count as a
core, so decode ran on every logical processor, which the measurement above
shows to be the slower choice under load. The standard library has no count
of physical cores, so the fallback is the one llama.cpp uses where it cannot
read the topology either: every thread up to four, and half of them beyond.
On a processor without simultaneous multithreading that leaves half the cores
idle for decode; a `decode_threads` set by the caller still wins.

A Linux CPU quota now bounds both counts: `cpu.max` under `/sys/fs/cgroup`
(cgroup v2), else `cpu/cpu.cfs_quota_us` and `cpu/cpu.cfs_period_us` (v1).
The bound is the quota divided by the period, rounded down, at least one, as
the Rust standard library's `available_parallelism` counts it: llama.cpp's
workers spin while they wait, so a thread beyond the quota is throttled rather
than idle. Only the files at the top of `/sys/fs/cgroup` are read, which is
the process's own cgroup inside a container with a cgroup namespace; on a
host, the quota of a cgroup further down is not followed.

### A value without quotes is read against every branch of its schema

316.16 reads a value a call format writes without a type mark by the type its
parameter declares, but only through a plain `type` field. A string parameter
declared as `anyOf` a string or null, as pydantic writes an optional string,
or through a `$ref`, still turned `01234` into `1234`. Now the types are
collected from the schema and from every schema its `anyOf`, `oneOf` and
`$ref` lead to, a `$ref` being followed where it points into the parameters'
own schema (`#/$defs/...`, `#/definitions/...`, or any other path in it), and
the rule stays as it was: the text is kept where some branch asks for a string
and none for the literal it spells. An `enum` of any branch that lists the
text keeps it too, and the parts of an object or a list are read against the
matching parts of every branch. A `$ref` that leads back to a schema already
on the path is not followed again, so a recursive schema ends. A `$ref` that
points outside the schema contributes no type: with no branch asking for a
string the value is the literal it spells, as before. `allOf` is not read;
the schemas the registry writes use none of these keywords.

### Closing a turn for want of room is one action

`close_unanswered` takes out the results written for the newest calls, drops
the calls and publishes that before it settles the cache, outside an action.
Where the settle failed, `inject_tool_result` raised with the turn half closed:
the call no longer owed, its reply empty, against principle 5. It now runs as
one action, so a failure puts the conversation and the cache back with the
call still owed. The test fails the cache at each of its operations while a
result closes the turn, and checks that the conversation is as before and
goes on.

### A long result on a cache that cannot shift drops only the history it needs

Where the cache cannot shift, an eviction brings the conversation down to
what a short question alone takes plus half the room after it (266.3), so
that the cache is decoded again only every few turns. A result that needs
room drops the oldest exchanges the same way, the open turn kept whole. Where
the open turn alone takes more than that bound, which a result longer than
about half the room does, no eviction reaches it, and every exchange went:
on gemma-4, 24 of 24 history items at a context of 3,000, where a cache
that shifts dropped 4 for the same result. Now, for a result, where the system turn and the open turn
alone pass the usual bound, the bound is what they take plus half the room
after them: the history kept is well below the room the open turn leaves it.
Where the open turn stays under the usual bound, nothing changes. Measured on
the fake context, context 3,000, 12 padded exchanges, a result 100 tokens
longer than the room left: 2 to 14 of 24 items go on the 41 test templates
that call tools, instead of all 24 on most of them. The reply to the result is
then generated with the newest history in view.

What it costs: the history kept is decoded again once, after the cut, where
before only the open turn's question and call were (gemma-4: 579 tokens
instead of 100). The reply's own eviction afterwards is unchanged: the
conversation then passes the eviction threshold, and on a cache that cannot
shift that eviction may still drop everything, the long turn included, so
the history kept serves the reply to the result. The rolling memory holds:
over 8 turns, every fourth with a result longer than the room left, a cache
that cannot shift decodes more than one shifting does (by more than two
exchanges and 80 tokens) in at most a third of the turns, on every test
template that calls tools, for a cache that cuts and for one that only
restores saved states.

Found while testing, not changed: on a cache that can neither shift nor cut,
a template that writes the declarations before the newest question
(mistral-small-3.2, ministral-2410) decodes every tool turn of the
conversation again on each turn (5 turns: 321, 406, 503, 600, 697 tokens,
against 214 each on a cache that shifts; plain exchanges cost 236 each). A
tool turn rests the cache after the call and after each result, and the two
newest saved states then no longer include the one saved before the
declarations, so the reply's settling goes back to the state pinned after the
system turn. The test marked those two cases as expected to fail; fixed
afterwards (337.11A), see "The state saved before moved declarations is kept".

### A history that ends on an owed call is decoded only as far as the results leave it

Follows "While a call is owed, the cache ends with its turn", whose "Not
changed" note this replaces. `begin` with a history that ends on a call
still owed a result decoded the conversation as the template renders it
without the result. qwen-3 and qwen-3-hermes then write an empty thought
before the call, since the call is the last message, and drop it once a
result follows; mistral-small-3.2 leaves out the declarations before the
question until a result follows. The result then took back 50 to 66 tokens,
by a cut or by restoring a saved state, and decoded them again (6 of 148
cases: each tool template, each reasoning mode, a cache that shifts and one
that cannot). Where some results of the round were in the history and one was
still owed, it was worse: the history ended on a result, so `begin` decoded
the generation prompt after it, which the next result took back, on most of
the templates that call tools.

Now `begin` with a call owed decodes the render only as far as it agrees
with the render that has a stand-in result for each owed call, to the last
token both share. The results then only append: the rest of the call's turn,
each result and the generation prompt are decoded once, with the last result,
as after a live call. The end-of-test check accepts a cache that holds an
owed call and no kept turn only where it ends exactly there, and its own test
shows that it fails on such a cache one token off. Measured with the same 148
cases and with one result of two already in the history, on every test
template that calls tools in each reasoning mode: no cut, no restore, and
every token decoded after `begin` is one the cache keeps.

## 2026-11-12: The user's decisions after the second review

### Prefill on the physical cores

337.4A. A `prefill_threads` left `None` is now as many as `decode_threads`,
and so one per physical core where that is not set either; it no longer takes
every thread the process may run on. The measurement under "Unset thread
counts follow the processors" decided it: on an idle machine every thread
prefilled about as fast as the physical cores (335.6 against 327.1 tok/s for
gemma-3-1b), and on a busy one it fell to 50 to 70 tok/s in 9 of 10 runs,
under half of the old 4 threads, while the physical cores stayed at 1.2 to
1.9 times the old speed; a load on every thread took up to 8.6 s instead of
1.0 s. On a processor with one thread per core the two are the same. The
count of usable threads is no longer needed, so `_physical_cores` returns the
cores alone.

### At most three unreadable calls in a row

337.5A. Retries after a call the library cannot read were bounded only by the
reply cap, and with a registry by `max_calls` where it is set. Without either,
a model that never wrote a readable call tried until the context was full: 59
tries and 1,534 tokens at a context of 8,192 on the stand-in, after which the
question was taken back. Now one reply reads at most
`MAX_UNREADABLE_CALLS_IN_A_ROW` (3) such calls in a row under every policy:
the third is answered with `do not call again`, and the generation after it
cannot open a call, so the model answers in words. A readable call in between
starts the count again. The constant is the wrapper's alone; nothing else
reads it.

### A single stop text is one stop text

337.9A. `reply(stop="END")` used to be split into its letters, so the answer
stopped at the first "E", "N" or "D"; the review fixes then refused a `str`.
Now a single `str` is taken as one stop text, as most chat interfaces take it,
and a list still gives several. Anything else is refused before the question
is written, as before.

### The CPU quota is read along the process's own control groups

The quota was read only at the top of `/sys/fs/cgroup`, which is the
process's own group inside a container but the root group on a host, where
no such file exists; a quota on the service or scope the process runs in, or
on a group above it, was missed (found by the review of this round's fixes).
Now the groups the process is in are taken from `/proc/self/cgroup`, every
group from each of them up to the root is read, cgroup v2's `cpu.max` and
v1's `cpu.cfs_quota_us` with `cpu.cfs_period_us`, and the tightest quota
bounds the cores. Where `/proc/self/cgroup` cannot be read, the top group
alone is read, as before.

### A call that spends the reply cap is not run

With a registry, a call whose own tokens used up the rest of
`max_reply_tokens` was still run and its result decoded, after which a
generation of 0 tokens closed the turn with empty text (found by the review of
this round's fixes: a cap of 27 tokens, the call's own 27). The tool's side
effects and the prefill went to a result the model never reads. Now such a
call is not run: the turn closes as one the reply cap cuts does, and the
reply ends with `token_limit`.

### The state saved before moved declarations is kept

337.11A, and the review's G2: the break of the rolling memory recorded under
"Found while testing, not changed". A cache that can neither shift nor cut
keeps the two newest states it rested in and one pinned after the system
turn. Where the template moves its declarations to the newest question, the
library rests the cache where the declarations begin, before they are decoded
again; a tool turn's own rests (after the call, after each result) then
pushed that state out, so the next question went back to the pinned state and
decoded the whole history again. Now `CacheSync.hold` keeps the state it
rested in at such a cut as a third saved state, for as long as the tokens
before it stay; a hold that only appends does not replace it. Measured on the
stand-in, ministral-2410 with the weather tool, a cache that can neither shift
nor cut: 1,387 tokens on the first turn, then 389 on each of the next five,
where every turn decoded the whole conversation before (1,535, 1,683, 1,831,
1,979). It costs one more saved copy of the model's state in RAM, only for
such a cache; caches that cut or shift save no states at all. The two cases
are no longer marked as expected to fail.

### A withheld call that opens with plain text ends the reply

337.10A. Where calls are not read (under `"withhold"`, in the rounds after a
refused call, and in a registry's last round), the token that opens a call
is never drawn. A template whose call opens with plain text (llama-3.2,
minicpm-5 and deepseek-v4.1 among the test templates) has no such token, so a
call the model wrote anyway reached the caller and the history as raw text,
for example `{"name": "weather", ...}`, against principle 7. Now
`ResolvedTemplate.call_boundary` keeps that opening as a mark there, and the
generation ends where such a call opens, as it ends at a turn's end written as
text: what the model wrote before it stays, the opening's tokens leave the
cache, and the template's closing is written whole. A reply that opens with a
call is therefore empty. A mention of the opening in prose is not a call, as
before: a plain-text opening counts only at the reply's start or right after
a tag.

## 2026-11-13: The user's decisions on the review of one-model cases (345.21A)

### Calls are read in the one shape their template writes

348.4A, review findings C7 to C10. Every call used to be tried as JSON first,
once or twice, under the keys `name` or `tool_name` and `arguments` or
`parameters` ("both are in the wild"), with arguments also accepted as JSON
text ("some models"), and only then read by the template's own delimiters.
That cost every non-JSON template one or two failed decodes per call (about 8
to 20% of reading a call on the stand-in) and wrote two families' key names
into the core. Now the load renders a probe call and decodes the JSON object
or array around the function's name; where it finds one, the keys the
template writes are kept: the one naming the function, the one holding the
arguments, and any others. Such a template's calls are read as JSON alone, by
those keys; an object with a key the template does not write is not a call.
Any other template's calls are read by its delimiters alone. Of the 52 test
templates, cohere-2 (`tool_name`, `parameters`), llama-3.2 (`name`,
`parameters`), ministral-2410, qwen-2.5, qwen-3 and qwen-3-hermes (`name`,
`arguments`) write JSON. Arguments written as a JSON object after the name
are read so only where an add-on file's call says so (`arguments_json`),
which only gpt-oss's does. Replaces the recorded rule "a body that is JSON
names itself": a model that writes JSON where its template does not, or under
other keys, is now answered that its call could not be read, instead of being
understood. The tests' JSON batches run on an invented template that writes a
turn's calls as one JSON array (`JSON_TOOLS`).

### A reply's header is read once at load, at tag edges

345.8, review findings C5 and C6. The header a channel template writes before
a reply's answer, naming its reader, was worked out for every generation from
the answer's and the thought's openings, and it could end inside a tag: on
cohere-2 the shared start of `<|START_TEXT|>` and the thought's opening was
`<|START_T`, on inkling `<|content_t`, so any reply opening with another tag
of that shape lost the tag from its answer. Now `ResolvedTemplate.reply_header`
works it out once for each reasoning mode, and neither of its ends may divide
one of the model's tags: an end that would is narrowed to the tags whole, so
cohere-2 and inkling read no header at all, while gpt-oss keeps
`<|channel|>`...`<|message|>` and muse-glimmer `to=`...`<|message|>`. The
other check C5 named, whether a thought closes with the token that ends a
turn, is one string comparison per reply; it is left where it is, since
moving it saves nothing measurable.

### Call ids keep nine characters

345.11, review finding B12. The library numbers the calls it writes as
`call` and a number, nine letters and digits in all, because the Mistral test
templates refuse any other id: ministral-2410 raises unless an id is exactly
9 alphanumeric characters, and mistral-small-3.2 refuses a short one such as
`c1`. Nine characters suit every other test template too, so the length is
the library's one rule for every model and needs no add-on key. The number was
written in five decimal digits, so the 100,000th call of a conversation
(`call100000`) was ten characters, which ministral-2410 refused. It is now
written in five base-36 digits (`call0000a` after `call00009`), nine
characters for the first 36**5 calls, about 60 million; ids read from a
history are taken back the same way, so every new id stays unused.

### A user tag after a past reply's own closing is found by the model's tags

345.12, review finding A7. Where a past reply closes with another tag than a
newest one (gpt-oss, hunyuan-3, hunyuan-memory among the test templates), the
user tag that follows was found as the longest shared end starting at a `<`
or `[`. It now starts where one of the model's own tags starts, from the
tokenizer's tag list the load already reads, so a family whose tags are
written otherwise is read the same way. The general rule that a shared
beginning stops where a tag ends at `>` or `]` or a word at whitespace
(`shared_tags_length`) is left as it is: it is the library's rule for every
template, not one model's, and replacing it would carry the tag list into
nine places for no measured gain.

### Only the shared reasoning switches are read from a template

348.2A and 345.16, review findings A1, A2, D1 and D2. The switch search
read `enable_thinking`, then `reasoning_effort` with `"none"` as off and,
where the default did not reason, `"high"` as on, then any variable whose name
held "think" or "reason". None of the 52 test templates needed the name search
(it cost up to 30 renders and 7 ms per load on 7 of them), and `"high"` was
the vocabulary of two families, hunyuan-3 and hunyuan-memory, written into the
core. Now the template is read for `enable_thinking`, and for a
`reasoning_effort` whose `"none"` turns reasoning off (the user's 248.4A,
unchanged); everything else is named in the add-on file's `reasoning_on`, and
the two Hunyuan add-on files (`llama_chat/addons/hunyuan-3.json`,
`hunyuan-memory.json`) name `{"reasoning_effort": "high"}`, which renders the
same prompts as before. A new load check guards the case the add-on is
missing for: a prompt with reasoning on that already closes the thought
(hunyuan-3 without its file writes `<think:opensource></think:opensource>`)
means the model cannot reason, so asking for reasoning is refused with a
pointer to `reasoning_on` instead of giving an answer that never thought.

### An add-on file's call format wins over the template's

348.3A, review finding A5, and the contract's invariant 2 changed with it: what
the template states, the template decides, except a call format the add-on
file names, which wins. `ResolvedTemplate._resolve_calls` takes the file's
call first, wherever it is given, tag-marked templates included, and checks
its closing against the template as before. Not done as proposed: removing the
rule that refuses a plain-text call opening found in the reply's own header
(`_opens_in_reply_header`). Measured: without it, gpt-oss without its add-on
file read its call texts from the template and could call tools, although
such a model may write the header in another order than its template, which
is guessing; with the rule it stays refused, naming the add-on file as the
remedy, as principle 2 asks. The rule now runs only where no add-on call is
given, so it costs a model with one nothing.

### Each test template lists its own tags

345.15, review finding D7. The tests' stand-in tokenizer read every text of a
tag's shape as a tag, by one pattern with twelve alternatives, one per tag
shape of a family. It thus took `<IMPORTANT>`, plain text in the tool
instructions of eight test templates, for a tag that no real vocabulary holds
(the real Qwen and Nemotron tokenizers split it into three to five ordinary
tokens). Now each test template's file states `tags`, the model's own tags as
the stand-in reads them: those its template writes, the opening forms of its
closing tags, and the tokens its file names besides (its end token, the tokens
that end a generation). The invented templates list theirs, and the stand-in
reads only listed tags, so a tag-shaped text the model does not know is
characters, as a real tokenizer reads it. `FakeContext(extra_tags=...)` adds
tags of a vocabulary that the template never writes, which one test needs. A
template a single test writes for itself lists none, and is still read by the
pattern. The name check (`tests/test_neutral.py`) reads the listed tags.

### A template whose reply cannot be split off is refused

345.17, review finding A8. Where neither a past reply nor a newest one could
be split off a render, the turn tags were guessed: the user's tags were
reused as the assistant's, "so generation still has an open and close pair".
None of the 52 test templates took that path, and a guess is what principle
2 rules out, so such a template now yields no fragments and is refused at
load as one without a usable template. The guard in `extract_tool_fragments`
against an assistant tag equal to the user's stays: a template may write the
two alike on its own.

### Unreached code removed where nothing needs it

345.18, review findings C14 and D9. A number given for a text parameter was
turned into its text before the tool ran ("a format that marks no strings
reads a number written as text as a number"); that reason no longer held,
since a value written without a type mark is read by the declared type when
the call is parsed. A number where text is asked is now the model's mistake,
answered as `wrong_arguments` so the model can write the call again. A test
that skipped where a typed tag was refused, which no test template takes, now
fails there instead. Kept, with their reasons: the tokenizing step for a tag
the tokenizer finds but cannot place (finding A19) hands the text to the
model's own tokenizer, which the load check verifies, so it guesses nothing;
the cache's and the rendering's unreached fallbacks (finding B11) need a test
template that takes them, or a proof that they cannot run, before they go.

### What every model no longer pays for

345.1 to 345.6, review findings B1 to B5, B9, A3, A4 and A15. Each moves work
that only some templates need to a check made once, at load or when a
conversation begins, so that the other templates skip it.

- **A repeated question (B2).** The fix of 316.5 rendered the whole
  conversation before any question that repeated an earlier message, for
  every model, although only ministral-2410 with tools writes such a question
  otherwise (it compares messages to find the newest question). Now one probe
  pair when a conversation begins tells whether the template writes a
  repeated question as any other; only where it does not is the repeat
  rendered whole. Measured on the stand-in with 20 exchanges: the first word
  after a repeated "Thanks!" 1.15 to 1.21 ms before, 0.38 ms now, the same as
  a new question; the review measured 16.6 ms against 1.2 ms on gemma-4 at 41
  messages.
- **The clock (B1, A15).** Every action rendered the conversation's opening
  again to see whether the date had moved, and a failed split was rendered a
  second time in case the clock moved between two renders. Both now run only
  where the template's text names `strftime_now` (`ResolvedTemplate.reads_clock`),
  6 of the 52 test templates; a template whose text cannot be read is taken
  to read the clock. A text search suffices: a template that does not name
  the function cannot call it. Reading the names through the template parser
  instead cost a compile per load, which tripled the test time.
- **A question's placement check (B3).** Each question rendered a probe
  conversation with the stand-in and one with the question; the stand-in's
  render is now made once per conversation start (`splice_probe`).
- **Saved states (B4, B5).** The opening's length for the pinned state, and a
  state saved before a reply's prompt at `begin`, are worked out only where
  the cache goes back to saved states, a cache that can neither shift nor cut
  (`CacheSync.keeps_saved_states`); before, every begin and declaration paid
  two full renders and tokenizations for them, 0.6 to 1.2 ms on the stand-in.
- **Windows (B9).** Where the declarations move with each newest question
  (ministral-2410 and mistral-small-3.2 with tools), the window check passed
  at load but failed at every reply, which then rendered whole after doing
  the window's work; such a conversation now renders no windows.
- **Load checks worked out again (A3, A4).** What a template writes after a
  conversation's last message was worked out by every load check that read
  renders as probes, and whether it declares tools in the system turn up to
  three times; each is now worked out once for each set of template
  variables. Renders at load over the 52 test templates: 2,448 before, 1,964
  now.

### A text placed by the gap between tags is found in polynomial time

Review finding H1 and the question of H8. Where a caller's text is written
otherwise than in any of its forms (minicpm-5 wraps an argument holding `<`,
`&` or a newline in CDATA, say), it is placed as what lies before the
template's next text, where exactly one such place leaves the rest of the
render matching. That search tried the rest of the render again for every
candidate place of every such text, so its cost grew about 3.5 times per
text: on the stand-in, a minicpm-5 history of 11 such past calls took 2.9 s
to begin and 3.6 s per reply, and every reply splits the history again. Now
`_walk` works out once, for each part and position, in how many ways (none,
one, or more) the rest matches, so the search is polynomial: 0.02 s at 11
calls, 0.2 s at 40. A text is placed only where the whole render matches in
exactly one way, which is stricter than before, where a candidate whose rest
matched in two ways was only skipped.

Of the JSON string forms a text is tried in, only `json.dumps(text,
ensure_ascii=False)` is left, which is how the renderer's `tojson` writes it.
Measured over the whole suite: the ASCII-escaped form matched only where it
equals that one, and the form with `<`, `>`, `&` and `'` escaped as `<`
and so on matched nothing, since no renderer writes it. The form that is left
is not needed for correctness, since the gap search finds the same span, but
for speed: without it a qwen-3 history of 9 such calls took 723 ms on the
search before this change.

### A round of results keeps its turn at every length

Review finding H2, a regression of the fix for G1. A result was written whole
wherever it fit, with no room kept for the results still owed after it, so a
first result that just fit (253 to 256 characters at a context of 400 on the
stand-in) left a one-character second result no room even cut to nothing,
and the turn closed, its results dropped. Now a result is written whole only
where each result still owed after it fits empty beside it, and where a later
result fits not even cut to nothing, the newest result written whole earlier
in the round is cut to make room for it, its call then listed as cut. Measured
over every first result from 200 to 299 characters: the turn stays open at
each length, where before it closed at four lengths, and at one after the
first part of this fix.
