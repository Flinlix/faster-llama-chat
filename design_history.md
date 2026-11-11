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
(`-1`) instead of to `total`. The triggers, the cache operations and the
choice between them still hold.

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
end-of-text token, belongs to no turn; every probe renders with reasoning off;
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
  content; a JSON body is read directly, under `name` or `tool_name` and
  `arguments` or `parameters`. Covered: JSON objects and arrays,
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
  call unrecognized rather than withheld.
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
  text; such a tag is held back while a call may follow, and is never shown.
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
conversation continues from theirs; no action clears them. The cache length
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

**The engine context warms itself.** `CacheContext.warmup` and the wrapper's
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

**264.4: A template of its own for tools.** At the user's choice (B). A GGUF
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

**270.13: an action on the thread whose reply is being read.** A reply holds
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
