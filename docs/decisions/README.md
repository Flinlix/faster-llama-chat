# Decision records

One record per decision that stands, foundations first. Each states the
decision as it stands now, its context, its consequences and the alternatives
it rejected. The dated log the records were written from is kept as
`archive/design_history.md`.

| No. | Decision | Summary |
|---|---|---|
| [0001](0001-the-cache-holds-the-templates-render.md) | The cache holds the template's render of the conversation, decoded only where it differs | The template renders the messages; each action decodes from the first token that differs. |
| [0002](0002-no-model-is-special.md) | No model is special | What one model needs lives in its add-on file or test template; a test keeps every other file free of model names. |
| [0003](0003-only-a-template-that-needs-something-pays-for-it.md) | Only a template that needs something pays for it | Features of some templates are read once at load or at begin; a trace test lists what a plain reply may run. |
| [0004](0004-the-library-renders-the-template.md) | The library renders the template; a context gives only its text | Jinja's immutable sandbox with the binding's settings, the template's refusal reason passed on, bounded memos. |
| [0005](0005-every-caller-text-is-placed-by-a-marker-render.md) | Every caller text is placed by a marker render | A second render with markers places each text exactly, so a typed tag stays text; otherwise refused. |
| [0006](0006-a-template-is-read-once-at-load.md) | A template is read once at load into one record | One properties record; tags are the model's own tokens; refused where reading would need a guess. |
| [0007](0007-templates-that-read-the-clock-follow-the-date.md) | A template that writes the date follows the clock | Only templates naming `strftime_now`; begin measures room on the longest date. |
| [0008](0008-a-template-for-tools-is-chosen-at-begin.md) | A model's template for tools writes every conversation begun with tools | Chosen at begin; a broken one refuses only tools; capabilities per kind of conversation. |
| [0009](0009-reasoning-is-switched-as-the-template-switches-it.md) | Reasoning is switched by the template's shared variables or the add-on file | `enable_thinking` or `reasoning_effort` "none"; `reasoning_on`; `reasoning` and `reasoning_level`. |
| [0010](0010-the-caller-sees-only-the-answer.md) | The caller sees only the answer the template marks as one | Thought and header filters by the model's own tags; `Reply.thought`; a thought cap. |
| [0011](0011-add-on-files.md) | What a model needs beyond its template comes from one JSON add-on file | Next to the GGUF naming the template, else shipped and matched by fingerprint; checked whole at load. |
| [0012](0012-model-sampling-value-by-value.md) | A model's sampling is merged value by value | Caller, add-on file, GGUF, then llama.cpp's default; penalties applied to the likeliest tokens alone. |
| [0013](0013-configuration-is-checked-where-it-is-made.md) | The configuration is frozen and checked where it is made | Types from annotations; no mutable setting; no reply cap by default. |
| [0014](0014-engine-context-settings.md) | The engine context follows llama.cpp's placement, uses the unified cache, and frees itself | `kv_unified`, cache on the layer's device, automatic flash attention, `gpu_layers=0` means no GPU, a finalizer. |
| [0015](0015-the-engine-keeps-its-own-micro-batch.md) | The engine keeps its own micro-batch unless the caller sets one | `micro_batch_tokens` trades device memory for a faster long prefill. |
| [0016](0016-the-warm-up-at-load.md) | Loading warms the engine | One micro-batch where layers run on a GPU, then two tokens and one. |
| [0017](0017-unset-thread-counts-follow-the-physical-cores.md) | Unset thread counts are one per physical core | Affinity, performance cores and CPU quotas respected; prefill follows decode. |
| [0018](0018-engine-log-output-through-python-logging.md) | The engine's log output goes through Python's logging | One callback maps llama.cpp's levels; each level means one thing across the library. |
| [0019](0019-engine-binding-floor-and-packaging.md) | llama-cpp-python 0.3.23 is the floor; the package installs as llama-cpp-conversation | One current name per symbol; the penalties signature read; Python 3.11. |
| [0020](0020-the-cache-context-protocol.md) | A cache context is one protocol, checked once where it is adopted | Two halves, an engine-neutral generation loop, token ids in a fixed range, load probes. |
| [0021](0021-one-owner-brings-the-cache-to-its-target.md) | One object brings the cache to a target built from the render alone | `CacheSync.hold`, versions, repair, and a left-out prompt stretch shifted out. |
| [0022](0022-eviction-drops-whole-oldest-exchanges.md) | Eviction drops whole oldest exchanges, never the system turn | Two triggers, shift edit and K-shift ordering, strategy chosen at load, count searched. |
| [0023](0023-without-a-shift-an-eviction-drops-well-below-the-threshold.md) | Where the cache cannot shift, an eviction drops well below the threshold | A fixed fraction of 0.5 of the room after a short question, so the cache is decoded again only every few turns. |
| [0024](0024-saved-states-on-a-cache-that-cannot-cut.md) | A cache that can neither shift nor cut goes back to a few saved states | Where states are kept, none before the first word, buffers reused. |
| [0025](0025-the-next-question-is-decoded-ahead-after-each-reply.md) | After each reply the conversation is decoded up to the next question | A stand-in for the question; a question decodes only its own tokens. |
| [0026](0026-renders-are-windows-of-the-newest-exchanges.md) | Actions render windows of the newest exchanges after an anchor | Used only where probes show the template writes them alike; render cost stays constant. |
| [0027](0027-the-work-after-a-reply-runs-after-its-stream-ends.md) | The work after a reply runs after its stream ends and gives way | A thread of its own; an early question answered first under conditions. |
| [0028](0028-a-turn-that-called-tools-keeps-its-generated-tokens.md) | A turn that called tools keeps the tokens it was generated as | Placed in the target in place of the template's version; refused where it cannot be placed. |
| [0029](0029-two-history-forms.md) | A history has two forms | Chat Completions messages, and beside them the generated turns with the model's fingerprint. |
| [0030](0030-a-begin-compares-with-the-cache.md) | A begin compares with the cache and is the one way to change settings | Decodes from the first difference; keeps the newest history that fits. |
| [0031](0031-conversations-switched-away-from-are-kept.md) | Caches of conversations switched away from are kept within a RAM bound | `kept_conversation_bytes`, 0 by default. |
| [0032](0032-every-question-gets-a-reply.md) | Every question gets a reply | Open turns closed, questions joined or cut, an answer guaranteed, timeouts on unread replies and tools. |
| [0033](0033-one-undo-path.md) | One undo path | One snapshot per generation and one state object put every failed, refused or cancelled action back. |
| [0034](0034-a-reply-is-recorded-as-the-template-writes-it.md) | A reply is recorded whole or not at all, as the template writes it | Answers holding thought tags, empty replies, refused replies, shortened replies. |
| [0035](0035-one-family-of-refusals.md) | Every refusal is one family; every other fault is an InternalError | `RefusalError` kinds with `field`; one entry point for public errors. |
| [0036](0036-one-report-of-what-an-action-changed.md) | Every action reports, and logs once, what it changed beyond what was asked | `Report` and `Reply` carry evictions, cuts, joins and closed turns. |
| [0037](0037-a-calls-format-is-recovered-from-the-templates-renders.md) | A call's format is recovered from the template's renders | Probes for tags, delimiters, ids and nesting; one shape per template; refused where it cannot be told. |
| [0038](0038-a-call-is-read-only-where-all-its-text-is-read.md) | Reading a call is its own module, and a call is readable only where all its text is read | Strict parsing; unmarked values read against the declared schema. |
| [0039](0039-tool-call-policies.md) | A request names what becomes of the model's calls | `refuse`, `yield`, `withhold` or a registry; the policy kept for a turn; withheld calls never drawn. |
| [0040](0040-every-call-is-answered-and-reported.md) | Every call is answered and reported | Failure wording, three unreadable calls in a row at most, call statuses. |
| [0041](0041-a-result-that-does-not-fit-evicts-then-is-cut-then-closes-the-turn.md) | A result too long for the room evicts, then is cut with a note, then closes the turn | Room kept for results still owed. |
| [0042](0042-call-ids.md) | A call keeps the model's id where new, else gets one of nine characters | `call` and five base-36 digits; never an id twice in one wrapper. |
| [0043](0043-tools-are-declared-as-specs-built-from-one-reading.md) | A tool is declared as a checked spec built from one reading of its signature | One decorator, one declaration form, tools on a thread with a timeout. |
| [0044](0044-actions-take-turns-and-readers-see-a-finished-copy.md) | Actions take turns under one lock; readers see a published copy | The reading thread cannot wait for itself; `close()` ends its replies first. |
| [0045](0045-one-way-to-do-each-thing.md) | The public interface has one way to do each thing | One read, one begin, no callback, internal types not exported. |
| [0046](0046-a-reply-streams-each-token-as-soon-as-it-can.md) | A reply streams each token as soon as nothing could still change it | Minimal hold-back, constant cost per token, text before decode on the processor. |
| [0047](0047-the-tests-try-to-break-the-rules.md) | The tests try to break the rules | Real templates on one stand-in, expectations from template files, checks after every test, replayed sequences. |
| [0048](0048-faults-that-show-themselves.md) | Faults show themselves | Loop variants, test time limits, `check_cache`, lint and types as tests, real-model runs. |
| [0049](0049-the-readme-keeps-to-what-a-caller-relies-on.md) | The README and the guides keep to what a caller relies on | A short README, a guide per task, an API reference from the docstrings; reasons live in these records. |
| [0050](0050-decisions-are-kept-as-one-record-each.md) | Each decision is one record; the dated log is an archive | Records updated with the code; the old log kept unchanged. |
| [0051](0051-releases-are-tagged-and-published-from-github.md) | A release is a version tag, released on GitHub | The tag must match `pyproject.toml`; the built package attached; `py.typed`. |
| [0052](0052-the-conversation-is-split-into-collaborators-by-job.md) | The conversation and the wrapper hand each job to a collaborator | Six modules of one job each; the actions, the undo path and the lock stay. |
| [0053](0053-dependencies-are-bounded-to-what-is-checked.md) | Dependencies are bounded to the releases the checks are run on | `jinja2` on the 3.1 line from 3.1.6; the tools below their next major version. |
| [0054](0054-what-is-public-and-what-a-version-promises.md) | What is public, and what a version number promises | A compatibility page; every public signature listed and compared by a test. |
| [0055](0055-measured-against-llama-cpp-pythons-own-chat.md) | Measured against llama-cpp-python's own chat interface | Later questions wait about half as long; the chat runs on where the plain one fills up. |
| [0056](0056-the-project-is-named-llama-cpp-conversation.md) | The project is named llama-cpp-conversation | One name for the repository, the distribution, the import and the site. |
| [0057](0057-the-site-is-built-with-sphinx.md) | The documentation site is built with Sphinx | Every warning an error; the README's sections included; the theme's support for the old tool ends in 2027. |
| [0058](0058-the-public-signatures-are-typed.md) | The public signatures are typed | Typed messages and conversation; a strictly checked caller in the tests. |
| [0059](0059-a-compact-cache-for-sliding-window-layers.md) | A compact cache for sliding-window layers, chosen by the caller | Much less memory; an eviction reads the kept history again after a reply; off by default. |
| [0060](0060-the-full-size-cache-is-not-checked.md) | The full-size cache keeps the gap llama.cpp leaves in its window layers; a kept conversation keeps every position | Nothing is read again for the gap; a kept conversation takes about twice the RAM. |
