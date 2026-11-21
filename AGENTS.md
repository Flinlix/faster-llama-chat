# llama-cpp-conversation

## Rules

llama-cpp-conversation keeps a chat model's conversation in its cache exactly
as the model's own template writes it, decodes only what changed, for any model
that ships a template, and refuses where it would have to guess. Each rule
names the tests that try to break it.

1. **The cache is the template's render, token for token.** Every model whose
   GGUF ships a chat template is encoded as that template renders it, without
   code particular to the model: the system prompt or its absence, past
   replies, the generation prompt in each reasoning mode, declared tools and a
   past turn that called tools. A turn that called tools during this
   conversation keeps the tokens it was generated as, except one whose tokens
   no longer fit the template's render after it was closed for lack of room.
   Where an action interrupts the work after a reply, the cache may hold
   less than the render until that action decodes the rest, before the model
   reads it. A family not yet covered gets a template in `tests/templates/`.
   Verified by:
   - `tests/test_conversations.py`: whole conversations on every test
     template, on a tokenizer of one token per character and on a merging
     one, hold exactly the template's own render before every reply, and a
     tag typed into any text stays text.
   - `tests/conftest.py`: after every test, the cache holds the template's
     render of the conversation, token for token, generated turns in their
     places; `tests/test_cache_check.py` shows that this check fails on a
     cache one token off or with a tag held as its letters.
   - `tests/conftest.py`: after every test, no turn kept as generated spells
     a tool result written after it.
   - `tests/test_real_templates.py` and the load check whenever a model
     loads.
2. **What cannot be recovered is refused, with its cause,** at load or where
   it is asked for, never approximated.
   Verified by:
   - `tests/test_real_templates.py`: every test template passes the load
     check, every property it reads is what the template's file states, and
     `begin` refuses exactly where `capabilities()` says it does not take
     something.
   - `tests/test_messages.py`: every form a history may not take is refused,
     naming where it stands.
3. **No model is special.** What a model needs beyond its chat template comes
   from a small JSON add-on file: the one next to the GGUF with its name, or
   one shipped in `llama_cpp_conversation/addons/` and matched by the template's
   fingerprint. It fills in only what the template cannot tell, except a call
   format it names, which wins. No file names a model, a family or a vendor,
   except the model's own files (a test template in `tests/templates/`, an
   add-on file in `llama_cpp_conversation/addons/`), the decision records in
   `docs/decisions/`, the README's install section and its list of supported
   models, and the install files (`install.sh`, `pyproject.toml`); llama.cpp
   and llama-cpp-python may be named in any file.
   Verified by:
   - `tests/test_model_sampling.py`, `tests/test_real_templates.py`: every
     shipped add-on file is read as a load reads it, for a template the tests
     run.
   - `tests/test_neutral.py`: no other file names a family or a publisher of
     the test templates or the add-on files, and the README's list of
     supported models is the table the test templates' files give.
4. **The memory rolls.** No action decodes more as the conversation grows,
   on any kind of cache; where a cache cannot shift, or the template writes
   the rest otherwise once the oldest exchange goes (as one that numbers its
   messages does), an eviction drops the conversation well below the
   threshold, so that it is decoded again only every few turns. A model that would need the whole conversation decoded
   again each turn is refused.
   Verified by:
   - `tests/test_conversations.py`: on every test template and every kind of
     cache, what each action decodes does not grow with the conversation, and
     a cache that cannot shift is decoded again only every few turns.
5. **Only a template that needs something pays for it.** What only some
   templates need is read once at load, or stated in the add-on file, and no
   question, reply or token of another template pays for it.
   Verified by:
   - `tests/test_wrapper.py`: a reply on a template without reasoning, tools
     or a clock runs only the functions listed as serving every template.
6. **A failed or refused action leaves the conversation and the cache as they
   were, and usable.**
   Verified by:
   - `tests/test_failures.py`: every action, failing at any point, on every
     kind of cache, leaves them as they were, a failed reply keeping only
     what it had produced.
7. **Every call the model makes is answered and reported**, and its turn
   closes, or waits for results until a new question closes it; where no
   room is left to answer, the turn closes on what the model wrote, each call
   reported.
   Verified by `tests/test_tool_loop.py` and `tests/test_conversations.py`.
8. **The caller sees only the answer the template marks as one.**
   Verified by `tests/test_reasoning.py` and `tests/test_generation.py`.
9. **Safe from any thread:** actions take turns, readers see a finished copy,
   a reply can be cancelled from elsewhere.
   Verified by `tests/test_threads.py` and `tests/test_tool_loop.py`.
10. **Beginning again from `conversation()` gives the same conversation.**
    `begin(**chat.conversation())` holds the same messages, the system message
    among them, tools and reasoning setting, each message unchanged and in its
    place, a call without a result included, and `conversation()` then
    returns what it returned before. Where the conversation holds more than a
    `begin` keeps, it begins with the newest part a `begin` keeps.
    Verified by:
    - `tests/test_conversations.py`: on every test template, with live calls,
      unanswered calls, evictions, restarts and later declarations.
    - `tests/test_sequences.py`: random sequences of actions on invented and
      real templates.
    - `tests/conftest.py`: after every test, what `conversation()` gives is
      taken by `begin`, and the conversation is its Chat Completions messages
      with each generated turn put back.
11. **Every question gets a reply.** After load, every question gets a
    reply; every error is a library error that names the argument and the
    cause, and it changes nothing; no state is reachable in which the model
    is ready but cannot answer. After any action, a question to a begun
    conversation is answered: an open turn is closed first, and a question
    after an unanswered one joins it where the template writes no two in a
    row, and where the two joined do not fit, the unanswered one goes and the
    question is asked alone. A question too long for the room the system
    prompt leaves under the eviction threshold keeps as much of its end as
    fits, possibly none of it, its beginning cut, with a warning logged. A
    reply holds at least one token of answer or a call, and a thought is
    closed in time to leave the answer its room. A question is refused only
    where the template does not write that question as its text; the refusal
    leaves the next question free.
    Verified by:
    - `tests/test_failures.py`: every state the failure sweep reaches is
      followed at once by a question, which is answered.
    - `tests/test_conversations.py`: on every test template, a question on an
      open turn, after an unanswered one, empty, asked twice, or filling
      all the room a system prompt leaves.
    - `tests/test_sequences.py`: random sequences of actions with hostile
      input, after each of which every error is a library error, a refused
      action changed nothing, and the next question gets a reply.
    - `tests/test_public_errors.py`: every public action raises only the
      library's errors, a fault inside it raised as `InternalError`.
    - `tests/conftest.py`: every error the library makes during a test says
      what went wrong.
12. **Every loop ends.** Each pass of a loop makes a measure smaller, or the
    loop takes the steps of an iterator; a pass that makes none raises
    `InternalError` naming the loop, instead of running on.
    Verified by `tests/test_loops.py`: every `while` loop of the library
    gives its measure at the start of each pass, or is one of the forms that
    end by construction.

## Tests

In an environment where this package is installed editable with its `test`,
`dev` and `docs` extras (`./install.sh --dev`, which also switches on
`hooks/pre-push`):

```
python -m pytest -q -n auto   # everything needing no model: rules, lint, types, prose, contract
python -m pytest -q -m slow   # many new random sequences of actions
python -m pytest -q -m gpu    # real model files named in LLAMA_CPP_CONVERSATION_TEST_MODELS
LLAMA_CPP_CONVERSATION_CHECK_EVERY_ACTION=1 python -m pytest -q -n auto   # every wrapper checked against a fresh begin
python -m sphinx -W --keep-going -b dirhtml docs site   # the documentation site
```

## Decisions

Why the code is as it is lives in `docs/decisions/`, one record per decision,
listed in its `README.md`. A decision made or revised gets a new record, or
its record is updated, in the same commit as the code. The code points to a
record by its file name. `docs/decisions/archive/design_history.md` is the
log the records were written from; it is no longer added to.
