"""Whole conversations through real chat templates: the cache always holds the template's render of what was sent.

Before every reply, the cache must read exactly as the template renders the
conversation the test itself sent, from an exchange on and with the generation
prompt after it, a turn that called tools held as it was generated. A
conversation the template refuses to render must be refused, never encoded.
Each check runs through a real template of every family in
``tests/templates/``, and through a small template of this file's own whose
renders fail the checks the shortcuts rest on, so that the paths an unknown
template takes are driven too.
"""

import datetime as real_datetime
import json
import math
from pathlib import Path

import pytest

from llama_chat import ChatWrapper, Config, ContextOverflowError
from llama_chat.reply_text import begins_in_thought
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

from .template_contexts import ANSWERS, QUESTIONS, WEATHER, ConversationContext, ScriptedCall

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_FIXTURES = {path.stem: json.loads(path.read_text()) for path in _TEMPLATES}
# A template of this file's own, with the conversation's length in its generation
# prompt, so that no window of the conversation stands for the whole render. A
# template that wrote anything of a message's text, its length say, beyond the
# text itself would be refused.
_FALLBACK = {
    "source": "written for this test", "eos_token": "<|end|>", "marks_thought": False,
    "can_reason": False, "calls_tools": False,
    "chat_template": "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}<|end|>"
                     "{%- endfor -%}{%- if add_generation_prompt -%}"
                     "<|assistant_{{ messages|length }}|>{%- endif -%}"}
_EVERY = sorted(_FIXTURES) + ["fallback"]
_TOOL_TEMPLATES = [stem for stem in sorted(_FIXTURES) if _FIXTURES[stem]["calls_tools"]]


def _fixture(stem):
    return _FALLBACK if stem == "fallback" else _FIXTURES[stem]


# ----- a conversation, and the render of what was sent ---------------------------
class _Run:
    """A conversation on one template, and what the test sent it, recorded by the test.

    ``seen`` holds the cache's text at the start of each reply, ``decoded``
    how many tokens had been decoded by then, and ``renders`` how many renders
    had been made.
    """

    def __init__(self, stem, context_size=4096, can_shift=True, replies=None, reasoning=False):
        self.fixture = _fixture(stem)
        probe = ChatWrapper(Config(context_size=4096), context=ConversationContext(self.fixture, []))
        self.answers = list(ANSWERS if replies is None else replies)
        self.context = ConversationContext(self.fixture, _as_generated(probe, self.answers, reasoning),
                                           can_shift=can_shift)
        self.chat = ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.75,
                                       min_reply_tokens=8), context=self.context)
        self.system = "Be brief." if self.chat.can_take_system_prompt else ""
        self.reasoning = reasoning
        self.seen, self.seen_tokens, self.decoded, self.renders = [], [], [], []
        self.context.before_reply = self._before_reply

    def _before_reply(self, text):
        self.seen.append(text)
        self.seen_tokens.append(list(self.context.cache))
        self.decoded.append(self.context.decoded)
        self.renders.append(len(self.context.rendered))

    def begin(self, history=(), tools=None):
        return self.chat.begin(self.system, list(history), tools=tools, reasoning=self.reasoning)

    def render(self, messages, tools=None):
        system = [{"role": "system", "content": self.system}] if self.system else []
        return self.chat._template.render(
            system + messages, [spec.as_declaration() for spec in tools] if tools else None,
            generation_prompt=True, reasoning=self.reasoning)


def _as_generated(probe, answers, reasoning):
    """Each answer as the model generates it: after a thought it closes, where the generation prompt opens one."""
    tags = probe._template.reasoning
    if tags is None or (reasoning and not probe.can_reason):
        return list(answers)
    opens = begins_in_thought(tags, probe._template.formatter(reasoning).generation_prompt())
    if reasoning:
        return [("" if opens else tags.thought_open) + _THOUGHT + tags.thought_close + answer
                if isinstance(answer, str) else answer for answer in answers]
    return [tags.thought_close + answer if opens and isinstance(answer, str) else answer
            for answer in answers]


_THOUGHT = "Erst denken."


def _exchanges(questions, answers, thought=None):
    return [[{"role": "user", "content": q},
             {"role": "assistant", "content": a, **({"reasoning_content": thought} if thought else {})}]
            for q, a in zip(questions, answers)]


def _assert_each_reply_followed_what_was_sent(run, questions, forms=(None,), first_reply=0):
    """Before reply ``k``: the render of the sent exchanges from one on, then question ``k``, where that exchange only moves forward."""
    oldest = 0
    for k, cached in enumerate(run.seen):
        if k < first_reply:
            continue
        found = None
        for first in range(oldest, k + 1):
            for thought in forms:
                sent = sum(_exchanges(questions[first:k], run.answers[first:k], thought), [])
                if cached == run.render(sent + [{"role": "user", "content": questions[k]}]):
                    found = first
            if found is not None:
                break
        assert found is not None, (k, cached)
        oldest = found


def _exchange_length(run, question, answer):
    """About how many tokens one exchange takes in the middle of a conversation."""
    two = sum(_exchanges([question] * 2, [answer] * 2), [])
    four = sum(_exchanges([question] * 4, [answer] * 4), [])
    tokens = lambda messages: len(run.context.tokenize(run.render(messages)))   # noqa: E731
    return (tokens(four) - tokens(two)) // 2


# ----- every reply starts from the render ---------------------------------------
@pytest.mark.parametrize("stem", _EVERY)
def test_every_reply_starts_from_the_templates_render_of_what_was_sent(stem):
    run = _Run(stem)
    run.begin()
    for question in QUESTIONS[:4]:
        run.chat.reply(question).finish()
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS)


@pytest.mark.parametrize("injected", [False, True], ids=["asked", "injected-then-answered"])
@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_question_waits_only_for_its_own_tokens(stem, injected):
    run = _Run(stem)
    run.begin()
    for question in QUESTIONS[:3]:      # more than a whole render could hide under the bound
        run.chat.reply(question).finish()
    waited = run.context.decoded
    rendered = len(run.context.rendered)
    if injected:
        run.chat.inject(QUESTIONS[3])
        run.chat.reply().finish()
    else:
        run.chat.reply(QUESTIONS[3]).finish()
    assert run.chat._conversation._splices
    # The question, the run of template text it joins, and what the template writes after it.
    assert run.decoded[-1] - waited <= (len(QUESTIONS[3]) + len(run.chat._conversation.prompt_text)
                                        + 40)
    # Nothing but small fixed probes is rendered while the question waits.
    assert max(run.context.rendered[rendered:run.renders[-1]]) <= 4


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_question_injected_over_the_threshold_waits_only_for_itself_and_evicts_after_its_reply(stem):
    question = "Some background you need not repeat. " * 6 + QUESTIONS[3]
    probe = _Run(stem)
    probe.begin()
    for asked in QUESTIONS[:3]:
        probe.chat.reply(asked).finish()
    held, length = len(probe.context.cache), len(probe.context.tokenize(question))
    # The threshold half way into the question, the room for the reply past it.
    run = _Run(stem)
    run.chat.close()
    size = held + length + 80
    run.chat = ChatWrapper(Config(context_size=size, min_reply_tokens=8,
                                  evict_above_fraction=(held + length // 2) / size),
                           context=run.context)
    run.begin()
    for asked in QUESTIONS[:3]:
        run.chat.reply(asked).finish()
    rendered = len(run.context.rendered)
    run.chat.inject(question)
    reply = run.chat.reply().finish()
    assert max(run.context.rendered[rendered:run.renders[-1]]) <= 4
    assert reply.n_evicted_messages


@pytest.mark.parametrize("stem", _EVERY)
def test_a_question_the_stand_in_cannot_stand_for_is_rendered_as_it_comes(stem):
    run = _Run(stem)
    run.begin()
    questions = ["Plain first.", "  spaced out \n"]
    for question in questions:
        run.chat.reply(question).finish()
    _assert_each_reply_followed_what_was_sent(run, questions)


# ----- evictions: whole exchanges, in place --------------------------------------
@pytest.mark.parametrize("stem", _EVERY)
@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_evictions_drop_whole_exchanges_and_decode_only_what_changed(stem, can_shift):
    probe = _Run(stem)
    probe.begin()
    exchange = _exchange_length(probe, QUESTIONS[0], ANSWERS[0])
    # Room for about six exchanges, so that decoding all kept history again is
    # more than a reply ever needs.
    run = _Run(stem, context_size=int((len(probe.context.cache) + 6 * exchange) / 0.75),
               can_shift=can_shift)
    run.begin()
    resets, evicted = run.context.resets, 0
    questions = QUESTIONS + [f"{q} Again." for q in QUESTIONS]
    run.answers = ANSWERS + ANSWERS
    run.context._replies = _as_generated(run.chat, run.answers, False)
    for question in questions:
        evicted += run.chat.reply(question).finish().n_evicted_messages
    assert evicted
    _assert_each_reply_followed_what_was_sent(run, questions)
    # What one reply decodes: its question, its reply again as a past reply, the
    # exchange that became the oldest, and the template text around them.
    bound = 4 * exchange + 80
    over = [k for k in range(1, len(run.decoded)) if run.decoded[k] - run.decoded[k - 1] > bound]
    if can_shift:
        assert run.context.resets == resets and not over
        assert any(edit[0] == "shift" for edit in run.context.edits)
    else:
        # Dropped well below the limit, so the cache is decoded again only every few turns.
        assert len(over) <= math.ceil(len(questions) / 3)


@pytest.mark.parametrize("stem, tools", [(stem, None) for stem in _EVERY]
                         + [(stem, [WEATHER]) for stem in _TOOL_TEMPLATES])
def test_an_eviction_on_a_cache_that_cannot_shift_goes_back_to_before_the_oldest_question(stem, tools):
    long_system = "Be brief. " + "Keep to the facts and leave out what was not asked. " * 6
    probe = _Run(stem)
    probe.system = long_system if probe.system else ""
    probe.begin(tools=tools)
    exchange = _exchange_length(probe, QUESTIONS[0], ANSWERS[0])
    # Room for the system turn twice over and a dozen exchanges, so that half the
    # threshold, where an eviction on such a cache stops, still holds some.
    run = _Run(stem, context_size=int((2 * len(probe.context.cache) + 12 * exchange) / 0.75),
               can_shift=False)
    run.system = probe.system
    run.begin(tools=tools)
    resets = run.context.resets
    questions = [f"{q} Round {n}." for n in range(12) for q in QUESTIONS]
    run.answers = ANSWERS * 12
    run.context._replies = _as_generated(run.chat, run.answers, False)
    evictions, asked = 0, []
    for question in questions:          # until two evictions are seen
        asked.append(question)
        evictions += bool(run.chat.reply(question).finish().n_evicted_messages)
        if evictions == 2:
            break
    assert evictions == 2
    assert run.context.resets == resets
    if tools is None:
        _assert_each_reply_followed_what_was_sent(run, asked)


@pytest.mark.parametrize("stem", _EVERY)
def test_a_system_prompt_longer_than_the_threshold_is_refused_at_begin(stem):
    run = _Run(stem, context_size=600)
    refusal = ContextOverflowError if run.chat.can_take_system_prompt else ValueError
    with pytest.raises(refusal):
        run.chat.begin("Keep it short. " * 80)
    assert not run.chat._conversation.begun


@pytest.mark.parametrize("stem", _EVERY)
def test_an_eviction_on_a_cache_that_cannot_shift_keeps_history_however_long_the_system_turn(stem):
    probe = _Run(stem)
    probe.system = probe.system and "Be brief. " + "Keep to the facts and leave out what was not asked. " * 12
    probe.begin()
    system = probe.chat._conversation._system_length()
    exchange = _exchange_length(probe, QUESTIONS[0], ANSWERS[0])
    # A threshold the system turn takes more than half of, with room for six exchanges.
    run = _Run(stem, context_size=int((system + 6 * exchange) / 0.75), can_shift=False)
    run.system = probe.system
    run.begin()
    questions = [f"{q} Round {n}." for n in range(3) for q in QUESTIONS]
    run.answers = ANSWERS * 3
    run.context._replies = _as_generated(run.chat, run.answers, False)
    evicted = 0
    for question in questions:
        evicted += run.chat.reply(question).finish().n_evicted_messages
        assert any(role == "user" for role, _ in run.chat.history())
    assert evicted


@pytest.mark.parametrize("stem", _EVERY)
def test_a_cache_that_cannot_be_cut_goes_back_to_a_saved_state_rather_than_decoding_all_again(stem):
    run = _Run(stem, can_shift=False)
    run.begin()
    resets = run.context.resets
    for question in QUESTIONS[:4]:
        run.chat.reply(question).finish()
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS)
    assert run.context.resets == resets


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_after_the_first_replies_only_windows_are_rendered(stem, can_shift):
    probe = _Run(stem)
    probe.begin()
    exchange = _exchange_length(probe, QUESTIONS[0], ANSWERS[0])
    run = _Run(stem, context_size=int((len(probe.context.cache) + 4 * exchange) / 0.75),
               can_shift=can_shift)
    run.begin()
    assert run.chat._conversation._windows
    window = (1 if run.system else 0) + 5         # the system turn, two exchanges, the stand-in
    whole = 0
    for question in QUESTIONS:
        rendered = len(run.context.rendered)
        run.chat.reply(question).finish()
        whole += max(run.context.rendered[rendered:]) > window
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS)
    # The first two replies follow no window yet; every later one is settled in one.
    assert whole <= 2


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_exchanges_that_repeat_word_for_word_are_kept_as_many_as_fit(stem):
    probe = _Run(stem)
    probe.begin()
    exchange = _exchange_length(probe, "Same again?", "Same answer.")
    size = int((len(probe.context.cache) + 6 * exchange) / 0.75)
    kept = {}
    for same in (True, False):
        letters = [chr(65 + i) for i in range(14)]
        questions = ["Same again?" if same else f"Same again {c}?" for c in letters]
        run = _Run(stem, context_size=size,
                   replies=["Same answer." if same else f"Same answer {c}." for c in letters])
        run.begin()
        counts = []
        for question in questions:
            run.chat.reply(question).finish()
            counts.append(sum(m["role"] == "user" for m in run.chat._conversation.messages))
        kept[same] = min(counts[8:])
    assert kept[True] >= kept[False]


@pytest.mark.parametrize("stem", _TOOL_TEMPLATES)
def test_declarations_that_move_with_each_question_cost_a_cache_that_cannot_be_cut_one_exchange_more(stem):
    costs = {}
    for can_shift in (True, False):
        run = _Run(stem, can_shift=can_shift)
        run.begin(tools=[WEATHER])
        for question in QUESTIONS[:6]:
            run.chat.reply(question).finish()
        costs[can_shift] = [b - a for a, b in zip(run.decoded[1:], run.decoded[2:])]
    exchange = _exchange_length(run, QUESTIONS[0], ANSWERS[0])
    assert all(cut <= shift + 2 * exchange for shift, cut in zip(costs[True], costs[False])), costs


@pytest.mark.parametrize("stem", _EVERY)
def test_a_reply_that_fails_part_way_is_kept_or_taken_back_and_the_next_follows_the_render(stem):
    run = _Run(stem)
    run.begin()
    for question in QUESTIONS[:2]:
        run.chat.reply(question).finish()
    generate = run.context.generate

    def failing_part_way(*args, out, **kwargs):
        for ch in "Partly":
            run.context.cache.append(ord(ch))
            out.token_ids.append(ord(ch))
            out.text += ch
            yield ch
        raise RuntimeError("the generation failed part-way")

    run.context.generate = failing_part_way
    with pytest.raises(RuntimeError, match="part-way"):
        run.chat.reply(QUESTIONS[2]).finish()
    run.context.generate = generate
    run.chat.reply(QUESTIONS[3]).finish()
    # Recorded where some of it was answer, taken back whole where none was.
    asked = any(m["content"] == QUESTIONS[2] for m in run.chat._conversation.messages)
    assert (QUESTIONS[2] in run.context.text_of(run.context.cache)) == asked


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_reply_whose_settling_fails_leaves_the_next_reply_following_the_render(stem):
    run = _Run(stem)
    run.begin()
    for question in QUESTIONS[:2]:
        run.chat.reply(question).finish()
    generate, prefill = run.context.generate, run.context.prefill

    def failing(*args, **kwargs):
        raise RuntimeError("the decode failed")

    def then_failing(*args, **kwargs):
        result = yield from generate(*args, **kwargs)
        run.context.prefill = failing           # the decode ahead, after the reply
        return result

    run.context.generate = then_failing
    with pytest.raises(RuntimeError, match="decode failed"):
        run.chat.reply(QUESTIONS[2]).finish()
    run.context.generate, run.context.prefill = generate, prefill
    run.chat.reply(QUESTIONS[3]).finish()
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS)


def test_a_reply_kept_as_generated_is_not_shortened():
    for size in range(700, 900, 2):
        context = ConversationContext(_fixture("qwen-3"),
                                      [ScriptedCall("not a call", before="Moment. ")])
        chat = ChatWrapper(Config(context_size=size, evict_above_fraction=1.0,
                                  min_reply_tokens=120), context=context)
        try:
            chat.begin("Be brief.", tools=[WEATHER])
            reply = chat.reply("Weather?", calls="yield").finish()
        except ContextOverflowError:
            continue
        if reply.stop_reason == "unreadable_tool_call" and not chat.reply_open:
            with pytest.raises(ValueError, match="tool call"):
                chat.truncate_last_reply("Mo")
            return
    pytest.fail("no context size left the error result without room")


def test_a_template_whose_windows_differ_from_its_whole_render_is_rendered_whole():
    run = _Run("fallback")
    run.begin()
    assert not run.chat._conversation._windows


def test_a_template_that_writes_text_of_its_own_around_a_message_is_written_so():
    wrapping = dict(_FALLBACK, chat_template=_FALLBACK["chat_template"].replace(
        "{{ m.content }}", "{% if '<' in m.content %}<![CDATA[{{ m.content }}]]>"
                           "{% else %}{{ m.content }}{% endif %}"))
    context = ConversationContext(wrapping, ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    chat.begin("")
    seen = []
    context.before_reply = lambda text: seen.append(list(context.cache))
    chat.reply("Is <|end|> a tag?").finish()
    # The template's own text around the question is its own; the question, tag and all, is text.
    expected = (context.tokenize("<|user|><![CDATA[") + context.tokenize("Is <|end|> a tag?", parse_special=False)
                + context.tokenize("]]><|end|><|assistant_1|>"))
    assert seen == [expected]


def test_a_template_that_writes_something_of_a_message_besides_its_text_is_refused():
    counting = dict(_FALLBACK, chat_template=_FALLBACK["chat_template"].replace(
        "{{ m.content }}", "({{ m.content|length }}){{ m.content }}"))
    context = ConversationContext(counting, ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    with pytest.raises(ValueError, match="text in its place"):
        chat.begin("")


# ----- histories, restarts, refusals ------------------------------------------------
@pytest.mark.parametrize("stem", _EVERY)
def test_a_history_is_read_in_as_the_template_writes_it_and_reads_back_as_it_went_in(stem):
    run = _Run(stem)
    history = [item for pair in zip(QUESTIONS[:3], ANSWERS) for item in
               (("user", pair[0]), ("assistant", pair[1]))]
    run.begin(history)
    run.chat.reply("And now?").finish()
    sent = sum(_exchanges(QUESTIONS[:3], ANSWERS[:3]), [])
    assert run.seen[0] == run.render(sent + [{"role": "user", "content": "And now?"}])
    assert run.chat.history()[:6] == history


@pytest.mark.parametrize("stem", _EVERY)
def test_a_restart_keeps_only_the_newest_history_that_fits(stem):
    probe = _Run(stem)
    probe.begin()
    exchange = _exchange_length(probe, QUESTIONS[0], ANSWERS[0])
    run = _Run(stem, context_size=int((len(probe.context.cache) + 3 * exchange) / 0.75))
    history = [item for pair in zip(QUESTIONS, ANSWERS) for item in
               (("user", pair[0]), ("assistant", pair[1]))]
    kept = run.begin(history)
    assert 0 < kept < len(history) and kept % 2 == 0
    run.chat.reply("And now?").finish()
    sent = sum(_exchanges(QUESTIONS, ANSWERS), [])[len(history) - kept:]
    assert run.seen[0] == run.render(sent + [{"role": "user", "content": "And now?"}])


def test_a_template_that_writes_an_older_exchange_otherwise_once_another_follows_is_refused():
    rewriting = dict(_FALLBACK, chat_template="<|count|>{{ messages|length }}" + _FALLBACK["chat_template"])
    context = ConversationContext(rewriting, ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    with pytest.raises(ValueError, match="older exchange"):
        chat.begin("")


def test_a_refused_restart_leaves_the_conversation_as_it_was_and_usable():
    run = _Run("qwen-2.5", context_size=300)
    run.begin()
    run.chat.reply(QUESTIONS[0]).finish()
    history, held = run.chat.history(), list(run.context.cache)
    with pytest.raises(ValueError):
        run.chat.begin("x" * 400)               # longer than the limit
    assert run.chat.history() == history and run.context.cache == held
    assert run.chat.reply(QUESTIONS[1]).finish().text == ANSWERS[1]


# ----- the clock a template reads -----------------------------------------------------
@pytest.fixture
def clock(monkeypatch):
    import llama_cpp.llama_chat_format as chat_format
    today = [real_datetime.datetime(2026, 11, 7, 23, 59)]

    class _Clock:
        @staticmethod
        def now():
            return today[0]

    monkeypatch.setattr(chat_format, "datetime", _Clock)
    return today


def test_a_conversation_begun_on_a_later_day_opens_with_that_days_date(clock):
    run = _Run("llama-3.2")
    run.begin()
    assert "07 Nov 2026" in run.context.text_of(run.context.cache)
    clock[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    run.begin()
    run.chat.reply(QUESTIONS[0]).finish()
    assert "08 Nov 2026" in run.seen[-1] and run.seen[-1] == run.render(
        [{"role": "user", "content": QUESTIONS[0]}])


def test_a_conversation_that_runs_past_midnight_follows_the_clock_decoding_it_again_once(clock):
    run = _Run("llama-3.2")
    run.begin()
    run.chat.reply(QUESTIONS[0]).finish()
    clock[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    resets = run.context.resets
    for question in QUESTIONS[1:4]:
        run.chat.reply(question).finish()
    assert all("08 Nov 2026" in cached for cached in run.seen[1:])
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS, first_reply=1)
    assert run.context.resets == resets
    # Once from the date on, then only the newest exchange each reply.
    cuts = [edit for edit in run.context.edits if edit[0] == "cut"]
    assert len([cut for cut in cuts if cut[1] < len(run.seen[0]) // 2]) == 1


# ----- reasoning ---------------------------------------------------------------------
def _thinking(stem):
    return [stem for stem in sorted(_FIXTURES) if _FIXTURES[stem]["can_reason"]]


@pytest.mark.parametrize("stem", _thinking("all"))
def test_with_reasoning_on_every_reply_starts_from_the_templates_render(stem):
    run = _Run(stem, reasoning=True)
    run.begin()
    answers = [run.chat.reply(question).finish().text for question in QUESTIONS[:3]]
    assert answers == ANSWERS[:3]                      # the thought kept out of each
    keeps = run.chat._template.reasoning.past_keeps_thought
    _assert_each_reply_followed_what_was_sent(run, QUESTIONS, forms=(_THOUGHT,) if keeps else (None,))


# ----- tools ------------------------------------------------------------------------
@pytest.mark.parametrize("stem", _TOOL_TEMPLATES)
def test_tools_declared_mid_conversation_are_written_where_the_template_writes_them(stem):
    run = _Run(stem)
    assert run.chat.can_call_tools
    run.begin()
    run.chat.reply(QUESTIONS[0]).finish()
    run.chat.declare_unknown_tools([WEATHER])
    run.chat.reply(QUESTIONS[1]).finish()
    sent = sum(_exchanges(QUESTIONS[:1], ANSWERS[:1]), [])
    assert run.seen[-1] == run.render(sent + [{"role": "user", "content": QUESTIONS[1]}], [WEATHER])


@pytest.mark.parametrize("stem", _TOOL_TEMPLATES)
def test_a_past_tool_turn_and_a_later_declaration_are_written_as_the_template_writes_them(stem):
    run = _Run(stem)
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", '{"weather": "sunny"}'), "Sunny."))
    clock = ToolSpec(name="get_time", description="The time",
                     parameters={"type": "object", "properties": {}})
    run.begin([("user", "Weather?"), turn], tools=[WEATHER])
    run.chat.reply(QUESTIONS[0]).finish()
    run.chat.declare_unknown_tools([clock])
    run.chat.reply(QUESTIONS[1]).finish()
    from llama_chat.messages import as_messages
    sent = as_messages([("user", "Weather?"), turn]) + sum(_exchanges(QUESTIONS[:1], ANSWERS[:1]), [])
    assert run.seen[-1] == run.render(sent + [{"role": "user", "content": QUESTIONS[1]}],
                                      [WEATHER, clock])
    assert isinstance(run.chat.history()[1], ToolTurn)


def _call_body(run, arguments):
    """A call's body as the template writes a past call: what the model generates between its tags."""
    template = run.chat._template
    rendered = template.render(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function",
              "function": {"name": WEATHER.name, "arguments": arguments}}]}], reasoning=False)
    opener, closer = (template.call_texts if template.call_texts is not None
                      else (run.context.token_text(t) for t in template.tool_tokens))
    if template._call_from_file:
        # The template shows no call it could be read from; the model writes the add-on file's.
        return opener, WEATHER.name + run.context.addon_call()["name_close"] + json.dumps(
            arguments), closer
    return opener, rendered.rsplit(opener, 1)[1].split(closer, 1)[0], closer


def _spaced(run, body):
    """``body`` spaced unlike the template writes it, where the call still reads back the same."""
    from llama_chat.call_parsing import parse_calls
    template = run.chat._template
    for spaced in (body.replace(": ", ":"), body.replace(":", ": "), body + " ", " " + body):
        try:
            read = parse_calls((template.call_lead_in + spaced) if template.call_texts else spaced,
                               template.call_body, template.call_lead_in)
        except Exception:
            continue
        if spaced != body and read == [ToolCall(WEATHER.name, {"city": "Bonn"})]:
            return spaced
    return body


def _rendered_turn(run, *after):
    """The template's render of a question, one call answered by its result, then ``after``."""
    return run.render([
        {"role": "user", "content": "Weather in Bonn?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call00000", "type": "function",
             "function": {"name": WEATHER.name, "arguments": {"city": "Bonn"}}}]},
        {"role": "tool", "name": WEATHER.name, "tool_call_id": "call00000",
         "content": '{"weather": "sunny"}'}, *after], [WEATHER])


@pytest.mark.parametrize("stem", _TOOL_TEMPLATES)
def test_a_live_call_keeps_its_tokens_and_the_conversation_goes_on_as_the_template_writes_it(stem):
    probe = _Run(stem)
    opener, written_body, closer = _call_body(probe, {"city": "Bonn"})
    body = _spaced(probe, written_body)          # as a model may space it, not as the template does
    run = _Run(stem, replies=[ScriptedCall(body), "Sunny in Bonn.", "You are welcome."])
    run.context._replies = [ScriptedCall(body)] + _as_generated(
        probe.chat, ["Sunny in Bonn.", "You are welcome."], False)
    run.begin(tools=[WEATHER])
    events = list(run.chat.reply("Weather in Bonn?", calls="yield"))
    call, = [event for event in events if isinstance(event, ToolCall)]
    assert call == ToolCall(WEATHER.name, {"city": "Bonn"})
    run.chat.inject_tool_result(call, '{"weather": "sunny"}')
    tags = run.chat._template.reasoning
    if tags and begins_in_thought(tags, run.context.text_of(run.context.cache)) \
            and not run.context._replies[0].startswith(tags.thought_close):
        run.context._replies[0] = tags.thought_close + run.context._replies[0]   # the model closes it
    assert run.chat.reply(calls="yield").finish().text == "Sunny in Bonn."
    run.chat.inject("Thanks.")
    run.chat.reply().finish()
    generated, answer = opener + body + closer, "Sunny in Bonn."
    before_call, before_answer, before_thanks = run.seen
    # The turn keeps the tokens it was generated as: what was cached, then the call,
    # and after its result the answer.
    assert before_answer.startswith(before_call + generated)
    resumed_with = before_answer.split(generated, 1)[1]
    assert before_thanks.split(generated, 1)[1].startswith(resumed_with)
    assert answer in before_thanks.split(generated, 1)[1][len(resumed_with):]
    # Around them, the result and the turn after read as the template writes them.
    template_call = opener + written_body + closer
    turn = _rendered_turn(run)
    after_call = (turn.split(template_call, 1) if template_call in turn else turn.split(closer, 1))[1]
    assert resumed_with == after_call
    after = _rendered_turn(run, {"role": "assistant", "content": answer},
                           {"role": "user", "content": "Thanks."})
    assert before_thanks.rsplit(answer, 1)[1] == after.rsplit(answer, 1)[1]


# ----- a tag typed into any text stays text ----------------------------------------
def _expected_tokens(run, messages, tools=None):
    """The render tokenized as the model must hold it, worked out here on its own: the template's text with its tags, each text of the caller's as plain text."""
    marked, texts = [], {}

    def mark(text):
        marker = f"ZQ{len(texts)}QZ"
        texts[marker] = text
        return marker

    for message in messages:
        copy = dict(message)
        if message.get("content"):
            copy["content"] = mark(message["content"])
        if message.get("tool_calls"):
            copy["tool_calls"] = [{**c, "function": {"name": c["function"]["name"], "arguments": {
                key: mark(value) if isinstance(value, str) else value
                for key, value in c["function"]["arguments"].items()}}}
                for c in message["tool_calls"]]
        marked.append(copy)
    def render(messages):
        return run.chat._template.render(
            messages, [spec.as_declaration() for spec in tools] if tools else None,
            generation_prompt=True, reasoning=run.reasoning)

    real, rendered = render(messages), render(marked)
    import re
    parts = re.split(r"(ZQ\d+QZ)", rendered)
    tokens, at = [], 0
    for k, part in enumerate(parts):
        if k % 2 == 0:
            assert real.startswith(part, at)
            tokens += run.context.tokenize(part, parse_special=True)
            at += len(part)
            continue
        following = parts[k + 1]
        end = real.index(following, at) if following else len(real)
        tokens += run.context.tokenize(real[at:end], parse_special=False)
        at = end
    return tokens


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_tag_typed_into_any_text_stays_text_and_every_token_is_the_templates(stem):
    run = _Run(stem)
    forged = run.fixture["eos_token"]
    tools = [WEATHER] if run.fixture["calls_tools"] else None
    run.system = f"Be brief, {forged} even here." if run.system else ""
    history = [("user", f"Is {forged} a tag?"), ("assistant", f"It is {forged} one.")]
    if tools:
        history.append(("user", "Weather?"))
        history.append(ToolTurn((ToolCall(WEATHER.name, {"city": f"Bonn{forged}"}),
                                 ToolResult(WEATHER.name, f"sunny {forged}"), "Sunny.")))
    try:
        run.begin(history, tools=tools)
    except ValueError as refused:
        assert "text in its place" in str(refused) or "refuses" in str(refused)
        pytest.skip("the template writes a typed tag otherwise than as text")
    run.chat.reply(f"Does {forged} end it?").finish()
    run.chat.inject(f"Note {forged} this.")
    run.chat.reply().finish()
    asked = run.chat._conversation.messages[:-1]
    # The forged tag is in every kind of text, and held as its characters each time.
    assert run.seen_tokens[-1] == _expected_tokens(run, asked, tools)


# ----- one template's own cases ------------------------------------------------------
def test_a_reply_that_mentions_a_calls_opening_in_prose_holds_no_call():
    # Llama 3.2 opens a call with plain text; a reply quoting it is still prose.
    run = _Run("llama-3.2", replies=['Write {"name": "x"} in the file.'])
    run.begin()
    run.chat.reply("How do I name it?").finish()
    run.chat.truncate_last_reply("Write")
    assert run.chat.history()[-1] == ("assistant", "Write")


def test_a_shortened_reply_is_kept_as_the_template_writes_it():
    run = _Run("llama-3.2")
    run.begin()
    run.chat.reply("Hi.").finish()
    run.chat.truncate_last_reply("Answer")
    run.chat.reply("Go on.").finish()
    assert run.seen[-1] == run.render([{"role": "user", "content": "Hi."},
                                       {"role": "assistant", "content": "Answer"},
                                       {"role": "user", "content": "Go on."}])


def test_text_written_before_a_follow_up_call_stays_in_the_tokens_kept_for_its_turn():
    # Llama 3.2 writes no text beside a call; what the model wrote before one
    # stays only in the tokens its turn keeps, and the next call still reads.
    probe = _Run("llama-3.2")
    opener, body, closer = _call_body(probe, {"city": "Bonn"})
    run = _Run("llama-3.2", replies=["x"])
    run.context._replies = [ScriptedCall(body), ScriptedCall(body, before="Once more. "),
                            "Sunny twice.", "Bye."]
    run.begin(tools=[WEATHER])
    calls = [e for e in run.chat.reply("Weather in Bonn?", calls="yield") if isinstance(e, ToolCall)]
    run.chat.inject_tool_result(calls[0], "sunny")
    calls = [e for e in run.chat.reply(calls="yield") if isinstance(e, ToolCall)]
    assert calls == [ToolCall(WEATHER.name, {"city": "Bonn"})]
    run.chat.inject_tool_result(calls[0], "sunny")
    assert run.chat.reply().finish().text == "Sunny twice."
    run.chat.reply("Thanks.").finish()
    assert "Once more. " + opener + body + closer in run.seen[-1]


def test_a_turn_kept_longer_than_the_template_writes_it_counts_against_the_limits():
    from tests.fake_context import FakeContext
    from tests.wrappers import weather_registry
    fake = FakeContext(calls_tools=True, gen_tokens=6, gen_text="Sonnig")
    chat = ChatWrapper(Config(context_size=800, evict_above_fraction=0.6, min_reply_tokens=20),
                       context=fake)
    chat.begin("persona", tools=weather_registry())
    for n in range(8):
        fake.call_script = ["weather{city:" + " " * 60 + "'Bonn'}"]      # far wider than written
        reply = chat.reply(f"Wetter {n}?", calls="run").finish()
        assert reply.text == "Sonnig"
        assert chat.total_tokens <= chat.config.evict_above_tokens
    assert len(chat.history()) < 8


def test_a_question_whose_words_change_how_the_template_opens_the_reply_is_refused():
    switching = dict(_FALLBACK, chat_template=_FALLBACK["chat_template"].replace(
        "<|assistant_{{ messages|length }}|>",
        "<|assistant|>{% if '/no_think' in messages[-1].content %}<|nothink|>{% endif %}"))
    context = ConversationContext(switching, ["Fine.", "Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    chat.begin("")
    with pytest.raises(ValueError, match="text in its place"):
        chat.reply("Hello /no_think").finish()
    assert chat.history() == [] and chat.reply("Hello").finish().text == "Fine."


def test_a_template_that_refuses_the_probe_for_rewritten_history_is_refused():
    short = dict(_FALLBACK, chat_template="{% if messages|length > 5 %}"
                 "{{ raise_exception('too long') }}{% endif %}" + _FALLBACK["chat_template"])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8),
                       context=ConversationContext(short, []))
    with pytest.raises(ValueError, match="cannot\\s+be told"):
        chat.begin("")
