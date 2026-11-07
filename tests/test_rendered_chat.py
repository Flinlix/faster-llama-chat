"""The template as the program: whole conversations through every real template, the cache always its render."""

import json
from pathlib import Path

import pytest

from llama_chat import Config
from llama_chat.rendered_chat import RenderedChat
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

from .template_contexts import ConversationContext

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_QUESTIONS = ["How are you?", "Name a colour.", "And a number?", "Why that one?",
              "Tell me more.", "Shorter, please.", "Thanks!", "One more thing?"]
_REPLIES = [f"Answer number {i}, kept short." for i in range(len(_QUESTIONS) + 2)]
_TOOL = ToolSpec(name="get_weather", description="Weather for a city",
                 parameters={"type": "object", "properties": {"city": {"type": "string"}}})


def _chat(path, context_size=4096, can_shift=True):
    context = ConversationContext(json.loads(path.read_text()), _REPLIES, can_shift=can_shift)
    chat = RenderedChat(Config(context_size=context_size, evict_above_fraction=0.75,
                               min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = lambda text: seen.append((text, chat.messages, list(chat._tools)))
    return context, chat, seen


def _rendered(context, chat, messages, tools):
    return context.render_with_model_template(
        messages, tools=[spec.as_declaration() for spec in tools] or None,
        generation_prompt=True, reasoning=False, **chat._template.template_variables)


def _system(chat):
    return "Be brief." if chat._template.has_system_turn else ""


def _assert_every_reply_started_from_the_render(context, chat, seen):
    assert seen
    for cached, messages, tools in seen:
        assert cached == _rendered(context, chat, messages, tools)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_every_reply_starts_from_the_templates_render(path):
    context, chat, seen = _chat(path)
    chat.begin(_system(chat))
    for question in _QUESTIONS[:4]:
        chat.reply(question).finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert chat.counts.splice_misses == 0


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_question_the_stand_in_cannot_stand_for_is_rendered_as_it_comes(path):
    context, chat, seen = _chat(path)
    chat.begin(_system(chat))
    chat.reply("Plain first.").finish()
    chat.reply("  spaced out \n").finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_evictions_drop_whole_exchanges_and_keep_the_render(path, can_shift):
    probe, chat, _ = _chat(path)
    chat.begin(_system(chat))
    budget = int((len(probe.cache) + 120) / 0.75)
    context, chat, seen = _chat(path, context_size=budget, can_shift=can_shift)
    chat.begin(_system(chat))
    opening = len(context.cache)
    for question in _QUESTIONS:
        chat.reply(question).finish()
    assert len(seen[-1][1]) < 2 * len(_QUESTIONS)
    _assert_every_reply_started_from_the_render(context, chat, seen)
    if can_shift:
        # Rolling: nothing the conversation opened with is decoded again.
        assert chat.counts.rebuilt == 0 and (chat.counts.earliest_cut or opening) >= opening


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_history_is_read_in_as_the_template_writes_it(path):
    context, chat, seen = _chat(path)
    history = [item for pair in zip(_QUESTIONS[:3], _REPLIES) for item in
               (("user", pair[0]), ("assistant", pair[1]))]
    chat.begin(_system(chat), history)
    chat.reply("And now?").finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert chat.history()[:6] == history


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if json.loads(p.read_text())["calls_tools"]],
                         ids=lambda p: p.stem)
def test_tools_declared_later_and_a_past_tool_turn_are_written_as_the_template_writes_them(path):
    context, chat, seen = _chat(path)
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", '{"weather": "sunny"}'), "Sunny."))
    try:
        chat.begin(_system(chat), [("user", "Weather?"), turn], tools=[_TOOL])
    except ValueError as refused:
        assert "refuses to write" in str(refused)
        pytest.skip("the template refuses a past tool turn")
    chat.reply(_QUESTIONS[0]).finish()
    chat.declare_unknown_tools([ToolSpec(name="get_time", description="The time",
                                         parameters={"type": "object", "properties": {}})])
    chat.reply(_QUESTIONS[1]).finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert isinstance(chat.history()[1], ToolTurn)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_question_waits_only_for_its_own_tokens(path):
    context, chat, seen = _chat(path)
    chat.begin(_system(chat))
    chat.reply(_QUESTIONS[0]).finish()
    before = chat.counts.decoded_waiting
    renders = chat.counts.renders_waiting
    chat.reply(_QUESTIONS[1]).finish()
    assert chat._splices
    if chat._splices:
        assert chat.counts.renders_waiting == renders
        # The question, the run of template text it joins, and what the template writes after it.
        assert chat.counts.decoded_waiting - before <= (len(_QUESTIONS[1]) + len(chat._prompt_text)
                                                        + 40)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_after_the_first_reply_only_windows_are_rendered_and_the_cache_stays_the_render(path, can_shift):
    probe, chat, _ = _chat(path)
    chat.begin(_system(chat))
    budget = int((len(probe.cache) + 250) / 0.75)       # room for a few exchanges
    context, chat, seen = _chat(path, context_size=budget, can_shift=can_shift)
    chat.begin(_system(chat))
    for question in _QUESTIONS:
        chat.reply(question).finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert chat._windows
    if all(m["content"] for m in chat.messages if m["role"] == "assistant"):
        # Where a reply is all thought, its text is empty and nothing anchors a window.
        assert len(seen[-1][1]) < 2 * len(_QUESTIONS)          # exchanges were dropped
        # The first reply follows no spliced question; every later one is settled in windows.
        assert chat.counts.settled_whole == 1


def test_a_template_that_writes_the_newest_exchanges_otherwise_in_a_window_renders_them_whole():
    path = Path(__file__).parent / "templates" / "llama-3.2.json"
    context, chat, seen = _chat(path)
    numbered = lambda messages, generation_prompt: "".join(       # noqa: E731
        f"<|eot_id|>[{n}/{len(messages)}]{m['content']}" for n, m in enumerate(messages))
    chat._render = numbered
    assert not chat._windows_are_exact()


def test_a_shortened_reply_is_kept_as_the_template_writes_it():
    path = Path(__file__).parent / "templates" / "llama-3.2.json"
    context, chat, seen = _chat(path)
    chat.begin("Be brief.")
    chat.reply("Hi.").finish()
    chat.truncate_last_reply("Answer")
    chat.reply("Go on.").finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert chat.messages[2]["content"] == "Answer"


def _fake_tool_chat():
    from tests.fake_context import FakeContext
    fake = FakeContext(calls_tools=True, gen_tokens=6, gen_text="Sonnig", tokens_spell_text=True)
    return fake, RenderedChat(Config(context_size=512, evict_above_fraction=0.9), context=fake)


def _fake_cache_is_the_render(fake, chat):
    rendered = fake.render_with_model_template(
        chat.messages, [spec.as_declaration() for spec in chat._tools] or None,
        generation_prompt=False)
    # Compared as text: a call keeps the tokens it was generated as, its tags among them.
    assert fake.text_of(fake.cache).startswith(rendered)


def test_a_call_is_run_and_the_reply_goes_on_after_its_result():
    from tests.wrappers import weather_registry
    fake, chat = _fake_tool_chat()
    chat.begin("persona", tools=weather_registry())
    fake.call_script = ["weather{city:'Bonn'}"]
    reply = chat.reply("wetter?", calls="run").finish()
    assert reply.tool_calls == [ToolCall("weather", {"city": "Bonn"})]
    assert reply.text == "Sonnig" and reply.stop_reason == "end_of_turn"
    assert [m["role"] for m in chat.messages] == ["system", "user", "assistant", "tool", "assistant"]
    _fake_cache_is_the_render(fake, chat)
    assert isinstance(chat.history()[1], ToolTurn)


def test_a_yielded_call_waits_for_its_result_and_then_goes_on():
    from tests.wrappers import weather_registry
    fake, chat = _fake_tool_chat()
    chat.begin("persona", tools=weather_registry())
    fake.call_script = ["weather{city:'Bonn'}"]
    asked = chat.reply("wetter?", calls="yield").finish()
    assert asked.stop_reason == "tool_call"
    with pytest.raises(ValueError, match="waiting"):
        chat.reply().finish()
    chat.inject_tool_result(asked.tool_calls[0], {"celsius": 18})
    assert chat.reply().finish().text == "Sonnig"
    _fake_cache_is_the_render(fake, chat)


def test_each_result_of_a_batch_answers_its_own_call():
    from llama_chat.rendered_chat import _as_messages
    turn = ToolTurn((ToolCall("a", {}), ToolCall("b", {}), ToolResult("a", "1"), ToolResult("b", "2"), "ok"))
    messages = _as_messages([("user", "q"), turn])
    calls = [call["id"] for m in messages for call in m.get("tool_calls", [])]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == calls


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_cache_that_cannot_be_cut_goes_back_to_a_saved_state_rather_than_decoding_all_again(path):
    context, chat, seen = _chat(path, can_shift=False)
    chat.begin(_system(chat))
    for question in _QUESTIONS[:4]:
        chat.reply(question).finish()
    _assert_every_reply_started_from_the_render(context, chat, seen)
    assert chat.counts.rebuilt == 0


def test_a_template_that_writes_an_older_exchange_otherwise_once_another_follows_is_refused():
    path = Path(__file__).parent / "templates" / "llama-3.2.json"
    context, chat, seen = _chat(path)
    numbered = lambda messages, generation_prompt: "".join(       # noqa: E731
        f"[{n}/{len(messages)}]{m['content']}" for n, m in enumerate(messages))
    chat._render = numbered
    with pytest.raises(ValueError, match="older exchange"):
        chat._ensure_history_stays_written()


def test_a_call_the_template_writes_otherwise_keeps_the_tokens_it_was_generated_as():
    from tests.wrappers import weather_registry
    fake, chat = _fake_tool_chat()
    chat.begin("persona", tools=weather_registry())
    fake.call_script = ["weather{ city : 'Bonn' }"]          # spaced unlike the template
    assert chat.reply("wetter?", calls="run").finish().text == "Sonnig"
    held = fake.text_of(fake.cache)
    assert "weather{ city : 'Bonn' }" in held
    assert "</result>Sonnig<turn]>\n" in held          # the rest as the template writes it
