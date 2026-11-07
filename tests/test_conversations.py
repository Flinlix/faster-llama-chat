"""Whole conversations through real chat templates: the cache always holds the template's render.

Before every reply, the cache must read exactly as the template renders the
messages that are still in it, with the generation prompt after them. A
conversation the template refuses to render must be refused, never encoded.
"""

import json
from pathlib import Path

import pytest

from llama_chat import ChatWrapper, Config
from llama_chat.reply_text import begins_in_thought
from llama_chat.addon_files import addon_file
from llama_chat.tools import ToolCall, ToolSpec

from .template_contexts import ConversationContext, HonestTagContext, ScriptedCall

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_QUESTIONS = ["How are you?", "Name a colour.", "And a number?", "Why that one?",
              "Tell me more.", "Shorter, please.", "Thanks!", "One more thing?"]
_REPLIES = [f"Answer number {i}, kept short." for i in range(len(_QUESTIONS) + 1)]


def _conversation(path, context_size):
    fixture = json.loads(path.read_text())
    context = ConversationContext(fixture, _REPLIES)
    chat = ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.75,
                              min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = lambda text: seen.append((text, chat.snapshot()))
    return context, chat, seen


def _rendered(context, rows, tools=None, variables=None):
    messages = [{"role": row["role"], "content": row["text"]} for row in rows
                if not (row["role"] == "system" and not row["text"])]
    return context.render_with_model_template(
        messages, tools=[spec.as_declaration() for spec in tools] if tools else None,
        generation_prompt=True, reasoning=False, **(variables or {}))


def _system_prompt(chat):
    return "Be brief." if chat._template.has_system_turn else ""


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_every_reply_starts_from_the_templates_render_of_what_the_cache_holds(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    for question in _QUESTIONS[:3]:
        chat.reply(question).finish()
    for cached, rows in seen:
        assert cached == _rendered(context, rows)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_after_evictions_the_cache_still_holds_the_templates_render(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    # Small enough that the oldest messages are evicted after a few turns,
    # even where a reply is all thought and leaves the conversation no longer.
    budget = int((len(context.cache) + 110) / 0.75)
    context, chat, seen = _conversation(path, context_size=budget)
    chat.begin(_system_prompt(chat))
    resets, early_cuts, evicted = context.resets, [], 0
    for question in _QUESTIONS:
        replies = [row for row in chat.snapshot() if row["role"] == "assistant"]
        newest = replies[-1]["pos_start"] if replies else len(context.cache)
        context.edits.clear()
        evicted += chat.reply(question).finish().n_evicted_messages
        for edit, amount in context.edits:      # a shift moves the newest reply down
            newest -= amount if edit == "shift" else 0
            if edit == "cut" and amount < newest:
                early_cuts.append((amount, newest))
    assert evicted
    # Rolling: no rebuild, and no cut reaches before the newest reply.
    assert context.resets == resets
    assert not early_cuts
    for cached, rows in seen:
        rendered = _rendered(context, rows)
        assert rendered is not None, [row["role"] for row in rows]
        assert cached == rendered


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_restart_keeps_only_history_the_template_renders(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    budget = int((len(context.cache) + 150) / 0.75)
    context, chat, seen = _conversation(path, context_size=budget)
    history = []
    for question, answer in zip(_QUESTIONS, _REPLIES):
        history += [("user", question), ("assistant", answer)]
    chat.begin(_system_prompt(chat), history)
    chat.reply("And now?").finish()
    cached, rows = seen[-1]
    rendered = _rendered(context, rows)
    assert rendered is not None, [row["role"] for row in rows]
    assert cached == rendered


def test_a_conversation_begun_on_a_later_day_opens_with_that_days_date(monkeypatch):
    import datetime as real_datetime

    import llama_cpp.llama_chat_format as chat_format

    today = [real_datetime.datetime(2026, 11, 7, 23, 59)]

    class _Clock:
        @staticmethod
        def now():
            return today[0]

    monkeypatch.setattr(chat_format, "datetime", _Clock)
    path = Path(__file__).parent / "templates" / "llama-3.2.json"
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin("Be brief.")
    assert "07 Nov 2026" in context.text_of(context.cache)
    today[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    chat.begin("Be brief.")
    chat.reply("How are you?").finish()
    cached, rows = seen[-1]
    assert "08 Nov 2026" in cached and cached == _rendered(context, rows)


def test_a_reply_that_mentions_a_calls_opening_in_prose_holds_no_call():
    # Llama 3.2 opens a call with plain text; a reply quoting it is still prose.
    fixture = json.loads((Path(__file__).parent / "templates" / "llama-3.2.json").read_text())
    context = ConversationContext(fixture, ['Write {"name": "x"} in the file.'])
    chat = ChatWrapper(Config(context_size=4096), context=context)
    chat.begin("Be brief.")
    chat.reply("How do I name it?").finish()
    chat.truncate_last_reply("Write")
    assert chat.snapshot()[-1]["text"] == "Write"


_TOOL = ToolSpec(name="get_weather", description="Weather for a city",
                 parameters={"type": "object", "properties": {"city": {"type": "string"}}})


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if json.loads(p.read_text())["calls_tools"]],
                         ids=lambda p: p.stem)
def test_tools_declared_mid_conversation_are_written_where_the_template_writes_them(path):
    context, chat, seen = _conversation(path, context_size=4096)
    assert chat.can_call_tools
    chat.begin(_system_prompt(chat))
    chat.reply(_QUESTIONS[0]).finish()
    chat.declare_unknown_tools([_TOOL])
    chat.reply(_QUESTIONS[1]).finish()
    cached, rows = seen[-1]
    assert cached == _rendered(context, rows, [_TOOL], chat._template.template_variables)


_THOUGHT = "Erst denken."


def _thinking_conversation(path, context_size=4096):
    """A conversation that reasons, each reply opening with a thought the template marks."""
    fixture = json.loads(path.read_text())
    probe = ChatWrapper(Config(context_size=context_size), context=ConversationContext(fixture, []))
    tags = probe._template.reasoning
    opens = begins_in_thought(tags, tags.generation_prompt)
    replies = [("" if opens else tags.thought_open) + _THOUGHT + tags.thought_close + reply
               for reply in _REPLIES]
    context = ConversationContext(fixture, replies)
    chat = ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.75,
                              min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = lambda text: seen.append((text, chat.snapshot()))
    return context, chat, seen


def _rendered_thinking(context, chat, rows) -> list[str]:
    """The template's render, with each past reply's thought in each form templates read it from."""
    keeps = chat._template.reasoning.past_keeps_thought
    forms = [lambda m: m]
    if keeps:
        forms = [lambda m: {**m, "reasoning_content": _THOUGHT},
                 lambda m: {**m, "content": [{"type": "thinking", "thinking": _THOUGHT},
                                             {"type": "text", "text": m["content"]}]}]
    renders = []
    for form in forms:
        messages = [form(m) if m["role"] == "assistant" else m
                    for m in ({"role": row["role"], "content": row["text"]} for row in rows
                              if not (row["role"] == "system" and not row["text"]))]
        renders.append(context.render_with_model_template(
            messages, generation_prompt=True, reasoning=True, **chat._template.template_variables))
    return renders


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if json.loads(p.read_text())["can_reason"]],
                         ids=lambda p: p.stem)
def test_with_reasoning_on_every_reply_starts_from_the_templates_render(path):
    context, chat, seen = _thinking_conversation(path)
    if not chat.can_reason:
        pytest.skip("the template's switch changes more than its system turn")
    chat.begin(_system_prompt(chat), reasoning=True)
    answers = [chat.reply(question).finish().text for question in _QUESTIONS[:3]]
    assert answers == _REPLIES[:3]                      # the thought kept out of each
    for cached, rows in seen:
        assert cached in _rendered_thinking(context, chat, rows)


def _call_body(context, template, arguments):
    """A call's body as the template writes a past call: what the model generates between its tags."""
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function",
              "function": {"name": _TOOL.name, "arguments": arguments}}]}], reasoning=False)
    opener, closer = (template.call_texts if template.call_texts is not None
                      else (context.token_text(t) for t in template.tool_tokens))
    return rendered.rsplit(opener, 1)[1].split(closer, 1)[0]


def _rendered_turn(context, chat, *after):
    """The template's render of a question, one call answered by its result, then ``after``."""
    system = [{"role": "system", "content": chat.system_prompt}] if chat.system_prompt else []
    messages = system + [
        {"role": "user", "content": "Weather in Bonn?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call00000", "type": "function",
             "function": {"name": _TOOL.name, "arguments": {"city": "Bonn"}}}]},
        {"role": "tool", "name": _TOOL.name, "tool_call_id": "call00000",
         "content": '{"weather": "sunny"}'}, *after]
    return context.render_with_model_template(
        messages, tools=[_TOOL.as_declaration()], generation_prompt=True, reasoning=False,
        **chat._template.template_variables)


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if json.loads(p.read_text())["calls_tools"]
                                  and not addon_file(None, json.loads(p.read_text())["chat_template"])["call"]],
                         ids=lambda p: p.stem)
def test_a_live_call_keeps_its_tokens_and_the_conversation_goes_on_as_the_template_writes_it(path):
    fixture = json.loads(path.read_text())
    probe = ChatWrapper(Config(context_size=4096), context=ConversationContext(fixture, []))
    body = _call_body(probe._context, probe._template, {"city": "Bonn"})
    context = ConversationContext(fixture, [ScriptedCall(body), "Sunny in Bonn.", "You are welcome."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = lambda text: seen.append(text)
    chat.begin(_system_prompt(chat), tools=[_TOOL])
    events = list(chat.reply("Weather in Bonn?", calls="yield"))
    call, = [event for event in events if isinstance(event, ToolCall)]
    assert call == ToolCall(_TOOL.name, {"city": "Bonn"})
    chat.inject_tool_result(call, '{"weather": "sunny"}')
    tags = chat._template.reasoning
    if tags and begins_in_thought(tags, context.text_of(context.cache)):
        context._replies[0] = tags.thought_close + context._replies[0]   # the model closes it
    assert chat.reply(calls="yield").finish().text == "Sunny in Bonn."
    chat.inject("Thanks.")
    chat.reply().finish()
    opener, closer = (probe._template.call_texts if probe._template.call_texts is not None
                      else (context.token_text(t) for t in probe._template.tool_tokens))
    written, answer = opener + body + closer, "Sunny in Bonn."
    before_call, before_answer, before_thanks = seen
    # The turn keeps the tokens it was generated as: what was cached, then the call,
    # and after its result the answer.
    assert before_answer.startswith(before_call + written)
    resumed_with = before_answer.split(written, 1)[1]
    assert before_thanks.split(written, 1)[1].startswith(resumed_with)
    assert answer in before_thanks.split(written, 1)[1][len(resumed_with):]
    # Around them, the result and the turn after read as the template writes them.
    resumed = _rendered_turn(context, chat)
    assert before_answer.split(written, 1)[1] == resumed.split(written, 1)[1]
    after = _rendered_turn(context, chat, {"role": "assistant", "content": answer},
                           {"role": "user", "content": "Thanks."})
    assert before_thanks.rsplit(answer, 1)[1] == after.rsplit(answer, 1)[1]


def _as_written(context, content):
    """``content`` as the template writes a user message's text, escapes and all."""
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "QQA" + content + "QQB"}], reasoning=False)
    return rendered.split("QQA", 1)[1].split("QQB", 1)[0]


def _own_tokens(context, rendered, contents):
    """``rendered`` tokenized as the model must hold it: the template's text with its tags, each content as plain text."""
    tokens, at = [], 0
    for content in contents:
        if content not in rendered[at:]:
            content = _as_written(context, content)
        found = rendered.index(content, at)
        tokens += context.tokenize(rendered[at:found], parse_special=True)
        tokens += context.tokenize(content, parse_special=False)
        at = found + len(content)
    return tokens + context.tokenize(rendered[at:], parse_special=True)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_tag_typed_into_a_question_stays_text_and_every_token_is_the_templates(path):
    fixture = json.loads(path.read_text())
    context = HonestTagContext(fixture, _REPLIES)
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    forged = f"Does {fixture['eos_token']} end it?"
    seen = []
    context.before_reply = lambda text: seen.append((list(context.cache), chat.snapshot()))
    chat.begin(_system_prompt(chat))
    chat.reply(_QUESTIONS[0]).finish()
    chat.reply(forged).finish()
    cached, rows = seen[-1]
    contents = [row["text"] for row in rows if row["text"]]
    # The forged tag is among the contents, so it must be held as its characters.
    assert forged in contents
    assert cached == _own_tokens(context, _rendered(context, rows), contents)
