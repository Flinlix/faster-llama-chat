"""Real chat templates of many families, each encoded as it renders itself.

``tests/templates/`` holds one template per family, read from a published
model file that each file names. The renderer is the one a model is loaded
with; only the tokenizer is a stand-in, one token per character, so what is
compared is the text.
"""

import json
from pathlib import Path

import pytest

from llama_chat import ChatWrapper, Config
from llama_chat.reply_text import ThoughtFilter, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

from .template_contexts import (ConversationContext, TaggedContext, TagTokenizingContext,
                                TemplateContext)

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_CONVERSATION = [("user", " ping "), ("assistant", " pong "), ("user", " again ")]


def _load(path):
    fixture = json.loads(path.read_text())
    context = TemplateContext(fixture)
    # Construction runs the check every model meets at load: a system
    # prompt, a past reply and a newest turn, in each reasoning mode.
    return fixture, context, ResolvedTemplate(context)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_the_template_loads_and_supports_what_it_is_known_to(path):
    fixture, _context, template = _load(path)
    assert (template.reasoning is not None) == fixture["marks_thought"]
    assert template.can_reason == fixture["can_reason"]
    assert (template._tool_calls is not None) == fixture["calls_tools"]


def _held_before_the_reply(path, history, question, system=None, tools=None, reasoning=False):
    """What a wrapper on ``path``'s template holds when it starts the reply to ``question``."""
    context = ConversationContext(json.loads(path.read_text()), ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = seen.append
    if system is None:
        system = "Be brief." if chat._template.has_system_turn else ""
    chat.begin(system, history, tools=tools, reasoning=reasoning)
    chat.reply(question).finish()
    return seen[0], context, chat


def _render(context, chat, messages, tools=None, reasoning=False):
    system = [{"role": "system", "content": chat.system_prompt}] if chat.system_prompt else []
    return chat._template.render(
        system + messages, [spec.as_declaration() for spec in tools] if tools else None,
        generation_prompt=True, reasoning=reasoning)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_two_user_turns_in_a_row_are_written_as_the_template_writes_them_or_refused(path):
    try:
        held, context, chat = _held_before_the_reply(path, [("user", " ping ")], " again ")
    except ValueError as refused:
        assert "two user turns" in str(refused)
        return
    assert held == _render(context, chat, [{"role": "user", "content": " ping "},
                                           {"role": "user", "content": " again "}])


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_conversation_without_a_system_prompt_opens_as_the_template_opens_it(path):
    _fixture, _context, template = _load(path)
    for reasoning in (False, True) if template.can_reason else (False,):
        history = _CONVERSATION[:2]
        held, context, chat = _held_before_the_reply(path, history, _CONVERSATION[2][1],
                                                     system="", reasoning=reasoning)
        messages = [{"role": r, "content": t} for r, t in _CONVERSATION]
        assert held == _render(context, chat, messages, reasoning=reasoning), reasoning


@pytest.mark.parametrize("path", [p for p in _TEMPLATES
                                  if json.loads(p.read_text())["can_reason"]],
                         ids=lambda p: p.stem)
def test_a_reply_that_reasons_streams_only_its_answer(path):
    _fixture, _context, template = _load(path)
    tags = template.reasoning
    opening = template.formatter(True).generation_prompt()
    inside = begins_in_thought(tags, opening)
    generated = ("" if inside else tags.thought_open) + "Denken" \
        + tags.thought_close + "Antwort."
    thought = ThoughtFilter(tags, inside)
    shown = "".join(thought.feed(piece) for piece in generated) + thought.flush()
    assert shown == "Antwort."
    assert thought.thought_pieces > 0


@pytest.mark.parametrize("stem, first_tag", [("cohere-aya", "<|END_RESPONSE|>"),
                                             ("qwen-3.6", "<|im_end|>")])
def test_a_reply_ends_at_the_first_tag_of_its_closing(stem, first_tag):
    fixture = json.loads((Path(__file__).parent / "templates" / f"{stem}.json").read_text())
    context = TagTokenizingContext(fixture)
    template = ResolvedTemplate(context)
    assert context.token_text(template.turn_end_token) == first_tag


@pytest.mark.parametrize("stem", ["gpt-oss", "muse-glimmer", "qwen-3.6"])
def test_a_thought_closing_with_the_turns_end_does_not_end_the_reply(stem):
    """gpt-oss closes a thought with the tag that also closes its turn."""
    from llama_chat.reply_text import AnswerOpening, turn_end_test
    fixture = json.loads((Path(__file__).parent / "templates" / f"{stem}.json").read_text())
    context = TagTokenizingContext(fixture)
    template = ResolvedTemplate(context)
    tags, turn_end = template.reasoning, template.turn_end_token
    formatter = template.formatter(True)
    inside = begins_in_thought(tags, formatter.generation_prompt())
    thought = ThoughtFilter(tags, inside)
    opening = AnswerOpening(formatter.answer_opening())
    ends = turn_end_test(turn_end, context.token_text(turn_end), tags, thought)
    reply = (("" if inside else tags.thought_open) + "Denken" + tags.thought_close
             + "Antwort." + template._fragments.assistant_suffix)
    shown = ""
    for token in context.tokenize(reply):
        if ends(token):
            break
        shown += opening.feed(thought.feed(context.token_text(token)))
    assert shown + opening.feed(thought.flush()) + opening.flush() == "Antwort."


# Families whose calls cannot be read, so tools are refused, and why. A family
# whose shipped add-on file says how it writes a call is not among them.
_TOOL = ToolSpec(name="get_weather", description="Weather for a city",
                 parameters={"type": "object", "properties": {"city": {"type": "string"}}})


def _tool_templates():
    for path in _TEMPLATES:
        fixture = json.loads(path.read_text())
        if fixture["calls_tools"]:
            yield path


def _reasoning_modes(template):
    return (False, True) if template.can_reason else (False,)


@pytest.mark.parametrize("path", list(_tool_templates()), ids=lambda p: p.stem)
def test_a_conversation_with_a_past_tool_turn_is_encoded_as_it_renders(path):
    _fixture, _context, template = _load(path)
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", "sunny"), "It is sunny."))
    for reasoning in _reasoning_modes(template):
        held, context, chat = _held_before_the_reply(path, [("user", "Weather?"), turn],
                                                     "Thanks.", tools=[_TOOL],
                                                     reasoning=reasoning)
        assert held == _render(context, chat, _past_turn_messages("sunny") + [
            {"role": "user", "content": "Thanks."}], [_TOOL], reasoning), reasoning


def _past_turn_messages(result: str) -> list[dict]:
    return [{"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call00000", "type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
            {"role": "tool", "name": "get_weather", "tool_call_id": "call00000",
             "content": result},
            {"role": "assistant", "content": "It is sunny." if result == "sunny" else "Sunny."}]


@pytest.mark.parametrize("path", list(_tool_templates()),
                         ids=lambda p: p.stem)
def test_a_call_written_as_the_template_writes_it_reads_back_as_made(path):
    fixture = json.loads(path.read_text())
    context = TaggedContext(fixture)
    template = ResolvedTemplate(context)
    template.ensure_tools_usable()
    call = context.addon_call()
    if call:
        # The add-on file says how the model writes a call, where the template
        # writes it in another order; a call in that form reads back.
        body = "get_weather" + call["name_close"] + '{"city": "Berlin"}'
    else:
        rendered = context.render_with_model_template(
            [{"role": "user", "content": "Weather?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "call00000", "type": "function", "function": {
                     "name": "get_weather", "arguments": {"city": "Berlin"}}}]}],
            tools=[_TOOL.as_declaration()], reasoning=False)
        boundary = template.call_boundary()
        opener, closer = (boundary.call_texts if boundary.call_texts is not None
                          else (context.token_text(t) for t in boundary.tool_tokens))
        # The last call: a declaration may show an example call first.
        body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    assert template.read_calls(body) == [ToolCall("get_weather", {"city": "Berlin"})]


@pytest.mark.parametrize("path", list(_tool_templates()), ids=lambda p: p.stem)
def test_a_call_is_read_by_its_texts_only_where_no_tag_of_its_own_marks_it(path):
    context = TaggedContext(json.loads(path.read_text()))
    template = ResolvedTemplate(context)
    boundary = template.call_boundary()
    if boundary.call_texts is None:
        return
    assert boundary.tool_tokens is None
    if boundary.opens_as_text:
        return                          # no tag at all
    tag = context.token_text(context.first_special_token(boundary.call_texts[0]))
    reply = template.fragments
    assert tag in reply.assistant_prefix + reply.assistant_suffix + reply.generation_prompt


def test_kimi_k3_ends_a_reply_where_its_closing_parts_from_a_call():
    context = TaggedContext(json.loads((Path(__file__).parent / "templates"
                                         / "kimi-k3.json").read_text()))
    template = ResolvedTemplate(context)
    assert template.call_boundary().turn_end_text == "<|close|>response<|sep|><|close|>"
    assert template.turn_end_token is None
    assert template._call_body.string_type == "string"


@pytest.mark.parametrize("path", list(_tool_templates()),
                         ids=lambda p: p.stem)
def test_tools_without_a_system_prompt_open_as_the_template_opens_them(path):
    held, context, chat = _held_before_the_reply(path, [], "Weather?", system="", tools=[_TOOL])
    assert held == _render(context, chat, [{"role": "user", "content": "Weather?"}], [_TOOL])


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if p.stem in ("gpt-oss", "muse-glimmer")],
                         ids=lambda p: p.stem)
def test_a_reply_routed_through_a_channel_streams_only_its_answer(path):
    from llama_chat.reply_text import AnswerOpening
    _fixture, _context, template = _load(path)
    formatter = template.formatter()
    assert formatter.answer_opening()
    generated = formatter.answer_opening() + "Antwort."
    thought, opening = ThoughtFilter(template.reasoning), AnswerOpening(formatter.answer_opening())
    shown = "".join(opening.feed(thought.feed(c)) for c in generated)
    shown += opening.feed(thought.flush()) + opening.flush()
    assert shown == "Antwort."


_QUOTED_RESULT = '{"weather": "sunny"}\nsecond line'


@pytest.mark.parametrize("path", list(_tool_templates()),
                         ids=lambda p: p.stem)
def test_a_result_with_quotes_is_written_as_the_template_writes_it(path):
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", _QUOTED_RESULT), "Sunny."))
    held, context, chat = _held_before_the_reply(path, [("user", "Weather?"), turn], "Thanks.",
                                                 tools=[_TOOL])
    assert held == _render(context, chat, _past_turn_messages(_QUOTED_RESULT) + [
        {"role": "user", "content": "Thanks."}], [_TOOL])


def test_an_add_on_files_call_format_is_ignored_where_the_template_shows_one():
    # Qwen 3.6 marks its calls with tags of their own; a file that says otherwise loses.
    fixture = json.loads((Path(__file__).parent / "templates" / "qwen-3.6.json").read_text())
    context = TaggedContext(fixture)
    context.addon_call = lambda: {"opening": "CALL:", "name_close": " ", "closing": "END"}
    template = ResolvedTemplate(context)
    assert template.tool_tokens is not None and template.call_boundary().call_texts is None


def test_an_add_on_file_closing_a_call_with_text_the_template_never_writes_is_refused():
    fixture = json.loads((Path(__file__).parent / "templates" / "gpt-oss.json").read_text())
    context = TaggedContext(fixture)
    context.addon_call = lambda: {"opening": "<|channel|>commentary to=functions.",
                                  "name_close": " <|constrain|>json<|message|>",
                                  "closing": "<|never|>"}
    with pytest.raises(ValueError, match="add-on file closes a call"):
        ResolvedTemplate(context)


_ARGUMENTS = [{}, {"title": "Zahnarzt"}, {"title": "Zahnarzt", "minutes": 30},
              {"where": {"city": "Bern"}}, {"days": [1, 2]},
              {"title": "a,b:c"}, {"flag": True}, {"nothing": None},
              {"inner": {"inner": {"n": 1}}}, {"rows": [{"x": 1, "y": "b"}, [2, "c"]]}]
# Arguments that do not come back as made, and why.
_LOST = {
    ("minimax-m3.1", "nothing"): "the template leaves a null out of the call",
}


@pytest.mark.parametrize("arguments", _ARGUMENTS, ids=lambda a: "-".join(a) or "none")
@pytest.mark.parametrize("path", [p for p in _tool_templates()
                                  if not TaggedContext(json.loads(p.read_text())).addon_call()],
                         ids=lambda p: p.stem)
def test_every_argument_shape_reads_back_as_the_template_wrote_it(path, arguments):
    context = TaggedContext(json.loads(path.read_text()))
    template = ResolvedTemplate(context)
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function",
              "function": {"name": "fn", "arguments": arguments}}]}], reasoning=False)
    boundary = template.call_boundary()
    opener, closer = (boundary.call_texts if boundary.call_texts is not None
                      else (context.token_text(t) for t in boundary.tool_tokens))
    body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    read = template.read_calls(body)
    if any((path.stem, key) in _LOST for key in arguments):
        assert read != [ToolCall("fn", arguments)], "now read back: drop it from _LOST"
    else:
        assert read == [ToolCall("fn", arguments)]
