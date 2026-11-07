"""Real chat templates of many families, each encoded as it renders itself.

``tests/templates/`` holds one template per family, read from a published
model file that each file names. The renderer is the one a model is loaded
with; only the tokenizer is a stand-in, one token per character, so what is
compared is the text.
"""

import json
from pathlib import Path

import pytest

from llama_chat.call_parsing import parse_calls
from llama_chat.reply_text import ThoughtFilter, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

from .template_contexts import TaggedContext, TagTokenizingContext, TemplateContext

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


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_two_user_turns_in_a_row_are_written_as_the_template_writes_them_or_refused(path):
    _fixture, context, template = _load(path)
    messages = [{"role": "user", "content": " ping "}, {"role": "user", "content": " again "}]
    rendered = context.render_with_model_template(messages, generation_prompt=True,
                                                  reasoning=False)
    formatter = template.formatter()
    ours = (template.preamble() + formatter.fragment("user", " ping ")
            + (template.user_turn_join or "") + formatter.fragment("user", " again ")
            + formatter.generation_prompt())
    if template.user_turn_join is None:
        assert rendered != ours
    else:
        assert rendered == ours


@pytest.mark.parametrize("stem, join", [("deepseek-r1-qwen3", "<｜Assistant｜>"), ("qwen-3.6", ""),
                                        ("mistral-3", None), ("deepseek-v4.1", None)])
def test_the_join_of_two_user_turns_is_read_from_the_template(stem, join):
    _fixture, _context, template = _load(Path(__file__).parent / "templates" / f"{stem}.json")
    assert template.user_turn_join == join


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_conversation_without_a_system_prompt_opens_as_the_template_opens_it(path):
    _fixture, context, template = _load(path)
    for reasoning in (False, True) if template.can_reason else (False,):
        formatter = template.formatter(reasoning)
        messages = [{"role": r, "content": t} for r, t in _CONVERSATION]
        rendered = context.render_with_model_template(
            messages, generation_prompt=True, reasoning=reasoning)
        ours = (template.preamble(reasoning)
                + "".join(formatter.fragment(r, t) for r, t in _CONVERSATION)
                + formatter.generation_prompt())
        assert ours == rendered


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
    fixture = json.loads(path.read_text())
    context = TaggedContext(fixture)
    template = ResolvedTemplate(context)
    template.ensure_tools_usable()
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", "sunny"), "It is sunny."))
    prompt = "Be brief." if template.has_system_turn else ""
    for reasoning in _reasoning_modes(template):
        formatter = template.formatter(reasoning)
        if template.declares_in_newest_turn:
            head, _, tail = (formatter.parts("system", "") if prompt
                             else (template.preamble(reasoning), "", ""))
            moving = template.newest_turn_declaration([_TOOL], reasoning)
        else:
            (head, tail), moving = template.system_turn_declaring([_TOOL], reasoning), ""
        past = template.past_tool_turn(turn, reasoning=reasoning, specs=[_TOOL])
        ours = (head + prompt + tail + formatter.fragment("user", "Weather?")
                + "".join(text for text, _ in past)
                + moving + formatter.fragment("user", "Thanks.") + formatter.generation_prompt())
        messages = [*([{"role": "system", "content": prompt}] if prompt else []),
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call00000", "type": "function", "function": {
                            "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
                    {"role": "tool", "name": "get_weather", "tool_call_id": "call00000",
                     "content": "sunny"},
                    {"role": "assistant", "content": "It is sunny."},
                    {"role": "user", "content": "Thanks."}]
        rendered = context.render_with_model_template(
            messages, tools=[_TOOL.as_declaration()], generation_prompt=True,
            reasoning=reasoning, **template.template_variables)
        assert ours == rendered, reasoning


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
        opener, closer = (template.call_texts if template.call_texts is not None
                          else (context.token_text(t) for t in template.tool_tokens))
        # The last call: a declaration may show an example call first.
        body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    assert parse_calls(body, template.call_body, template.call_lead_in) == [
        ToolCall("get_weather", {"city": "Berlin"})]


@pytest.mark.parametrize("path", list(_tool_templates()), ids=lambda p: p.stem)
def test_a_call_is_read_by_its_texts_only_where_no_tag_of_its_own_marks_it(path):
    context = TaggedContext(json.loads(path.read_text()))
    template = ResolvedTemplate(context)
    if template.call_texts is None:
        return
    assert template.tool_tokens is None
    if template.call_opens_as_text:
        return                          # no tag at all
    tag = context.token_text(context.first_special_token(template.call_texts[0]))
    reply = template.fragments
    assert tag in reply.assistant_prefix + reply.assistant_suffix + reply.generation_prompt


def test_kimi_k3_ends_a_reply_where_its_closing_parts_from_a_call():
    context = TaggedContext(json.loads((Path(__file__).parent / "templates"
                                         / "kimi-k3.json").read_text()))
    template = ResolvedTemplate(context)
    assert template.turn_end_text == "<|close|>response<|sep|><|close|>"
    assert template.turn_end_token is None
    assert template.call_body.string_type == "string"


@pytest.mark.parametrize("path", list(_tool_templates()),
                         ids=lambda p: p.stem)
def test_tools_without_a_system_prompt_open_as_the_template_opens_them(path):
    fixture = json.loads(path.read_text())
    context = TaggedContext(fixture)
    template = ResolvedTemplate(context)
    formatter = template.formatter()
    if template.declares_in_newest_turn:
        opening = template.preamble() + template.newest_turn_declaration([_TOOL])
    else:
        opening = template.preamble_declaring([_TOOL])
    ours = opening + formatter.fragment("user", "Weather?") + formatter.generation_prompt()
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "Weather?"}], tools=[_TOOL.as_declaration()],
        generation_prompt=True, reasoning=False, **template.template_variables)
    assert ours == rendered


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
    from llama_chat.resolved_template import written_between
    context = TaggedContext(json.loads(path.read_text()))
    template = ResolvedTemplate(context)
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", _QUOTED_RESULT), "Sunny."))
    past = "".join(text for text, _ in template.past_tool_turn(turn))
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "Weather?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
         {"role": "tool", "name": "get_weather", "tool_call_id": "call00000",
          "content": _QUOTED_RESULT},
         {"role": "assistant", "content": "Sunny."}, {"role": "user", "content": "Thanks."}],
        reasoning=False, **template.template_variables)
    assert past in rendered
    prefix, suffix = template.result_wrapping("get_weather")
    assert prefix + written_between(_QUOTED_RESULT, prefix, suffix) in rendered


def test_an_add_on_files_call_format_is_ignored_where_the_template_shows_one():
    # Qwen 3.6 marks its calls with tags of their own; a file that says otherwise loses.
    fixture = json.loads((Path(__file__).parent / "templates" / "qwen-3.6.json").read_text())
    context = TaggedContext(fixture)
    context.addon_call = lambda: {"opening": "CALL:", "name_close": " ", "closing": "END"}
    template = ResolvedTemplate(context)
    assert template.tool_tokens is not None and template.call_texts is None


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
    opener, closer = (template.call_texts if template.call_texts is not None
                      else (context.token_text(t) for t in template.tool_tokens))
    body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    read = parse_calls(body, template.call_body, template.call_lead_in)
    if any((path.stem, key) in _LOST for key in arguments):
        assert read != [ToolCall("fn", arguments)], "now read back: drop it from _LOST"
    else:
        assert read == [ToolCall("fn", arguments)]
