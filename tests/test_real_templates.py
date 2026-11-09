"""Real chat templates of many families, each through the check a model meets at load, and what that load reads from it.

``tests/templates/`` holds one template per family, read from a published
model file that each file names. The renderer is the one a model is loaded
with; only the tokenizer is a stand-in, one token per character and one per
tag, so what is compared is the text. Whole conversations on these templates
are in ``test_conversations.py``.
"""

import pytest

from llama_chat import ChatWrapper, Config
from llama_chat.addon_files import addon_file
from llama_chat.reply_text import ReplyText, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tokenizing import first_special_token
from llama_chat.tools import ToolCall, ToolRegistry

from .chat_templates import STEMS, WEATHER, fixture
from .fake_context import FakeContext, call_marks

_TOOL_STEMS = [stem for stem in STEMS if fixture(stem)["calls_tools"]]
_REASONING_STEMS = [stem for stem in STEMS if fixture(stem)["can_reason"]]


def _load(stem):
    context = FakeContext(fixture(stem))
    # Construction runs the check every model meets at load: a system
    # prompt, a past reply and a newest turn, in each reasoning mode.
    return context, ResolvedTemplate(context)


@pytest.mark.parametrize("stem", STEMS)
def test_the_template_loads_and_supports_what_it_is_known_to(stem):
    known = fixture(stem)
    _context, template = _load(stem)
    assert (template.reasoning is not None) == known["marks_thought"]
    assert template.can_reason == known["can_reason"]
    assert _describes_a_call(template) == known["calls_tools"]


def _describes_a_call(template):
    """Whether the template writes a call at all, whatever else tool calling lacks."""
    try:
        template.ensure_tools_usable()
    except ValueError as missing:
        return "describes no tool call" not in str(missing)
    return True


@pytest.mark.parametrize("stem", _REASONING_STEMS)
def test_a_reply_that_reasons_streams_only_its_answer(stem):
    _context, template = _load(stem)
    tags, prompt = template.reasoning, template.generation_prompt(True)
    inside = begins_in_thought(tags, prompt)
    generated = ("" if inside else tags.thought_open) + "Denken" \
        + tags.thought_close + "Antwort."
    reply = ReplyText(tags, template.answer_opening(True), prompt, [])
    shown = "".join(reply.feed(piece) for piece in generated) + reply.finish("end_of_turn")
    assert shown == "Antwort."
    assert reply.thought == "Denken" and reply.thought_pieces > 0


@pytest.mark.parametrize("stem, first_tag", [("cohere-aya", "<|END_RESPONSE|>"),
                                             ("qwen-3.6", "<|im_end|>")])
def test_a_reply_ends_at_the_first_tag_of_its_closing(stem, first_tag):
    context, template = _load(stem)
    assert context.token_text(template.turn_end_token) == first_tag


@pytest.mark.parametrize("stem", _REASONING_STEMS)
def test_a_thought_closing_with_the_turns_end_does_not_end_the_reply(stem):
    """A thought may close with the tag that also closes the turn, as gpt-oss's does; only the turn's own closing ends the reply."""
    context, template = _load(stem)
    tags, turn_end = template.reasoning, template.turn_end_token
    prompt = template.generation_prompt(True)
    inside = begins_in_thought(tags, prompt)
    reply_text = ReplyText(tags, template.answer_opening(True), prompt, [])
    # A reply that ends at a text rather than a tag is ended by the generation loop.
    ends = (reply_text.turn_end_test(turn_end, context.token_text(turn_end))
            if turn_end is not None else lambda token: False)
    reply = (("" if inside else tags.thought_open) + "Denken" + tags.thought_close + "Antwort."
             + (template.fragments.assistant_suffix if turn_end is not None else ""))
    shown = ""
    for token in context.tokenize(reply):
        if ends(token):
            break
        shown += reply_text.feed(context.token_text(token))
    assert shown + reply_text.finish("end_of_turn") == "Antwort."


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_a_call_written_as_the_template_writes_it_reads_back_as_made(stem):
    context, template = _load(stem)
    template.ensure_tools_usable()
    call = context.addon_call()
    if call:
        # The add-on file says how the model writes a call, where the template
        # writes it in another order; a call in that form reads back.
        body = "get_weather" + call["name_close"] + '{"city": "Berlin"}'
    else:
        rendered = context.stated_render(
            [{"role": "user", "content": "Weather?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "call00000", "type": "function", "function": {
                     "name": "get_weather", "arguments": {"city": "Berlin"}}}]}],
            tools=[WEATHER.as_declaration()])
        opener, closer = call_marks(template, context)
        # The last call: a declaration may show an example call first.
        body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    assert template.read_calls(body) == [ToolCall("get_weather", {"city": "Berlin"})]


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_a_call_is_read_by_its_texts_only_where_no_tag_of_its_own_marks_it(stem):
    context, template = _load(stem)
    boundary = template.call_boundary()
    if boundary.call_texts is None:
        return
    assert boundary.tool_tokens is None
    if boundary.opens_as_text:
        return                          # no tag at all
    tag = context.token_text(first_special_token(context, boundary.call_texts[0]))
    reply = template.fragments
    assert tag in reply.assistant_prefix + reply.assistant_suffix + reply.generation_prompt


def test_kimi_k3_ends_a_reply_where_its_closing_parts_from_a_call():
    context, template = _load("kimi-k3")
    assert template.call_boundary().turn_end_text == "<|close|>response<|sep|><|close|>"
    assert template.turn_end_token is None
    # A value written with the type a string is written with reads back as the text it is.
    rendered = context.stated_render(
        [{"role": "user", "content": "Weather?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "42"}}}]}],
        tools=[WEATHER.as_declaration()])
    opener, closer = call_marks(template, context)
    body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    assert template.read_calls(body) == [ToolCall("get_weather", {"city": "42"})]


@pytest.mark.parametrize("stem", ["gpt-oss", "muse-glimmer"])
def test_a_reply_routed_through_a_channel_streams_only_its_answer(stem):
    _context, template = _load(stem)
    opening = template.answer_opening()
    assert opening
    generated = opening + "Antwort."
    reply = ReplyText(template.reasoning, opening, template.generation_prompt(), [])
    shown = "".join(reply.feed(c) for c in generated) + reply.finish("end_of_turn")
    assert shown == "Antwort."


def test_an_add_on_files_call_format_is_ignored_where_the_template_shows_one():
    # Qwen 3.6 marks its calls with tags of their own; a file that says otherwise loses.
    context = FakeContext(fixture("qwen-3.6"))
    context.addon_call = lambda: {"opening": "CALL:", "name_close": " ", "closing": "END"}
    boundary = ResolvedTemplate(context).call_boundary()
    assert boundary.tool_tokens is not None and boundary.call_texts is None


def test_an_add_on_files_call_format_makes_a_template_that_hides_its_call_callable():
    # gpt-oss writes a call's opening as plain text its replies also hold.
    known = fixture("gpt-oss")
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")

    without = FakeContext(known, ["Fine."])
    without.addon_call = lambda: {}
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=without)
    assert not chat.can_call_tools
    with pytest.raises(ValueError, match="add-on file can say how it writes one"):
        chat.begin("", tools=registry)

    context = FakeContext(known, ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    assert chat.can_call_tools
    chat.begin("", tools=registry)          # the whole chain is checked, and passes
    # Read with the spelling the file gives.
    written = "clock" + context.addon_call()["name_close"] + "{}"
    assert ResolvedTemplate(context).read_calls(written) == [ToolCall("clock", {})]


def test_an_add_on_file_closing_a_call_with_text_the_template_never_writes_is_refused():
    context = FakeContext(fixture("gpt-oss"))
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
@pytest.mark.parametrize("stem", [stem for stem in _TOOL_STEMS
                                  if not addon_file(None, fixture(stem)["chat_template"])["call"]])
def test_every_argument_shape_reads_back_as_the_template_wrote_it(stem, arguments):
    context, template = _load(stem)
    rendered = context.stated_render(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function",
              "function": {"name": "fn", "arguments": arguments}}]}])
    opener, closer = call_marks(template, context)
    body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    read = template.read_calls(body)
    if any((stem, key) in _LOST for key in arguments):
        assert read != [ToolCall("fn", arguments)], "now read back: drop it from _LOST"
    else:
        assert read == [ToolCall("fn", arguments)]
