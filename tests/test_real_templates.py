"""Real chat templates of many families, each through the check a model meets at load, and what that load reads from it.

``tests/templates/`` holds one template per family, read from a published
model file that each file names. The renderer is the one a model is loaded
with; only the tokenizer is a stand-in, one token per character and one per
tag, so what is compared is the text. Whole conversations on these templates
are in ``test_conversations.py``.
"""

import json

import pytest

from llama_chat import ChatWrapper, Config
from llama_chat.addon_files import addon_file, template_fingerprint
from llama_chat.reply_text import ReplyText, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tokenizing import first_special_token
from llama_chat.tools import ToolCall, ToolRegistry

from .chat_templates import KNOWN_KEYS, REQUIRED_KEYS, STEMS, WEATHER, expected, fixture
from .fake_context import FakeContext, call_marks, shipped_addon_call, written_call

_TOOL_STEMS = [stem for stem in STEMS if fixture(stem)["calls_tools"]]
_REASONING_STEMS = [stem for stem in STEMS if fixture(stem)["can_reason"]]
# The templates that show no call, whose shipped add-on file says how the model writes one.
_ADDON_CALL_STEMS = [stem for stem in _TOOL_STEMS if shipped_addon_call(fixture(stem)["chat_template"])]


def _load(stem):
    context = FakeContext(fixture(stem))
    # Construction runs the check every model meets at load: a system
    # prompt, a past reply and a newest turn, in each reasoning mode.
    return context, ResolvedTemplate(context, shipped_addon_call(fixture(stem)["chat_template"]))


@pytest.mark.parametrize("stem", STEMS)
def test_a_test_templates_file_states_every_key_it_must_and_no_unknown_one(stem):
    keys = set(fixture(stem))
    assert not sorted(REQUIRED_KEYS - keys) and not sorted(keys - KNOWN_KEYS)


@pytest.mark.parametrize("stem", STEMS)
def test_the_template_loads_and_supports_what_it_is_known_to(stem):
    known = fixture(stem)
    _context, template = _load(stem)
    assert (template.reasoning is not None) == known["marks_thought"]
    assert template.can_reason == known["can_reason"]
    assert _describes_a_call(template) == known["calls_tools"]
    assert template.answer_opening() == expected(known, "answer_opening")


# The templates that write nothing after a reply: each loads only because the
# tag that opens the next turn ends a generation on its model.
_ENDING_TAG_STEMS = [stem for stem in STEMS if expected(fixture(stem), "ends_generation")]


@pytest.mark.parametrize("stem", _ENDING_TAG_STEMS)
def test_a_reply_closed_by_nothing_is_refused_at_load_unless_the_next_turns_tag_ends_a_generation(stem):
    known = fixture(stem)
    context, template = _load(stem)
    assert not template.fragments.assistant_suffix.strip()
    next_turn = template.fragments.user_prefix.strip()
    assert context.ends_generation(first_special_token(context, next_turn))
    with pytest.raises(ValueError, match="writes nothing after a reply") as refused:
        ResolvedTemplate(FakeContext({**known, "ends_generation": []}),
                         shipped_addon_call(known["chat_template"]))
    assert repr(context.token_text(first_special_token(context, next_turn))) in str(refused.value)


@pytest.mark.parametrize("stem", _REASONING_STEMS)
def test_reasoning_can_be_turned_off_exactly_where_the_template_writes_its_off_otherwise(stem):
    context, template = _load(stem)
    question = [{"role": "user", "content": "x"}]
    on, off = (context.stated_render(question, generation_prompt=True, reasoning=reasoning)
               for reasoning in (True, False))
    assert template.can_turn_off_reasoning == (on != off)


@pytest.mark.parametrize("stem", _REASONING_STEMS)
def test_a_template_with_or_without_an_off_says_so_and_still_begins_without_reasoning(stem):
    _context, template = _load(stem)
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8),
                       context=FakeContext(fixture(stem)))
    assert chat.can_reason and chat.can_turn_off_reasoning == template.can_turn_off_reasoning
    chat.begin("", reasoning=False)             # nothing that works today is refused


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


@pytest.mark.parametrize("stem", STEMS)
def test_a_reply_ends_at_the_first_tag_of_its_closing(stem):
    context, template = _load(stem)
    turn_end = template.turn_end_token
    found = context.token_text(turn_end) if turn_end is not None else None
    assert found == expected(fixture(stem), "turn_end_token")


@pytest.mark.parametrize("stem", _REASONING_STEMS)
def test_a_thought_closing_with_the_turns_end_does_not_end_the_reply(stem):
    """A thought may close with the tag that also closes the turn; only the turn's own closing ends the reply."""
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
    call = shipped_addon_call(fixture(stem)["chat_template"])
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
    assert boundary.opens_as_text == expected(fixture(stem), "call_opens_as_text")
    if boundary.call_texts is None:
        return
    assert boundary.tool_tokens is None
    if boundary.opens_as_text:
        return                          # no tag at all
    tag = context.token_text(first_special_token(context, boundary.call_texts[0]))
    reply = template.fragments
    assert tag in reply.assistant_prefix + reply.assistant_suffix + reply.generation_prompt


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_a_reply_ends_where_its_closing_parts_from_a_call(stem):
    _context, template = _load(stem)
    assert template.call_boundary().turn_end_text == expected(fixture(stem), "turn_end_text")


@pytest.mark.parametrize("stem", [stem for stem in STEMS if expected(fixture(stem), "answer_opening")])
def test_a_reply_routed_through_a_channel_streams_only_its_answer(stem):
    _context, template = _load(stem)
    opening = template.answer_opening()
    generated = opening + "Antwort."
    reply = ReplyText(template.reasoning, opening, template.generation_prompt(), [])
    shown = "".join(reply.feed(c) for c in generated) + reply.finish("end_of_turn")
    assert shown == "Antwort."


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_an_add_on_files_call_format_is_ignored_where_the_template_shows_one(stem):
    _context, template = _load(stem)
    if template.call_boundary().tool_tokens is None:
        return                          # no tag of its own marks a call
    # The template marks its calls with tags of their own; a file that says otherwise loses.
    boundary = ResolvedTemplate(FakeContext(fixture(stem)),
                                {"opening": "CALL:", "name_close": " ", "closing": "END"}).call_boundary()
    assert boundary.tool_tokens is not None and boundary.call_texts is None


@pytest.mark.parametrize("stem", _ADDON_CALL_STEMS)
def test_an_add_on_files_call_format_makes_a_template_that_hides_its_call_callable(tmp_path, stem):
    known = fixture(stem)
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The time.", name="clock")

    # An add-on file of its own that names the template and says nothing of a call.
    (tmp_path / "model.json").write_text(json.dumps(
        {"templates": [template_fingerprint(known["chat_template"])]}))
    without = FakeContext(known, ["Fine."])
    chat = ChatWrapper(Config(model_path=str(tmp_path / "model.gguf"), context_size=4096,
                              min_reply_tokens=8), context=without)
    assert not chat.can_call_tools
    with pytest.raises(ValueError, match="add-on file can say how it writes one"):
        chat.begin("", tools=registry.specs())

    context = FakeContext(known, ["Fine."])
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=context)
    assert chat.can_call_tools
    chat.begin("", tools=registry.specs())      # the whole chain is checked, and passes
    # Read with the spelling the file gives.
    call = shipped_addon_call(known["chat_template"])
    written = "clock" + call["name_close"] + "{}"
    assert ResolvedTemplate(context, call).read_calls(written) == [ToolCall("clock", {})]


@pytest.mark.parametrize("stem", _ADDON_CALL_STEMS)
def test_an_add_on_file_closing_a_call_with_text_the_template_never_writes_is_refused(stem):
    known = fixture(stem)
    with pytest.raises(ValueError, match="add-on file closes a call"):
        ResolvedTemplate(FakeContext(known), {**shipped_addon_call(known["chat_template"]),
                                              "closing": "<|never|>"})


_ARGUMENTS = [{}, {"title": "Zahnarzt"}, {"title": "Zahnarzt", "minutes": 30},
              {"where": {"city": "Bern"}}, {"days": [1, 2]},
              {"title": "a,b:c"}, {"flag": True}, {"nothing": None},
              {"inner": {"inner": {"n": 1}}}, {"rows": [{"x": 1, "y": "b"}, [2, "c"]]}]
_ARGUMENT_STEMS = [stem for stem in _TOOL_STEMS
                   if not addon_file(None, fixture(stem)["chat_template"])["call"]]


def _call_body(context, template, arguments):
    """The body of a past call to ``fn`` with ``arguments``, as the template writes it."""
    return written_call(template, context, "fn", arguments)[1]


@pytest.mark.parametrize("arguments", _ARGUMENTS, ids=lambda a: "-".join(a) or "none")
@pytest.mark.parametrize("stem", _ARGUMENT_STEMS)
def test_every_argument_shape_reads_back_as_the_template_wrote_it(stem, arguments):
    context, template = _load(stem)
    read = template.read_calls(_call_body(context, template, arguments))
    if set(arguments) & set(expected(fixture(stem), "lost_arguments")):
        assert read != [ToolCall("fn", arguments)], "now read back: drop it from the template's file"
    else:
        assert read == [ToolCall("fn", arguments)]


@pytest.mark.parametrize("stem", _ARGUMENT_STEMS)
def test_a_text_that_reads_as_a_number_reads_back_as_text_where_the_template_writes_it_as_one(stem):
    context, template = _load(stem)
    as_text = _call_body(context, template, {"city": "42"})
    if as_text == _call_body(context, template, {"city": 42}):
        return                          # written alike, so it cannot be told from the number
    assert template.read_calls(as_text) == [ToolCall("fn", {"city": "42"})]


@pytest.mark.parametrize("stem", STEMS)
def test_an_empty_reply_is_written_where_the_templates_file_says(stem):
    context = FakeContext(fixture(stem))
    replied = [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]
    written = (context.stated_render(replied) is not None
               or context.stated_render(replied + [{"role": "user", "content": "y"}],
                                        generation_prompt=True) is not None)
    assert written == expected(fixture(stem), "writes_empty_reply")


@pytest.mark.parametrize("stem", STEMS)
def test_a_template_that_reads_the_clock_writes_the_date_as_its_file_says(stem, clock):
    known = fixture(stem)
    date_format = expected(known, "date_format")
    assert (date_format is not None) == ("strftime_now" in known["chat_template"])
    if date_format is not None:
        rendered = FakeContext(known).stated_render([{"role": "user", "content": "x"}],
                                                    generation_prompt=True)
        assert clock[0].strftime(date_format) in rendered
