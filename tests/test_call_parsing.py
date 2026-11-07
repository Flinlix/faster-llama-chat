"""Reading a call back out of what a model generated."""

import json

import pytest

from llama_chat.call_parsing import parse_calls
from llama_chat.config import Config
from llama_chat.template import CallBody, ToolCallFragments
from llama_chat.template_extract import (extract_call_body, extract_fragments,
                                         extract_tool_fragments)
from llama_chat.tools import ToolCall
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import FakeContext
from tests.wrappers import FAKE_CALL_BODY


ELEMENT = CallBody(name_close=">\n<", key_open="parameter=", key_close=">\n",
                   pair_sep="\n</parameter>\n<parameter=",
                   body_close="\n</parameter>\n</function>\n",
                   empty_close="/function>\n")


def _element_body(name, arguments):
    """What an element-shaped model generates between its call tags."""
    parts = [f"\n<function={name}>\n"]
    for key, value in arguments.items():
        parts.append(f"<parameter={key}>\n{value}\n</parameter>\n")
    parts.append("</function>\n")
    return "".join(parts)


def _json_object(name, arguments):
    return json.dumps({"name": name, "arguments": arguments})


def _json_array(name, arguments):
    return json.dumps([{"name": name, "arguments": arguments}])


def _python_call(name, arguments):
    return f"{name}(" + ", ".join(
        f"{k}={json.dumps(v)}" for k, v in arguments.items()) + ")"


def _attribute_xml(name, arguments):
    body = "".join(f'<parameter name="{k}">{json.dumps(v)}</parameter>\n'
                   for k, v in arguments.items())
    return f'<invoke name="{name}">\n{body}</invoke>'


def _key_value_tags(name, arguments):
    return name + "".join(
        f"<arg_key>{k}</arg_key><arg_value>"
        + (v if isinstance(v, str) else json.dumps(v)) + "</arg_value>"
        for k, v in arguments.items())


SHAPES = {
    "key value tags": _key_value_tags,
    "json object": _json_object,
    "json array": _json_array,
    "python call": _python_call,
    "attribute xml": _attribute_xml,
    "delimited": lambda n, a: f"call:{n}{{" + ",".join(
        f"{k}:" + (f"<|\"|>{v}<|\"|>" if isinstance(v, str) else json.dumps(v))
        for k, v in a.items()) + "}",
}


ARGUMENTS = [{}, {"title": "Zahnarzt"}, {"title": "Zahnarzt", "minutes": 30},
             {"where": {"city": "Bern"}}, {"days": [1, 2]},
             {"title": "a,b:c"}, {"flag": True}, {"nothing": None}]


_KEY_VALUE = CallBody(name_close="", key_open="<arg_key>", key_close="</arg_key><arg_value>",
                      pair_sep="</arg_value><arg_key>", body_close="</arg_value>",
                      empty_close="")


_ELEMENTS = CallBody(name_close='">|', key_open="<", key_close=">", pair_sep="|</{key}>|<",
                     body_close="|</{key}>|", empty_close="", key_in_close=True,
                     member_open="|<{key}>", member_close="|</{key}>",
                     item_open="|<item>", item_close="|</item>")


# ----- reading a call back -------------------------------------------------
def test_a_call_parses_into_a_name_and_typed_arguments():
    call, = parse_calls("fn:calendar_add{title:'Zahnarzt',minutes:30,all_day:false}",
                      FAKE_CALL_BODY, "fn:")
    assert call == ToolCall(name="calendar_add",
                            arguments={"title": "Zahnarzt", "minutes": 30,
                                       "all_day": False})


def test_a_quoted_argument_may_hold_the_separators():
    call, = parse_calls("fn:say{text:'a,b:c{d}'}", FAKE_CALL_BODY, "fn:")
    assert call.arguments == {"text": "a,b:c{d}"}


def test_nested_objects_and_lists_survive():
    call, = parse_calls("fn:x{where:{city:'Bern'},days:[1,2]}", FAKE_CALL_BODY, "fn:")
    assert call.arguments == {"where": {"city": "Bern"}, "days": [1, 2]}


@pytest.mark.parametrize("body", ["", "fn:", "fn:name", "fn:{a:1}"])
def test_something_that_is_not_a_call_is_refused(body):
    with pytest.raises(ValueError):
        parse_calls(body, FAKE_CALL_BODY, "fn:")


# ----- a call that names itself --------------------------------------------
def test_a_json_call_needs_no_grammar_at_all():
    call, = parse_calls('{"name": "calendar_add", '
                      '"arguments": {"title": "Zahnarzt", "minutes": 30}}', None)
    assert call == ToolCall("calendar_add", {"title": "Zahnarzt", "minutes": 30})


@pytest.mark.parametrize("raw", [
    '{"name": "x", "parameters": {"a": 1}}',        # the other spelling
    '{"name": "x", "arguments": "{\\"a\\": 1}"}',   # arguments nested as text
])
def test_the_shapes_the_json_family_comes_in(raw):
    assert parse_calls(raw, None)[0].arguments == {"a": 1}


def test_without_a_grammar_anything_but_json_is_refused():
    with pytest.raises(ValueError, match="no call_body"):
        parse_calls("fn:clock{}", None, "fn:")


# ----- the same parser, an unlike model ------------------------------------
@pytest.mark.parametrize("arguments", [
    {},
    {"title": "Zahnarzt"},
    {"title": "Zahnarzt", "minutes": 30},
])
def test_an_element_shaped_model_reads_with_the_same_parser(arguments):
    call, = parse_calls(_element_body("calendar_add", arguments), ELEMENT,
                      "\n<function=")
    assert call == ToolCall("calendar_add", arguments)


# ----- the shapes a model may write a call in ------------------------------
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("arguments", ARGUMENTS)
def test_every_shape_round_trips_through_recovery_and_parsing(shape, arguments):
    """One model family's call shape survives recovery and parsing intact.

    Render a call the way that family does, recover the call body from the
    rendering, and read the call back equal to what went in.
    """
    emit = SHAPES[shape]

    def render(messages, tools=None, **_options):
        out = []
        for m in messages:
            calls = m.get("tool_calls", ())
            inner = "".join("<tc>" + emit(c["function"]["name"],
                                          c["function"]["arguments"]) + "</tc>"
                            for c in calls)
            out.append(f"<[turn>{m['role']}\n{inner}{m['content'].strip()}"
                       + ("" if calls else "<turn]>\n"))
        return "".join(out)

    tools = extract_tool_fragments(render, extract_fragments(render))
    call_body = extract_call_body(render, tools.call_open, "</tc>")
    lead_in = tools.call_open[len("<tc>"):]

    call, = parse_calls(emit("calendar_add", arguments), call_body, lead_in)
    assert call == ToolCall("calendar_add", arguments)


def test_a_grammar_may_be_supplied_where_the_probes_cannot_recover_one():
    # The same escape hatch `fragments=` is for a template that cannot be read.
    fake = FakeContext(calls_tools=True)
    spelled = CallBody(name_close="(", key_open="", key_close="=",
                       pair_sep=",", body_close=")", empty_close=")",
                       quote_open='"', quote_close='"')
    wrapper = ChatWrapper(Config(context_size=512), context=fake,
                          tool_calls=ToolCallFragments(
                              call_open="<call>fn:", call_close="}</call>",
                              body=spelled))
    assert wrapper._template.call_body is spelled


# ----- a model that writes a batch -----------------------------------------
def test_a_json_array_yields_every_call_it_holds():
    calls = parse_calls('[{"name": "clock", "arguments": {}}, '
                        '{"name": "calendar_add", "arguments": {"title": "x"}}]',
                        None)
    assert calls == [ToolCall("clock", {}),
                     ToolCall("calendar_add", {"title": "x"})]


def test_an_array_holding_something_that_is_not_a_call_is_not_a_batch():
    # All or nothing: a half-read batch would silently drop work.
    with pytest.raises(ValueError):
        parse_calls('[{"name": "clock", "arguments": {}}, 42]', None)


# ----- reading a body the model got wrong ----------------------------------
def test_an_empty_argument_value_reads_as_an_empty_string():
    """A key the model wrote no value for is an empty string, not a crash.

    The container test used to be ``text[:1] in "{["``, and an empty string is
    a substring of every string, so an empty value took the object branch and
    indexed past its own end.
    """
    assert parse_calls("fn:clock{city:}", FAKE_CALL_BODY, "fn:") == [
        ToolCall("clock", {"city": ""})]


def test_an_argument_with_no_separator_at_all_is_refused():
    with pytest.raises(ValueError, match="an argument with no value"):
        parse_calls("fn:clock{city}", FAKE_CALL_BODY, "fn:")


def test_a_batch_entry_that_is_not_a_call_voids_the_whole_batch():
    # All or nothing: half a batch would silently drop what the model asked for.
    with pytest.raises(ValueError):
        parse_calls('[{"name":"clock","arguments":{}}, 7]', None)


def test_a_batch_entry_with_no_usable_name_voids_the_whole_batch():
    with pytest.raises(ValueError):
        parse_calls('[{"name":"clock","arguments":{}}, {"arguments":{}}]', None)


def test_a_value_that_only_looks_like_a_json_string_is_read_as_written():
    # It opens and closes with a quote, so JSON is tried first; it is not valid
    # JSON, so the model's own quoting decides instead.
    assert parse_calls("""fn:say{text:"a"b"}""", FAKE_CALL_BODY, "fn:") == [
        ToolCall("say", {"text": '"a"b"'})]


def test_nested_arguments_that_are_not_json_are_not_a_call():
    # Some models nest the arguments as JSON text; text that is not JSON means
    # this was never a call worth acting on.
    with pytest.raises(ValueError):
        parse_calls('{"name":"clock","arguments":"not json"}', None)


def test_a_nested_object_written_in_the_model_s_own_delimiters():
    # The value is neither JSON nor quoted, so it is read with the same
    # delimiters again, one level down.
    assert parse_calls("fn:plan{where:{city:'Bonn',day:2}}", FAKE_CALL_BODY, "fn:") == [
        ToolCall("plan", {"where": {"city": "Bonn", "day": 2}})]


def test_a_list_value_written_in_the_model_s_own_delimiters():
    assert parse_calls("fn:plan{days:['Mo','Di',3]}", FAKE_CALL_BODY, "fn:") == [
        ToolCall("plan", {"days": ["Mo", "Di", 3]})]


def test_a_grammar_with_no_pair_separator_reads_one_argument():
    # A model that never writes two arguments has no separator to recover, and
    # the whole body is then a single pair.
    single = CallBody(name_close="(", key_open="", key_close="=", pair_sep="",
                      body_close=")", empty_close=")",
                      quote_open='"', quote_close='"')
    assert parse_calls('fn:clock(city="Bonn")', single, "fn:") == [
        ToolCall("clock", {"city": "Bonn"})]


# ----- reading values back out of a call -----------------------------------
def test_a_bare_string_that_looks_nested_stays_a_string():
    call, = parse_calls("f<arg_key>x</arg_key><arg_value>{x}</arg_value>", _KEY_VALUE)
    assert call == ToolCall("f", {"x": "{x}"})


def test_whitespace_after_the_last_value_is_not_part_of_it():
    call, = parse_calls("f<arg_key>city</arg_key><arg_value>Berlin</arg_value>\n", _KEY_VALUE)
    assert call == ToolCall("f", {"city": "Berlin"})


def test_a_call_body_that_cannot_end_a_name_is_refused():
    with pytest.raises(ValueError, match="name_close or a key_open"):
        CallBody(name_close="", key_open="", key_close=":", pair_sep=",",
                 body_close="", empty_close="")


def test_a_json_call_may_name_its_function_as_tool_name():
    body = '[{"tool_call_id": "0", "tool_name": "clock", "parameters": {"zone": "CET"}}]'
    assert parse_calls(body, None) == [ToolCall("clock", {"zone": "CET"})]


def test_a_value_after_its_written_type_is_read_by_that_type():
    body = CallBody(name_close='"', key_open="<a ", key_close=' type="', pair_sep="</a><a ",
                    body_close="</a>", empty_close='"', type_close='">', string_type="string")
    text = ('clock"<a zone type="string">12</a><a n type="number">3</a>'
            '<a o type="object">{"x": [1]}</a>')
    assert parse_calls(text, body) == [ToolCall("clock", {"zone": "12", "n": 3, "o": {"x": [1]}})]


def test_a_value_closed_by_a_tag_naming_its_key_is_read_to_that_tag():
    body = CallBody(name_close='">', key_open="<", key_close=">", pair_sep="</{key}><",
                    body_close="</{key}>", empty_close="", key_in_close=True)
    text = 'clock"><zone>CET</zone><n>3</n>'
    assert parse_calls(text, body) == [ToolCall("clock", {"zone": "CET", "n": 3})]


def test_nested_values_in_the_model_s_own_elements_are_read_by_depth():
    text = ('f">|<a>|<a>|<a>1|</a>|</a>|</a>|<rows>|<item>|<x>1|</x>|<y>b|</y>|</item>'
            '|<item>|<item>2|</item>|</item>|</rows>|')
    assert parse_calls(text, _ELEMENTS) == [ToolCall("f", {"a": {"a": {"a": 1}},
                                                          "rows": [{"x": 1, "y": "b"}, [2]]})]


def test_a_string_that_opens_like_a_member_stays_a_string():
    assert parse_calls('f">|<a>|<b oops|</a>|', _ELEMENTS) == [ToolCall("f", {"a": "|<b oops"})]


def test_an_object_whose_only_key_is_the_items_name_reads_as_the_list_it_is_written_as():
    assert parse_calls('f">|<a>|<item>1|</item>|</a>|', _ELEMENTS) == [ToolCall("f", {"a": [1]})]


def test_nested_delimiters_come_whole_or_not_at_all():
    with pytest.raises(ValueError, match="nested delimiters"):
        CallBody(name_close="(", key_open="", key_close="=", pair_sep=",", body_close=")",
                 empty_close=")", member_open="<{key}>", member_close="</{key}>")
