"""Reading a call back out of what a model generated."""

import pytest

from llama_chat.call_parsing import parse_calls
from llama_chat.template import CallBody
from llama_chat.tools import ToolCall
from tests.invented_templates import CALL_BODY


_KEY_VALUE = CallBody(name_close="", key_open="<key_is>", key_close="</key_is><value_is>",
                      pair_sep="</value_is><key_is>", body_close="</value_is>",
                      empty_close="")


_ELEMENTS = CallBody(name_close='">|', key_open="<", key_close=">", pair_sep="|</{key}>|<",
                     body_close="|</{key}>|", empty_close="", key_in_close=True,
                     member_open="|<{key}>", member_close="|</{key}>",
                     item_open="|<entry>", item_close="|</entry>")


# ----- reading a call back -------------------------------------------------
def test_a_call_parses_into_a_name_and_typed_arguments():
    call, = parse_calls("fn:calendar_add{title:'Zahnarzt',minutes:30,all_day:false}",
                      CALL_BODY, "fn:")
    assert call == ToolCall(name="calendar_add",
                            arguments={"title": "Zahnarzt", "minutes": 30,
                                       "all_day": False})


@pytest.mark.parametrize("written", ["01234", "1e5", "nan", "True", "None", "null"])
def test_a_value_written_without_a_type_mark_keeps_its_spelling_where_the_schema_asks_for_text(
        written):
    schema = {"say": {"type": "object", "properties": {"text": {"type": "string"},
                                                       "items": {"type": "array",
                                                                 "items": {"type": "string"}}}}}
    call, = parse_calls(f"fn:say{{text:{written},items:[01234,nan]}}", CALL_BODY, "fn:", schema)
    assert call.arguments == {"text": written, "items": ["01234", "nan"]}
    assert all(type(value) is str for value in [call.arguments["text"], *call.arguments["items"]])


_TEXT_OR_NOTHING = [{"type": "string"}, {"type": "null"}]


@pytest.mark.parametrize("text, items, definitions, read_as", [
    ({"anyOf": _TEXT_OR_NOTHING}, {"type": "array", "items": {"anyOf": _TEXT_OR_NOTHING}}, {},
     "01234"),
    ({"oneOf": _TEXT_OR_NOTHING}, {"anyOf": [{"type": "array", "items": {"oneOf": _TEXT_OR_NOTHING}},
                                             {"type": "null"}]}, {}, "01234"),
    ({"$ref": "#/$defs/Text"}, {"type": "array", "items": {"$ref": "#/$defs/Text"}},
     {"$defs": {"Text": {"anyOf": _TEXT_OR_NOTHING}}}, "01234"),
    ({"$ref": "#/definitions/Text"}, {"$ref": "#/definitions/Texts"},
     {"definitions": {"Text": {"anyOf": _TEXT_OR_NOTHING},
                      "Texts": {"type": "array", "items": {"$ref": "#/definitions/Text"}}}},
     "01234"),
    # A branch that asks for the number it spells reads it as that number.
    ({"anyOf": [{"$ref": "#/$defs/Number"}, {"$ref": "#/$defs/Text"}]},
     {"type": "array", "items": {"type": ["string", "null"]}},
     {"$defs": {"Number": {"type": "number"}, "Text": {"type": "string"}}}, 1234),
], ids=["any-of", "one-of", "ref-to-defs", "ref-to-definitions", "any-of-refs"])
def test_a_value_written_without_a_type_mark_keeps_its_spelling_where_any_branch_asks_for_text(
        text, items, definitions, read_as):
    schema = {"say": {"type": "object", "properties": {"text": text, "items": items},
                      **definitions}}
    call, = parse_calls("fn:say{text:01234,items:[01234,null]}", CALL_BODY, "fn:", schema)
    assert call.arguments == {"text": read_as, "items": ["01234", None]}


def test_a_value_written_without_a_type_mark_is_the_literal_it_spells_where_the_schema_allows_it():
    schema = {"set": {"type": "object", "properties": {
        "n": {"type": "integer"}, "x": {"type": ["string", "null"]},
        "on": {"type": ["boolean", "string"]}, "pick": {"enum": ["01", 2]}}}}
    call, = parse_calls("fn:set{n:01234,x:null,on:true,pick:01,free:2}", CALL_BODY, "fn:", schema)
    assert call.arguments == {"n": 1234, "x": None, "on": True, "pick": "01", "free": 2}


def test_a_quoted_argument_may_hold_the_separators():
    call, = parse_calls("fn:say{text:'a,b:c{d}'}", CALL_BODY, "fn:")
    assert call.arguments == {"text": "a,b:c{d}"}


def test_nested_objects_and_lists_survive():
    call, = parse_calls("fn:x{where:{city:'Bern'},days:[1,2]}", CALL_BODY, "fn:")
    assert call.arguments == {"where": {"city": "Bern"}, "days": [1, 2]}


@pytest.mark.parametrize("body", ["", "fn:", "fn:name", "fn:{a:1}"])
def test_something_that_is_not_a_call_is_refused(body):
    with pytest.raises(ValueError):
        parse_calls(body, CALL_BODY, "fn:")


# ----- a call that names itself --------------------------------------------
_JSON_KEYS = ("name", "arguments")


def test_a_json_call_is_read_by_the_keys_its_template_names():
    call, = parse_calls('{"name": "calendar_add", '
                      '"arguments": {"title": "Zahnarzt", "minutes": 30}}', None,
                        json_keys=_JSON_KEYS)
    assert call == ToolCall("calendar_add", {"title": "Zahnarzt", "minutes": 30})


def test_a_json_number_reaches_a_text_parameter_as_the_model_wrote_it():
    schema = {"f": {"type": "object", "properties": {
        "zip": {"type": "string"}, "version": {"type": "string"}, "count": {"type": "number"}}}}
    call, = parse_calls('{"name": "f", "arguments": {"zip": 12345, "version": 3.10, "count": 3.10}}',
                        None, json_keys=_JSON_KEYS, parameters=schema)
    assert call.arguments == {"zip": "12345", "version": "3.10", "count": 3.1}
    assert type(call.arguments["count"]) is float


def test_a_json_number_without_a_schema_stays_a_number_and_a_numbered_name_is_no_call():
    call, = parse_calls('{"name": "f", "arguments": {"n": 12, "x": 1e5}}', None, json_keys=_JSON_KEYS)
    assert call.arguments == {"n": 12, "x": 100000.0} and type(call.arguments["n"]) is int
    with pytest.raises(ValueError):
        parse_calls('{"name": 7, "arguments": {}}', None, json_keys=_JSON_KEYS)


@pytest.mark.parametrize("raw", [
    '{"name": "x", "parameters": {"a": 1}}',        # keys the template does not name
    '{"name": "x", "arguments": "{\\"a\\": 1}"}',   # arguments nested as text
    'fn:x{a:1}',                                    # not JSON at all
])
def test_a_call_in_another_shape_than_its_templates_json_is_unreadable(raw):
    with pytest.raises(ValueError):
        parse_calls(raw, CALL_BODY, "fn:", json_keys=_JSON_KEYS)


def test_a_json_call_is_not_read_where_the_template_writes_none():
    with pytest.raises(ValueError):
        parse_calls('{"name": "x", "arguments": {}}', CALL_BODY, "fn:")


@pytest.mark.parametrize("raw", [
    '{"name": "x", "arguments": [1, 2]}',
    '{"name": "x", "arguments": null}',
    '{"name": "x", "parameters": null}',
    '{"name": "x", "arguments": "[1, 2]"}',
    '{"name": "x", "arguments": "null"}',
])
def test_arguments_that_are_no_object_make_the_call_unreadable(raw):
    """Read as no arguments, the tool would run with none of what the model gave."""
    with pytest.raises(ValueError):
        parse_calls(raw, None, json_keys=_JSON_KEYS)


_NESTED_TOO_DEEPLY = "[" * 150 + "]" * 150
_NESTED_PAST_THE_RECURSION_LIMIT = "[" * 3000 + "]" * 3000


@pytest.mark.parametrize("deep", [_NESTED_TOO_DEEPLY, _NESTED_PAST_THE_RECURSION_LIMIT])
@pytest.mark.parametrize("written", ['{"name": "f", "arguments": {"a": %s}}',
                                     "fn:f{a:%s}"], ids=["json", "the-model-s-own"])
def test_arguments_nested_too_deeply_make_the_call_unreadable_on_every_path(written, deep):
    json_keys = _JSON_KEYS if written.startswith("{") else None
    with pytest.raises(ValueError, match="nest"):
        parse_calls(written % deep, CALL_BODY, "fn:", json_keys=json_keys)


def test_without_a_grammar_or_json_keys_a_call_is_refused():
    with pytest.raises(ValueError, match="no call_body"):
        parse_calls("fn:clock{}", None, "fn:")


# ----- a model that writes a batch -----------------------------------------
def test_a_json_array_yields_every_call_it_holds():
    calls = parse_calls('[{"name": "clock", "arguments": {}}, '
                        '{"name": "calendar_add", "arguments": {"title": "x"}}]',
                        None, json_keys=_JSON_KEYS)
    assert calls == [ToolCall("clock", {}),
                     ToolCall("calendar_add", {"title": "x"})]


# ----- reading a body the model got wrong ----------------------------------
def test_an_empty_argument_value_reads_as_an_empty_string():
    """A key the model wrote no value for is an empty string, not a crash.

    The container test used to be ``text[:1] in "{["``, and an empty string is
    a substring of every string, so an empty value took the object branch and
    indexed past its own end.
    """
    assert parse_calls("fn:clock{city:}", CALL_BODY, "fn:") == [
        ToolCall("clock", {"city": ""})]


def test_an_argument_with_no_separator_at_all_is_refused():
    with pytest.raises(ValueError, match="an argument with no value"):
        parse_calls("fn:clock{city}", CALL_BODY, "fn:")


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
    assert parse_calls("""fn:say{text:"a"b"}""", CALL_BODY, "fn:") == [
        ToolCall("say", {"text": '"a"b"'})]


def test_nested_arguments_that_are_not_json_are_not_a_call():
    # Some models nest the arguments as JSON text; text that is not JSON means
    # this was never a call worth acting on.
    with pytest.raises(ValueError):
        parse_calls('{"name":"clock","arguments":"not json"}', None)


def test_a_nested_object_written_in_the_model_s_own_delimiters():
    # The value is neither JSON nor quoted, so it is read with the same
    # delimiters again, one level down.
    assert parse_calls("fn:plan{where:{city:'Bonn',day:2}}", CALL_BODY, "fn:") == [
        ToolCall("plan", {"where": {"city": "Bonn", "day": 2}})]


def test_a_list_value_written_in_the_model_s_own_delimiters():
    assert parse_calls("fn:plan{days:['Mo','Di',3]}", CALL_BODY, "fn:") == [
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
    call, = parse_calls("f<key_is>x</key_is><value_is>{x}</value_is>", _KEY_VALUE)
    assert call == ToolCall("f", {"x": "{x}"})


def test_whitespace_after_the_last_value_is_not_part_of_it():
    call, = parse_calls("f<key_is>city</key_is><value_is>Berlin</value_is>\n", _KEY_VALUE)
    assert call == ToolCall("f", {"city": "Berlin"})


def test_a_call_body_that_cannot_end_a_name_is_refused():
    with pytest.raises(ValueError, match="name_close or a key_open"):
        CallBody(name_close="", key_open="", key_close=":", pair_sep=",",
                 body_close="", empty_close="")


def test_a_json_call_may_name_its_function_and_arguments_by_other_keys():
    body = '[{"tool_call_id": "0", "tool_name": "clock", "parameters": {"zone": "CET"}}]'
    assert parse_calls(body, None, json_keys=("tool_name", "parameters", "tool_call_id")) == [
        ToolCall("clock", {"zone": "CET"})]


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
    text = ('f">|<a>|<a>|<a>1|</a>|</a>|</a>|<rows>|<entry>|<x>1|</x>|<y>b|</y>|</entry>'
            '|<entry>|<entry>2|</entry>|</entry>|</rows>|')
    assert parse_calls(text, _ELEMENTS) == [ToolCall("f", {"a": {"a": {"a": 1}},
                                                          "rows": [{"x": 1, "y": "b"}, [2]]})]


def test_a_string_that_opens_like_a_member_stays_a_string():
    assert parse_calls('f">|<a>|<b oops|</a>|', _ELEMENTS) == [ToolCall("f", {"a": "|<b oops"})]


def test_an_object_whose_only_key_is_the_items_name_reads_as_the_list_it_is_written_as():
    assert parse_calls('f">|<a>|<entry>1|</entry>|</a>|', _ELEMENTS) == [ToolCall("f", {"a": [1]})]


def test_nested_delimiters_come_whole_or_not_at_all():
    with pytest.raises(ValueError, match="nested delimiters"):
        CallBody(name_close="(", key_open="", key_close="=", pair_sep=",", body_close=")",
                 empty_close=")", member_open="<{key}>", member_close="</{key}>")


def test_a_number_nested_in_a_json_value_of_a_delimited_call_is_read_as_written():
    schema = {"f": {"type": "object", "properties": {"addr": {"type": "object", "properties": {
        "zip": {"type": "string"}}}}}}
    call, = parse_calls('fn:f{addr:{"zip": 12345}}', CALL_BODY, "fn:", parameters=schema)
    assert call.arguments == {"addr": {"zip": "12345"}}


def test_a_number_too_long_to_read_makes_the_call_unreadable_instead_of_infinite():
    schema = {"f": {"type": "object", "properties": {"n": {"type": "number"}}}}
    with pytest.raises(ValueError, match="too long"):
        parse_calls('{"name": "f", "arguments": {"n": 1' + "0" * 5000 + "}}", None,
                    json_keys=_JSON_KEYS, parameters=schema)
