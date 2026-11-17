"""A history in the Chat Completions form begin takes, as the messages a template is given, and back."""

import itertools
import logging

import pytest

from llama_chat.history_forms import (answer_in_order, completions_messages, read_messages,
                                      without_unreadable)
from llama_chat.messages import (UNREADABLE_CALL, call_id, exchange_count, exchange_starts,
                                 history_count, next_call_number)
from llama_chat.tools import ToolCall
from tests.chat_messages import Result, opened, tool_turn
from llama_chat.errors import InvalidArgumentError, ArgumentTypeError
from tests.wrappers import tool_wrapper


def test_kept_history_opens_on_each_user_turn():
    messages = [{"role": r, "content": ""} for r in
                ("system", "user", "assistant", "user", "assistant")]
    assert exchange_starts(messages) == [1, 3]


def test_an_exchange_is_what_an_eviction_drops_at_once():
    """Counted as the oldest exchange is dropped: up to the next user message, until none is left."""
    for system, roles in itertools.product((True, False), itertools.product(
            ("user", "assistant", "tool"), repeat=4)):
        messages = [{"role": "system", "content": ""}] * system + [
            {"role": r, "content": ""} for r in roles]
        drops, rest = 0, list(messages[system:])
        while rest:
            following = next((i for i, m in enumerate(rest) if m["role"] == "user" and i > 0),
                             len(rest))
            rest, drops = rest[following:], drops + 1
        assert exchange_count(messages) == drops, roles


_QUESTION = {"role": "user", "content": "q"}
_HISTORIES = [
    [_QUESTION, {"role": "assistant", "content": "a"}],
    [{"role": "system", "content": "s"}, _QUESTION, {"role": "assistant", "content": "a"}],
    [{"role": "system", "content": ""}, _QUESTION],
    [_QUESTION, {"role": "assistant", "content": "a", "reasoning_content": "t"}],
    [_QUESTION, *tool_turn("Let me look.", ToolCall("a", {"x": 1, "b": [2]}), Result("a", "1"), "ok")],
    [_QUESTION, *tool_turn(ToolCall("a", {}), ToolCall("b", {}), Result("a", "1"),
                          Result("b", "2"), "ok")],
    [_QUESTION, *tool_turn(ToolCall("a", {}))],                       # still owed a result
    [_QUESTION, *tool_turn(ToolCall("a", {}), Result("a", "1"))],     # open after it
    # Left without a result, or without a reply after it, before a later question.
    [_QUESTION, *tool_turn(ToolCall("a", {})), _QUESTION, {"role": "assistant", "content": "a"}],
    [_QUESTION, *tool_turn(ToolCall("a", {}), Result("a", "1")), _QUESTION,
     *tool_turn(ToolCall("b", {}), Result("b", "2"), "ok")],
]


def _read(history):
    return answer_in_order(read_messages(history), "")


@pytest.mark.parametrize("history", [opened("", h) for h in _HISTORIES])
def test_every_history_in_the_form_comes_back_unchanged_but_for_the_spelling_of_arguments(history):
    back = completions_messages(_read(history), joins_questions=False)
    assert back == [_normalized(m) for m in history]
    assert completions_messages(_read(back), joins_questions=False) == back


def _normalized(message):
    """``message`` as the form gives it back: arguments compact, a reply that makes calls and says nothing null."""
    if not message.get("tool_calls"):
        return message
    calls = [{**c, "function": {**c["function"], "arguments": c["function"]["arguments"]
                                .replace(", ", ",").replace(": ", ":")}}
             for c in message["tool_calls"]]
    return {**message, "content": message["content"] or None, "tool_calls": calls}


def test_each_result_answers_its_own_call_and_is_named_for_it():
    turn = tool_turn(ToolCall("a", {}), ToolCall("b", {}), Result("a", "1"), Result("b", "2"), "ok")
    messages = _read(opened("", [_QUESTION, *turn]))
    calls = [(c["id"], c["function"]["name"]) for m in messages for c in m.get("tool_calls", [])]
    assert [(m["tool_call_id"], m["name"]) for m in messages if m["role"] == "tool"] == calls


def _call(arguments, name="f", call="call00000"):
    return [_QUESTION, {"role": "assistant", "content": None, "tool_calls": [
        {"id": call, "type": "function", "function": {"name": name, "arguments": arguments}}]}]


@pytest.mark.parametrize("arguments, reads", [
    ('{"x":1}', {"x": 1}), ('{ "x" : 1 }', {"x": 1}), ("", {}), ("{}", {}),
    ("null", None), ("  ", None), ("[1]", None), ('"x"', None), ("{", None), ("not json", None)])
def test_arguments_that_are_no_json_object_make_a_call_that_could_not_be_read(arguments, reads):
    call, = read_messages(_call(arguments))[1]["tool_calls"]
    if reads is None:
        assert UNREADABLE_CALL in call and call["id"] == "call00000"
    else:
        assert UNREADABLE_CALL not in call and call["function"]["arguments"] == reads


def test_a_call_that_could_not_be_read_is_left_out_with_its_result_and_its_id_given():
    history = [*_call("{"), {"role": "tool", "tool_call_id": "call00000", "content": "r"},
               {"role": "assistant", "content": "a"}]
    kept, gone = without_unreadable(read_messages(history), set(), joins_questions=False)
    assert kept == [_QUESTION, {"role": "assistant", "content": "a"}] and gone == {"call00000"}


def test_an_unreadable_call_owed_next_is_answered_and_one_behind_a_readable_call_waits():
    unreadable = {"id": "call00001", "type": "function", UNREADABLE_CALL: True,
                  "function": {"name": "unknown", "arguments": {}}}
    readable = {"id": "call00000", "type": "function", "function": {"name": "f", "arguments": {}}}
    calling = {"role": "assistant", "content": "", "tool_calls": [unreadable, readable]}
    answered = answer_in_order([_QUESTION, calling], "failed")
    assert [m.get("tool_call_id") for m in answered] == [None, None, "call00001"]
    waiting = {**calling, "tool_calls": [readable, unreadable]}
    assert answer_in_order([_QUESTION, waiting], "failed") == [_QUESTION, waiting]


@pytest.mark.parametrize("history, cause", [
    ([{"role": "developer", "content": "s"}], "role"),
    ([{"role": "user", "content": [{"type": "text", "text": "q"}]}], "list of parts"),
    ([{"role": "user", "content": "q", "name": "n"}], "'name'"),
    ([_QUESTION, {"role": "system", "content": "s"}], "only open"),
    ([_QUESTION, {"role": "assistant", "content": "a", "refusal": None}], "'refusal'"),
    ([_QUESTION, {"role": "assistant", "content": None}], "null"),
    (_call("{}", call=""), "'id'"),
    ([*_call("{}")[:1], {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "custom", "custom": {"name": "f", "input": "x"}}]}], "'custom'"),
    ([*_call("{}"), {"role": "tool", "content": "r"}], "tool_call_id"),
    ([*_call("{}"), {"role": "tool", "tool_call_id": "call00000", "content": 18}], "result of"),
    ([*_call("{}"), {"role": "tool", "tool_call_id": "call00000", "name": "f", "content": "r"}],
     "'name'"),
])
def test_what_the_form_holds_otherwise_is_refused_naming_the_field(history, cause):
    with pytest.raises(ValueError, match=cause):
        _read(history)


@pytest.mark.parametrize("history, cause", [
    ([{"role": "user", "content": 5}], "messages\\[0\\].content"),
    (["user"], "messages\\[0\\]"),
    (_call({"x": 1}), "arguments"),
])
def test_a_value_of_another_type_is_refused_naming_where_it_stands(history, cause):
    with pytest.raises(TypeError, match=cause):
        _read(history)


@pytest.mark.parametrize("history, cause", [
    ([*_call("{}"), {"role": "tool", "tool_call_id": "other", "content": "r"}], "before the result"),
    ([_QUESTION, {"role": "tool", "tool_call_id": "c", "content": "r"}], "answers no call"),
    ([*_call("{}"), {"role": "assistant", "content": "a"}], "before its result"),
    ([*_call("{}"), {"role": "tool", "tool_call_id": "call00000", "content": "r"},
      *_call("{}")], "two calls"),
])
def test_results_out_of_the_order_of_their_calls_are_refused(history, cause):
    with pytest.raises(ValueError, match=cause):
        _read(history)


def test_every_call_id_of_a_long_conversation_is_nine_letters_and_digits_and_new():
    numbers = [0, 9, 10, 35, 36, 99_999, 100_000, 1_000_000, 36 ** 5 - 1]
    ids = [call_id(n) for n in numbers]
    assert all(len(found) == 9 and found.isalnum() for found in ids)
    assert len(set(ids)) == len(ids)
    start = next_call_number(ids)
    assert len(call_id(start)) == 9 and call_id(start) not in ids


def test_ids_from_a_history_never_push_the_next_one_past_nine_characters():
    start = next_call_number(["callzzzzz", "call00000", "callbacks"])
    assert all(len(call_id(n)) == 9 for n in range(start, start + 1000))
    assert not {call_id(n) for n in range(start, start + 1000)} & {"callzzzzz", "call00000",
                                                                  "callbacks"}


def test_the_count_of_history_messages_leaves_out_only_the_system_message():
    kinds = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
             {"role": "tool", "name": "f", "tool_call_id": "c", "content": "r"},
             {"role": "system", "content": "s"}]
    for length in range(5):
        for messages in itertools.product(kinds, repeat=length):
            assert history_count(list(messages)) == sum(m["role"] != "system" for m in messages)


def test_a_reply_whose_calls_all_went_goes_on_in_the_next_reply_even_one_with_calls_left():
    """Its text and thought come before the next reply's, so no reply stands between a call and its result."""
    unreadable = {"id": "u", "type": "function", UNREADABLE_CALL: True,
                  "function": {"name": "unknown", "arguments": {}}}
    readable = {"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}
    messages = [_QUESTION,
                {"role": "assistant", "content": "Ich versuche es.", "reasoning_content": "eins",
                 "tool_calls": [unreadable]},
                {"role": "tool", "name": "unknown", "tool_call_id": "u", "content": "failed"},
                {"role": "assistant", "content": "Nochmal.", "reasoning_content": "zwei",
                 "tool_calls": [dict(unreadable, id="v"), readable]},
                {"role": "tool", "name": "unknown", "tool_call_id": "v", "content": "failed"},
                {"role": "tool", "name": "f", "tool_call_id": "c", "content": "12:00"},
                {"role": "assistant", "content": "Fertig."}]
    kept, gone = without_unreadable(messages, set(), joins_questions=False)
    assert gone == {"u", "v"}
    assert kept[1] == {"role": "assistant", "content": "Ich versuche es.\n\nNochmal.",
                       "reasoning_content": "eins\n\nzwei", "tool_calls": [readable]}
    _read(completions_messages(messages, joins_questions=False))      # a history begin takes


def test_replies_whose_calls_all_went_in_a_row_go_on_in_the_next_reply_as_one():
    unreadable = {"id": "u", "type": "function", UNREADABLE_CALL: True,
                  "function": {"name": "unknown", "arguments": {}}}
    messages = [_QUESTION]
    for n, said in enumerate(("Ich schaue.", "", "Nochmal.")):
        messages += [{"role": "assistant", "content": said, "tool_calls": [dict(unreadable, id=f"u{n}")]},
                     {"role": "tool", "name": "unknown", "tool_call_id": f"u{n}", "content": "failed"}]
    messages.append({"role": "assistant", "content": "Fertig."})
    kept, gone = without_unreadable(messages, set(), joins_questions=False)
    assert gone == {"u0", "u1", "u2"}
    assert kept == [_QUESTION, {"role": "assistant", "content": "Ich schaue.\n\nNochmal.\n\nFertig."}]


def test_a_call_left_out_shares_no_id_with_one_kept():
    with pytest.raises(ValueError, match="two calls"):
        read_messages([*_call("{", call="x"), {"role": "tool", "tool_call_id": "x", "content": "r"},
                       {"role": "assistant", "content": "a"}, *_call("{}", call="x")])


def test_a_call_name_of_the_wrong_type_is_refused_naming_where_it_stands():
    from llama_chat.errors import ArgumentTypeError
    from tests.wrappers import tool_wrapper
    wrapper, _ = tool_wrapper()
    called = {"type": "function", "id": "c1", "function": {"name": 5, "arguments": "{}"}}
    with pytest.raises(ArgumentTypeError, match=r"function\.name must be a str"):
        wrapper.begin([{"role": "user", "content": "a"},
                       {"role": "assistant", "content": "", "tool_calls": [called]}])


_FUNCTION = {"name": "f", "arguments": "{}"}


def _calling(call) -> list[dict]:
    return [_QUESTION, {"role": "assistant", "content": None, "tool_calls": [call]}]


@pytest.mark.parametrize("call, cause", [
    ({"id": "c", "type": "function", "function": _FUNCTION, "index": 0}, "'index'"),
    ({"id": "c", "type": "custom", "function": _FUNCTION}, "'custom'"),
    ({"type": "function", "function": _FUNCTION}, "'id'"),
    ({"id": "", "type": "function", "function": _FUNCTION}, "'id'"),
    ({"id": "c", "type": "function", "function": {**_FUNCTION, "strict": True}}, "'strict'"),
    ({"id": "c", "type": "function"}, "function holds 'name' and 'arguments'"),
    ({"id": "c", "type": "function", "function": "f"}, "function holds 'name' and 'arguments'"),
])
def test_a_call_in_a_form_not_taken_is_refused_as_messages_naming_where_it_stands_and_why(call, cause):
    with pytest.raises(InvalidArgumentError) as refused:
        read_messages(_calling(call))
    assert refused.value.field == "messages"
    assert "messages[1].tool_calls[0]" in str(refused.value)
    assert cause in str(refused.value)


def test_a_call_without_its_type_is_read_as_a_function_call():
    typed = {"id": "c", "type": "function", "function": _FUNCTION}
    untyped = {"id": "c", "function": _FUNCTION}
    assert read_messages(_calling(untyped)) == read_messages(_calling(typed))


def test_a_call_without_its_arguments_is_read_as_a_call_with_no_arguments():
    without = {"id": "c", "type": "function", "function": {"name": "f"}}
    empty = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    assert read_messages(_calling(without)) == read_messages(_calling(empty))


@pytest.mark.parametrize("field", ["name", "arguments"])
@pytest.mark.parametrize("given", [5, 1.5, True, ["f"], {"x": 1}, b"f"])
def test_a_call_field_of_the_wrong_type_is_refused_naming_the_type_it_was_given(field, given):
    function = {"name": "f", "arguments": "{}", field: given}
    with pytest.raises(ArgumentTypeError) as refused:
        read_messages(_calling({"id": "c", "type": "function", "function": function}))
    assert f"messages[1].tool_calls[0].function.{field} must be" in str(refused.value)
    assert str(refused.value).endswith(f"not {type(given).__name__}")


@pytest.mark.parametrize("function", [{"name": "f", "arguments": "{"}, {"name": "", "arguments": "{}"}])
def test_a_call_that_could_not_be_read_is_said_once_at_info(function, caplog):
    with caplog.at_level(logging.INFO, logger="llama_chat"):
        read_messages(_calling({"id": "c", "type": "function", "function": function}))
    said = [r for r in caplog.records if r.name == "llama_chat.history_forms"]
    assert len(said) == 1 and said[0].levelno == logging.INFO
    assert "could not be read" in said[0].getMessage()


_RESULTS_NOT_TAKEN = [
    ({"role": "tool", "content": "r"}, "'tool_call_id'"),
    ({"role": "tool", "tool_call_id": "", "content": "r"}, "'tool_call_id'"),
    ({"role": "tool", "tool_call_id": 7, "content": "r"}, "'tool_call_id'"),
    ({"role": "tool", "tool_call_id": ["call00000"], "content": "r"}, "'tool_call_id'"),
    ({"role": "tool", "tool_call_id": "call00000", "content": 18}, "result of 'call00000'"),
    ({"role": "tool", "tool_call_id": "call00000", "content": None}, "result of 'call00000'"),
]


@pytest.mark.parametrize("result, cause", _RESULTS_NOT_TAKEN, ids=[
    "no id", "empty id", "id a number", "id a list", "content a number", "content null"])
def test_a_result_in_a_form_not_taken_is_refused_as_messages_naming_where_it_stands_and_why(
        result, cause):
    wrapper, _ = tool_wrapper()
    before = wrapper.conversation()
    with pytest.raises(InvalidArgumentError) as refused:
        wrapper.begin([*_call("{}"), result])
    assert refused.value.field == "messages"
    assert "messages[2]" in str(refused.value) and cause in str(refused.value)
    assert wrapper.conversation() == before


@pytest.mark.parametrize("history", [
    [*_call("{}"), {"role": "tool", "tool_call_id": "other", "content": "r"}],
    [_QUESTION, {"role": "tool", "tool_call_id": "c", "content": "r"}],
], ids=["another call's id", "no call owed"])
def test_a_result_out_of_the_order_of_its_calls_is_refused_as_messages(history):
    with pytest.raises(InvalidArgumentError) as refused:
        _read(history)
    assert refused.value.field == "messages"
