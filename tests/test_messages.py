"""A history as begin takes it, and as the messages a template is given."""

import itertools

from llama_chat.messages import (as_history, as_messages, call_id, exchange_count,
                                 exchange_starts, next_call_number)
from llama_chat.tools import ToolCall
from tests.chat_messages import Result, tool_turn


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


def test_every_history_comes_back_as_it_went_in():
    histories = [
        [_QUESTION, {"role": "assistant", "content": "a"}],
        [_QUESTION, {"role": "assistant", "content": "a", "reasoning_content": "t"}],
        [_QUESTION, *tool_turn("Let me look.", ToolCall("a", {"x": 1}), Result("a", "1"), "ok")],
        [_QUESTION, *tool_turn(ToolCall("a", {}), ToolCall("b", {}), Result("a", "1"),
                              Result("b", "2"), "ok")],
        [_QUESTION, *tool_turn(ToolCall("a", {}))],                  # still owed a result
        [_QUESTION, *tool_turn(ToolCall("a", {}), Result("a", "1"))],    # open after it
        # Left without a result, or without a reply after it, before a later question.
        [_QUESTION, *tool_turn(ToolCall("a", {})), _QUESTION, {"role": "assistant", "content": "a"}],
        [_QUESTION, *tool_turn(ToolCall("a", {}), Result("a", "1")), _QUESTION,
         *tool_turn(ToolCall("b", {}), Result("b", "2"), "ok")],
    ]
    for history in histories:
        back = as_history(as_messages(history))
        # Each call comes back with the id it was given, and goes in again under it.
        assert [_without_ids(item) for item in back] == history
        assert as_history(as_messages(back)) == back


def test_a_message_given_as_a_pair_comes_back_as_its_dictionary():
    pairs = [("user", "q"), ("assistant", "a")]
    assert as_history(as_messages(pairs)) == [_QUESTION, {"role": "assistant", "content": "a"}]
    assert all(isinstance(item, dict) for item in as_history(as_messages(pairs)))


def _without_ids(message):
    """``message`` as it was given: its calls without the ids they were given, a result without the call and function it answers."""
    if message["role"] == "tool":
        return {key: value for key, value in message.items() if key not in ("tool_call_id", "name")}
    if "tool_calls" not in message:
        return message
    return {**message, "tool_calls": [{key: value for key, value in call.items() if key != "id"}
                                      for call in message["tool_calls"]]}


def test_each_result_of_a_batch_answers_its_own_call():
    turn = tool_turn(ToolCall("a", {}), ToolCall("b", {}), Result("a", "1"),
                     Result("b", "2"), "ok")
    messages = as_messages([("user", "q"), *turn])
    calls = [call["id"] for m in messages for call in m.get("tool_calls", [])]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == calls


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


def test_the_count_of_history_messages_is_what_reading_the_history_back_gives():
    import itertools

    from llama_chat.messages import as_history, history_count

    calling = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call00000", "type": "function", "function": {"name": "f", "arguments": {}}}]}
    kinds = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, calling,
             {"role": "tool", "name": "f", "tool_call_id": "call00000", "content": "r"},
             {"role": "system", "content": "s"}]
    for length in range(6):
        for messages in itertools.product(kinds, repeat=length):
            try:
                history = as_history(list(messages))
            except ValueError:
                continue                # a result before its call is no history
            assert history_count(list(messages)) == len(history)
