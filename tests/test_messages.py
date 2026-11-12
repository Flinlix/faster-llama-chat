"""A history as begin takes it, and as the messages a template is given."""

import itertools
from dataclasses import replace

from llama_chat.messages import as_history, as_messages, exchange_count, exchange_starts
from llama_chat.tools import ToolCall, ToolResult, ToolTurn


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
        [_QUESTION, ToolTurn(("Let me look.", ToolCall("a", {"x": 1}), ToolResult("a", "1"), "ok"))],
        [_QUESTION, ToolTurn((ToolCall("a", {}), ToolCall("b", {}), ToolResult("a", "1"),
                              ToolResult("b", "2"), "ok"))],
        [_QUESTION, ToolTurn((ToolCall("a", {}),))],                  # still owed a result
        [_QUESTION, ToolTurn((ToolCall("a", {}), ToolResult("a", "1")))],    # open after it
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


def _without_ids(item):
    if not isinstance(item, ToolTurn):
        return item
    return ToolTurn(tuple(replace(step, id="") if isinstance(step, ToolCall) else step
                          for step in item.steps))


def test_each_result_of_a_batch_answers_its_own_call():
    turn = ToolTurn((ToolCall("a", {}), ToolCall("b", {}), ToolResult("a", "1"),
                     ToolResult("b", "2"), "ok"))
    messages = as_messages([("user", "q"), turn])
    calls = [call["id"] for m in messages for call in m.get("tool_calls", [])]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == calls
