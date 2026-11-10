"""A history as begin takes it, and as the messages a template is given."""

from llama_chat.messages import as_history, as_messages, exchange_starts
from llama_chat.tools import ToolCall, ToolResult, ToolTurn


def test_kept_history_opens_on_each_user_turn():
    messages = [{"role": r, "content": ""} for r in
                ("system", "user", "assistant", "user", "assistant")]
    assert exchange_starts(messages) == [1, 3]


def test_every_history_comes_back_as_it_went_in():
    histories = [
        [("user", "q"), ("assistant", "a")],
        [("user", "q"), ToolTurn(("Let me look.", ToolCall("a", {"x": 1}), ToolResult("a", "1"), "ok"))],
        [("user", "q"), ToolTurn((ToolCall("a", {}), ToolCall("b", {}), ToolResult("a", "1"),
                                  ToolResult("b", "2"), "ok"))],
        [("user", "q"), ToolTurn((ToolCall("a", {}),))],                  # still owed a result
        [("user", "q"), ToolTurn((ToolCall("a", {}), ToolResult("a", "1")))],    # open after it
    ]
    for history in histories:
        assert as_history(as_messages(history)) == history


def test_each_result_of_a_batch_answers_its_own_call():
    turn = ToolTurn((ToolCall("a", {}), ToolCall("b", {}), ToolResult("a", "1"),
                     ToolResult("b", "2"), "ok"))
    messages = as_messages([("user", "q"), turn])
    calls = [call["id"] for m in messages for call in m.get("tool_calls", [])]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == calls
