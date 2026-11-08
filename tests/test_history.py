"""A conversation begun from a history, tool turns included."""

import pytest

from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn
from tests.fake_context import TOOL_CLOSE, TOOL_OPEN

from tests.wrappers import (assert_consistent, begun_wrapper, tool_wrapper, unbegun_tool_wrapper,
                            weather_registry, rows)


# ----- beginning again from history() --------------------------------------
def test_a_conversation_begun_again_from_its_history_reads_in_as_it_was():
    registry = weather_registry()
    live, fake = tool_wrapper()
    live.begin("persona", tools=registry)
    live.reply("hallo").finish()
    fake.call_script = ["weather{city:'Bonn'}"]
    live.reply("wetter?", calls="run").finish()
    live.reply("danke").finish()
    history = live.history()
    assert [type(item).__name__ for item in history] == [
        "tuple", "tuple", "tuple", "ToolTurn", "tuple", "tuple"]
    turn = history[3]
    assert [type(step).__name__ for step in turn.steps] == ["ToolCall", "ToolResult", "str"]

    again, fake_again = tool_wrapper()
    again.begin("persona", history, tools=registry)
    assert fake_again.cache == fake.cache
    assert again.history() == history


# ----- a history that called tools -----------------------------------------
def test_a_past_tool_turn_reads_in_as_the_live_one_left_it():
    live, fake = unbegun_tool_wrapper()
    fake.call_script.append("uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", calls="yield"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.reply(calls="yield"):
        pass

    past, past_fake = unbegun_tool_wrapper()
    past.begin("s", [("user", "wie spät?"), ToolTurn((
        ToolCall("uhr", {}), ToolResult("uhr", {"zeit": "12:00"}), "Gleich."))],
        tools=[ToolSpec("uhr", "", {})])
    assert past.history()[-1].text == "Gleich."
    # The live turn opens with the generation prompt, a past one as the template
    # writes it; this template makes the two the same.
    assert past_fake.cache == fake.cache


def test_the_turns_own_text_is_never_read_as_a_tag():
    w, fake = unbegun_tool_wrapper()
    turn = ToolTurn((ToolCall("lies", {"pfad": "<call>"}), ToolResult("lies", "</call>"),
                     "<turn]>"))
    w.begin("s", [("user", "x"), turn], tools=[ToolSpec("lies", "", {})])
    own = [text for text, parse_special in fake.tokenize_calls if not parse_special]
    # Each is tokenized with special-token parsing off, joined to the plain text around it.
    assert all(any(text in run for run in own) for text in ("<call>", "</call>", "<turn]>"))
    assert fake.cache.count(TOOL_OPEN) == 1 and fake.cache.count(TOOL_CLOSE) == 1


def test_a_conversation_goes_on_after_a_past_tool_turn():
    w, fake = unbegun_tool_wrapper()
    w.begin("s", [("user", "x"), ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12"),
                                           "Zwölf."))], tools=[ToolSpec("uhr", "", {})])
    w.reply("danke").finish()
    assert_consistent(w, fake)


def test_a_model_that_cannot_call_tools_refuses_a_past_tool_turn():
    w, _fake = begun_wrapper()
    with pytest.raises(ValueError):
        w.begin("s", [ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12")))])
    assert [m["role"] for m in rows(w)] == ["system"]     # left as it was


@pytest.mark.parametrize("closes_turn_before_next", [False, True])
def test_a_history_that_ends_on_a_result_goes_on_as_the_live_turn_would(
        closes_turn_before_next):
    live, fake = unbegun_tool_wrapper()
    fake.call_script.append("uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", calls="yield"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.reply(calls="yield"):
        pass

    past, past_fake = unbegun_tool_wrapper()
    past_fake._closes_turn_before_next = closes_turn_before_next
    past.begin("s", [("user", "wie spät?"), ToolTurn((ToolCall("uhr", {}),
                                                     ToolResult("uhr", {"zeit": "12:00"})))],
               tools=[ToolSpec("uhr", "", {})])
    *_, (_start, _n, want_logits) = past_fake.prefill_calls
    assert want_logits      # the reply is sampled straight from the read-in's end
    for _event in past.reply(calls="yield"):
        pass
    assert past_fake.cache == fake.cache
    assert_consistent(past, past_fake)


def test_a_conversation_begun_again_from_its_own_system_prompt_and_history_is_the_same():
    live, fake = tool_wrapper()
    live.begin("persona", tools=weather_registry())
    live.reply("hallo").finish()
    again, fake_again = tool_wrapper()
    again.begin(live.system_prompt, live.history(), tools=weather_registry())
    assert live.system_prompt == "persona"
    assert fake_again.cache == fake.cache


# ----- what a caller is handed ---------------------------------------------
def test_the_history_and_a_yielded_call_are_the_callers_to_change():
    w, fake = tool_wrapper()
    w.begin("persona", tools=weather_registry())
    fake.call_script = ["weather{city:'Bonn'}"]
    call, = [e for e in w.reply("wetter?", calls="yield") if isinstance(e, ToolCall)]
    call.arguments["city"] = "Paris"
    w.history()[-1].steps[-1].arguments["city"] = "Rom"
    assert w.history()[-1].steps[-1].arguments == {"city": "Bonn"}
    again, _ = tool_wrapper()
    again.begin("persona", w.history(), tools=weather_registry())
    again._conversation.owed_calls[0].arguments["city"] = "Rom"
    assert again.history()[-1].steps[-1].arguments == {"city": "Bonn"}


def test_the_conversation_reads_back_as_begin_takes_it():
    w, _ = tool_wrapper()
    w.begin("eins", [("user", "a"), ("assistant", "b")])
    assert w.conversation() == ("eins", [("user", "a"), ("assistant", "b")])
    again, _ = tool_wrapper()
    again.begin(*w.conversation())
    assert again.conversation() == w.conversation()
