"""A conversation begun from a history, tool turns included."""

import pytest

from llama_chat.config import Config
from llama_chat.tools import Thought, ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import FakeContext, ScriptedCall
from tests.internals import pending_calls_held
from tests.invented_templates import (CALL_TAGS, REASONING_KEEPING_THOUGHTS,
                                      TOOLS_CLOSING_OPEN_TURNS)

from tests.wrappers import (begun_wrapper, tool_wrapper, unbegun_tool_wrapper,
                            weather_registry, rows)


# ----- beginning again from conversation() ---------------------------------
def test_a_conversation_begun_again_from_its_history_reads_in_as_it_was():
    registry = weather_registry()
    live, fake = tool_wrapper(context_size=2048)
    live.begin("persona", tools=registry.specs())
    live.reply("hallo").finish()
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    live.reply("wetter?", tool_call_policy=registry).finish()
    live.reply("danke").finish()
    history = live.conversation()["messages"]
    assert [type(item).__name__ for item in history] == [
        "dict", "dict", "dict", "ToolTurn", "dict", "dict"]
    turn = history[3]
    assert [type(step).__name__ for step in turn.steps] == ["ToolCall", "ToolResult", "str"]

    again, fake_again = tool_wrapper(context_size=2048)
    again.begin("persona", history, tools=registry.specs())
    assert fake_again.cache == fake.cache
    assert again.conversation()["messages"] == history


def test_begin_from_the_conversation_alone_restores_its_thoughts_tools_and_reasoning():
    """Nothing is passed again by hand: the declarations, the reasoning switch and every thought the template keeps come back."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Uhrzeit.", name="uhr", parameters={"type": "object"})
    later = ToolSpec("datum", "Datum.", {"type": "object"})
    replies = ["<thought>Gruss</thought>Hallo.",
               ScriptedCall("fn:uhr{}", before="<thought>Erst fragen</thought>"),
               "<thought>Jetzt antworten</thought>Zwölf."]
    live_fake = FakeContext(REASONING_KEEPING_THOUGHTS, replies=list(replies))
    live = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=live_fake)
    live.begin("persona", tools=registry.specs(), reasoning=True)
    live.reply("hallo").finish()
    live.declare_tools([later])
    live.reply("wie spät?", tool_call_policy=registry).finish()
    saved = live.conversation()
    assert saved["system_prompt"] == "persona" and saved["reasoning"] is True
    assert [spec.name for spec in saved["tools"]] == ["uhr", "datum"]
    assert saved["messages"][1] == {"role": "assistant", "content": "Hallo.",
                                    "reasoning_content": "Gruss"}
    assert saved["messages"][3].steps[0] == Thought("Erst fragen")

    again_fake = FakeContext(REASONING_KEEPING_THOUGHTS)
    again = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=again_fake)
    again.begin(**saved)
    assert again.conversation() == saved
    # Written where the template writes a past thought, which a plain history would leave out.
    held = again_fake.text_of(again_fake.cache)
    assert '<thought>Gruss</thought>Hallo.' in held and '<thought>Erst fragen</thought>' in held
    assert held.startswith(live_fake.text_of(live_fake.cache).split('<call>')[0])


# ----- a history that called tools -----------------------------------------
def test_a_past_tool_turn_reads_in_as_the_live_one_left_it():
    live, fake = unbegun_tool_wrapper()
    fake.call_script.append("fn:uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", tool_call_policy="yield"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.reply(tool_call_policy="yield"):
        pass

    past, past_fake = unbegun_tool_wrapper()
    past.begin("s", [("user", "wie spät?"), ToolTurn((
        ToolCall("uhr", {}), ToolResult("uhr", {"zeit": "12:00"}), "Gleich."))],
        tools=[ToolSpec("uhr", "", {})])
    assert past.conversation()["messages"][-1].text == "Gleich."
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
    opening, closing = (fake.tokenize(tag) for tag in CALL_TAGS)
    assert fake.cache.count(*opening) == 1 and fake.cache.count(*closing) == 1


def test_a_conversation_goes_on_after_a_past_tool_turn():
    w, _fake = unbegun_tool_wrapper()
    w.begin("s", [("user", "x"), ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12"),
                                           "Zwölf."))], tools=[ToolSpec("uhr", "", {})])
    w.reply("danke").finish()


def test_a_model_that_cannot_call_tools_refuses_a_past_tool_turn():
    w, _fake = begun_wrapper()
    with pytest.raises(ValueError):
        w.begin("s", [ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12")))])
    assert [m["role"] for m in rows(w)] == ["system"]     # left as it was


@pytest.mark.parametrize("closes_turn_before_next", [False, True])
def test_a_history_that_ends_on_a_result_goes_on_as_the_live_turn_would(
        closes_turn_before_next):
    live, fake = unbegun_tool_wrapper()
    fake.call_script.append("fn:uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", tool_call_policy="yield"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.reply(tool_call_policy="yield"):
        pass

    past, past_fake = (unbegun_tool_wrapper(TOOLS_CLOSING_OPEN_TURNS) if closes_turn_before_next
                       else unbegun_tool_wrapper())
    past.begin("s", [("user", "wie spät?"), ToolTurn((ToolCall("uhr", {}),
                                                     ToolResult("uhr", {"zeit": "12:00"})))],
               tools=[ToolSpec("uhr", "", {})])
    *_, (_start, _n, want_logits) = past_fake.prefill_calls
    assert want_logits      # the reply is sampled straight from the read-in's end
    for _event in past.reply(tool_call_policy="yield"):
        pass
    assert past_fake.cache == fake.cache


def test_a_conversation_begun_again_from_its_own_system_prompt_and_history_is_the_same():
    live, fake = tool_wrapper()
    live.begin("persona", tools=weather_registry().specs())
    live.reply("hallo").finish()
    again, fake_again = tool_wrapper()
    again.begin(**live.conversation())
    assert live.conversation()["system_prompt"] == "persona"
    assert fake_again.cache == fake.cache


# ----- the id of every call --------------------------------------------------
def _calls_in(history) -> list[ToolCall]:
    return [step for item in history if isinstance(item, ToolTurn) for step in item.steps
            if isinstance(step, ToolCall)]


def _past_exchange(n: int) -> list:
    return [{"role": "user", "content": f"frage {n}"},
            ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", f"{n}:00"), f"Es ist {n}."))]


def test_every_call_has_an_id_the_history_and_the_reply_agree_on():
    w, fake = tool_wrapper(context_size=2048)
    w.begin("persona", tools=weather_registry())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    handed, = [e for e in w.reply("wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert handed.id and w.pending_calls == [handed]
    w.inject_tool_result(handed, {"celsius": 18})
    w.reply(tool_call_policy="yield").finish()
    fake.call_script = ["fn:weather{city:'Rom'}"]
    reply = w.reply("und in Rom?", tool_call_policy=weather_registry()).finish()

    held = _calls_in(w.conversation()["messages"])
    assert held[0] == handed and held[1] == reply.tool_calls[0]
    assert len({call.id for call in held}) == 2


def test_a_call_after_a_begin_that_shortened_the_history_gets_an_id_of_its_own():
    """The kept calls carry the ids they were written with; a new one must take none of them."""
    history = [item for n in range(6) for item in _past_exchange(n)]
    w, fake = unbegun_tool_wrapper()
    kept = w.begin("s", history, tools=[ToolSpec("uhr", "", {})])
    assert kept < len(history)
    ids = {call.id for call in _calls_in(w.conversation()["messages"])}
    fake.call_script.append("fn:uhr{}")
    handed, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert handed.id and handed.id not in ids


def test_a_conversation_begun_again_keeps_the_ids_of_its_calls():
    history = [item for n in range(2) for item in _past_exchange(n)]
    w, _ = unbegun_tool_wrapper()
    w.begin("s", history, tools=[ToolSpec("uhr", "", {})])
    again, fake = unbegun_tool_wrapper()
    again.begin(**w.conversation())
    assert again.conversation() == w.conversation()
    ids = {call.id for call in _calls_in(w.conversation()["messages"])}
    fake.call_script.append("fn:uhr{}")
    handed, = [e for e in again.reply("wie spät?", tool_call_policy="yield")
               if isinstance(e, ToolCall)]
    assert handed.id not in ids


def test_a_history_that_gives_one_id_to_two_calls_is_refused():
    w, fake = unbegun_tool_wrapper()
    twice = [("user", "a"), ToolTurn((ToolCall("uhr", {}, id="eins"), ToolResult("uhr", "1"),
                                      ToolCall("uhr", {}, id="eins"), ToolResult("uhr", "2"),
                                      "Fertig."))]
    with pytest.raises(ValueError, match="'eins'"):
        w.begin("s", twice, tools=[ToolSpec("uhr", "", {})])


# ----- what a caller is handed ---------------------------------------------
def test_the_history_and_a_yielded_call_are_the_callers_to_change():
    w, fake = tool_wrapper()
    w.begin("persona", tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    call, = [e for e in w.reply("wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    call.arguments["city"] = "Paris"
    w.conversation()["messages"][-1].steps[-1].arguments["city"] = "Rom"
    assert w.conversation()["messages"][-1].steps[-1].arguments == {"city": "Bonn"}
    again, _ = tool_wrapper()
    again.begin("persona", w.conversation()["messages"], tools=weather_registry().specs())
    pending_calls_held(again)[0].arguments["city"] = "Rom"
    assert again.conversation()["messages"][-1].steps[-1].arguments == {"city": "Bonn"}


def test_the_conversation_reads_back_as_begin_takes_it():
    w, _ = tool_wrapper()
    w.begin("eins", [("user", "a"), ("assistant", "b")])
    assert w.conversation() == {"system_prompt": "eins",
                                "messages": [{"role": "user", "content": "a"},
                                             {"role": "assistant", "content": "b"}],
                                "tools": None, "reasoning": False}
    again, _ = tool_wrapper()
    again.begin(**w.conversation())
    assert again.conversation() == w.conversation()


def test_a_tool_turn_whose_result_answers_no_call_is_refused():
    call = ToolCall("clock", {})
    with pytest.raises(ValueError):
        ToolTurn((ToolResult("clock", "12:00"),))
    with pytest.raises(ValueError):
        ToolTurn((call, ToolResult("clock", "12:00"), ToolResult("clock", "12:01")))
    ToolTurn(("Let me look.", call, ToolResult("clock", "12:00"), "It is noon."))


# ----- the forms a history message takes ------------------------------------
def test_a_history_in_the_usual_dictionaries_reads_in_as_the_same_pairs():
    pairs, _ = begun_wrapper()
    pairs.begin("s", [("user", "Hallo"), ("assistant", "Hi."), ["user", "Wie gehts?"]])
    dicts, _ = begun_wrapper()
    dicts.begin("s", [{"role": "user", "content": "Hallo"},
                      {"role": "assistant", "content": "Hi."},
                      {"role": "user", "content": "Wie gehts?"}])
    assert rows(dicts) == rows(pairs)
    assert [row["text"] for row in rows(dicts)] == ["s", "Hallo", "Hi.", "Wie gehts?"]


@pytest.mark.parametrize("item, error, cause", [
    (("human", "Hallo"), ValueError, "'human'"),
    ({"role": "human", "content": "Hallo"}, ValueError, "'human'"),
    (("system", "Sei kurz."), ValueError, "begin"),
    (("tool", "12:00"), ValueError, "ToolTurn"),
    ({"role": "user"}, ValueError, "content"),
    ({"role": "user", "content": "Hallo", "name": "Ada"}, ValueError, "'name'"),
    ({"role": "user", "content": "Hallo", "reasoning_content": "x"}, ValueError,
     "a reply also"),
    ("Hallo", TypeError, "str"),
    (("user", "Hallo", "zu viel"), TypeError, "tuple"),
], ids=["unknown-role", "unknown-role-dict", "system", "tool", "no-content", "other-key",
        "thought-of-a-question", "bare-text", "three-values"])
def test_a_history_message_of_another_form_or_role_is_refused_naming_it(item, error, cause):
    w, fake = begun_wrapper()
    w.reply("eins").finish()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(error, match=cause):
        w.begin("s", [("user", "a"), item])
    assert w.conversation() == before and fake.cache == held      # left as it was


def test_a_tool_turn_closed_by_an_empty_answer_begins_again_closed():
    registry = weather_registry()
    live, _ = tool_wrapper(context_size=2048, replies=[ScriptedCall("fn:weather{city:'Bonn'}"), ""])
    live.begin("persona", tools=registry.specs())
    live.reply("wetter?", tool_call_policy=registry).finish()
    assert not live.turn_open
    saved = live.conversation()
    again, _ = tool_wrapper(context_size=2048)
    again.begin(**saved)
    assert not again.turn_open
    assert again.conversation() == saved
    again.reply("danke").finish()           # a question follows the closed turn


def test_a_call_after_beginning_again_takes_no_id_an_evicted_call_had():
    """Within one wrapper no two calls share an id, whether made by the model or given to begin without one."""
    from tests.invented_templates import TOOLS
    registry = weather_registry()
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=2048, evict_above_fraction=0.9, max_history_exchanges=1),
                    context=fake)
    w.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    seen = {call.id for call in w.reply("wetter?", tool_call_policy=registry).finish().tool_calls}
    w.reply("hallo").finish()
    assert not _calls_in(w.conversation()["messages"])     # the tool turn was evicted
    w.begin(**w.conversation())
    fake.call_script = ["fn:weather{city:'Rom'}"]
    made, = w.reply("und Rom?", tool_call_policy=registry).finish().tool_calls
    assert made.id not in seen
    seen.add(made.id)
    w.begin("persona", [("user", "a"), ToolTurn((ToolCall("weather", {"city": "Ulm"}),
                                                 ToolResult("weather", "kalt"), "Kalt."))],
            tools=registry.specs())
    given, = _calls_in(w.conversation()["messages"])
    assert given.id not in seen


def test_begin_counts_the_items_kept_as_they_were_given():
    """A reply given after a tool turn that ends on a result closes that turn, and counts as the item it was."""
    history = [("user", "x"), ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12"))),
               ("assistant", "Zwölf.")]
    w, _ = unbegun_tool_wrapper()
    assert w.begin("s", history, tools=[ToolSpec("uhr", "", {})]) == len(history)
    turn = w.conversation()["messages"][-1]
    assert [type(step).__name__ for step in turn.steps] == ["ToolCall", "ToolResult", "str"]
    w, _ = unbegun_tool_wrapper()
    longer = [item for n in range(6) for item in _past_exchange(n)] + history
    kept = w.begin("s", longer, tools=[ToolSpec("uhr", "", {})])
    assert 3 <= kept < len(longer)
    assert w.conversation()["messages"][0] == longer[-kept]
