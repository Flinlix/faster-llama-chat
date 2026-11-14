"""A conversation begun from a history, tool turns included."""

import pytest

from llama_chat.config import Config
from llama_chat.tools import ToolCall, ToolRegistry, ToolSpec
from llama_chat.wrapper import ChatWrapper
from llama_chat.messages import as_messages
from tests.chat_messages import Result, tool_turn
from tests.fake_context import FakeContext, ScriptedCall
from tests.internals import pending_calls_held
from tests.invented_templates import (CALL_TAGS, REASONING_KEEPING_THOUGHTS, TOOLS,
                                      TOOLS_CLOSING_OPEN_TURNS)

from tests.wrappers import (add_tools, begun_wrapper, tool_wrapper, unbegun_tool_wrapper,
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
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant", "tool",
                                            "assistant", "user", "assistant"]
    assert history[3]["tool_calls"][0]["function"] == {"name": "weather",
                                                       "arguments": {"city": "Bonn"}}

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
    add_tools(live, [later])
    live.reply("wie spät?", tool_call_policy=registry).finish()
    saved = live.conversation()
    assert saved["system_prompt"] == "persona" and saved["reasoning"] is True
    assert [spec.name for spec in saved["tools"]] == ["uhr", "datum"]
    assert saved["messages"][1] == {"role": "assistant", "content": "Hallo.",
                                    "reasoning_content": "Gruss"}
    assert saved["messages"][3]["reasoning_content"] == "Erst fragen"

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
    past.begin("s", [("user", "wie spät?"), *tool_turn(
        ToolCall("uhr", {}), Result("uhr", {"zeit": "12:00"}), "Gleich.")],
        tools=[ToolSpec("uhr", "", {})])
    assert past.conversation()["messages"][-1] == {"role": "assistant", "content": "Gleich."}
    # The live turn opens with the generation prompt, a past one as the template
    # writes it; this template makes the two the same.
    assert past_fake.cache == fake.cache


def test_the_turns_own_text_is_never_read_as_a_tag():
    w, fake = unbegun_tool_wrapper()
    turn = tool_turn(ToolCall("lies", {"pfad": "<call>"}), Result("lies", "</call>"), "<turn]>")
    w.begin("s", [("user", "x"), *turn], tools=[ToolSpec("lies", "", {})])
    own = [text for text, parse_special in fake.tokenize_calls if not parse_special]
    # Each is tokenized with special-token parsing off, joined to the plain text around it.
    assert all(any(text in run for run in own) for text in ("<call>", "</call>", "<turn]>"))
    opening, closing = (fake.tokenize(tag) for tag in CALL_TAGS)
    assert fake.cache.count(*opening) == 1 and fake.cache.count(*closing) == 1


def test_a_conversation_goes_on_after_a_past_tool_turn():
    w, _fake = unbegun_tool_wrapper()
    w.begin("s", [("user", "x"), *tool_turn(ToolCall("uhr", {}), Result("uhr", "12"),
                                           "Zwölf.")], tools=[ToolSpec("uhr", "", {})])
    w.reply("danke").finish()


def test_a_model_that_cannot_call_tools_refuses_a_past_tool_turn():
    w, _fake = begun_wrapper()
    with pytest.raises(ValueError):
        w.begin("s", [*tool_turn(ToolCall("uhr", {}), Result("uhr", "12"))])
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
    past.begin("s", [("user", "wie spät?"), *tool_turn(ToolCall("uhr", {}),
                                                     Result("uhr", {"zeit": "12:00"}))],
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
    return [ToolCall(call["function"]["name"], call["function"]["arguments"], call["id"])
            for message in history for call in message.get("tool_calls", ())]


def _past_exchange(n: int) -> list:
    return [{"role": "user", "content": f"frage {n}"},
            *tool_turn(ToolCall("uhr", {}), Result("uhr", f"{n}:00"), f"Es ist {n}.")]


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
    kept = w.begin("s", history, tools=[ToolSpec("uhr", "", {})]).n_kept_messages
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
    twice = [("user", "a"), *tool_turn(ToolCall("uhr", {}, id="eins"), Result("uhr", "1"),
                                      ToolCall("uhr", {}, id="eins"), Result("uhr", "2"),
                                      "Fertig.")]
    with pytest.raises(ValueError, match="'eins'"):
        w.begin("s", twice, tools=[ToolSpec("uhr", "", {})])


# ----- what a caller is handed ---------------------------------------------
def test_the_history_and_a_yielded_call_are_the_callers_to_change():
    w, fake = tool_wrapper()
    w.begin("persona", tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    call, = [e for e in w.reply("wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    call.arguments["city"] = "Paris"
    w.conversation()["messages"][-1]["tool_calls"][-1]["function"]["arguments"]["city"] = "Rom"
    assert w.conversation()["messages"][-1]["tool_calls"][-1]["function"]["arguments"] == {"city": "Bonn"}
    again, _ = tool_wrapper()
    again.begin("persona", w.conversation()["messages"], tools=weather_registry().specs())
    pending_calls_held(again)[0].arguments["city"] = "Rom"
    assert again.conversation()["messages"][-1]["tool_calls"][-1]["function"]["arguments"] == {"city": "Bonn"}


def test_the_conversation_reads_back_as_begin_takes_it():
    w, _ = tool_wrapper()
    w.begin("eins", [("user", "a"), ("assistant", "b")])
    assert w.conversation() == {"system_prompt": "eins",
                                "messages": [{"role": "user", "content": "a"},
                                             {"role": "assistant", "content": "b"}],
                                "tools": None, "reasoning": False, "generated_turns": None}
    again, _ = tool_wrapper()
    again.begin(**w.conversation())
    assert again.conversation() == w.conversation()


def test_a_result_that_answers_no_call_owed_one_is_refused():
    call, question = ToolCall("clock", {}), ("user", "q")
    with pytest.raises(ValueError, match="answers no call"):
        as_messages([question, *tool_turn(Result("clock", "12:00"))])
    with pytest.raises(ValueError, match="answers no call"):
        as_messages([question, *tool_turn(call, Result("clock", "12:00"), Result("clock", "12:01"))])
    as_messages([question, *tool_turn("Let me look.", call, Result("clock", "12:00"), "It is noon.")])


def test_a_result_out_of_order_or_naming_another_function_is_refused():
    turn = tool_turn(ToolCall("clock", {}, "c1"), ToolCall("date", {}, "c2"),
                     Result("clock", "12:00"), Result("date", "1.1."))
    turn[1]["tool_call_id"] = "c2"
    with pytest.raises(ValueError, match="order of their calls"):
        as_messages([("user", "q"), *turn])
    turn[1].update(tool_call_id="c1", name="date")
    with pytest.raises(ValueError, match="names the function 'date'"):
        as_messages([("user", "q"), *turn])


def test_a_reply_between_a_call_and_its_result_is_refused():
    with pytest.raises(ValueError, match="before its result"):
        as_messages([("user", "q"), *tool_turn(ToolCall("clock", {})), ("assistant", "Gleich."),
                     *tool_turn(Result("clock", "12:00"))])


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
    (("tool", "12:00"), ValueError, "tool message"),
    ({"role": "user"}, ValueError, "content"),
    ({"role": "user", "content": "Hallo", "name": "Ada"}, ValueError, "'name'"),
    ({"role": "user", "content": "Hallo", "reasoning_content": "x"}, ValueError,
     "may hold \\[\\]"),
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


def test_a_call_left_without_a_result_before_a_later_question_stays_in_its_place():
    registry = weather_registry()
    history = [("user", "wetter?"), *tool_turn(ToolCall("weather", {"city": "Bonn"})),
               ("user", "egal, danke")]
    live, _ = tool_wrapper(context_size=2048)
    live.begin("persona", history, tools=registry.specs())
    assert not live.turn_open
    live.reply().finish()
    saved = live.conversation()
    assert saved["messages"][1:2] == tool_turn(ToolCall("weather", {"city": "Bonn"}, "call00000"))
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]
    again, _ = tool_wrapper(context_size=2048)
    again.begin(**saved)
    assert again.conversation() == saved


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
    w.begin("persona", [("user", "a"), *tool_turn(ToolCall("weather", {"city": "Ulm"}),
                                                 Result("weather", "kalt"), "Kalt.")],
            tools=registry.specs())
    given, = _calls_in(w.conversation()["messages"])
    assert given.id not in seen


def test_begin_counts_the_messages_kept_and_a_reply_after_a_result_closes_its_turn():
    history = [("user", "x"), *tool_turn(ToolCall("uhr", {}), Result("uhr", "12")),
               ("assistant", "Zwölf.")]
    w, _ = unbegun_tool_wrapper()
    assert w.begin("s", history, tools=[ToolSpec("uhr", "", {})]).n_kept_messages == len(history)
    assert [m["role"] for m in w.conversation()["messages"]] == ["user", "assistant", "tool",
                                                                  "assistant"]
    assert not w.turn_open
    w, _ = unbegun_tool_wrapper()
    longer = [item for n in range(6) for item in _past_exchange(n)] + history
    kept = w.begin("s", longer, tools=[ToolSpec("uhr", "", {})]).n_kept_messages
    assert 3 <= kept < len(longer)
    assert w.conversation()["messages"][0] == longer[-kept]


@pytest.mark.parametrize("context_size", [4096, 8192])
def test_a_long_question_into_a_context_full_of_short_exchanges_renders_a_few_times_however_many_go(
        context_size):
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=context_size), context=fake)
    w.begin("sys", [m for _ in range(4000) for m in (("user", "ok"), ("assistant", "fine"))])
    before = len(fake.rendered)
    reply = w.reply("z" * int(context_size * 0.6)).finish()
    # Dropping hundreds of exchanges costs a few renders, not one each.
    assert reply.n_evicted_messages > 100 and len(fake.rendered) - before < 40


@pytest.mark.parametrize("history, named", [
    ([("user", "q"), 42], "int"),
    ([("user", "q"), *tool_turn(ToolCall("clock", ["not", "a", "dict"]), Result("clock", "1"))],
     "arguments"),
])
def test_a_history_item_or_call_of_another_type_is_refused_naming_it(history, named):
    with pytest.raises(TypeError, match=named):
        as_messages(history)


def test_beginning_with_a_history_that_fits_renders_as_often_however_long_it_is():
    def renders(n):
        fake = FakeContext(TOOLS)
        w = ChatWrapper(Config(context_size=32768), context=fake)
        before = len(fake.rendered)
        kept = w.begin("sys", [m for i in range(n) for m in (("user", f"q{i}"), ("assistant", f"a{i}"))]
                       ).n_kept_messages
        assert kept == 2 * n
        return len(fake.rendered) - before

    assert renders(200) == renders(50)


def test_a_history_whose_older_exchanges_are_long_is_not_rendered_whole_at_begin():
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=32768), context=fake)
    history = [m for i in range(300) for m in (("user", f"q{i} " + "x" * 2000), ("assistant", f"a{i}"))]
    history += [("user", "thanks"), ("assistant", "you are welcome"), ("user", "bye"), ("assistant", "bye")]
    before = len(fake.rendered)
    w.begin("sys", history)
    assert max(fake.rendered[before:]) < 100     # never near the 604 messages of the whole


def test_an_unreadable_call_with_a_name_is_refused_and_one_beside_other_calls_is_taken():
    named = tool_turn(ToolCall("", {}, status="unreadable"), Result("unknown", "e"))
    named[0]["tool_calls"][0]["function"] = {"name": "weather", "arguments": {"city": "Bonn"}}
    with pytest.raises(ValueError, match="no name or arguments"):
        as_messages([("user", "q"), *named])
    # An entry of a batch the model wrote unreadably stands beside the calls it could read.
    as_messages([("user", "q"), *tool_turn(ToolCall("weather", {}), ToolCall("", {}, status="unreadable"),
                                           Result("weather", "r"), Result("unknown", "e"))])


def test_a_history_cannot_leave_an_unreadable_call_owed():
    w, _fake = tool_wrapper(context_size=2048)
    with pytest.raises(ValueError, match="cannot leave one owed"):
        w.begin("persona", [("user", "x"), *tool_turn(ToolCall("", {}, status="unreadable", text="t"))])
