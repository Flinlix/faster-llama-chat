"""A conversation begun from a history, tool turns included."""

import json

import pytest

from llama_chat.config import Config
from llama_chat.errors import ContextOverflowError
from llama_chat.tool_registry import ToolRegistry
from llama_chat.tools import ToolCall, ToolSpec
from llama_chat.wrapper import ChatWrapper
from tests import internals
from tests.chat_messages import Result, tool_turn, opened, after_system
from tests.fake_context import FakeContext, ScriptedCall
from tests.internals import pending_calls_held
from tests.invented_templates import (CALL_TAGS, PLAIN, REASONING_KEEPING_THOUGHTS, TOOLS,
                                      TOOLS_CLOSING_OPEN_TURNS)

from tests.wrappers import (add_tools, begun_wrapper, tool_wrapper, unbegun_tool_wrapper,
                            weather_registry, rows)


# ----- beginning again from conversation() ---------------------------------
def test_a_conversation_begun_again_from_its_history_reads_in_as_it_was():
    registry = weather_registry()
    live, fake = tool_wrapper(context_size=2048)
    live.begin(opened("persona"), tools=registry.specs())
    live.reply("hallo").finish()
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    live.reply("wetter?", tool_call_policy=registry).finish()
    live.reply("danke").finish()
    history = after_system(live.conversation()["messages"])
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant", "tool",
                                            "assistant", "user", "assistant"]
    assert history[3]["tool_calls"][0]["function"] == {"name": "weather",
                                                       "arguments": '{"city":"Bonn"}'}

    again, fake_again = tool_wrapper(context_size=2048)
    again.begin(opened("persona", history), tools=registry.specs())
    assert fake_again.cache == fake.cache
    assert after_system(again.conversation()["messages"]) == history


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
    live.begin(opened("persona"), tools=registry.specs(), reasoning=True)
    live.reply("hallo").finish()
    add_tools(live, [later])
    live.reply("wie spät?", tool_call_policy=registry).finish()
    saved = live.conversation()
    assert saved["messages"][0] == {"role": "system", "content": "persona"}
    assert saved["reasoning"] is True
    assert [spec.name for spec in saved["tools"]] == ["uhr", "datum"]
    assert saved["messages"][2] == {"role": "assistant", "content": "Hallo.",
                                    "reasoning_content": "Gruss"}
    assert saved["messages"][4]["reasoning_content"] == "Erst fragen"

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
    live.begin(opened("s"), tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", tool_call_policy="yield"):
        call = event
    live.inject_tool_result(call.id, {"zeit": "12:00"})
    for _event in live.reply(tool_call_policy="yield"):
        pass

    past, past_fake = unbegun_tool_wrapper()
    past.begin(opened("s", [{"role": "user", "content": "wie spät?"}, *tool_turn(
        ToolCall("uhr", {}), Result("uhr", {"zeit": "12:00"}), "Gleich.")]), tools=[ToolSpec("uhr", "", {})])
    assert after_system(past.conversation()["messages"])[-1] == {"role": "assistant", "content": "Gleich."}
    # The live turn opens with the generation prompt, a past one as the template
    # writes it; this template makes the two the same.
    assert past_fake.cache == fake.cache


def test_the_turns_own_text_is_never_read_as_a_tag():
    w, fake = unbegun_tool_wrapper()
    turn = tool_turn(ToolCall("lies", {"pfad": "<call>"}), Result("lies", "</call>"), "<turn]>")
    w.begin(opened("s", [{"role": "user", "content": "x"}, *turn]), tools=[ToolSpec("lies", "", {})])
    own = [text for text, parse_special in fake.tokenize_calls if not parse_special]
    # Each is tokenized with special-token parsing off, joined to the plain text around it.
    assert all(any(text in run for run in own) for text in ("<call>", "</call>", "<turn]>"))
    opening, closing = (fake.tokenize(tag) for tag in CALL_TAGS)
    assert fake.cache.count(*opening) == 1 and fake.cache.count(*closing) == 1


def test_a_conversation_goes_on_after_a_past_tool_turn():
    w, _fake = unbegun_tool_wrapper()
    w.begin(opened("s", [{"role": "user", "content": "x"}, *tool_turn(ToolCall("uhr", {}), Result("uhr", "12"),
                                           "Zwölf.")]), tools=[ToolSpec("uhr", "", {})])
    w.reply("danke").finish()


def test_a_model_that_cannot_call_tools_refuses_a_past_tool_turn():
    w, _fake = begun_wrapper()
    with pytest.raises(ValueError):
        w.begin(opened("s", [*tool_turn(ToolCall("uhr", {}), Result("uhr", "12"))]))
    assert [m["role"] for m in rows(w)] == ["system"]     # left as it was


@pytest.mark.parametrize("closes_turn_before_next", [False, True])
def test_a_history_that_ends_on_a_result_goes_on_as_the_live_turn_would(
        closes_turn_before_next):
    live, fake = unbegun_tool_wrapper()
    fake.call_script.append("fn:uhr{}")
    live.begin(opened("s"), tools=[ToolSpec("uhr", "", {})])
    for event in live.reply("wie spät?", tool_call_policy="yield"):
        call = event
    live.inject_tool_result(call.id, {"zeit": "12:00"})
    for _event in live.reply(tool_call_policy="yield"):
        pass

    past, past_fake = (unbegun_tool_wrapper(TOOLS_CLOSING_OPEN_TURNS) if closes_turn_before_next
                       else unbegun_tool_wrapper())
    past.begin(opened("s", [{"role": "user", "content": "wie spät?"}, *tool_turn(ToolCall("uhr", {}),
                                                     Result("uhr", {"zeit": "12:00"}))]), tools=[ToolSpec("uhr", "", {})])
    *_, (_start, _n, want_logits) = past_fake.prefill_calls
    assert want_logits      # the reply is sampled straight from the read-in's end
    for _event in past.reply(tool_call_policy="yield"):
        pass
    assert past_fake.cache == fake.cache


def test_a_conversation_begun_again_from_its_own_system_prompt_and_history_is_the_same():
    live, fake = tool_wrapper()
    live.begin(opened("persona"), tools=weather_registry().specs())
    live.reply("hallo").finish()
    again, fake_again = tool_wrapper()
    again.begin(**live.conversation())
    assert live.conversation()["messages"][0] == {"role": "system", "content": "persona"}
    assert fake_again.cache == fake.cache


# ----- the id of every call --------------------------------------------------
def _calls_in(history) -> list[ToolCall]:
    return [ToolCall(call["function"]["name"], json.loads(call["function"]["arguments"]), call["id"])
            for message in history for call in message.get("tool_calls", ())]


def _past_exchange(n: int) -> list:
    return [{"role": "user", "content": f"frage {n}"},
            *tool_turn(ToolCall("uhr", {}), Result("uhr", f"{n}:00"), f"Es ist {n}.")]


def test_every_call_has_an_id_the_history_and_the_reply_agree_on():
    w, fake = tool_wrapper(context_size=2048)
    w.begin(opened("persona"), tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    handed, = [e for e in w.reply("wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert handed.id and w.pending_calls == [handed]
    w.inject_tool_result(handed.id, {"celsius": 18})
    w.reply(tool_call_policy="yield").finish()
    fake.call_script = ["fn:weather{city:'Rom'}"]
    reply = w.reply("und in Rom?", tool_call_policy=weather_registry()).finish()

    held = _calls_in(after_system(w.conversation()["messages"]))
    assert held[0] == handed and held[1] == reply.tool_calls[0]
    assert len({call.id for call in held}) == 2


def test_a_call_after_a_begin_that_shortened_the_history_gets_an_id_of_its_own():
    """The kept calls carry the ids they were written with; a new one must take none of them."""
    history = [item for n in range(6) for item in _past_exchange(n)]
    w, fake = unbegun_tool_wrapper()
    kept = w.begin(opened("s", history), tools=[ToolSpec("uhr", "", {})]).n_kept_messages
    assert kept < len(history)
    ids = {call.id for call in _calls_in(after_system(w.conversation()["messages"]))}
    fake.call_script.append("fn:uhr{}")
    handed, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert handed.id and handed.id not in ids


def test_a_conversation_begun_again_keeps_the_ids_of_its_calls():
    history = [item for n in range(2) for item in _past_exchange(n)]
    w, _ = unbegun_tool_wrapper()
    w.begin(opened("s", history), tools=[ToolSpec("uhr", "", {})])
    again, fake = unbegun_tool_wrapper()
    again.begin(**w.conversation())
    assert again.conversation() == w.conversation()
    ids = {call.id for call in _calls_in(after_system(w.conversation()["messages"]))}
    fake.call_script.append("fn:uhr{}")
    handed, = [e for e in again.reply("wie spät?", tool_call_policy="yield")
               if isinstance(e, ToolCall)]
    assert handed.id not in ids


def test_a_history_that_gives_one_id_to_two_calls_is_refused():
    w, fake = unbegun_tool_wrapper()
    twice = [{"role": "user", "content": "a"}, *tool_turn(ToolCall("uhr", {}, id="eins"), Result("uhr", "1"),
                                      ToolCall("uhr", {}, id="eins"), Result("uhr", "2"),
                                      "Fertig.")]
    with pytest.raises(ValueError, match="'eins'"):
        w.begin(opened("s", twice), tools=[ToolSpec("uhr", "", {})])


# ----- what a caller is handed ---------------------------------------------
def test_the_history_and_a_yielded_call_are_the_callers_to_change():
    w, fake = tool_wrapper()
    w.begin(opened("persona"), tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    call, = [e for e in w.reply("wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    call.arguments["city"] = "Paris"
    w.conversation()["messages"][-1]["tool_calls"][-1]["function"]["arguments"] = '{"city":"Rom"}'
    assert w.conversation()["messages"][-1]["tool_calls"][-1]["function"]["arguments"] \
        == '{"city":"Bonn"}'
    again, _ = tool_wrapper()
    given = w.conversation()["messages"]
    again.begin(given, tools=weather_registry().specs())
    given[-1]["content"] = "changed"
    pending_calls_held(again)[0].arguments["city"] = "Rom"
    assert again.conversation()["messages"] == w.conversation()["messages"]


def test_the_conversation_reads_back_as_begin_takes_it():
    w, _ = tool_wrapper()
    w.begin(opened("eins", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    assert w.conversation() == {"messages": [{"role": "system", "content": "eins"},
                                             {"role": "user", "content": "a"},
                                             {"role": "assistant", "content": "b"}],
                                "tools": None, "reasoning": False, "generated_turns": None}
    again, _ = tool_wrapper()
    again.begin(**w.conversation())
    assert again.conversation() == w.conversation()


@pytest.mark.parametrize("item, error, cause", [
    (("user", "Hallo"), TypeError, "chat message dictionary"),
    ({"role": "human", "content": "Hallo"}, ValueError, "'human'"),
    ({"role": "system", "content": "Sei kurz."}, ValueError, "only open"),
    ({"role": "tool", "content": "12:00"}, ValueError, "tool call"),
    ({"role": "user"}, ValueError, "content"),
    ({"role": "user", "content": "Hallo", "name": "Ada"}, ValueError, "'name'"),
    ({"role": "user", "content": "Hallo", "reasoning_content": "x"}, ValueError,
     "'reasoning_content'"),
    ("Hallo", TypeError, "chat message dictionary"),
    (("user", "Hallo", "zu viel"), TypeError, "tuple"),
], ids=["pair", "unknown-role", "system", "result-without-tools", "no-content", "other-key",
        "thought-of-a-question", "bare-text", "three-values"])
def test_a_history_message_of_another_form_or_role_is_refused_naming_it(item, error, cause):
    w, fake = begun_wrapper()
    w.reply("eins").finish()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(error, match=cause):
        w.begin(opened("s", [{"role": "user", "content": "a"}, item]))
    assert w.conversation() == before and fake.cache == held      # left as it was


def test_a_tool_turn_closed_by_an_empty_answer_begins_again_closed():
    registry = weather_registry()
    live, _ = tool_wrapper(context_size=2048, replies=[ScriptedCall("fn:weather{city:'Bonn'}"), ""])
    live.begin(opened("persona"), tools=registry.specs())
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
    history = [{"role": "user", "content": "wetter?"}, *tool_turn(ToolCall("weather", {"city": "Bonn"})),
               {"role": "user", "content": "egal, danke"}]
    live, _ = tool_wrapper(context_size=2048)
    live.begin(opened("persona", history), tools=registry.specs())
    assert not live.turn_open
    live.reply().finish()
    saved = live.conversation()
    assert saved["messages"][2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call00000", "type": "function",
         "function": {"name": "weather", "arguments": '{"city":"Bonn"}'}}]}
    assert [m["role"] for m in saved["messages"]] == ["system", "user", "assistant", "user",
                                                      "assistant"]
    again, _ = tool_wrapper(context_size=2048)
    again.begin(**saved)
    assert again.conversation() == saved


def test_a_call_after_beginning_again_takes_no_id_an_evicted_call_had():
    """Within one wrapper no two calls share an id, a call the model makes after a begin included."""
    from tests.invented_templates import TOOLS
    registry = weather_registry()
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=2048, evict_above_fraction=0.9, max_history_exchanges=1),
                    context=fake)
    w.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    seen = {call.id for call in w.reply("wetter?", tool_call_policy=registry).finish().tool_calls}
    w.reply("hallo").finish()
    assert not _calls_in(after_system(w.conversation()["messages"]))     # the tool turn was evicted
    w.begin(**w.conversation())
    fake.call_script = ["fn:weather{city:'Rom'}"]
    made, = w.reply("und Rom?", tool_call_policy=registry).finish().tool_calls
    assert made.id not in seen


def test_begin_counts_the_messages_kept_and_a_reply_after_a_result_closes_its_turn():
    history = [{"role": "user", "content": "x"}, *tool_turn(ToolCall("uhr", {}), Result("uhr", "12")),
               {"role": "assistant", "content": "Zwölf."}]
    w, _ = unbegun_tool_wrapper()
    assert w.begin(opened("s", history), tools=[ToolSpec("uhr", "", {})]).n_kept_messages == len(history)
    assert [m["role"] for m in after_system(w.conversation()["messages"])] == ["user", "assistant", "tool",
                                                                  "assistant"]
    assert not w.turn_open
    w, _ = unbegun_tool_wrapper()
    longer = [item for n in range(6) for item in _past_exchange(n)] + history
    kept = w.begin(opened("s", longer), tools=[ToolSpec("uhr", "", {})]).n_kept_messages
    assert 3 <= kept < len(longer)
    assert after_system(w.conversation()["messages"])[0] == longer[-kept]


@pytest.mark.counts_work
@pytest.mark.parametrize("context_size", [4096, 8192])
def test_a_long_question_into_a_context_full_of_short_exchanges_renders_a_few_times_however_many_go(
        context_size):
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=context_size), context=fake)
    w.begin(opened("sys", [m for _ in range(4000) for m in ({"role": "user", "content": "ok"}, {"role": "assistant", "content": "fine"})]))
    before = len(fake.rendered)
    reply = w.reply("z" * int(context_size * 0.6)).finish()
    # Dropping hundreds of exchanges costs a few renders, not one each, some of them gone
    # for the question and the rest after its reply.
    _decoded, after = internals.done_after_the_newest_reply(w)
    assert reply.n_evicted_messages + after > 100 and len(fake.rendered) - before < 40


def test_beginning_with_a_history_that_fits_renders_as_often_however_long_it_is():
    def renders(n):
        fake = FakeContext(TOOLS)
        w = ChatWrapper(Config(context_size=32768), context=fake)
        before = len(fake.rendered)
        kept = w.begin(opened("sys", [m for i in range(n) for m in ({"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"})])).n_kept_messages
        assert kept == 2 * n
        return len(fake.rendered) - before

    assert renders(200) == renders(50)


def test_a_history_whose_older_exchanges_are_long_is_not_rendered_whole_at_begin():
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=32768), context=fake)
    history = [m for i in range(300) for m in ({"role": "user", "content": f"q{i} " + "x" * 2000}, {"role": "assistant", "content": f"a{i}"})]
    history += [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "you are welcome"}, {"role": "user", "content": "bye"}, {"role": "assistant", "content": "bye"}]
    before = len(fake.rendered)
    w.begin(opened("sys", history))
    assert max(fake.rendered[before:]) < 100     # never near the 604 messages of the whole


def test_a_history_may_leave_a_call_that_could_not_be_read_owed_and_it_is_left_out():
    w, _fake = tool_wrapper(context_size=2048)
    w.begin(opened("persona", [{"role": "user", "content": "x"},
                               *tool_turn(ToolCall("", {}, status="unreadable", text="t"))]),
            tools=weather_registry().specs())
    assert not w.turn_open and w.awaits_reply
    assert after_system(w.conversation()["messages"]) == [{"role": "user", "content": "x"}]
    assert w.begin(opened("persona", [{"role": "user", "content": "x"}, *tool_turn(
        ToolCall("", {}, status="unreadable", text="t"))]),
        tools=weather_registry().specs()).n_left_out_calls == 1


@pytest.mark.parametrize("turns", [
    {"turns": []},
    {"model": "m", "turns": [{"tokens": [1], "span": 1, "messages": [], "count": 2}]},
    {"model": "m", "turns": [{"tokens": ["x"], "span": 1, "count": 2, "digest": "d",
                              "messages": [{"role": "assistant", "content": ""}]}]},
    {"model": "m", "turns": [{"tokens": [1], "span": 1, "count": 2, "digest": "d",
                              "messages": [{"role": "user", "content": "q"}]}]},
    {"model": "m", "turns": [{"tokens": [1], "span": 2, "count": 2, "digest": "d",
                              "messages": [{"role": "assistant", "content": ""}]}]},
    {"model": "m", "turns": "none"},
], ids=["no-model", "no-digest", "tokens-of-text", "a-question-held", "span-past-its-messages",
        "turns-not-a-list"])
def test_generated_turns_not_as_a_conversation_gives_them_are_refused_naming_them(turns):
    w, fake = begun_wrapper()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ValueError) as refused:
        w.begin(opened("s", [{"role": "user", "content": "a"}]), generated_turns=turns)
    assert refused.value.field == "generated_turns"
    assert w.conversation() == before and fake.cache == held


def test_a_begin_takes_back_a_conversation_whose_result_was_cut_and_cuts_a_long_one_itself():
    """A begin gives the newest exchange the room a running conversation gives it."""
    from tests.invented_templates import JSON_TOOLS
    clock = ToolSpec("clock", "time", {"type": "object", "properties": {}})
    fake = FakeContext(JSON_TOOLS)
    w = ChatWrapper(Config(context_size=600), context=fake)
    w.begin(opened("persona"), tools=[clock])
    fake.call_script = ['[{"name": "clock", "arguments": {}}]']
    list(w.reply("wie spät?", tool_call_policy="yield"))
    assert w.inject_tool_result(w.pending_calls[0].id, "x" * 5000).tool_results_cut
    saved = w.conversation()
    again = ChatWrapper(Config(context_size=600), context=FakeContext(JSON_TOOLS))
    again.begin(**saved)
    assert again.conversation() == saved
    whole = [*saved["messages"][:-1], {**saved["messages"][-1], "content": "x" * 5000}]
    report = again.begin(whole, tools=[clock])
    assert [call.name for call in report.tool_results_cut] == ["clock"]
    assert again.conversation()["messages"][-1]["content"].endswith("characters kept)")
    again.reply().finish()


def test_a_call_whose_arguments_are_no_object_is_left_out_and_a_later_result_for_it_refused():
    w, _ = unbegun_tool_wrapper()
    history = [{"role": "user", "content": "wie spät?"},
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "kaputt", "type": "function",
                    "function": {"name": "uhr", "arguments": "{nicht json"}}]}]
    report = w.begin(opened("s", history), tools=[ToolSpec("uhr", "", {})])
    assert report.n_left_out_calls == 1 and not w.turn_open and w.awaits_reply
    with pytest.raises(ValueError, match="'kaputt' could not be read") as refused:
        w.inject_tool_result("kaputt", "12:00")
    assert refused.value.field == "call_id"
    w.reply().finish()                  # the question is answered all the same


def test_a_begin_decodes_only_from_where_the_conversation_differs_from_the_cache():
    """The memory rolls across a begin: beginning again, or with a newer last message, decodes only that."""
    registry = weather_registry()
    w, fake = tool_wrapper(context_size=2048)
    w.begin(opened("persona"), tools=registry.specs())
    w.reply("hallo").finish()
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    w.reply("wetter?", tool_call_policy=registry).finish()
    saved, held = w.conversation(), len(fake.cache)
    decoded = w._conversation.decoded
    w.begin(**saved)
    assert w._conversation.decoded == decoded and len(fake.cache) == held
    edited = [*saved["messages"][:-1], {"role": "assistant", "content": "Anders."}]
    w.begin(edited, tools=saved["tools"], generated_turns=saved["generated_turns"])
    assert 0 < w._conversation.decoded - decoded < 20


def test_a_begin_that_goes_on_with_the_cached_conversation_keeps_it_to_the_eviction_limit():
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=0.9), context=fake)
    w.begin(opened("s"))
    for n in range(30):
        w.reply(f"Frage {n} " + "x" * 20).finish()
    saved = w.conversation()
    assert w.total_tokens > 1000                # more than half the context
    report = w.begin(**saved)
    assert report.n_evicted_messages == 0 and w.conversation() == saved
    fresh = ChatWrapper(Config(context_size=2000, evict_above_fraction=0.9),
                        context=FakeContext(TOOLS))
    assert fresh.begin(**saved).evicted_for == "half_context"


def test_a_history_given_as_text_is_refused_naming_the_form():
    w, _ = begun_wrapper()
    with pytest.raises(TypeError, match="list of chat message dictionaries"):
        w.begin("You are a terse assistant.")


def test_begin_reports_only_the_turns_it_kept_as_generated():
    registry = weather_registry()
    live, fake = tool_wrapper(context_size=2048)
    live.begin(opened("persona"), tools=registry.specs())
    for city in ("Bonn", "Rom"):
        fake.call_script = [f"fn:weather{{city:'{city}'}}"]
        live.reply("wetter?", tool_call_policy=registry).finish()
    saved = live.conversation()
    assert sum(len(record["kept"]) for record in saved["generated_turns"]["turns"]) == 2
    limited = ChatWrapper(Config(context_size=2048, max_history_exchanges=1),
                          context=FakeContext(TOOLS))
    report = limited.begin(**saved)
    assert report.n_turns_restored == 1


def test_a_call_made_in_the_room_a_reply_leaves_begins_again_and_its_result_closes_the_turn():
    """The live call took room the reply leaves; begun again it keeps it, and a result that does not fit closes the turn as it would have live."""
    for length in range(150, 300, 4):
        fake = FakeContext(TOOLS)
        fake.call_script = ["fn:weather{city:'Bonn'}"]
        config = Config(context_size=300, min_reply_tokens=16, evict_above_fraction=0.9)
        live = ChatWrapper(config, context=fake)
        live.begin(opened("s"), tools=weather_registry().specs())
        try:
            reply = live.reply("q" * length, tool_call_policy="yield").finish()
        except ContextOverflowError:
            break
        if not live.pending_calls or live.total_tokens <= config.context_size - config.min_reply_tokens:
            continue
        saved = live.conversation()
        again = ChatWrapper(config, context=FakeContext(TOOLS))
        again.begin(**saved)
        assert again.conversation() == saved
        report = again.inject_tool_result(reply.tool_calls[0].id, "sunny")
        assert report.turn_closed and not again.turn_open
        return
    pytest.fail("no question left the call in the room a reply leaves")


def test_a_reply_that_ran_to_the_end_of_the_context_begins_again():
    """Rule 10: a reply may fill the context to its end, as no reply follows it until the next question evicts it."""
    fake = FakeContext(PLAIN, gen_tokens=400)
    config = Config(context_size=200, evict_above_fraction=1.0, min_reply_tokens=16)
    w = ChatWrapper(config, context=fake)
    w.begin(opened("s"))
    assert w.reply("Erzähl.").finish().stop_reason == "token_limit"
    saved = w.conversation()
    again = ChatWrapper(config, context=FakeContext(PLAIN))
    again.begin(**saved)
    assert again.conversation() == saved
    w.begin(**saved)
    assert w.conversation() == saved and w.reply("Und?").finish()


def test_two_questions_in_a_row_are_joined_at_a_begin_where_the_template_writes_no_two():
    """As a running conversation joins them, so a history kept with both, as a server keeps it, begins again."""
    from tests.invented_templates import REFUSES_TWO_USER_TURNS
    w = ChatWrapper(Config(context_size=500), context=FakeContext(REFUSES_TWO_USER_TURNS, gen_tokens=3))
    report = w.begin(opened("s", [{"role": "user", "content": "eins"},
                                  {"role": "user", "content": "zwei"}]))
    assert report.question_joined
    assert after_system(w.conversation()["messages"]) == [
        {"role": "user", "content": "eins\n\nzwei"}]
    assert w.reply().finish().n_generated_tokens
