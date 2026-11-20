"""A conversation begun from a history, tool turns included."""

import copy
import json

import pytest

from llama_cpp_conversation.config import Config
from llama_cpp_conversation.errors import ContextOverflowError, InvalidArgumentError, TemplateRefusedError
from llama_cpp_conversation.tool_registry import ToolRegistry
from llama_cpp_conversation.tools import ToolCall, ToolSpec
from llama_cpp_conversation.wrapper import ChatWrapper
from tests import internals
from tests.chat_messages import Result, tool_turn, opened, after_system
from llama_cpp_conversation.testing.fake_context import FakeContext, ScriptedCall
from tests.internals import pending_calls_held
from llama_cpp_conversation.testing.invented_templates import (CALL_TAGS, PLAIN, REASONING_KEEPING_THOUGHTS, TOOLS,
                                      TOOLS_CLOSING_OPEN_TURNS, JSON_TOOLS, REFUSES_A_TEXT,
                                      REFUSES_TWO_USER_TURNS)

from tests.wrappers import (add_tools, begun_wrapper, tool_wrapper, unbegun_tool_wrapper,
                            weather_registry, rows, ask_through_a_tool_call, with_check_first)
from tests.chat_templates import STEMS, WEATHER, fixture
from llama_cpp_conversation.history_forms import (completions_messages, generated_turn_records, read_messages,
                                      with_generated_turns)
from llama_cpp_conversation.messages import GENERATED_TURN, UNREADABLE_CALL, THOUGHT_KEY
from llama_cpp_conversation.rendering import GeneratedTurn
from llama_cpp_conversation.result_cutting import cut_result


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
                                "tools": None, "reasoning": False, "reasoning_level": None,
                                "generated_turns": None}
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
    from llama_cpp_conversation.testing.invented_templates import TOOLS
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
    from llama_cpp_conversation.testing.invented_templates import JSON_TOOLS
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
    from llama_cpp_conversation.testing.invented_templates import REFUSES_TWO_USER_TURNS
    w = ChatWrapper(Config(context_size=500), context=FakeContext(REFUSES_TWO_USER_TURNS, gen_tokens=3))
    report = w.begin(opened("s", [{"role": "user", "content": "eins"},
                                  {"role": "user", "content": "zwei"}]))
    assert report.question_joined
    assert after_system(w.conversation()["messages"]) == [
        {"role": "user", "content": "eins\n\nzwei"}]
    assert w.reply().finish().n_generated_tokens


def test_two_turns_kept_as_generated_around_questions_joined_are_one_record():
    """A record that joins two questions takes in the turn kept before them, the branch that merges records."""
    from llama_cpp_conversation.history_forms import generated_turn_records
    from llama_cpp_conversation.messages import GENERATED_TURN, UNREADABLE_CALL
    from llama_cpp_conversation.rendering import GeneratedTurn
    unreadable = {"id": "u", "type": "function", UNREADABLE_CALL: True,
                  "function": {"name": "unknown", "arguments": {}}}
    called = {"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}
    messages = [{"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "", "tool_calls": [unreadable],
                 GENERATED_TURN: GeneratedTurn((1, 2), 1, True)},
                {"role": "tool", "name": "unknown", "tool_call_id": "u", "content": "failed"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "", "tool_calls": [called]},
                {"role": "tool", "name": "f", "tool_call_id": "c", "content": "12:00"},
                {"role": "assistant", "content": "Done.", GENERATED_TURN: GeneratedTurn((3, 4, 5), 3, False)}]
    assert [len(record["kept"]) for record in generated_turn_records(messages, True)] == [2]


@pytest.mark.parametrize("broken", [
    lambda turn: turn.update(tokens=["x"]),
    lambda turn: turn.update(span=99),
    lambda turn: turn.update(start=-1),
], ids=["tokens as text", "span past the messages", "start before them"])
def test_a_kept_turn_broken_in_one_field_is_refused_naming_generated_turns(broken):
    from llama_cpp_conversation.errors import InvalidArgumentError
    from llama_cpp_conversation.history_forms import ensure_generated_turns
    from tests.wrappers import weather_registry
    wrapper, fake = tool_wrapper(context_size=2048)
    registry = weather_registry()
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    wrapper.reply("wetter?", tool_call_policy=registry).finish()
    records = json.loads(json.dumps(wrapper.conversation()["generated_turns"]))
    broken(records["turns"][0]["kept"][0])
    with pytest.raises(InvalidArgumentError) as refused:
        ensure_generated_turns(records)
    assert refused.value.field == "generated_turns"


def test_saved_turns_stored_with_their_keys_sorted_are_restored():
    """The record is JSON, and a store may write its keys in another order."""
    from tests.wrappers import weather_registry
    wrapper, fake = tool_wrapper(context_size=2048)
    registry = weather_registry()
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    wrapper.reply("wetter?", tool_call_policy=registry).finish()
    saved = wrapper.conversation()
    stored = {**saved, "generated_turns": json.loads(json.dumps(saved["generated_turns"],
                                                                sort_keys=True))}
    again, _ = tool_wrapper(context_size=2048)
    assert again.begin(**stored).n_turns_restored == 1


def test_a_begin_cuts_every_result_no_kept_turn_stands_for_and_none_a_kept_turn_stands_for():
    """A result written by the caller before the kept turn, one inside it and one after it, begun again in a smaller context."""
    clock = ToolSpec("clock", "time", {"type": "object", "properties": {}})
    fake = FakeContext(JSON_TOOLS)
    live = ChatWrapper(Config(context_size=3000), context=fake)
    written = [{"role": "user", "content": "wie spät?"},
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "a", "type": "function", "function": {"name": "clock", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "a", "content": "A" * 300}]
    live.begin(opened("persona", written), tools=[clock])
    fake.call_script = ['[{"name": "clock", "arguments": {}}]'] * 2
    for text in ("B" * 300, "C" * 300):
        list(live.reply(tool_call_policy="yield"))
        live.inject_tool_result(live.pending_calls[0].id, text)
    saved = live.conversation()
    (first, last), = internals.generated_turn_spans(live)
    inside = {saved["messages"][i]["tool_call_id"] for i in range(first, last + 1)
              if saved["messages"][i]["role"] == "tool"}
    outside = {m["tool_call_id"] for m in saved["messages"] if m["role"] == "tool"} - inside
    assert inside and len(outside) == 2

    again = ChatWrapper(Config(context_size=800), context=FakeContext(JSON_TOOLS))
    report = again.begin(**saved)
    assert report.n_turns_restored == 1
    assert {call.id for call in report.tool_results_cut} == outside
    kept = again.conversation()["messages"]
    assert kept[first:last + 1] == saved["messages"][first:last + 1]
    assert again.reply().finish().n_generated_tokens


def _owed_call(name: str, arguments: dict) -> list[dict]:
    """A question and a reply calling ``name``, its result still owed."""
    return [{"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0abcdefg", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}]}]


@pytest.mark.parametrize("stem", [stem for stem in STEMS if fixture(stem).get("calls_tools")])
def test_a_begin_with_a_call_owed_holds_nothing_past_the_conversation_as_it_stands(stem):
    """The cache holds no token of what follows the owed call, such as the opening of the next turn, which a question may write otherwise than a result."""
    fake = FakeContext(fixture(stem))
    w = ChatWrapper(Config(context_size=4096), context=fake)
    try:
        w.begin(opened("s", _owed_call(WEATHER.name, {"city": "Bonn"})), tools=[WEATHER])
    except ValueError:
        pytest.skip("the template does not take this history")
    assert w.pending_calls
    standing = fake.stated_render(internals.rendered_messages(w), internals.declarations(w),
                                  generation_prompt=False, reasoning=internals.reasons(w),
                                  for_tools=internals.uses_tool_template(w))
    assert standing.startswith(fake.text_of(fake.cache))


def test_a_begin_with_a_call_owed_is_taken_where_the_template_refuses_a_result_that_names_no_call():
    """The results the cache is settled against name their calls, as every result the library writes does."""
    names_its_call = {**TOOLS, "chat_template": TOOLS["chat_template"].replace(
        "{%- if role == 'tool' -%}",
        "{%- if role == 'tool' -%}{%- if m.name is not string -%}"
        "{{- raise_exception('a result names its call') -}}{%- endif -%}", 1)}
    assert names_its_call != TOOLS
    w = ChatWrapper(Config(context_size=500), context=FakeContext(names_its_call, gen_tokens=3))
    w.begin(opened("s", _owed_call("weather", {"city": "Bonn"})),
            tools=weather_registry().specs())
    assert [call.id for call in w.pending_calls] == ["c0abcdefg"]
    w.inject_tool_result("c0abcdefg", "sonnig")
    assert w.reply().finish().n_generated_tokens


def test_a_history_the_template_refuses_is_refused_naming_messages_and_the_templates_reason():
    w = ChatWrapper(Config(context_size=500), context=FakeContext(REFUSES_A_TEXT, gen_tokens=3))
    with pytest.raises(ValueError) as refused:
        w.begin(opened("s", [{"role": "user", "content": "a"},
                             {"role": "assistant", "content": "refused"}]))
    assert refused.value.field == "messages"
    assert "this text is refused" in str(refused.value)


def test_a_begin_with_the_same_messages_kept_as_other_generated_tokens_keeps_the_given_tokens():
    given, given_fake = tool_wrapper(tokens_spell_text=False)
    ask_through_a_tool_call(given, given_fake)
    conversation = given.conversation()
    other, other_fake = tool_wrapper(tokens_spell_text=False)
    other.begin(opened("persona"))
    other.reply("hallo").finish()           # the model generates other tokens from here on
    ask_through_a_tool_call(other, other_fake)
    held = other.conversation()
    assert held["messages"] == conversation["messages"]
    assert held["generated_turns"] != conversation["generated_turns"]
    other.begin(**conversation)
    assert other.conversation() == conversation
    other.reply("und?").finish()


def _long_conversation(fake):
    wrapper = ChatWrapper(Config(context_size=2000, evict_above_fraction=0.9), context=fake)
    wrapper.begin(opened("s"))
    for n in range(30):
        wrapper.reply(f"Frage {n} " + "x" * 20).finish()
    assert wrapper.total_tokens > 1000              # more than half the context
    return wrapper


@pytest.mark.parametrize("goes_on", [True, False], ids=["goes on with the cache", "another history"])
def test_a_begin_past_the_eviction_limit_keeps_up_to_it_only_where_it_goes_on_with_the_cache(goes_on):
    wrapper = _long_conversation(FakeContext(TOOLS))
    saved = wrapper.conversation()
    if goes_on:
        newer = [message for n in range(12) for message in (
            {"role": "user", "content": f"Neu {n} " + "y" * 20},
            {"role": "assistant", "content": "reply"})]
        history = saved["messages"] + newer
    else:
        history = _long_conversation(FakeContext(TOOLS)).conversation()["messages"]
        history = [history[0]] + [{**m, "content": m["content"].replace("Frage", "Andere")}
                                  for m in history[1:]] * 2
    fresh = ChatWrapper(Config(context_size=2000, evict_above_fraction=0.9), context=FakeContext(TOOLS))
    assert fresh.begin(history).evicted_for == "half_context"
    report = wrapper.begin(history)
    kept = wrapper.conversation()["messages"]
    assert report.n_evicted_messages > 0 and kept[-1] == history[-1]
    if goes_on:
        # Its oldest part is what the cache holds already: it keeps up to the eviction limit.
        assert report.evicted_for == "room"
        assert len(kept) > len(fresh.conversation()["messages"])
        assert 1000 < wrapper.total_tokens <= 1800
    else:
        assert report.evicted_for == "half_context"
        assert kept == fresh.conversation()["messages"]
    wrapper.reply("und?").finish()


@pytest.mark.parametrize("exchanges_before", [1, 2])
def test_two_questions_in_a_row_after_earlier_exchanges_are_joined_in_their_own_place(exchanges_before):
    """The joined question takes the place of the first of the two, wherever the two stand, and the messages before them stay as they were."""
    earlier = [message for n in range(exchanges_before)
               for message in ({"role": "user", "content": f"Frage {n}"},
                               {"role": "assistant", "content": f"Antwort {n}"})]
    w = ChatWrapper(Config(context_size=1000),
                    context=FakeContext(REFUSES_TWO_USER_TURNS, gen_tokens=3))
    report = w.begin(opened("s", earlier + [{"role": "user", "content": "eins"},
                                            {"role": "user", "content": "zwei"}]))
    assert report.question_joined
    assert after_system(w.conversation()["messages"]) == earlier + [
        {"role": "user", "content": "eins\n\nzwei"}]
    assert w.reply().finish().n_generated_tokens


def test_an_open_turn_with_an_unreadable_entry_begins_again_with_the_answer_it_was_given():
    """The library's own answer to the unreadable entry, written after the caller's result, is part of the turn and comes back as it was, the cache unchanged."""
    fake = FakeContext(JSON_TOOLS)
    w = ChatWrapper(Config(context_size=2048), context=fake)
    w.begin(opened("persona"), tools=[ToolSpec("clock", "", {"type": "object", "properties": {}})])
    fake.call_script = ['[{"name": "clock", "arguments": {}}, {"no": "call"}]']
    list(w.reply("wie spät?", tool_call_policy="yield"))
    w.inject_tool_result(w.pending_calls[0].id, "12:00")
    assert w.turn_open and not w.pending_calls
    live, held, saved = list(fake.cache), internals.rendered_messages(w), w.conversation()
    w.begin(**saved)
    assert internals.rendered_messages(w) == held and fake.cache == live
    assert w.conversation() == saved
    assert w.reply().finish().n_generated_tokens


def _turn_left_open_until_a_question(w, fake):
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [step for step in w.reply("wie spät?", tool_call_policy="yield")
             if isinstance(step, ToolCall)]
    w.inject_tool_result(call.id, "12:00")


@pytest.mark.parametrize("call_turn", [ask_through_a_tool_call, _turn_left_open_until_a_question],
                         ids=["closed by its reply", "closed by the next question"])
def test_a_turn_kept_as_generated_stays_kept_when_the_exchanges_after_it_are_taken_from_the_history(
        call_turn):
    """A record is found by the turn's own messages and the question before them, so that a client that drops later messages, as a regenerate does, keeps it."""
    w, fake = tool_wrapper(context_size=2048)
    call_turn(w, fake)
    w.reply("Und?").finish()
    w.reply("Und dann?").finish()
    kept = internals.generated_turns_held(w)
    assert kept
    saved = w.conversation()
    for dropped in (2, 4):
        w.begin(**{**saved, "messages": saved["messages"][:-dropped]})
        assert internals.generated_turns_held(w) == kept


_SYSTEM = {"role": "system", "content": "persona"}


def _question(text):
    return {"role": "user", "content": text}


def _reply(text, calls=(), turn=None):
    """A reply, ending the generated turn ``turn`` (its tokens and span) where one is given."""
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = list(calls)
    if turn is not None:
        tokens, span = turn
        message[GENERATED_TURN] = GeneratedTurn(tokens, span, bool(calls))
    return message


def _call(call_id):
    return {"id": call_id, "type": "function", "function": {"name": "clock", "arguments": {}}}


def _unreadable(call_id):
    return {"id": call_id, "type": "function", UNREADABLE_CALL: True,
            "function": {"name": "unknown", "arguments": {}}}


def _result(call_id, name="clock"):
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": f"Ergebnis {call_id}"}


# Each history, whether it joins questions, and for each generated turn the Chat Completions
# messages it is found by, as a slice of those messages: the question before it, its own
# messages and results, and the message after it where it gives none of its own. ``None`` is
# a turn with nothing given back to find it by.
_HISTORIES_WITH_KEPT_TURNS = {
    "closed by its answer": (
        [_SYSTEM, _question("Q1"), _reply("", [_call("c")]), _result("c"),
         _reply("Antwort", turn=((11, 12, 13), 3)), _question("Q2"), _reply("A2")],
        False, [(1, 5)]),
    "two exchanges": (
        [_SYSTEM, _question("Q1"), _reply("", [_call("c")]), _result("c"),
         _reply("Antwort", turn=((11, 12), 3)), _question("Q2"), _reply("", [_call("d")]),
         _result("d"), _reply("Antwort 2", turn=((21, 22), 3))],
        False, [(1, 5), (5, 9)]),
    "open, closed by a question": (
        [_SYSTEM, _question("Q1"), _reply("", [_call("c")], turn=((11,), 1)), _result("c"),
         _question("Q2"), _reply("A2")],
        False, [(1, 4)]),
    "nothing of its own, at the end": (
        [_SYSTEM, _question("Q1"), _reply("", [_unreadable("u")], turn=((11,), 1)),
         _result("u", "unknown")],
        False, [(1, 2)]),
    "nothing of its own, a question and a reply after it": (
        [_SYSTEM, _question("Q1"), _reply("", [_unreadable("u")], turn=((11,), 1)),
         _result("u", "unknown"), _question("Q2"), _reply("A2")],
        False, [(1, 3)]),
    "nothing of its own, the newest question after it": (
        [_SYSTEM, _question("Q1"), _reply("", [_unreadable("u")], turn=((11,), 1)),
         _result("u", "unknown"), _question("Q2")],
        False, [(1, 3)]),
    "nothing of its own, then a turn after the question that closed it": (
        [_question("Q1"), _reply("", [_unreadable("u")], turn=((11,), 1)), _result("u", "unknown"),
         _question("Q2"), _reply("", [_call("c")]), _result("c"),
         _reply("Fertig", turn=((21, 22), 3))],
        False, [(0, 2), (1, 5)]),
    "nothing to find it by, then a turn": (
        [_reply("", [_unreadable("u")], turn=((11,), 1)), _result("u", "unknown"),
         _question("Q2"), _reply("", [_call("c")]), _result("c"),
         _reply("Fertig", turn=((21, 22), 3))],
        False, [None, (0, 4)]),
    "a turn going on from the results of the one before": (
        [_SYSTEM, _question("Q1"), _reply("", [_call("c")], turn=((11,), 1)), _result("c"),
         _reply("Antwort", turn=((21,), 1)), _question("Q2"), _reply("A2")],
        False, [(1, 5), (1, 5)]),
}


def _altered(message):
    """``message``, a Chat Completions message, with its text changed."""
    return {**message, "content": (message.get("content") or "") + " (geändert)"}


def _turns_put_back(given, records, joins):
    """The tokens of each generated turn :func:`with_generated_turns` puts back into ``given``."""
    put, _n = with_generated_turns(read_messages(given), records, joins)
    return [message[GENERATED_TURN].tokens for message in put if GENERATED_TURN in message]


@pytest.mark.parametrize("name", list(_HISTORIES_WITH_KEPT_TURNS))
def test_a_generated_turn_is_found_by_exactly_the_messages_its_record_names(name):
    """A turn is put back in the messages it is found by alone, and in none of them with one of those messages changed, so that a history cut or edited elsewhere keeps it."""
    messages, joins, windows = _HISTORIES_WITH_KEPT_TURNS[name]
    given = completions_messages(messages, joins)
    records = generated_turn_records(messages, joins)
    turns = [m[GENERATED_TURN].tokens for m in messages if GENERATED_TURN in m]
    assert _turns_put_back(given, records, joins) == [t for t, w in zip(turns, windows) if w]
    for tokens, window in zip(turns, windows):
        if window is None:
            continue
        lo, hi = window
        assert tokens in _turns_put_back(given[lo:hi], records, joins), \
            f"the turn is not found in messages {lo} to {hi - 1}, those it is found by"
        for i in range(lo, hi):
            edited = given[:i] + [_altered(given[i])] + given[i + 1:]
            assert tokens not in _turns_put_back(edited, records, joins), \
                f"the turn is found with message {i}, one it is found by, changed"


def test_each_generated_turn_stays_kept_when_begin_is_given_the_history_up_to_its_end():
    """A client that cuts the history after a turn, as a regenerate does, keeps that turn and every one before it, each record standing on its own."""
    w, fake = tool_wrapper(context_size=4096)
    ask_through_a_tool_call(w, fake)
    fake.call_script = ["fn:clock{}"]
    call, = [step for step in w.reply("und jetzt?", tool_call_policy="yield")
             if isinstance(step, ToolCall)]
    w.inject_tool_result(call.id, "12:05")
    w.reply(tool_call_policy="yield").finish()
    w.reply("Danke").finish()
    kept, spans = internals.generated_turns_held(w), internals.generated_turn_spans(w)
    assert len(kept) == 2
    saved = w.conversation()
    assert len(saved["messages"]) == len(internals.rendered_messages(w))
    for length in range(1, len(saved["messages"]) + 1):
        w.begin(**{**saved, "messages": saved["messages"][:length]})
        assert internals.generated_turns_held(w) == [
            turn for turn, (_first, last) in zip(kept, spans) if last < length], \
            f"a turn is lost from the history cut to {length} messages"


def _a_turn_kept_as_generated():
    """A wrapper whose conversation keeps one turn that called a tool as generated."""
    registry = weather_registry()
    wrapper, fake = tool_wrapper(context_size=2048)
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    wrapper.reply("wetter?", tool_call_policy=registry).finish()
    return wrapper


def _changed_inside(value) -> None:
    """Change every list and dict nested in ``value`` in place."""
    if isinstance(value, dict):
        for item in list(value.values()):
            _changed_inside(item)
        value["changed"] = True
    elif isinstance(value, list):
        for item in list(value):
            _changed_inside(item)
        value.append("changed")


def test_changing_what_conversation_gives_inside_a_kept_turns_messages_changes_nothing_held():
    """The kept turns ``conversation()`` gives are the caller's to change, down to a call's arguments."""
    wrapper = _a_turn_kept_as_generated()
    given = wrapper.conversation()
    before = copy.deepcopy(given)
    assert any(m.get("tool_calls") for r in given["generated_turns"]["turns"] for m in r["messages"])
    for record in given["generated_turns"]["turns"]:
        for message in record["messages"]:
            for value in message.values():
                _changed_inside(value)
    assert wrapper.conversation() == before


def test_a_kept_turn_given_back_with_a_lone_surrogate_in_any_of_its_messages_is_left_out():
    """Such a record is one changed since it was given out; the messages alone begin the conversation."""
    wrapper = _a_turn_kept_as_generated()
    saved = wrapper.conversation()
    (record,) = saved["generated_turns"]["turns"]
    for at in range(len(record["messages"])):
        stored = {**saved, "generated_turns": json.loads(json.dumps(saved["generated_turns"]))}
        held = stored["generated_turns"]["turns"][0]["messages"][at]
        held["content"] += "\ud800"
        again, _ = tool_wrapper(context_size=2048)
        report = again.begin(**stored)
        assert report.generated_turns_ignored and report.n_turns_restored == 0
        assert again.conversation()["messages"] == saved["messages"]


def _first_in_role(record: dict, role: str) -> dict:
    """The first message of ``record`` in ``role``."""
    return next(message for message in record["messages"] if message["role"] == role)


def _first_call(record: dict) -> dict:
    """The first call a reply of ``record`` makes."""
    return next(message for message in record["messages"] if message.get("tool_calls"))["tool_calls"][0]


_BROKEN = {
    "a turn standing for no message": lambda r: r["kept"][0].update(span=0),
    "a record found by no message": lambda r: r.update(count=0),
    "a message that is not an object": lambda r: r["messages"].__setitem__(0, "q"),
    "a question without text": lambda r: _first_in_role(r, "user").update(content=None),
    "a result answering a numbered call": lambda r: _first_in_role(r, "tool").update(tool_call_id=7),
    "a result naming no tool": lambda r: _first_in_role(r, "tool").update(name=None),
    "a call's arguments as text": lambda r: _first_call(r)["function"].update(arguments="{}"),
    "a call without an id": lambda r: _first_call(r).update(id=None),
}


@pytest.mark.parametrize("broken", list(_BROKEN.values()), ids=list(_BROKEN))
def test_a_kept_turns_record_broken_in_one_field_is_refused_by_begin_naming_generated_turns(broken):
    """A record ``conversation()`` gave, one field put in a form it never gives, is refused, not left out as one changed since."""
    registry = weather_registry()
    live, fake = tool_wrapper(context_size=2048)
    live.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    live.reply("wetter?", tool_call_policy=registry).finish()
    saved = live.conversation()
    saved["generated_turns"] = json.loads(json.dumps(saved["generated_turns"]))
    broken(saved["generated_turns"]["turns"][0])
    again, again_fake = tool_wrapper(context_size=2048)
    again.begin(opened("persona"), tools=registry.specs())
    before, held = again.conversation(), list(again_fake.cache)
    with pytest.raises(InvalidArgumentError) as refused:
        again.begin(**saved)
    assert refused.value.field == "generated_turns"
    assert again.conversation() == before and again_fake.cache == held


def test_a_turn_kept_as_generated_begins_again_in_its_own_place_where_a_later_exchange_reads_alike():
    """A turn whose Chat Completions messages read as a later exchange does is put back where it stood when it was saved, not on the newer exchange."""
    w, fake = tool_wrapper(context_size=2048)
    w.begin(opened("persona"))
    fake.call_script = ["fn:nonsense with no name"]     # an unreadable call, left out of the messages
    w.reply("los", tool_call_policy="refuse").finish()
    w.reply("weiter").finish()
    w.reply("los").finish()                             # reads as the turn kept above
    spans, saved = internals.generated_turn_spans(w), w.conversation()
    messages = saved["messages"]
    assert len(spans) == 1 and messages[1:3] == messages[5:7]
    again, _ = tool_wrapper(context_size=2048)
    again.begin(**saved)
    assert internals.generated_turn_spans(again) == spans
    assert again.conversation() == saved


def test_a_begin_cuts_an_older_result_only_once_every_newer_one_of_its_exchange_keeps_nothing():
    """Three results of one round, begun in contexts from roomy to tight: the newest are cut first, each to nothing before an older one loses a character."""
    clock = ToolSpec("clock", "time", {"type": "object", "properties": {}})
    ids = ["a", "b", "c"]
    texts = {"a": "A" * 300, "b": "B" * 300, "c": "C" * 300}
    history = [{"role": "user", "content": "wie spät?"},
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": i, "type": "function", "function": {"name": "clock", "arguments": "{}"}}
                   for i in ids]},
               *[{"role": "tool", "tool_call_id": i, "content": texts[i]} for i in ids]]
    older_cut = 0
    for context_size in range(200, 1400, 10):
        w = ChatWrapper(Config(context_size=context_size), context=FakeContext(JSON_TOOLS))
        try:
            report = w.begin(opened("persona", history), tools=[clock])
        except ContextOverflowError:
            continue
        held = {m["tool_call_id"]: m["content"] for m in w.conversation()["messages"]
                if m["role"] == "tool"}
        if set(held) != set(ids):
            continue
        cut = [i for i in ids if held[i] != texts[i]]
        assert {call.id for call in report.tool_results_cut} == set(cut)
        for older, newer in zip(ids, ids[1:]):
            if held[older] != texts[older]:
                assert held[newer] == cut_result(texts[newer], 0), (context_size, held)
                older_cut += 1
    assert older_cut, "no context size made an older result be cut"


CLOCK = ToolSpec("clock", "time", {"type": "object", "properties": {}})


def _calling_clock(*ids: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "clock", "arguments": "{}"}} for i in ids]}


def _clock_result(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


@pytest.mark.parametrize("newest", [
    [_calling_clock("a"), _clock_result("a", "A" * 2000), {"role": "assistant", "content": "fertig"}],
    [_calling_clock("a", "b"), _clock_result("a", "A" * 2000), _clock_result("b", "")],
    [_calling_clock("a", "b"), _clock_result("a", "A" * 2000), _clock_result("b", ""), {"role": "assistant", "content": "fertig"}],
], ids=["answered", "an-empty-result-after-it", "an-empty-result-and-an-answer-after-it"])
def test_a_begin_cuts_the_long_result_of_its_newest_exchange_wherever_it_stands_and_changes_nothing_else(newest):
    """An answer or an empty result after the long one is passed over, not taken for the end of the results to cut, and an empty result is not cut."""
    history = opened("persona", [{"role": "user", "content": "wie spät?"}, *newest])
    wrapper = ChatWrapper(Config(context_size=400), context=FakeContext(JSON_TOOLS))
    report = wrapper.begin(history, tools=[CLOCK])
    assert [call.id for call in report.tool_results_cut] == ["a"]
    kept = wrapper.conversation()["messages"]
    assert len(kept) == len(history)
    for given, held in zip(history, kept):
        if given.get("tool_call_id") == "a":
            assert held["content"].endswith(" of 2000 characters kept)")
            assert "A" * 10 in held["content"]
        else:
            assert held.get("content") == given.get("content")
    wrapper.reply("und?").finish()


def test_a_begin_cuts_no_result_of_an_exchange_before_its_newest():
    """Where the newest exchange's results cut to nothing still leave too little room, an older exchange's result is never cut to make it; the begin keeps less of the history, or refuses."""
    cases = [(min_reply_tokens, old, new) for min_reply_tokens in (320, 340)
             for old in range(20, 301, 20) for new in (0, 1, 10, 50)]
    for min_reply_tokens, old, new in cases:
        history = opened("persona", [
            {"role": "user", "content": "q0"}, _calling_clock("c0"), _clock_result("c0", "A" * old),
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "q1"}, _calling_clock("c1"), _clock_result("c1", "B" * new)])
        # The room beside the reply is below the half of the context a begin keeps history in.
        wrapper = ChatWrapper(Config(context_size=600, min_reply_tokens=min_reply_tokens),
                              context=FakeContext(JSON_TOOLS))
        try:
            report = wrapper.begin(history, tools=[CLOCK])
        except ContextOverflowError:
            continue
        assert {call.id for call in report.tool_results_cut} <= {"c1"}, (min_reply_tokens, old, new)
        kept = wrapper.conversation()["messages"]
        assert all(m["content"] == "A" * old for m in kept if m.get("tool_call_id") == "c0")


def _first_calling(record: dict) -> dict:
    """The first reply of ``record`` that makes a call."""
    return next(message for message in record["messages"] if message.get("tool_calls"))


_HELD_IN_NO_FORM = {
    "a question put in a role no turn holds": lambda r: _first_in_role(r, "user").update(role="system"),
    "a reply with a key no reply holds": lambda r: _first_calling(r).update(name="helper"),
    "a reply whose thought is not text": lambda r: _first_calling(r).update({THOUGHT_KEY: 7}),
}


@pytest.mark.parametrize("broken", list(_HELD_IN_NO_FORM.values()), ids=list(_HELD_IN_NO_FORM))
def test_a_kept_turns_record_holding_a_message_in_a_form_no_record_holds_is_refused_naming_generated_turns(broken):
    """A record holds only questions, replies and results, each with the keys and the types ``conversation()`` gives it."""
    registry = weather_registry()
    live, fake = tool_wrapper(context_size=2048)
    live.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    live.reply("wetter?", tool_call_policy=registry).finish()
    saved = live.conversation()
    saved["generated_turns"] = json.loads(json.dumps(saved["generated_turns"]))
    broken(saved["generated_turns"]["turns"][0])
    again, again_fake = tool_wrapper(context_size=2048)
    again.begin(opened("persona"), tools=registry.specs())
    before, held = again.conversation(), list(again_fake.cache)
    with pytest.raises(InvalidArgumentError) as refused:
        again.begin(**saved)
    assert refused.value.field == "generated_turns"
    assert again.conversation() == before and again_fake.cache == held


def test_a_result_the_template_writes_only_before_its_reply_is_not_named_as_the_refused_message():
    """A history up to a result counts as written where the template writes it with the generation prompt after it."""
    says = with_check_first(TOOLS,
                       "{%- if messages[-1].role == 'tool' and not add_generation_prompt -%}"
                       "{{ raise_exception('a result is followed by its reply') }}{%- endif -%}"
                       "{%- for m in messages -%}"
                       "{%- if m.role == 'user' and loop.previtem is defined "
                       "and loop.previtem.role == 'tool' -%}"
                       "{{ raise_exception('a question does not follow a result') }}{%- endif -%}"
                       "{%- if m.content == 'verboten' -%}"
                       "{{ raise_exception('this word is not written') }}{%- endif -%}"
                       "{%- endfor -%}")
    chat = ChatWrapper(Config(context_size=1024), context=FakeContext(says))
    history = opened("s", [{"role": "user", "content": "Wetter?"},
                           *tool_turn(ToolCall("weather", {"city": "Bonn"}), Result("weather", "sonnig")),
                           {"role": "assistant", "content": "Sonnig."},
                           {"role": "user", "content": "verboten"}])
    with pytest.raises(TemplateRefusedError) as refused:
        chat.begin(history, tools=weather_registry().specs())
    message = str(refused.value)
    assert refused.value.field == "messages"
    assert "the user message 'verboten' where it stands" in message
    assert "this word is not written" in message and "tool message" not in message


def test_a_call_of_the_history_that_writes_a_key_twice_is_left_out_as_one_the_model_wrote_so_would_be():
    """Read alike, given or generated: the value meant cannot be told."""
    w, _ = tool_wrapper()
    history = [{"role": "user", "content": "Zeit?"},
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "c1", "type": "function",
                    "function": {"name": "clock", "arguments": '{"zone": "CET", "zone": "UTC"}'}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "12:00"},
               {"role": "assistant", "content": "Zwölf."}]
    report = w.begin(opened("s", history))
    assert report.n_left_out_calls == 1
    assert all("tool_calls" not in m for m in w.conversation()["messages"])
