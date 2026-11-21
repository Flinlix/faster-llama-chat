"""Cutting the newest reply back, and the saved states a cache that cannot shift returns to."""

import pytest

from tests import internals
from tests.chat_messages import opened, Result, tool_turn

from llama_cpp_conversation.config import Config
from llama_cpp_conversation.resolved_template import ResolvedTemplate
from llama_cpp_conversation.tools import ToolSpec, ToolCall
from llama_cpp_conversation.wrapper import ChatWrapper
from tests.chat_templates import STEMS, expected, fixture, ANSWERS, QUESTIONS, WEATHER
from llama_cpp_conversation.testing.fake_context import FakeContext, ScriptedCall, shipped_addon_call, written_call
from tests.internals import token_id_range
from llama_cpp_conversation.testing.invented_templates import PLAIN, REASONING, TOOLS
from tests.wrappers import (assert_consistent, begun_wrapper, held_text, newest_turn, tool_wrapper,
                            reasoning_wrapper, rendered_reply, rows, ask_through_a_tool_call)
from llama_cpp_conversation import OutOfOrderError, ContextOverflowError, TemplateRefusedError
from llama_cpp_conversation.errors import InvalidArgumentError
from llama_cpp_conversation.tool_registry import ToolRegistry
def _without_restore(fake):
    class Wrapped:
        def __getattr__(self, name):
            if name == "restore_state":
                raise AttributeError(name)
            return getattr(fake, name)
    return Wrapped()


# ----- cutting the newest reply --------------------------------------------
def test_a_cut_keeps_the_prefix_and_the_next_request_appends_at_the_cache_end():
    w, _fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    w.truncate_last_reply("Eins.")
    assert rows(w)[-1]["text"] == "Eins."
    w.reply("weiter").finish()


def test_a_reply_generated_in_other_pieces_than_its_text_tokenizes_into_is_kept_as_its_text():
    """Same text, other tokens: the template's past reply is the text, tokenized."""
    w, fake = begun_wrapper(gen_tokens=4, gen_text="ok!!", tokens_spell_text=False)
    reply = w.reply("hi").finish()
    assert rendered_reply(fake, "ok!!") in held_text(fake)
    assert not any(fake.is_generated_only(token) for token in fake.cache)   # none is kept
    assert reply.text == "ok!!"


@pytest.mark.parametrize("spelled", [True, False], ids=["spelling-text", "other-pieces"])
def test_every_token_id_the_stand_in_hands_out_lies_where_a_context_may_have_one(spelled):
    w, fake = begun_wrapper(gen_tokens=40, gen_text="Ein langer Satz, der viele Token braucht.",
                            tokens_spell_text=spelled)
    sampled, drawn = fake.sampled_tokens, []

    def recorded(*args, **kwargs):
        for token in sampled(*args, **kwargs):
            drawn.append(token)
            yield token

    fake.sampled_tokens = recorded
    for question in ("eins", "zwei", "drei"):
        w.reply(question).finish()
    handed_out = set(drawn) | set(fake.cache) | set(fake.special_tokens())
    assert handed_out <= set(token_id_range())
    assert spelled or any(fake.is_generated_only(token) for token in drawn)


def test_a_model_whose_template_renders_nothing_is_refused_at_load_naming_why():
    refuses_everything = dict(PLAIN, chat_template="{{ raise_exception('nothing') }}")
    with pytest.raises(ValueError, match="reply can be told from the question"):
        ChatWrapper(Config(context_size=500), context=FakeContext(refuses_everything))


def test_a_model_whose_template_drops_every_reply_is_refused_at_load_naming_why():
    drops = dict(PLAIN, tags=["<|user_says|>", "<|assistant_says|>", "<|said|>"],
                 eos_token="<|said|>",
                 chat_template="{%- for m in messages if m.role != 'assistant' -%}"
                               "<|{{ m.role }}_says|>{{ m.content }}<|said|>{%- endfor -%}"
                               "{%- if add_generation_prompt -%}<|assistant_says|>{%- endif -%}")
    with pytest.raises(ValueError, match="reply can be told from the question"):
        ChatWrapper(Config(context_size=500), context=FakeContext(drops))


def test_a_context_of_your_own_that_reports_no_template_text_is_refused_at_load():
    """Its renders alone would pass every check, but no add-on file could name its template."""
    class _NoTemplateText(FakeContext):
        def chat_template_text(self, template_name=None):
            return None

    with pytest.raises(ValueError, match="no chat template"):
        ChatWrapper(Config(context_size=500), context=_NoTemplateText(PLAIN))


def test_the_tokens_that_stay_are_not_decoded_again():
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    before = list(fake.prefill_calls)
    w.truncate_last_reply("Eins.")
    (start, n, _logits), = fake.prefill_calls[len(before):]
    # From the end of what stays: the closing and the next question's opening.
    assert fake.text_of(fake.cache[:start]).endswith("Eins.")
    assert n == len(fake.tokenize("<turn]>\n<[turn>"))


def test_keeping_the_whole_text_changes_nothing():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    cache = list(fake.cache)
    w.truncate_last_reply("Hallo")
    assert fake.cache == cache


def test_a_text_that_is_not_a_prefix_is_refused_and_changes_nothing():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    cache = list(fake.cache)
    with pytest.raises(ValueError, match="prefix"):
        w.truncate_last_reply("Tschüss")
    assert fake.cache == cache


def test_only_an_assistant_reply_can_be_cut():
    w, _fake = begun_wrapper()
    w.inject("nebenbei", "user")
    with pytest.raises(ValueError, match="assistant"):
        w.truncate_last_reply("")


def test_a_prefilled_reply_keeps_the_templates_opening_when_cut():
    w, fake = begun_wrapper()
    w.inject("Eins. Zwei.", "assistant")
    w.truncate_last_reply("Eins.")
    assert held_text(fake).endswith(rendered_reply(fake, "Eins.").rstrip("\n") + "\n<[turn>")
    assert "Zwei" not in held_text(fake)


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
def test_a_cut_to_a_reply_the_template_does_not_write_is_refused(stem):
    """These templates write no empty reply, so a reply cut to nothing could be neither held nor followed."""
    fake = FakeContext(fixture(stem), gen_tokens=7, gen_text="Antwort")
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=1.0), context=fake)
    w.begin(opened(""))
    w.reply("Hallo").finish()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ValueError, match="keep more"):
        w.truncate_last_reply("")
    assert w.conversation() == before and fake.cache == held
    w.truncate_last_reply("Ant")
    assert w.reply("Weiter").finish().text == "Antwort"


def test_a_cut_to_nothing_stands_where_the_template_writes_an_empty_reply():
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    w.truncate_last_reply("")
    assert rows(w)[-1] == {"role": "assistant", "text": ""}
    w.reply("weiter").finish()


def test_a_reply_whose_call_waits_for_its_result_cannot_be_cut():
    w, fake = begun_wrapper(TOOLS)
    fake.call_script.append("fn:uhr{}")
    w.begin(opened("s"), tools=[ToolSpec("uhr", "", {})])
    for _event in w.reply("wie spät?", tool_call_policy="yield"):
        pass
    with pytest.raises(ValueError, match="reply is open"):
        w.truncate_last_reply("")


def _fails_to_save():
    raise RuntimeError("no state saved this time")


@pytest.mark.parametrize("saves_state", [True, False])
def test_a_cut_on_a_cache_that_cannot_shift_restores_or_rebuilds(saves_state):
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw", can_shift=False)
    if not saves_state:
        fake.save_state = _fails_to_save
        w.begin(opened("anders"))           # no state saved before fits this opening
    w.reply("hi").finish()
    resets = fake.resets
    w.truncate_last_reply("Eins.")
    # From the state saved before the reply where there is one, else rebuilt.
    assert fake.resets == resets + (not saves_state)
    assert rows(w)[-1]["text"] == "Eins."


# ----- a past reply on a cache that cannot shift ---------------------------
def test_a_past_reply_on_a_cache_that_cannot_shift_comes_back_from_its_saved_state():
    w, fake = reasoning_wrapper(can_shift=False)
    resets = fake.resets
    prefilled = fake.total_prefilled()
    w.reply("hi").finish()
    # No rebuild: the request and the past reply are all that is decoded again.
    assert fake.resets == resets
    assert fake.total_prefilled() - prefilled < 2 * len(newest_turn(fake)) + 20
    assert_consistent(w, fake)
    w.truncate_last_reply("Ant")
    assert fake.resets == resets


def test_a_cache_that_fails_to_save_a_state_is_rebuilt_instead():
    w, fake = reasoning_wrapper(can_shift=False)
    fake.save_state = _fails_to_save
    w.begin(opened("anders"), reasoning=True)   # no state saved before fits this opening
    resets = fake.resets
    w.reply("hi").finish()
    assert fake.resets == resets + 1


# ----- saved states of a cache that cannot shift ---------------------------
def test_a_saved_state_the_cache_refuses_is_rebuilt_from_instead():
    w, fake = reasoning_wrapper(can_shift=False)
    w.reply("hi").finish()

    def refuse(state, pos):
        raise RuntimeError("the cache refused the saved state")
    fake.restore_state = refuse
    resets = fake.resets
    w.truncate_last_reply("Ant")
    assert fake.resets == resets + 1


def test_a_cache_that_takes_a_cut_needs_no_saved_state():
    w, fake = reasoning_wrapper(can_shift=False)
    fake.can_cut_tail = True
    w.begin(opened("s"), reasoning=True)
    saved, resets = fake.saved_states, fake.resets
    w.reply("hi").finish()
    w.truncate_last_reply("Ant")
    assert fake.saved_states == saved and fake.resets == resets


def test_a_cut_leaves_a_state_for_the_next_reply():
    w, fake = reasoning_wrapper(can_shift=False)
    w.reply("hi").finish()
    saved = fake.saved_states
    w.truncate_last_reply("Ant")
    assert fake.saved_states == saved + 1


@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_what_a_reply_decodes_again_is_counted(can_shift):
    w, fake = reasoning_wrapper(can_shift=can_shift)
    w.reply("eins").finish()
    before, ahead = fake.total_prefilled(), internals.done_after_the_newest_reply(w)[0]
    reply = w.reply("hi").finish()
    # What was decoded ahead for it after the reply before counts, and what is decoded
    # ahead after it counts for the next.
    assert reply.n_prefilled_tokens == (fake.total_prefilled() - before + ahead
                                        - internals.done_after_the_newest_reply(w)[0])


@pytest.mark.parametrize("refusal", ["none", "raises"])
def test_a_context_of_your_own_that_can_neither_shift_cut_nor_save_is_refused(refusal):
    """It could go back to an earlier point only by decoding the whole conversation again."""
    class _NoWayBack(FakeContext):
        def save_state(self):
            if refusal == "raises":
                raise RuntimeError("no state to save")

    with pytest.raises(ValueError, match="neither shift"):
        ChatWrapper(Config(context_size=500),
                    context=_NoWayBack(PLAIN, can_shift=False, can_cut_tail=False))
    for can_shift, can_cut_tail in ((True, False), (False, True)):
        ChatWrapper(Config(context_size=500), context=_NoWayBack(
            PLAIN, can_shift=can_shift, can_cut_tail=can_cut_tail)).begin(opened("s"))


def test_a_context_that_saves_states_must_restore_them():
    with pytest.raises(TypeError, match="restore_state"):
        ChatWrapper(Config(context_size=500),
                    context=_without_restore(FakeContext(PLAIN, can_shift=False)))


# ----- a reply ended by its closing text -----------------------------------
@pytest.mark.parametrize("stem", [stem for stem in STEMS if expected(fixture(stem), "turn_end_text")])
def test_a_reply_ended_by_its_closing_text_is_cut_back_as_its_cache_allows(stem):
    """A cache that cannot cut at its end goes back to a saved state instead."""
    # A template whose reply ends where a text its closing opens with is written.
    known = fixture(stem)
    closing = ResolvedTemplate(FakeContext(known)).call_boundary().turn_end_text
    for can_shift in (True, False):
        # The reply, then its closing, which leaves the reply again.
        fake = FakeContext(known, ["rep" + closing, "Fine."], can_shift=can_shift)
        w = ChatWrapper(Config(context_size=2000, min_reply_tokens=8), context=fake)
        w.begin(opened("s"))
        assert w.reply("hi").finish().text == "rep"
        assert fake.edit_count("cut" if can_shift else "restore") == 1
        w.reply("again").finish()
        assert_consistent(w, fake)


def test_a_cache_context_whose_list_of_tags_leaves_one_out_is_refused_at_load_naming_why():
    class _FewTags(FakeContext):
        def special_tokens(self):
            return ()

    with pytest.raises(ValueError, match="list every tag"):
        ChatWrapper(Config(context_size=500), context=_FewTags(PLAIN))


def test_a_reply_cut_to_nothing_after_its_call_keeps_neither_its_text_nor_the_result_in_the_turn():
    """The cut searches for the call's closing only before the reply's own text, which may spell it."""
    from llama_cpp_conversation.tool_registry import ToolRegistry
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(context_size=2048, writes_tags=False)
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    fake.replies = ["Es ist </call> zwölf."]
    shown = "".join(step for step in wrapper.reply("wie spät?", tool_call_policy=registry)
                    if isinstance(step, str))
    newest = wrapper.conversation()["messages"][-1]["content"]
    wrapper.truncate_last_reply(shown[:len(shown) - len(newest)])
    held = held_text(fake)
    assert "Es ist" not in held and held.count("<result>") == 1


@pytest.mark.parametrize("stem", STEMS)
def test_on_a_cache_that_cannot_be_cut_a_reply_written_again_goes_back_no_further_than_its_question(stem):
    """A state is kept where the reply begins, so that a reply the template writes otherwise than it was generated decodes only itself."""
    fake = FakeContext(fixture(stem), ANSWERS, can_shift=False)
    chat = ChatWrapper(Config(context_size=4096, min_reply_tokens=8, min_answer_tokens=None,
                              answer_room_tokens=None), context=fake)
    system = "Be brief." if chat.capabilities(tools=False).can_take_system_prompt else ""
    chat.begin(opened(system))
    for asked, question in enumerate(QUESTIONS[:3]):
        edits = len(fake.edits)
        if asked % 2:
            chat.inject(question)
            chat.reply().finish()
        else:
            chat.reply(question).finish()
        held, written = internals.held_tokens(chat), fake.tokenize(question, parse_special=False)
        starts = [at for at in range(len(held) - len(written) + 1)
                  if held[at:at + len(written)] == written]
        question_end = starts[-1] + len(written)
        assert all(edit[1] >= question_end for edit in fake.edits[edits:] if edit[0] == "restore")


def test_a_conversation_begun_without_a_message_has_no_reply_to_cut():
    wrapper = ChatWrapper(Config(context_size=500), context=FakeContext(PLAIN))
    wrapper.begin(opened(""))
    assert wrapper.conversation()["messages"] == []
    with pytest.raises(OutOfOrderError, match="not an assistant reply") as refused:
        wrapper.truncate_last_reply("")
    assert refused.value.field is None
    assert wrapper.conversation()["messages"] == []


def test_a_cut_refused_while_a_call_waits_names_the_cut_as_the_action_refused():
    wrapper, fake = begun_wrapper(TOOLS)
    fake.call_script.append("fn:uhr{}")
    wrapper.begin(opened("s"), tools=[ToolSpec("uhr", "", {})])
    for _event in wrapper.reply("wie spät?", tool_call_policy="yield"):
        pass
    with pytest.raises(OutOfOrderError, match="^cannot truncate a reply while a reply is open"):
        wrapper.truncate_last_reply("")


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
def test_every_cut_refused_for_its_text_names_keep_text(stem):
    """A text that is not a prefix and a reply the template does not write shortened are both about ``keep_text``."""
    fake = FakeContext(fixture(stem), gen_tokens=7, gen_text="Antwort")
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=1.0), context=fake)
    w.begin(opened(""))
    w.reply("Hallo").finish()
    for keep_text, error in (("Tschüss", InvalidArgumentError), ("", TemplateRefusedError)):
        with pytest.raises(error) as refused:
            w.truncate_last_reply(keep_text)
        assert refused.value.field == "keep_text", (keep_text, refused.value)


def _closed_without_its_call():
    """A wrapper whose newest reply keeps its tokens as generated, its unreadable call gone for want of room."""
    for size in range(200, 900, 2):
        context = FakeContext(TOOLS, [ScriptedCall("not a call", before="Moment. ")])
        chat = ChatWrapper(Config(context_size=size, evict_above_fraction=1.0,
                                  min_reply_tokens=120), context=context)
        try:
            chat.begin(opened("Be brief."), tools=[WEATHER])
            reply = chat.reply("Weather?", tool_call_policy="yield").finish()
        except ContextOverflowError:
            continue
        if reply.stop_reason == "unreadable_tool_call" and not chat.turn_open:
            return chat
    pytest.fail("no context size left the error result without room")


def test_a_reply_whose_tokens_as_generated_hold_a_call_that_went_is_refused_naming_keep_text():
    chat = _closed_without_its_call()
    before = chat.conversation()
    with pytest.raises(TemplateRefusedError, match="cannot be cut at keep_text") as refused:
        chat.truncate_last_reply("Mo")
    assert refused.value.field == "keep_text"
    assert chat.conversation() == before


def test_a_reply_cut_to_nothing_after_a_call_that_went_leaves_an_older_turns_call_as_it_was():
    """Only a call of the newest reply's own turn can hold what stays of its tokens; an older exchange's call is not one."""
    saved = _closed_without_its_call().conversation()
    older = [{"role": "user", "content": "Wetter?"}, *tool_turn(
        ToolCall(WEATHER.name, {"city": "Bonn"}), Result(WEATHER.name, "sonnig"), "Sonnig.")]
    chat = ChatWrapper(Config(context_size=4096, evict_above_fraction=1.0),
                       context=FakeContext(TOOLS))
    chat.begin(**{**saved, "messages": opened("Be brief.", older) + saved["messages"][1:]})
    assert any(m.get("tool_calls") for m in chat.conversation()["messages"])
    assert internals.generated_turns_held(chat)
    chat.truncate_last_reply("")
    assert chat.conversation()["messages"][-1] == {"role": "assistant", "content": ""}
    assert not internals.generated_turns_held(chat)
    assert chat.reply("weiter").finish().text


def _turn_after_calls(calls: int, answer: str, **fake_kw):
    """A wrapper whose newest turn ran ``calls`` calls and then answered ``answer``, the shown text and the turn's tokens as generated."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(context_size=4096, **fake_kw)
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:clock{}"] * calls
    fake.replies = [answer]
    shown = "".join(step for step in wrapper.reply("wie spät?", tool_call_policy=registry)
                    if isinstance(step, str))
    generated, = internals.generated_turns_held(wrapper)
    return wrapper, fake, shown, generated[0]


def _spelled(fake, tokens) -> bytes:
    return b"".join(fake.token_bytes(token) for token in tokens)


@pytest.mark.parametrize("fake_kw", [{"tokens_spell_text": False}, {"merges": True}],
                         ids=["generated-ids", "merging"])
def test_a_reply_cut_in_its_answer_keeps_every_whole_generated_token_before_the_cut_and_tokenizes_only_the_rest(fake_kw):
    answer = "Es ist zwölf Uhr."
    for keep_chars in range(1, len(answer)):
        wrapper, fake, shown, before = _turn_after_calls(1, answer, **fake_kw)
        newest = wrapper.conversation()["messages"][-1]["content"]
        assert newest == answer
        wrapper.truncate_last_reply(shown[:len(shown) - len(answer) + keep_chars])
        (after, _text, _on_call), = internals.generated_turns_held(wrapper)
        whole = _spelled(fake, before)
        target = whole[:len(whole) - len(answer.encode()) + len(answer[:keep_chars].encode())]
        assert _spelled(fake, after) == target, keep_chars
        kept = max(n for n in range(len(before) + 1) if target.startswith(_spelled(fake, before[:n])))
        assert after[:kept] == before[:kept], keep_chars
        rest = target[len(_spelled(fake, before[:kept])):].decode()
        assert after[kept:] == fake.tokenize(rest, parse_special=False), keep_chars


def _answered_with_a_tag_inside(answer: str, **fake_kw):
    """A wrapper whose newest turn called a tool and then answered ``answer``, its tags generated as the model's own tags, and the text shown."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    fake = FakeContext(REASONING, writes_tags=True, **fake_kw)
    wrapper = ChatWrapper(Config(context_size=4096), context=fake)
    wrapper.begin(opened("persona"), tools=registry.specs())
    fake.call_script, fake.replies = ["fn:clock{}"], [answer]
    shown = "".join(step for step in wrapper.reply("wie spät?", tool_call_policy=registry)
                    if isinstance(step, str))
    return wrapper, shown


@pytest.mark.parametrize("keep_chars", range(0, len("Es istzwölf Uhr.")))
def test_a_reply_with_a_tag_the_model_wrote_inside_its_text_is_cut_anywhere_in_that_text(keep_chars):
    """A tag inside the answer is no text: the cut counts the text alone, and keeps the tag only where it stands before the cut."""
    wrapper, shown = _answered_with_a_tag_inside("Es ist</thought>zwölf Uhr.")
    assert shown == "Es istzwölf Uhr."
    wrapper.truncate_last_reply(shown[:keep_chars])
    assert wrapper.conversation()["messages"][-1]["content"] == shown[:keep_chars]
    assert wrapper.reply("weiter").finish().text


@pytest.mark.parametrize("answer, keep", [("Ja.<thought>Ja.</thought>", "Ja"), ("ha<thought>ha</thought>", "h"),
                                          ("a<thought>a</thought>a", "a")])
def test_a_cut_whose_text_would_begin_inside_a_thought_after_the_answer_is_refused(answer, keep):
    """A thought after the answer that ends as the answer does is hidden text, which the cut may not take for the answer."""
    wrapper, _shown = _answered_with_a_tag_inside(answer)
    before = wrapper.conversation()
    with pytest.raises(TemplateRefusedError) as refused:
        wrapper.truncate_last_reply(keep)
    assert refused.value.field == "keep_text"
    assert wrapper.conversation() == before


_REASONING_TOOL_STEMS = [stem for stem in STEMS
                         if fixture(stem)["calls_tools"] and fixture(stem)["marks_thought"]]


def _answered_on(stem: str, answer: str):
    """A wrapper on the test template ``stem`` whose newest turn called a tool and then answered ``answer``, its tags generated as the model's own, and the template's thought marks."""
    known = fixture(stem)
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    fake = FakeContext(known, writes_tags=True)
    wrapper = ChatWrapper(Config(context_size=4096), context=fake)
    wrapper.begin(opened("persona") if wrapper.capabilities(tools=True).can_take_system_prompt else [],
                  tools=registry.specs())
    call = shipped_addon_call(known["chat_template"])
    fake.call_script = [("clock" + call["name_close"] + "{}") if call
                        else written_call(None, fake, "clock", {})[1]]
    fake.replies = [answer]
    wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    return wrapper


@pytest.mark.parametrize("stem", _REASONING_TOOL_STEMS)
def test_on_every_template_a_cut_whose_text_would_begin_inside_a_thought_after_the_answer_is_refused(stem):
    """The thought marks are found by their text, whatever tokens spell them."""
    marks = ResolvedTemplate(FakeContext(fixture(stem))).properties.reasoning
    if marks is None:
        pytest.skip("the library reads no thought marks off this template")
    wrapper = _answered_on(stem, f"Ja.{marks.thought_open.strip()}Ja.{marks.thought_close.strip()}")
    before = wrapper.conversation()
    content = before["messages"][-1]["content"] or ""
    if not content.startswith("Ja"):
        pytest.skip("the template shows this answer otherwise")
    with pytest.raises(TemplateRefusedError):
        wrapper.truncate_last_reply("Ja")
    assert wrapper.conversation() == before


@pytest.mark.parametrize("merges", [False, True], ids=["one-per-character", "merging"])
def test_a_reply_after_a_call_is_cut_to_nothing_where_one_token_holds_a_space_and_its_first_letter(merges):
    """On a merging tokenizer the token after the thought's closing holds the space before the answer and its first letter, which the shown text does not start with."""
    wrapper, shown = _answered_with_a_tag_inside("</thought> Es ist zwölf.", merges=merges)
    wrapper.truncate_last_reply("")
    assert wrapper.conversation()["messages"][-1]["content"] == ""
    assert wrapper.reply("weiter").finish().text


@pytest.mark.parametrize("calls", [1, 2])
def test_a_reply_cut_to_nothing_keeps_its_turn_as_generated_up_to_the_closing_of_its_newest_call(calls):
    """The answer spells a call's closing as text, which the cut does not take for the call's."""
    answer = "Es ist </call> zwölf."
    wrapper, fake, shown, before = _turn_after_calls(calls, answer, tokens_spell_text=False)
    try:
        wrapper.truncate_last_reply(shown[:len(shown) - len(answer)])
    except TemplateRefusedError:
        pytest.fail("the invented template writes an empty reply after a call")
    (after, _text, on_call), = internals.generated_turns_held(wrapper)
    whole = _spelled(fake, before)
    closing = b"</call>"
    end = whole.rfind(closing, 0, len(whole) - len(answer.encode())) + len(closing)
    assert on_call and after == before[:len(after)] and _spelled(fake, after) == whole[:end]


def _states_saved(stem: str, n_exchanges: int, ends_on_question: bool) -> list[int]:
    """How many states a cache that saves states has saved after a begin from ``n_exchanges`` exchanges, and after the reply that follows."""
    fake = FakeContext(fixture(stem), ANSWERS, can_shift=False)
    chat = ChatWrapper(Config(context_size=16384, min_reply_tokens=8), context=fake)
    system = "Be brief." if chat.capabilities(tools=False).can_take_system_prompt else ""
    history = [message for n in range(n_exchanges) for message in (
        {"role": "user", "content": f"Old question {n}?"},
        {"role": "assistant", "content": f"Old answer {n}."})]
    if ends_on_question:
        history.append({"role": "user", "content": "A note."})
    chat.begin(opened(system, history))
    saved = [fake.saved_states]
    chat.reply(None if ends_on_question else "Question?").finish()
    return saved + [fake.saved_states]


@pytest.mark.parametrize("ends_on_question", [False, True], ids=["answered", "asked"])
@pytest.mark.parametrize("stem", STEMS)
def test_a_cache_that_saves_states_saves_no_more_of_them_as_the_history_grows(stem, ends_on_question):
    """The states kept where the newest exchanges begin are two, not one per exchange: a begin and its reply save as many for a long history as for a short one."""
    short = _states_saved(stem, 4, ends_on_question)
    long = _states_saved(stem, 24, ends_on_question)
    assert all(n_long <= n_short for n_short, n_long in zip(short, long)), (short, long)


class _SplitsACharacter(FakeContext):
    """A context whose model generates each ``ä`` in two tokens, the second holding its last byte and the character after it, as a byte level tokenizer may."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, tokens_spell_text=False, **kwargs)
        self._split_bytes: dict[int, bytes] = {}

    def _generated_run(self, text: str) -> list[int]:
        tokens = super()._generated_run(text)
        for at in range(len(text) - 1):
            if text[at] == "ä":
                first, second = tokens[at], tokens[at + 1]
                self._split_bytes[first] = "ä".encode()[:1]
                self._split_bytes[second] = "ä".encode()[1:] + text[at + 1].encode()
                self._generated_chars[first] = ""
                self._generated_chars[second] = "ä" + text[at + 1]
        return tokens

    def token_bytes(self, token: int) -> bytes:
        return self._split_bytes.get(token, super().token_bytes(token))


@pytest.mark.parametrize("split", [False, True], ids=["one token per character", "a character split"])
def test_a_reply_kept_as_generated_keeps_its_generated_tokens_up_to_each_cut(split):
    answer = "Es ist wärmer."
    for cut in range(1, len(answer)):
        fake = _SplitsACharacter(TOOLS) if split else FakeContext(TOOLS, tokens_spell_text=False)
        wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9), context=fake)
        ask_through_a_tool_call(wrapper, fake, answer)
        (generated, text, _), = internals.generated_turns_held(wrapper)
        assert text == answer
        wrapper.truncate_last_reply(answer[:cut])
        assert wrapper.conversation()["messages"][-1]["content"] == answer[:cut]
        (kept, _, _), = internals.generated_turns_held(wrapper)
        spelled = lambda tokens, fake=fake: b"".join(fake.token_bytes(t) for t in tokens)  # noqa: E731
        assert spelled(generated).endswith(answer.encode())
        assert spelled(kept).decode().endswith(answer[:cut]), cut
        # All but the token the cut falls in, and the one before where it splits a character.
        same = len(generated) - (len(answer) - cut) - 1
        assert kept[:same] == generated[:same], cut
        wrapper.reply("und?").finish()


@pytest.mark.parametrize("calls", [1, 2])
def test_a_reply_cut_to_nothing_keeps_its_turn_up_to_its_call_where_its_answer_spells_the_result_after_the_call(calls):
    """The answer spells the call's closing and the result's text after it, which the cut does not take for the call's own."""
    answer = "Es ist </call><result>clock(12:00) zwölf."
    wrapper, fake, shown, before = _turn_after_calls(calls, answer, tokens_spell_text=False)
    wrapper.truncate_last_reply(shown[:len(shown) - len(answer)])
    (after, _text, on_call), = internals.generated_turns_held(wrapper)
    whole = _spelled(fake, before)
    closing = b"</call>"
    end = whole.rfind(closing, 0, len(whole) - len(answer.encode())) + len(closing)
    assert on_call and after == before[:len(after)] and _spelled(fake, after) == whole[:end]


def _decoded_by_a_begin_changing_the_newest_exchange(n_exchanges: int, newest: str) -> int:
    """What a begin decodes on a cache that cannot shift, after a begin of the same history whose newest exchange asks otherwise, ending on ``newest``."""
    def history(question: str) -> list[dict]:
        older = [m for n in range(n_exchanges) for m in (
            {"role": "user", "content": f"Frage {n}?"}, {"role": "assistant", "content": f"Antwort {n}."})]
        ending = tool_turn(ToolCall(WEATHER.name, {"city": "Bonn"})) if newest == "pending call" else []
        return opened("s", older + [{"role": "user", "content": question}] + ending)
    fake = FakeContext(TOOLS, can_shift=False)
    chat = ChatWrapper(Config(context_size=16384), context=fake)
    chat.begin(history("Wetter?"), tools=[WEATHER])
    n_before = len(fake.prefill_calls)
    chat.begin(history("Wetter heute?"), tools=[WEATHER])
    return sum(n for _, n, _ in fake.prefill_calls[n_before:])


@pytest.mark.parametrize("newest", ["question", "pending call"])
def test_a_begin_that_changes_only_the_newest_exchange_decodes_no_more_for_a_longer_history(newest):
    """The first begin keeps a state where its newest exchange begins, which the second goes back to."""
    short, long = (_decoded_by_a_begin_changing_the_newest_exchange(n, newest) for n in (2, 20))
    assert long <= short, (short, long)
