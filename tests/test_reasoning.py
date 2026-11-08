"""Reasoning as the template turns it on: the switch, the thought kept out, stop strings."""


import pytest

from llama_chat.config import Config
from llama_chat.template import Reasoning
from llama_chat.tools import ToolSpec
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import EMPTY_THOUGHT, FakeContext, THINK_ON

from tests.wrappers import (THOUGHTFUL, assert_consistent, held_text, reasoning_wrapper,
                            rendered_close, rendered_prompt, rendered_reply, system_turn, rows)
class _KeepsThought(FakeContext):
    """A template that writes a past reply's thought into the conversation."""

    def render_with_model_template(self, messages, tools=None, **options):
        kept = [dict(m, content="<thought>" + m["reasoning_content"] + "</thought>"
                     + m["content"]) if m.get("reasoning_content") and
                not m.get("tool_calls") else m for m in messages]
        return super().render_with_model_template(kept, tools, **options)


# ----- reasoning, as the template turns it on ------------------------------
def test_the_switch_and_the_thought_tags_are_recovered_from_the_template():
    w, _fake = reasoning_wrapper()
    assert w.can_reason
    assert w._template.reasoning == Reasoning(
        thought_open="<thought>", thought_close="</thought>",
        generation_prompt="<[turn>model\n")
    assert w._template.fragments.generation_prompt == "<[turn>model\n" + EMPTY_THOUGHT


def test_a_model_without_a_switch_refuses_to_reason_and_keeps_its_conversation():
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext())
    w.begin("s", [("user", "hi")])
    assert not w.can_reason
    with pytest.raises(ValueError, match="reasoning"):
        w.begin("t", reasoning=True)
    assert [m["text"] for m in rows(w)] == ["s", "hi"]


def test_a_conversation_that_reasons_opens_its_system_turn_with_the_switch():
    for reasoning in (True, False):
        w, fake = reasoning_wrapper(reasoning)
        assert system_turn(w).startswith("<[turn>system\n" + THINK_ON) == reasoning
        assert system_turn(w).endswith("s<turn]>")


def test_with_tools_declared_the_switch_is_rendered_with_them():
    w, fake = reasoning_wrapper()
    specs = [ToolSpec("uhr", "", {})]
    w.begin("s", tools=specs, reasoning=True)
    assert THINK_ON in system_turn(w) and "<decl>uhr</decl>" in system_turn(w)


def test_a_reply_that_does_not_reason_opens_with_what_the_template_adds():
    w, fake = reasoning_wrapper(False, gen_text="Antwort.", gen_tokens=8)
    held = []
    generate = fake.generate
    fake.generate = lambda *a, **k: (held.append(held_text(w)), (yield from generate(*a, **k)))[1]
    w.reply("hi").finish()
    assert rendered_prompt(fake, reasoning=False).endswith(EMPTY_THOUGHT)
    # Generated after the empty thought, kept without it, as a past reply is.
    assert held[0].endswith(EMPTY_THOUGHT)
    assert rendered_reply(fake, "Antwort.", w._conversation._state.reasoning) in held_text(w)
    assert EMPTY_THOUGHT not in held_text(w)
    assert_consistent(w, fake)


def test_the_thought_is_kept_out_of_the_stream_and_out_of_the_conversation():
    w, fake = reasoning_wrapper()
    stream = w.reply("hi")
    assert "".join(stream) == "Antwort."
    assert stream.reply.text == "Antwort."
    assert stream.reply.n_reasoning_tokens == len("<thought>Denken</thought>")
    assert rows(w)[-1]["text"] == "Antwort."
    assert rendered_reply(fake, "Antwort.", w._conversation._state.reasoning) in held_text(w)
    assert "Denken" not in held_text(w)
    assert_consistent(w, fake)
    w.reply("weiter").finish()
    assert_consistent(w, fake)


def test_a_thought_the_model_has_without_being_asked_is_kept_out_too():
    w, fake = reasoning_wrapper(False)
    assert w.reply("hi").finish().text == "Antwort."
    assert_consistent(w, fake)


def test_the_eviction_after_a_reply_is_sized_by_the_past_reply():
    config = Config(context_size=200, evict_above_fraction=0.8)
    thought = "<thought>" + "x" * 60 + "</thought>ok"
    w, fake = reasoning_wrapper(config=config, gen_text=thought, gen_tokens=len(thought))
    w.inject("hist", "user")
    w.inject("hist", "assistant")
    w.reply("hi").finish()
    assert_consistent(w, fake)
    assert [m["text"] for m in rows(w)].count("hist") == 2


def test_a_reply_that_called_a_tool_is_kept_as_generated():
    w, fake = reasoning_wrapper()
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})], reasoning=True)
    for event in w.reply("wie spät?", calls="yield"):
        call = event
    w.inject_tool_result(call, {"zeit": "12:00"})
    generated = list(fake.cache)
    for _event in w.reply(calls="yield"):
        pass
    assert fake.cache[:len(generated)] == generated          # the turn as it was generated
    assert rows(w)[-1]["text"] == "Antwort."
    assert_consistent(w, fake)


def test_a_reply_kept_without_its_thought_is_cut_as_the_past_reply_it_is():
    w, fake = reasoning_wrapper()
    w.reply("hi").finish()
    w.truncate_last_reply("Ant")
    assert rendered_reply(fake, "Ant", w._conversation._state.reasoning) in held_text(w)
    assert "Antwort" not in held_text(w)
    assert_consistent(w, fake)


def test_a_reply_that_is_all_thought_is_removed():
    w, fake = reasoning_wrapper()
    reply = w.reply("hi", max_reply_tokens=12).finish()   # cut inside the thought
    assert reply.text == "" and reply.stop_reason == "token_limit"
    assert [m["role"] for m in rows(w)] == ["system", "user"]
    assert_consistent(w, fake)
    w.reply("weiter").finish()
    assert_consistent(w, fake)


def test_a_reply_cut_inside_a_thought_tag_keeps_none_of_it():
    w, fake = reasoning_wrapper(False, gen_text="<thou", gen_tokens=5)
    stream = w.reply("hi", max_reply_tokens=3)
    assert "".join(stream) == ""
    assert_consistent(w, fake)


def test_the_eviction_comes_before_the_past_reply_so_its_decode_carries_the_shift():
    config = Config(context_size=300, evict_above_fraction=0.5)
    thought = "<thought>" + "x" * 60 + "</thought>ok"
    w, fake = reasoning_wrapper(config=config, gen_text=thought, gen_tokens=len(thought))
    for _ in range(3):
        w.inject("hist", "user")
        w.inject("hist", "assistant")
    evictions, at_eviction = len(fake.evictions), []
    evict = fake.apply_eviction
    fake.apply_eviction = lambda ev: (at_eviction.append(len(fake.prefill_calls)), evict(ev))
    w.reply("hi").finish()
    assert len(fake.evictions) > evictions
    # The last cache edit is a decode after the eviction, which carries its shift.
    assert at_eviction[-1] < len(fake.prefill_calls)
    assert_consistent(w, fake)


@pytest.mark.parametrize("reasoning", [False, True])
def test_a_resumed_reply_goes_on_where_the_result_wrapping_ends_it(reasoning):
    """Nothing is added after the result: the template's wrapping ends the turn as it ends it."""
    w, fake = reasoning_wrapper(reasoning, gen_text="ok", gen_tokens=2)
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})], reasoning=reasoning)
    for event in w.reply("wie spät?", calls="yield"):
        call = event
    w.inject_tool_result(call, "12:00")
    before = len(fake.cache)
    list(w.reply(calls="yield"))
    assert fake.cache[before:before + 2] == fake.tokenize("ok")
    assert_consistent(w, fake)


# ----- stop strings and a reasoning reply ----------------------------------
def test_a_stop_string_inside_the_thought_does_not_end_the_reply():
    thoughtful = "<thought>a\n\nb</thought>Antwort.\n\nRest"
    w, fake = reasoning_wrapper(gen_text=thoughtful, gen_tokens=len(thoughtful))
    reply = w.reply("hi", stop=["\n\n"]).finish()
    assert reply.text == "Antwort." and reply.stop_reason == "stop_string"
    assert rows(w)[-1]["text"] == "Antwort."
    assert_consistent(w, fake)


def test_a_reply_whose_template_keeps_its_thought_keeps_the_tokens_it_had():
    fake = _KeepsThought(can_reason=True, calls_tools=True,
                         gen_text=THOUGHTFUL, gen_tokens=len(THOUGHTFUL))
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    assert w._template.reasoning.past_keeps_thought
    w.begin("s", reasoning=True)
    w.reply("hi").finish()
    assert rows(w)[-1]["text"] == "Antwort."
    assert rendered_prompt(fake, reasoning=True) + THOUGHTFUL in held_text(w)
    assert_consistent(w, fake)


def test_a_reply_decoded_again_counts_what_was_decoded():
    w, fake = reasoning_wrapper()
    before = fake.total_prefilled()
    reply = w.reply("hi").finish()
    assert reply.n_prefilled_tokens == fake.total_prefilled() - before


# ----- a reply kept with its thought ---------------------------------------
def test_a_reply_kept_with_its_thought_holds_the_tokens_its_text_tokenizes_into():
    w, fake = reasoning_wrapper(keeps_thought=True, tokens_spell_text=False)
    assert w._template.reasoning.past_keeps_thought
    reply = w.reply("hi").finish()
    assert reply.text == "Antwort."
    assert (rendered_prompt(fake, reasoning=True) + THOUGHTFUL
            + rendered_close(fake, reasoning=True)) in held_text(w)
    assert all(t < 1000 for t in fake.cache)          # none of the generated ids is kept
    assert_consistent(w, fake)


# ----- a header naming another reader --------------------------------------
def test_a_header_naming_another_reader_is_never_shown():
    from llama_chat.reply_text import AnswerOpening, reply_header
    from llama_chat.template import Reasoning
    tags = Reasoning(thought_open=" to=self<|message|>", thought_close="<|eom|>",
                     generation_prompt="")
    header = reply_header(" to=user<|message|>", tags)
    assert header == ("to=", "<|message|>")
    opening = AnswerOpening(" to=user<|message|>", header)
    assert opening.feed(" to=get_wea") + opening.feed("ther") == ""
    assert opening.flush(False) == ""                     # cut at a call: never shown
    opening = AnswerOpening(" to=user<|message|>", header)
    assert opening.feed(" to=user<|message|>Hallo") == "Hallo"
    opening = AnswerOpening(" to=user<|message|>", header)
    assert opening.feed(" to=notes<|message|>Text") == "Text"


def test_no_header_is_read_where_either_opening_leaves_no_reader_between_its_shared_ends():
    from llama_chat.reply_text import reply_header
    from llama_chat.template import Reasoning
    def tags(thought_open):
        return Reasoning(thought_open=thought_open, thought_close="<|eom|>", generation_prompt="")
    assert reply_header("<|a|>user<|a|>", tags("<|a|>")) is None
    assert reply_header("<|a|>", tags("<|a|>self<|a|>")) is None
    assert reply_header("<|a|>user<|b|>", tags("<|a|>self<|b|>")) == ("<|a|>", "<|b|>")
