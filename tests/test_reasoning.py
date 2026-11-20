"""Reasoning as the template turns it on: the switch, the thought kept out, stop strings."""

import re
import json
import sys

import pytest

from tests.chat_messages import opened
from tests.chat_templates import STEMS

from llama_chat.addon_files import template_fingerprint
from llama_chat.config import Config
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.template import Reasoning
from llama_chat.tools import ToolSpec
from llama_chat.wrapper import ChatWrapper
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import (EMPTY_THOUGHT, PLAIN, REASONING, REASONING_KEEPING_THOUGHTS,
                                      THINK_ON)

from tests.wrappers import (THOUGHTFUL, UNGUARDED, assert_consistent, held_text, reasoning_wrapper,
                            rendered_close, rendered_prompt, rendered_reply, system_turn, rows)
from llama_chat.reply_text import reply_header
# ----- reasoning, as the template turns it on ------------------------------
def test_the_switch_and_the_thought_tags_are_recovered_from_the_template():
    w, _fake = reasoning_wrapper()
    assert w.capabilities(tools=False).can_reason
    template = ResolvedTemplate(FakeContext(REASONING))
    assert template.properties.reasoning == Reasoning(
        thought_open="<thought>", thought_close="</thought>",
        generation_prompt="<[turn>model\n", can_turn_off=True)
    assert template.properties.fragments.generation_prompt == "<[turn>model\n" + EMPTY_THOUGHT


def test_a_model_without_a_switch_refuses_to_reason_and_keeps_its_conversation():
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext(PLAIN))
    w.begin(opened("s", [{"role": "user", "content": "hi"}]))
    assert not w.capabilities(tools=False).can_reason
    with pytest.raises(ValueError, match="reasoning"):
        w.begin(opened("t"), reasoning=True)
    assert [m["text"] for m in rows(w)] == ["s", "hi"]


def test_reasoning_can_be_turned_off_with_a_switch_and_where_no_thought_is_marked():
    for template in (REASONING, PLAIN):
        w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                        context=FakeContext(template))
        assert w.capabilities(tools=False).can_turn_off_reasoning


def test_a_conversation_that_reasons_opens_its_system_turn_with_the_switch():
    for reasoning in (True, False):
        w, fake = reasoning_wrapper(reasoning)
        assert system_turn(fake).startswith("<[turn>system\n" + THINK_ON) == reasoning
        assert system_turn(fake).endswith("s<turn]>")


def test_with_tools_declared_the_switch_is_rendered_with_them():
    w, fake = reasoning_wrapper()
    specs = [ToolSpec("uhr", "", {})]
    w.begin(opened("s"), tools=specs, reasoning=True)
    assert THINK_ON in system_turn(fake) and "<decl>uhr</decl>" in system_turn(fake)


def test_a_reply_that_does_not_reason_opens_with_what_the_template_adds():
    w, fake = reasoning_wrapper(False, gen_text="Antwort.", gen_tokens=8)
    held = []
    sampled = fake.sampled_tokens
    fake.sampled_tokens = lambda *a, **k: (held.append(held_text(fake)), (yield from sampled(*a, **k)))[1]
    w.reply("hi").finish()
    assert rendered_prompt(fake, reasoning=False).endswith(EMPTY_THOUGHT)
    # Generated after the empty thought, kept without it, as a past reply is.
    assert held[0].endswith(EMPTY_THOUGHT)
    assert rendered_reply(fake, "Antwort.", reasoning=False) in held_text(fake)
    assert EMPTY_THOUGHT not in held_text(fake)


def test_the_thought_is_kept_out_of_the_stream_and_out_of_the_conversation():
    w, fake = reasoning_wrapper()
    stream = w.reply("hi")
    assert "".join(stream) == "Antwort."
    assert stream.reply.text == "Antwort."
    assert stream.reply.n_reasoning_tokens == len("<thought>Denken</thought>")
    assert rows(w)[-1]["text"] == "Antwort."
    assert rendered_reply(fake, "Antwort.", reasoning=True) in held_text(fake)
    assert "Denken" not in held_text(fake)
    assert_consistent(w, fake)
    w.reply("weiter").finish()


def test_the_reply_hands_back_the_thought_it_kept_out_of_the_answer():
    w, _fake = reasoning_wrapper()
    reply = w.reply("hi").finish()
    assert (reply.text, reply.thought) == ("Antwort.", "Denken")
    plain, _ = reasoning_wrapper(False, template=PLAIN, gen_text="Antwort.")
    assert plain.reply("hi").finish().thought == ""


def test_a_reply_of_several_generations_hands_back_each_thought_in_order():
    from llama_chat.tool_registry import ToolRegistry
    from llama_chat.testing.fake_context import ScriptedCall
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Uhrzeit.", name="uhr", parameters={"type": "object"})
    fake = FakeContext(REASONING_KEEPING_THOUGHTS, replies=[
        ScriptedCall("fn:uhr{}", before="<thought>Erst fragen</thought>"),
        "<thought>Jetzt antworten</thought>Zwölf."])
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"), tools=registry.specs(), reasoning=True)
    reply = w.reply("wie spät?", tool_call_policy=registry).finish()
    assert reply.text == "Zwölf."
    assert reply.thought == "Erst fragen\n\nJetzt antworten"


def test_a_thought_the_model_has_without_being_asked_is_kept_out_too():
    w, _fake = reasoning_wrapper(False)
    assert w.reply("hi").finish().text == "Antwort."


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
    fake.call_script.append("fn:uhr{}")
    w.begin(opened("s"), tools=[ToolSpec("uhr", "", {})], reasoning=True)
    for event in w.reply("wie spät?", tool_call_policy="yield"):
        call = event
    w.inject_tool_result(call.id, {"zeit": "12:00"})
    generated = list(fake.cache)
    for _event in w.reply(tool_call_policy="yield"):
        pass
    assert fake.cache[:len(generated)] == generated          # the turn as it was generated
    assert rows(w)[-1]["text"] == "Antwort."


def test_a_reply_kept_without_its_thought_is_cut_as_the_past_reply_it_is():
    w, fake = reasoning_wrapper()
    w.reply("hi").finish()
    w.truncate_last_reply("Ant")
    assert rendered_reply(fake, "Ant", reasoning=True) in held_text(fake)
    assert "Antwort" not in held_text(fake)


def test_a_reply_that_is_all_thought_is_removed():
    w, fake = reasoning_wrapper(config=Config(context_size=500, evict_above_fraction=1.0,
                                              **UNGUARDED))
    reply = w.reply("hi", max_reply_tokens=12).finish()   # cut inside the thought
    assert reply.text == "" and reply.stop_reason == "token_limit"
    assert [m["role"] for m in rows(w)] == ["system", "user"]
    assert_consistent(w, fake)
    w.reply("weiter").finish()


def test_a_reply_cut_inside_a_thought_tag_keeps_none_of_it():
    w, _fake = reasoning_wrapper(False, gen_text="<thou", gen_tokens=5)
    stream = w.reply("hi", max_reply_tokens=3)
    assert "".join(stream) == ""


def test_the_eviction_comes_before_the_past_reply_so_its_decode_carries_the_shift():
    config = Config(context_size=200, evict_above_fraction=0.5)
    thought = "<thought>" + "x" * 60 + "</thought>ok"
    w, fake = reasoning_wrapper(config=config, gen_text=thought, gen_tokens=len(thought))
    for _ in range(3):
        w.inject("hist", "user")
        w.inject("hist", "assistant")
    evictions, at_eviction = fake.edit_count("shift"), []
    evict = fake.apply_eviction
    fake.apply_eviction = lambda start, end: (at_eviction.append(len(fake.prefill_calls)),
                                              evict(start, end))
    w.reply("hi").finish()
    assert fake.edit_count("shift") > evictions
    # The last cache edit is a decode after the eviction, which carries its shift.
    assert at_eviction[-1] < len(fake.prefill_calls)


@pytest.mark.parametrize("reasoning", [False, True])
def test_a_resumed_reply_goes_on_where_the_result_wrapping_ends_it(reasoning):
    """Nothing is added after the result: the template's wrapping ends the turn as it ends it."""
    w, fake = reasoning_wrapper(reasoning, gen_text="ok", gen_tokens=2)
    fake.call_script.append("fn:uhr{}")
    w.begin(opened("s"), tools=[ToolSpec("uhr", "", {})], reasoning=bool(reasoning))
    for event in w.reply("wie spät?", tool_call_policy="yield"):
        call = event
    w.inject_tool_result(call.id, "12:00")
    before = len(fake.cache)
    list(w.reply(tool_call_policy="yield"))
    assert fake.cache[before:before + 2] == fake.tokenize("ok")


# ----- a thought marked by tags of another shape ----------------------------
_TURNS = ("{%- for m in messages -%}<|{{ m.role }}_says|>"
          "{{ m.content.split('◁/ponder▷')[-1] if m.role == 'assistant' else m.content }}<|said|>"
          "{%- endfor -%}")


@pytest.mark.parametrize("prompt", [
    # Only the prompt without reasoning shows the thought, empty.
    "{%- if add_generation_prompt -%}<|assistant_says|>"
    "{%- if not enable_thinking -%}◁ponder▷◁/ponder▷{%- endif -%}{%- endif -%}",
    # Only a past reply shows it, by leaving it out.
    "{%- if add_generation_prompt -%}<|assistant_says|>{%- endif -%}",
], ids=["empty-thought", "removed-thought"])
def test_a_thought_whose_tags_are_not_angle_brackets_is_read_by_the_models_own_tags(prompt):
    template = ResolvedTemplate(FakeContext(
        {"chat_template": _TURNS + prompt, "eos_token": "<|said|>",
         "render_variables": {"off": {"enable_thinking": False}, "on": {"enable_thinking": True}}}))
    assert (template.properties.reasoning.thought_open, template.properties.reasoning.thought_close) == \
        ("◁ponder▷", "◁/ponder▷")


# ----- stop strings and a reasoning reply ----------------------------------
def test_a_stop_string_inside_the_thought_does_not_end_the_reply():
    thoughtful = "<thought>a\n\nb</thought>Antwort.\n\nRest"
    w, _fake = reasoning_wrapper(gen_text=thoughtful, gen_tokens=len(thoughtful))
    reply = w.reply("hi", stop=["\n\n"]).finish()
    assert reply.text == "Antwort." and reply.stop_reason == "stop_string"
    assert rows(w)[-1]["text"] == "Antwort."


# ----- a reply kept with its thought ---------------------------------------
def test_a_reply_kept_with_its_thought_holds_the_tokens_its_text_tokenizes_into():
    w, fake = reasoning_wrapper(template=REASONING_KEEPING_THOUGHTS, tokens_spell_text=False)
    assert ResolvedTemplate(FakeContext(REASONING_KEEPING_THOUGHTS)).properties.reasoning.past_keeps_thought
    reply = w.reply("hi").finish()
    assert reply.text == "Antwort."
    assert (rendered_prompt(fake, reasoning=True) + THOUGHTFUL
            + rendered_close(fake, reasoning=True)) in held_text(fake)
    assert max(fake.cache) <= sys.maxunicode          # none of the generated ids is kept


# ----- a header naming another reader --------------------------------------
def test_a_header_naming_another_reader_is_never_shown():
    from llama_chat.reply_text import ReplyText, reply_header
    from llama_chat.template import Reasoning
    tags = Reasoning(thought_open=" to=self<|body|>", thought_close="<|call_ends|>",
                     generation_prompt="")
    assert reply_header(" to=user<|body|>", tags) == ("to=", "<|body|>")
    header = reply_header(" to=user<|body|>", tags)
    reply = ReplyText(tags, " to=user<|body|>", header, "", [])
    assert reply.feed(" to=get_wea") + reply.feed("ther") == ""
    assert reply.finish("tool_call") == ""                # held at a call: never shown
    reply = ReplyText(tags, " to=user<|body|>", header, "", [])
    assert reply.feed(" to=user<|body|>Hallo") + reply.finish("end_of_turn") == "Hallo"
    reply = ReplyText(tags, " to=user<|body|>", header, "", [])
    assert reply.feed(" to=notes<|body|>Text") + reply.finish("end_of_turn") == "Text"


def test_no_header_is_read_where_either_opening_leaves_no_reader_between_its_shared_ends():
    from llama_chat.reply_text import reply_header
    from llama_chat.template import Reasoning
    def tags(thought_open):
        return Reasoning(thought_open=thought_open, thought_close="<|call_ends|>", generation_prompt="")
    assert reply_header("<|a|>user<|a|>", tags("<|a|>")) is None
    assert reply_header("<|a|>", tags("<|a|>self<|a|>")) is None
    assert reply_header("<|a|>user<|b|>", tags("<|a|>self<|b|>")) == ("<|a|>", "<|b|>")


def test_a_header_never_divides_one_of_the_models_tags():
    from llama_chat.reply_text import reply_header
    from llama_chat.template import Reasoning
    thought = Reasoning(thought_open="<|OPEN_THOUGHT|>", thought_close="<|SHUT_THOUGHT|>",
                        generation_prompt="")
    assert reply_header("<|OPEN_TEXT|>", thought) == ("<|OPEN_T", "T|>")
    assert reply_header("<|OPEN_TEXT|>", thought,
                        ("<|OPEN_TEXT|>", "<|OPEN_THOUGHT|>", "<|SHUT_THOUGHT|>")) is None


# ----- reasoning turned on as the add-on file says -------------------------
# A template whose default does not reason, yet writes an empty thought for the
# effort "none", so that the template alone reads the default as on.
_DEFAULT_WITHOUT_THOUGHT = {
    "chat_template": _TURNS.replace("◁/ponder▷", "</reason>")
                     + "{%- if add_generation_prompt -%}<|assistant_says|>"
                     "{%- if reasoning_effort == 'none' -%}<reason></reason>"
                     "{%- elif reasoning_effort == 'high' -%}<reason>{%- endif -%}{%- endif -%}",
    "eos_token": "<|said|>",
    "render_variables": {"off": {"reasoning_effort": "none"}, "on": {"reasoning_effort": "high"}}}
_QUESTION = [{"role": "user", "content": "hi"}]


def test_without_an_add_on_file_such_a_template_reasons_as_its_default_writes():
    template = ResolvedTemplate(FakeContext(_DEFAULT_WITHOUT_THOUGHT))
    assert template.render(_QUESTION, generation_prompt=True,
                           reasoning=True).endswith("<|assistant_says|>")


def test_the_add_on_files_reasoning_on_is_what_reasoning_on_renders():
    template = ResolvedTemplate(FakeContext(_DEFAULT_WITHOUT_THOUGHT),
                                reasoning_on={"reasoning_effort": "high"})
    assert template.render(_QUESTION, generation_prompt=True,
                           reasoning=True).endswith("<|assistant_says|><reason>")
    assert template.render(_QUESTION, generation_prompt=True,
                           reasoning=False).endswith("<|assistant_says|><reason></reason>")
    assert template.properties.can_reason


def test_a_reasoning_on_that_writes_the_prompt_as_off_does_is_refused_at_load():
    with pytest.raises(ValueError, match="reasoning_on"):
        ResolvedTemplate(FakeContext(_DEFAULT_WITHOUT_THOUGHT),
                         reasoning_on={"reasoning_effort": "none"})


def test_a_wrapper_takes_reasoning_on_from_the_add_on_file_next_to_the_model(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps(
        {"templates": [template_fingerprint(_DEFAULT_WITHOUT_THOUGHT["chat_template"])],
         "reasoning_on": {"reasoning_effort": "high"}}))
    (tmp_path / "m.gguf").touch()
    fake = FakeContext(_DEFAULT_WITHOUT_THOUGHT)
    w = ChatWrapper(Config(model_path=str(tmp_path / "m.gguf"), context_size=500,
                           evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}]), reasoning=True)
    assert held_text(fake).endswith("<|assistant_says|><reason>")


@pytest.mark.parametrize("addon, recommended, cause", [
    ({"sampler": {}}, {}, "may hold only"),
    ({"reasoning_on": {"reasoning_effort": "none"}}, {}, "reasoning_on"),
    ({}, {"top_p": 2.0}, "the model's own recommendation"),
], ids=["malformed-file", "reasoning-on-as-off", "refused-gguf-sampling"])
def test_the_add_on_file_and_the_models_own_sampling_are_refused_by_the_constructor(
        tmp_path, addon, recommended, cause):
    """The refusals the constructor's docstring names reach its caller as ValueError."""
    fingerprint = template_fingerprint(_DEFAULT_WITHOUT_THOUGHT["chat_template"])
    (tmp_path / "m.json").write_text(json.dumps({"templates": [fingerprint], **addon}))
    (tmp_path / "m.gguf").touch()
    fake = FakeContext(_DEFAULT_WITHOUT_THOUGHT)
    fake.recommended = recommended
    with pytest.raises(ValueError, match=cause):
        ChatWrapper(Config(model_path=str(tmp_path / "m.gguf"), context_size=500,
                           evict_above_fraction=1.0), context=fake)


def test_a_reasoning_on_written_as_off_is_refused_naming_the_setting_its_fault_and_the_remedy():
    with pytest.raises(ValueError) as refused:
        ResolvedTemplate(FakeContext(_DEFAULT_WITHOUT_THOUGHT),
                         reasoning_on={"reasoning_effort": "none"})
    message = str(refused.value)
    assert "the add-on file's reasoning_on writes the prompt as reasoning off does" in message
    assert "name the template variables that turn this model's reasoning on" in message


def test_a_template_that_refuses_a_message_it_does_not_know_takes_the_reasoning_on_its_add_on_file_names():
    checked = {**_DEFAULT_WITHOUT_THOUGHT, "chat_template":
               "{%- for m in messages if m.role not in ('system', 'user', 'assistant', 'tool')"
               " or m.content is not defined -%}"
               "{{ raise_exception('a message this template does not write') }}{%- endfor -%}"
               + _DEFAULT_WITHOUT_THOUGHT["chat_template"]}
    template = ResolvedTemplate(FakeContext(checked), reasoning_on={"reasoning_effort": "high"})
    assert template.properties.can_reason
    assert template.render([{"role": "user", "content": "hi"}], generation_prompt=True,
                           reasoning=True).endswith("<|assistant_says|><reason>")


@pytest.mark.parametrize("answer, thought, tags, header", [
    ("<|a|>user<|b|>", "<|a|>self<|b|>", ("<|a|>", "<|b|>"), ("<|a|>", "<|b|>")),
    ("<|a|>user<|a|>", "<|a|>self<|a|>", ("<|a|>",), ("<|a|>", "<|a|>")),
    ("<|ab|>user<|ab|><|z|>", "<|ac|>self<|z|>", ("<|ab|>", "<|ac|>", "<|z|>"), None),
    ("<|OPEN_TEXT|>", "<|OPEN_THOUGHT|>", ("<|OPEN_TEXT|>", "<|OPEN_THOUGHT|>"), None),
])
def test_a_headers_ends_move_only_off_a_tag_they_would_divide(answer, thought, tags, header):
    found = reply_header(answer, Reasoning(thought, "<|call_ends|>", ""), tags)
    assert found == header
    if found:
        bounds = (len(found[0]), len(answer) - len(found[1]))
        assert not any(at < bound < at + len(tag) for tag in tags for bound in bounds
                       for at in (m.start() for m in re.finditer(re.escape(tag), answer)))


@pytest.mark.timed
def test_streaming_a_long_thought_takes_time_in_proportion_to_it():
    import time
    from llama_chat.testing.invented_templates import _invented

    def timed(words: int) -> float:
        """The shorter of two replies thinking ``words`` words, in this thread's processor time."""
        times = []
        for _ in range(2):
            fake = FakeContext(_invented(can_reason=True),
                               replies=["<thought>" + "denk " * words + "</thought>Antwort."])
            w = ChatWrapper(Config(context_size=words * 6 + 2000, answer_room_tokens=None),
                            context=fake)
            w.begin(opened("s"), reasoning=True)
            started = time.thread_time()
            assert w.reply("q").finish().text == "Antwort."
            times.append(time.thread_time() - started)
        return min(times)

    timed(500)                                  # warmed up
    # Four times the thought takes about four times as long, not sixteen times.
    assert timed(8000) < 8 * timed(2000)


@pytest.mark.parametrize("where", ["config", "reply"])
def test_a_thought_cap_closes_the_thought_there_and_the_reply_still_answers(where):
    thought = "<thought>" + "x" * 60 + "</thought>Antwort."
    config = Config(context_size=500, evict_above_fraction=1.0,
                    max_thought_tokens=5 if where == "config" else None)
    w, fake = reasoning_wrapper(config=config, gen_text=thought, gen_tokens=len(thought))
    reply = w.reply("hi", **({"max_thought_tokens": 5} if where == "reply" else {})).finish()
    assert reply.thought == "x" * 5
    # The scripted model writes on as it was going to; all of it after the closing is answer.
    assert reply.text == "x" * 55 + "</thought>Antwort."


# ----- effort levels ---------------------------------------------------------------
def _levelled_wrapper(stem: str):
    from tests.chat_templates import fixture
    fake = FakeContext(fixture(stem), ["Antwort."])
    return ChatWrapper(Config(context_size=8192), context=fake), fake


def _levels(stem: str) -> tuple[str, ...]:
    return _levelled_wrapper(stem)[0].capabilities(tools=False).reasoning_levels


def _stems_with_levels():
    from tests.chat_templates import STEMS
    return [stem for stem in sorted(STEMS) if _levels(stem)]


_WITH_LEVELS = _stems_with_levels()


@pytest.mark.parametrize("stem", sorted(STEMS))
def test_the_levels_are_those_the_templates_file_states(stem):
    """Read from the template where it tells levels apart, else from the add-on file shipped for it."""
    from tests.chat_templates import expected, fixture
    assert _levels(stem) == tuple(expected(fixture(stem), "reasoning_levels"))


@pytest.mark.parametrize("stem", _WITH_LEVELS)
def test_a_conversation_at_each_level_is_written_with_it_and_begins_again_at_it(stem):
    held = {}
    for level in _levels(stem):
        w, fake = _levelled_wrapper(stem)
        w.begin(opened("s"), reasoning=True, reasoning_level=level)
        w.reply("Frage?").finish()
        saved = w.conversation()
        assert (saved["reasoning"], saved["reasoning_level"]) == (True, level)
        w.begin(**saved)
        assert w.conversation() == saved
        held[level] = fake.text_of(fake.cache)
    # A level the template writes otherwise than another is held otherwise too.
    assert len(set(held.values())) > 1


def test_a_level_the_template_does_not_know_or_one_without_reasoning_is_refused_by_name_and_changes_nothing():
    from llama_chat.errors import InvalidArgumentError, UnsupportedError
    stem = _WITH_LEVELS[0]
    w, _fake = _levelled_wrapper(stem)
    w.begin(opened("s"), reasoning=True)
    before = w.conversation()
    known = ", ".join(map(repr, _levels(stem)))
    with pytest.raises(UnsupportedError, match=re.escape(known)) as refused:
        w.begin(opened("s"), reasoning=True, reasoning_level="QQnolevel")
    assert refused.value.field == "reasoning_level"
    with pytest.raises(InvalidArgumentError) as refused:
        w.begin(opened("s"), reasoning_level=_levels(stem)[0])
    assert refused.value.field == "reasoning_level"
    assert w.conversation() == before


def test_a_template_without_levels_reasons_as_it_does_by_default_or_not_at_all():
    from llama_chat.errors import UnsupportedError
    from tests.chat_templates import STEMS, expected, fixture
    stem = next(stem for stem in sorted(STEMS)
                if fixture(stem)["can_reason"] and not expected(fixture(stem), "reasoning_levels"))
    w, _fake = _levelled_wrapper(stem)
    assert w.capabilities(tools=False).reasoning_levels == ()
    w.begin(opened("s"), reasoning=True)
    assert (w.conversation()["reasoning"], w.conversation()["reasoning_level"]) == (True, None)
    w.begin(opened("s"), reasoning=False)
    assert w.conversation()["reasoning"] is False
    with pytest.raises(UnsupportedError):
        w.begin(opened("s"), reasoning=True, reasoning_level="high")
