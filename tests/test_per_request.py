"""Settings one request brings with it, and cutting the newest reply short."""

from types import SimpleNamespace

import pytest

from llama_chat.config import Config, Sampling
from llama_chat.context import GenerationAccumulator, KVContext
from llama_chat.tools import ToolSpec
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import TOOL_CLOSE, TOOL_OPEN, FakeContext


def _wrapper(**fake_kw):
    fake = FakeContext(**fake_kw)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    return w, fake


def _assert_consistent(w, fake):
    """The physical cache equals the concatenation of the message table."""
    tokens = [t for m in w._table.messages for t in m.token_ids]
    assert fake.cache == tokens
    assert w.snapshot()[-1]["pos_end"] == len(fake.cache)


# ----- sampling per request ----------------------------------------------
def test_an_empty_sampling_overrides_nothing():
    assert Sampling().overrides == {}
    config = Config(temperature=0.7)
    assert Sampling().applied_to(config) == config


def test_a_sampling_replaces_exactly_the_fields_it_sets():
    applied = Sampling(temperature=0.2, seed=7).applied_to(
        Config(temperature=0.7, top_p=0.9))
    assert (applied.temperature, applied.seed, applied.top_p) == (0.2, 7, 0.9)


@pytest.mark.parametrize("bad", [{"top_p": 0.0}, {"top_p": 1.5}])
def test_an_impossible_top_p_is_refused_where_it_is_built(bad):
    with pytest.raises(ValueError):
        Sampling(**bad)


def test_a_seed_the_config_refuses_is_refused_for_one_request_too():
    with pytest.raises(ValueError, match="seed"):
        Sampling(seed=-1).applied_to(Config())


@pytest.mark.parametrize("entry", ["reply_without_tools", "reply_with_tools"])
def test_every_entry_point_hands_its_sampling_to_the_generation(entry):
    w, fake = _wrapper()
    sampling = Sampling(temperature=0.1)
    getattr(w, entry)("hi", sampling=sampling)
    assert fake.samplings == [sampling]


def test_a_stream_and_a_resume_hand_it_over_as_well():
    w, fake = _wrapper()
    sampling = Sampling(top_p=0.5)
    list(w.reply_without_tools_stream("hi", sampling=sampling))
    list(w.reply_until_tool("again", sampling=sampling))
    assert fake.samplings == [sampling, sampling]


def _stub_context(chains):
    """A real KVContext over a backend that records which chain it samples from."""
    ctx = object.__new__(KVContext)
    tokens = iter([0, 1, 2])
    ctx._backend = SimpleNamespace(
        sample=lambda sampler, llama_ctx: (chains["sampled"].append(sampler),
                                           next(tokens))[1],
        is_eog=lambda vocab, tok: False,
        decode=lambda *args: None,
        token_to_piece_bytes=lambda vocab, tok: b"x",
        free_sampler=lambda sampler: chains["freed"].append(sampler),
        sampler_reset=lambda sampler: chains.setdefault("reset", []).append(sampler),
    )
    ctx._sampler, ctx._vocab, ctx._llama_context = "loaded", None, None
    ctx._max_batch_tokens = 8
    ctx._config = Config()
    ctx._build_sampler = lambda config: ("own", config.temperature)
    return ctx


def test_a_generation_without_overrides_keeps_the_chain_built_at_load():
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    list(ctx.generate(0, 2, [], out=GenerationAccumulator(), sampling=Sampling()))
    assert set(chains["sampled"]) == {"loaded"} and chains["freed"] == []


def test_an_override_gets_its_own_chain_and_frees_it_even_when_abandoned():
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    stream = ctx.generate(0, 3, [], out=GenerationAccumulator(),
                          sampling=Sampling(temperature=0.3))
    next(stream)
    stream.close()
    assert chains["sampled"] == [("own", 0.3)]
    assert chains["freed"] == [("own", 0.3)]


def test_every_generation_starts_its_penalties_and_its_seed_afresh():
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    list(ctx.generate(0, 1, [], out=GenerationAccumulator()))
    list(ctx.generate(1, 1, [], out=GenerationAccumulator(), sampling=Sampling()))
    assert chains["reset"] == ["loaded", "loaded"]


# ----- cutting the newest reply ------------------------------------------
def test_a_cut_keeps_the_prefix_and_the_cache_agrees():
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply_without_tools("hi")
    w.truncate_last_reply("Eins.")
    assert w.snapshot()[-1]["text"] == "Eins."
    _assert_consistent(w, fake)


def test_after_a_cut_the_next_request_appends_at_the_cache_end():
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply_without_tools("hi")
    w.truncate_last_reply("Eins.")
    w.reply_without_tools("weiter")
    _assert_consistent(w, fake)


def test_a_reply_generated_in_other_pieces_than_its_text_tokenizes_into_is_kept_as_its_text():
    """Same text, other tokens: the template's past reply is the text, tokenized."""
    w, fake = _wrapper(gen_tokens=4, gen_text="ok!!", tokens_spell_text=False)
    reply = w.reply_without_tools("hi")
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "ok!!")
    assert reply.text == "ok!!"
    _assert_consistent(w, fake)


def test_a_reply_after_an_empty_assistant_tag_is_kept_as_its_text_too():
    """A template may open a reply with nothing at all, as Mistral's does."""
    from dataclasses import replace
    from tests.fake_context import DEFAULT_FRAGMENTS

    class _TagsOnly(FakeContext):
        def render_with_model_template(self, *args, **kwargs):
            return None             # no template: the tags are the ones supplied

    fake = _TagsOnly(gen_tokens=4, gen_text="ok!!", tokens_spell_text=False)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake,
                    fragments=replace(DEFAULT_FRAGMENTS, assistant_prefix=""))
    w.begin("s")
    w.reply_without_tools("hi")
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "ok!!")
    _assert_consistent(w, fake)


def test_the_tokens_that_stay_are_not_decoded_again():
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply_without_tools("hi")
    before = list(fake.prefill_calls)
    w.truncate_last_reply("Eins.")
    (start, n, _logits), = fake.prefill_calls[len(before):]
    closing = fake.tokenize(w._formatter.assistant_close())
    assert n == len(closing)
    assert start == w.snapshot()[-1]["pos_start"] + w.snapshot()[-1]["n_tokens"] - n


def test_keeping_the_whole_text_changes_nothing():
    w, fake = _wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply_without_tools("hi")
    cache = list(fake.cache)
    w.truncate_last_reply("Hallo")
    assert fake.cache == cache


def test_a_text_that_is_not_a_prefix_is_refused_and_changes_nothing():
    w, fake = _wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply_without_tools("hi")
    cache = list(fake.cache)
    with pytest.raises(ValueError, match="prefix"):
        w.truncate_last_reply("Tschüss")
    assert fake.cache == cache


def test_only_an_assistant_reply_can_be_cut():
    w, _fake = _wrapper()
    w.inject("nebenbei", "user")
    with pytest.raises(ValueError, match="assistant"):
        w.truncate_last_reply("")


def test_a_prefilled_reply_keeps_the_templates_opening_when_cut():
    w, fake = _wrapper()
    w.inject("Eins. Zwei.", "assistant")
    w.truncate_last_reply("Eins.")
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Eins.")
    _assert_consistent(w, fake)


def test_a_reply_that_called_a_tool_cannot_be_cut():
    w, fake = _wrapper(calls_tools=True)
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    for _event in w.reply_until_tool("wie spät?"):
        pass
    with pytest.raises(ValueError, match="tool call"):
        w.truncate_last_reply("")


def test_a_seed_out_of_range_is_refused_before_anything_is_prefilled():
    with pytest.raises(ValueError, match="seed"):
        Sampling(seed=2**40)


@pytest.mark.parametrize("saves_state", [True, False])
def test_a_cut_on_a_cache_that_cannot_shift_restores_or_rebuilds(saves_state):
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw", can_shift=False)
    if not saves_state:
        fake.save_state = lambda: None
        w.begin("s")
    w.reply_without_tools("hi")
    rebuilds = fake.rebuilds
    w.truncate_last_reply("Eins.")
    # From the state saved before the reply where there is one, else rebuilt.
    assert fake.rebuilds == rebuilds + (not saves_state)
    assert w.snapshot()[-1]["text"] == "Eins."
    _assert_consistent(w, fake)


# ----- reasoning, as the template turns it on ------------------------------
from llama_chat.template import Reasoning  # noqa: E402
from tests.fake_context import EMPTY_THOUGHT, THINK_ON  # noqa: E402

_THOUGHTFUL = "<thought>Denken</thought>Antwort."


def _reasoning(reasoning=True, config=None, **fake_kw):
    fake_kw = {"gen_tokens": len(_THOUGHTFUL), "gen_text": _THOUGHTFUL, **fake_kw}
    fake = FakeContext(can_reason=True, calls_tools=True, **fake_kw)
    w = ChatWrapper(config or Config(context_size=500, evict_above_fraction=1.0),
                    context=fake)
    w.begin("s", reasoning=reasoning)
    return w, fake


def test_the_switch_and_the_thought_tags_are_recovered_from_the_template():
    w, _fake = _reasoning()
    assert w.can_reason
    assert w._template.reasoning == Reasoning(
        thought_open="<thought>", thought_close="</thought>",
        system_prefix="<|turn>system\n" + THINK_ON,
        generation_prompt="<|turn>model\n")
    assert w._template.fragments.generation_prompt == "<|turn>model\n" + EMPTY_THOUGHT


def test_a_model_without_a_switch_refuses_to_reason_and_keeps_its_conversation():
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext())
    w.begin("s", [("user", "hi")])
    assert not w.can_reason
    with pytest.raises(ValueError, match="reasoning"):
        w.begin("t", reasoning=True)
    assert [m["text"] for m in w.snapshot()] == ["s", "hi"]


def test_a_conversation_that_reasons_opens_its_system_turn_with_the_switch():
    for reasoning in (True, False):
        w, fake = _reasoning(reasoning)
        prefix, content, suffix = w._formatter.parts("system", "s")
        assert prefix.endswith(THINK_ON) == reasoning
        assert w._table.messages[0].token_ids == fake.tokenize_fragment(
            prefix, content, suffix, add_special=True)


def test_with_tools_declared_the_switch_is_rendered_with_them():
    w, fake = _reasoning()
    specs = [ToolSpec("uhr", "", {})]
    w.begin("s", tools=specs, reasoning=True)
    head, tail = w._template.system_turn_declaring(specs, reasoning=True)
    assert THINK_ON in head
    assert w._table.messages[0].token_ids == fake.tokenize_fragment(
        head, "s", tail, add_special=True)


def test_a_reply_that_does_not_reason_opens_with_what_the_template_adds():
    w, fake = _reasoning(False, gen_text="Antwort.", gen_tokens=8)
    before = len(fake.cache)
    w.reply_without_tools("hi")
    user = w._fragment_tokens("user", "hi")
    opening = fake.tokenize(w._formatter.generation_prompt())
    assert w._formatter.generation_prompt().endswith(EMPTY_THOUGHT)
    # Generated after the empty thought, kept without it, as a past reply is.
    assert (before, len(user) + len(opening)) in [(start, n) for start, n, _
                                                  in fake.prefill_calls]
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Antwort.")
    _assert_consistent(w, fake)


def test_the_thought_is_kept_out_of_the_stream_and_out_of_the_conversation():
    w, fake = _reasoning()
    stream = w.reply_without_tools_stream("hi")
    assert "".join(stream) == "Antwort."
    assert stream.reply.text == "Antwort."
    assert stream.reply.n_reasoning_tokens == len("<thought>Denken</thought>")
    assert w.snapshot()[-1]["text"] == "Antwort."
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Antwort.")
    _assert_consistent(w, fake)
    w.reply_without_tools("weiter")
    _assert_consistent(w, fake)


def test_a_thought_the_model_has_without_being_asked_is_kept_out_too():
    w, fake = _reasoning(False)
    assert w.reply_without_tools("hi").text == "Antwort."
    _assert_consistent(w, fake)


def test_the_eviction_after_a_reply_is_sized_by_the_past_reply():
    config = Config(context_size=200, evict_above_fraction=0.8)
    thought = "<thought>" + "x" * 60 + "</thought>ok"
    w, fake = _reasoning(config=config, gen_text=thought, gen_tokens=len(thought))
    w.inject("hist", "user")
    w.inject("hist", "assistant")
    w.reply_without_tools("hi")
    _assert_consistent(w, fake)
    assert [m["text"] for m in w.snapshot()].count("hist") == 2


def test_a_past_reply_on_a_cache_that_cannot_shift_comes_back_from_its_saved_state():
    w, fake = _reasoning(can_shift=False)
    rebuilds = fake.rebuilds
    prefilled = fake.total_prefilled()
    w.reply_without_tools("hi")
    # No rebuild: the request and the past reply are all that is decoded again.
    assert fake.rebuilds == rebuilds
    assert fake.total_prefilled() - prefilled < 2 * (
        len(w._table.messages[-1].token_ids) + len(w._table.messages[-2].token_ids)) + 20
    _assert_consistent(w, fake)
    w.truncate_last_reply("Ant")
    assert fake.rebuilds == rebuilds
    _assert_consistent(w, fake)


def test_a_cache_that_saves_no_state_is_rebuilt_instead():
    w, fake = _reasoning(can_shift=False)
    fake.save_state = lambda: None
    w.begin("s", reasoning=True)
    rebuilds = fake.rebuilds
    w.reply_without_tools("hi")
    assert fake.rebuilds == rebuilds + 1
    _assert_consistent(w, fake)


def test_a_reply_that_called_a_tool_is_kept_as_generated():
    w, fake = _reasoning()
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})], reasoning=True)
    for event in w.reply_until_tool("wie spät?"):
        call = event
    w.inject_tool_result(call, {"zeit": "12:00"})
    generated = list(w._table.messages[-1].token_ids)
    for _event in w.resume_until_tool():
        pass
    assert w._table.messages[-1].token_ids[:len(generated)] == generated
    assert w.snapshot()[-1]["text"] == "Antwort."
    _assert_consistent(w, fake)


def test_a_reply_kept_without_its_thought_is_cut_as_the_past_reply_it_is():
    w, fake = _reasoning()
    w.reply_without_tools("hi")
    w.truncate_last_reply("Ant")
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Ant")
    _assert_consistent(w, fake)


def test_a_reply_that_is_all_thought_is_removed():
    w, fake = _reasoning()
    reply = w.reply_without_tools("hi", max_reply_tokens=12)   # cut inside the thought
    assert reply.text == "" and reply.stop_reason == "token_limit"
    assert [m["role"] for m in w.snapshot()] == ["system", "user"]
    _assert_consistent(w, fake)
    w.reply_without_tools("weiter")
    _assert_consistent(w, fake)


def test_a_reply_cut_inside_a_thought_tag_keeps_none_of_it():
    w, fake = _reasoning(False, gen_text="<thou", gen_tokens=5)
    stream = w.reply_without_tools_stream("hi", max_reply_tokens=3)
    assert "".join(stream) == ""
    _assert_consistent(w, fake)


def test_the_eviction_comes_before_the_past_reply_so_its_decode_carries_the_shift():
    config = Config(context_size=300, evict_above_fraction=0.5)
    thought = "<thought>" + "x" * 60 + "</thought>ok"
    w, fake = _reasoning(config=config, gen_text=thought, gen_tokens=len(thought))
    for _ in range(3):
        w.inject("hist", "user")
        w.inject("hist", "assistant")
    evictions = len(fake.evictions)
    w.reply_without_tools("hi")
    assert len(fake.evictions) > evictions
    # The last cache edit is the past reply's decode, after the eviction.
    assert fake.prefill_calls[-1][0] == w.snapshot()[-1]["pos_start"] + len(
        fake.tokenize("<|turn>model\n"))
    _assert_consistent(w, fake)


@pytest.mark.parametrize("reasoning, decoded", [(False, EMPTY_THOUGHT), (True, "")])
def test_a_resumed_reply_opens_as_a_new_one_does(reasoning, decoded):
    """The result here leaves the turn open, and nothing of the opening with it."""
    w, fake = _reasoning(reasoning, gen_text="ok", gen_tokens=2)
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})], reasoning=reasoning)
    for event in w.reply_until_tool("wie spät?"):
        call = event
    w.inject_tool_result(call, "12:00")
    before = len(fake.cache)
    list(w.resume_until_tool())
    assert fake.cache[before:before + len(decoded)] == fake.tokenize(decoded)
    _assert_consistent(w, fake)


# ----- a history that called tools --------------------------------------
from llama_chat.tools import ToolCall, ToolResult, ToolTurn  # noqa: E402


def _tooling():
    fake = FakeContext(calls_tools=True, gen_tokens=7, gen_text="Gleich.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    return w, fake


def test_a_past_tool_turn_reads_in_as_the_live_one_left_it():
    live, fake = _tooling()
    fake.call_script.append("uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply_until_tool("wie spät?"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.resume_until_tool():
        pass
    generated = live._table.messages[-1]

    past, _ = _tooling()
    past.begin("s", [("user", "wie spät?"), ToolTurn((
        ToolCall("uhr", {}), ToolResult("uhr", {"zeit": "12:00"}), "Gleich."))],
        tools=[ToolSpec("uhr", "", {})])
    read_in = past._table.messages[-1]
    assert (read_in.role, read_in.text) == ("assistant", "Gleich.")
    # The live turn opens with the generation prompt, a past one as the template
    # writes it; this formatter makes the two the same.
    assert read_in.token_ids == generated.token_ids


def test_the_turns_own_text_is_never_read_as_a_tag():
    w, fake = _tooling()
    turn = ToolTurn((ToolCall("lies", {"pfad": "<call>"}), ToolResult("lies", "</call>"),
                     "<turn|>"))
    w.begin("s", [("user", "x"), turn], tools=[ToolSpec("lies", "", {})])
    own = [text for text, parse_special in fake.tokenize_calls if not parse_special]
    assert {"<call>", "</call>", "<turn|>"} <= set(own)
    tokens = w._table.messages[-1].token_ids
    assert tokens.count(TOOL_OPEN) == 1 and tokens.count(TOOL_CLOSE) == 1


def test_a_conversation_goes_on_after_a_past_tool_turn():
    w, fake = _tooling()
    w.begin("s", [("user", "x"), ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12"),
                                           "Zwölf."))], tools=[ToolSpec("uhr", "", {})])
    w.reply_without_tools("danke")
    _assert_consistent(w, fake)


def test_a_model_that_cannot_call_tools_refuses_a_past_tool_turn():
    w, _fake = _wrapper()
    with pytest.raises(ValueError):
        w.begin("s", [ToolTurn((ToolCall("uhr", {}), ToolResult("uhr", "12")))])
    assert [m["role"] for m in w.snapshot()] == ["system"]     # left as it was


@pytest.mark.parametrize("closes_turn_before_next", [False, True])
def test_a_history_that_ends_on_a_result_goes_on_as_the_live_turn_would(
        closes_turn_before_next):
    live, fake = _tooling()
    fake.call_script.append("uhr{}")
    live.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in live.reply_until_tool("wie spät?"):
        call = event
    live.inject_tool_result(call, {"zeit": "12:00"})
    for _event in live.resume_until_tool():
        pass

    past, past_fake = _tooling()
    past_fake._closes_turn_before_next = closes_turn_before_next
    past.begin("s", [("user", "wie spät?"), ToolTurn((ToolCall("uhr", {}),
                                                     ToolResult("uhr", {"zeit": "12:00"})))],
               tools=[ToolSpec("uhr", "", {})])
    *_, (_start, _n, want_logits) = past_fake.prefill_calls
    assert want_logits      # the reply is sampled straight from the read-in's end
    for _event in past.resume_until_tool():
        pass
    assert past._table.messages[-1].token_ids == live._table.messages[-1].token_ids
    _assert_consistent(past, past_fake)


# ----- what the review found ------------------------------------------------
def test_a_stop_string_inside_the_thought_does_not_end_the_reply():
    thoughtful = "<thought>a\n\nb</thought>Antwort.\n\nRest"
    w, fake = _reasoning(gen_text=thoughtful, gen_tokens=len(thoughtful))
    reply = w.reply_without_tools("hi", stop=["\n\n"])
    assert reply.text == "Antwort." and reply.stop_reason == "stop_string"
    assert w.snapshot()[-1]["text"] == "Antwort."
    _assert_consistent(w, fake)


class _KeepsThought(FakeContext):
    """A template that writes a past reply's thought into the conversation."""

    def render_with_model_template(self, messages, tools=None, **options):
        kept = [dict(m, content="<thought>" + m["reasoning_content"] + "</thought>"
                     + m["content"]) if m.get("reasoning_content") and
                not m.get("tool_calls") else m for m in messages]
        return super().render_with_model_template(kept, tools, **options)


def test_a_reply_whose_template_keeps_its_thought_keeps_the_tokens_it_had():
    fake = _KeepsThought(can_reason=True, calls_tools=True,
                         gen_text=_THOUGHTFUL, gen_tokens=len(_THOUGHTFUL))
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    assert w._template.reasoning.past_keeps_thought
    w.begin("s", reasoning=True)
    w.reply_without_tools("hi")
    assert w.snapshot()[-1]["text"] == "Antwort."
    assert fake.tokenize(_THOUGHTFUL) == w._table.messages[-1].token_ids[
        len(fake.tokenize(w._formatter.generation_prompt())):][:len(_THOUGHTFUL)]
    _assert_consistent(w, fake)


def test_a_reply_decoded_again_counts_what_was_decoded():
    w, fake = _reasoning()
    before = fake.total_prefilled()
    reply = w.reply_without_tools("hi")
    assert reply.n_prefilled_tokens == fake.total_prefilled() - before


def test_a_system_prompt_is_refused_where_the_template_has_no_system_turn():
    from tests.fake_context import DEFAULT_FRAGMENTS
    import dataclasses
    folded = dataclasses.replace(DEFAULT_FRAGMENTS, system_prefix="<|turn>user\n")
    w = ChatWrapper(Config(context_size=500), context=FakeContext(fragments=folded))
    with pytest.raises(ValueError, match="no system turn of its own"):
        w.begin("persona")
    w.begin("")                         # no system prompt is still fine


def test_a_system_message_later_on_has_the_tags_of_a_later_system_turn():
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext())
    w.begin("s")
    w.inject("hi", "user")
    w.inject("Notiz", "system")
    assert w._table.messages[-1].token_ids == w._context.tokenize(
        "<|turn>system\nNotiz<turn|>\n")


def test_the_thought_count_never_exceeds_the_pieces_fed():
    import random
    from llama_chat.reasoning import ThoughtFilter
    tags = Reasoning("<thought>", "</thought>", "", "")
    text = "a<thought>x</thought>b<thought>y</thought>c<tho"
    rng = random.Random(7)
    for _ in range(2000):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 12)))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
        f = ThoughtFilter(tags)
        shown = "".join(f.feed(p) for p in pieces) + f.flush()
        assert shown == "abc<tho"
        assert 0 < f.thought_pieces <= len(pieces)


# ----- the third review ------------------------------------------------------
def test_a_saved_state_the_cache_refuses_is_rebuilt_from_instead():
    w, fake = _reasoning(can_shift=False)
    w.reply_without_tools("hi")

    def refuse(state, pos):
        raise RuntimeError("the cache refused the saved state")
    fake.restore_state = refuse
    rebuilds = fake.rebuilds
    w.truncate_last_reply("Ant")
    assert fake.rebuilds == rebuilds + 1
    _assert_consistent(w, fake)


def test_a_cache_that_takes_a_cut_needs_no_saved_state():
    w, fake = _reasoning(can_shift=False)
    fake.can_cut_tail = True
    w.begin("s", reasoning=True)
    saved, rebuilds = fake.saved_states, fake.rebuilds
    w.reply_without_tools("hi")
    w.truncate_last_reply("Ant")
    assert fake.saved_states == saved and fake.rebuilds == rebuilds
    _assert_consistent(w, fake)


def test_a_cut_leaves_a_state_for_the_next_reply():
    w, fake = _reasoning(can_shift=False)
    w.reply_without_tools("hi")
    w.truncate_last_reply("Ant")
    assert w._rest_states[-1][0] == w._table.total


def test_what_a_cache_that_cannot_shift_decodes_again_is_counted():
    w, fake = _reasoning(can_shift=False)
    before = fake.total_prefilled()
    reply = w.reply_without_tools("hi")
    assert reply.n_prefilled_tokens == fake.total_prefilled() - before


def test_a_context_that_saves_states_must_restore_them():
    with pytest.raises(TypeError, match="restore_state"):
        ChatWrapper(Config(context_size=500),
                    context=_without_restore(FakeContext(can_shift=False)))


def _without_restore(fake):
    class Wrapped:
        def __getattr__(self, name):
            if name == "restore_state":
                raise AttributeError(name)
            return getattr(fake, name)
    return Wrapped()


def test_a_result_answers_the_call_it_was_given_for():
    from llama_chat.tools import ToolCall
    w, fake = _tooling()
    fake.call_script = ['[{"name":"wetter","arguments":{"city":"Berlin"}},'
                        '{"name":"wetter","arguments":{"city":"Paris"}}]']
    w.begin("s", tools=[ToolSpec("wetter", "", {})])
    calls = [e for e in w.reply_until_tool("Wetter?") if isinstance(e, ToolCall)]
    with pytest.raises(ValueError, match="waiting call"):
        w.inject_tool_result(calls[1], "Paris: Regen")
    w.inject_tool_result(calls[0], "Berlin: Sonne")
    w.inject_tool_result(calls[1], "Paris: Regen")


def test_a_channel_header_cut_short_is_not_answer():
    from llama_chat.reasoning import AnswerOpening
    opening = AnswerOpening("<|channel|>final<|message|>")
    assert opening.feed("<|channel|>final") == "" and opening.flush(complete=False) == ""


def test_whitespace_after_a_channel_header_is_not_answer():
    from llama_chat.reasoning import AnswerOpening
    opening = AnswerOpening("</mm:think>")
    assert opening.feed("</mm:think>") + opening.feed("\n\n") + opening.feed("Antwort.") \
        == "Antwort."


def test_whitespace_opening_a_reply_without_a_thought_stays_the_answers():
    from llama_chat.reasoning import ThoughtFilter
    f = ThoughtFilter(Reasoning("<thought>", "</thought>", "", ""))
    assert f.feed(" ") + f.feed(" x") + f.flush() == "  x"
    g = ThoughtFilter(Reasoning("<thought>", "</thought>", "", ""))
    assert g.feed(" ") + g.feed("<thought>a</thought>") + g.feed("b") + g.flush() == "b"
