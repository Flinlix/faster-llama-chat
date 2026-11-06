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


def test_the_tokens_that_stay_are_not_decoded_again():
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw", tokens_spell_text=True)
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


def test_a_cut_on_a_cache_that_cannot_shift_rebuilds_it_from_the_conversation():
    w, fake = _wrapper(gen_tokens=8, gen_text="Eins. Zw", can_shift=False)
    w.reply_without_tools("hi")
    rebuilds = fake.rebuilds
    w.truncate_last_reply("Eins.")
    assert fake.rebuilds == rebuilds + 1
    assert w.snapshot()[-1]["text"] == "Eins."
    _assert_consistent(w, fake)


# ----- what the conversation keeps of a reply ----------------------------
def _keeping(recorded_text, config=None, **fake_kw):
    fake = FakeContext(**fake_kw)
    w = ChatWrapper(config or Config(context_size=500, evict_above_fraction=1.0),
                    context=fake, recorded_text=recorded_text)
    w.begin("s")
    return w, fake


def _answer(text):
    return text.split("|")[-1]


def test_a_reply_the_option_changes_is_kept_as_a_past_reply_and_the_cache_agrees():
    w, fake = _keeping(_answer, gen_tokens=15, gen_text="Denken|Antwort.")
    w.reply_without_tools("hi")
    assert w.snapshot()[-1]["text"] == "Antwort."
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Antwort.")
    _assert_consistent(w, fake)
    w.reply_without_tools("weiter")
    _assert_consistent(w, fake)


def test_only_what_differs_from_the_generated_reply_is_decoded_again():
    w, fake = _keeping(_answer, gen_tokens=15, gen_text="Denken|Antwort.",
                       tokens_spell_text=True)
    before = len(fake.prefill_calls)
    w.reply_without_tools("hi")
    *_, (start, n, _logits) = fake.prefill_calls[before:]
    opening = fake.tokenize(w._formatter.assistant_open())
    kept = w._fragment_tokens("assistant", "Antwort.")
    assert (start, n) == (w.snapshot()[-1]["pos_start"] + len(opening),
                          len(kept) - len(opening))


def test_the_eviction_after_a_reply_is_sized_by_what_is_kept():
    config = Config(context_size=200, evict_above_fraction=0.8)

    def history_left(recorded_text):
        w, fake = _keeping(recorded_text, config, gen_tokens=60, gen_text="x" * 60)
        w.inject("hist", "user")
        w.inject("hist", "assistant")
        w.reply_without_tools("hi")
        _assert_consistent(w, fake)
        return [m["text"] for m in w.snapshot()].count("hist")

    assert history_left(None) < 2           # the whole reply would evict history
    assert history_left(lambda text: "ok") == 2


def test_a_reply_kept_as_nothing_is_removed():
    w, fake = _keeping(lambda text: "", gen_tokens=8, gen_text="Denken")
    w.reply_without_tools("hi")
    assert [m["role"] for m in w.snapshot()] == ["system", "user"]
    _assert_consistent(w, fake)
    w.reply_without_tools("weiter")
    _assert_consistent(w, fake)


def test_a_reply_kept_on_a_cache_that_cannot_shift_rebuilds_it():
    w, fake = _keeping(_answer, gen_tokens=15, gen_text="Denken|Antwort.",
                       can_shift=False)
    rebuilds = fake.rebuilds
    w.reply_without_tools("hi")
    assert fake.rebuilds == rebuilds + 1
    _assert_consistent(w, fake)


def test_an_option_that_raises_leaves_the_reply_as_generated():
    def broken(text):
        raise RuntimeError("kaputt")
    w, fake = _keeping(broken, gen_tokens=15, gen_text="Denken|Antwort.")
    w.reply_without_tools("hi")
    assert w.snapshot()[-1]["text"] == "Denken|Antwort."
    _assert_consistent(w, fake)


def test_a_reply_that_called_a_tool_is_kept_as_generated():
    w, fake = _keeping(lambda text: "", calls_tools=True)
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    for event in w.reply_until_tool("wie spät?"):
        call = event
    w.inject_tool_result(call, {"zeit": "12:00"})
    for _event in w.resume_until_tool():
        pass
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_a_kept_reply_is_cut_as_the_past_reply_it_now_is():
    w, fake = _keeping(lambda text: "Eins. Zwei.", gen_tokens=8, gen_text="Denken")
    w.reply_without_tools("hi")
    w.truncate_last_reply("Eins. ")
    assert w._table.messages[-1].token_ids == w._fragment_tokens("assistant", "Eins. ")
    _assert_consistent(w, fake)


# ----- what opens the system turn ----------------------------------------
from llama_chat.template import TemplateFormatter  # noqa: E402


class _Opening(TemplateFormatter):
    def system_open(self):
        return "<mode>"


def test_the_system_turn_opens_with_what_the_formatter_adds():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake,
                    formatter=_Opening)
    w.begin("s")
    prefix, content, suffix = w._formatter.parts("system", "s")
    assert w._table.messages[0].token_ids == fake.tokenize_fragment(
        prefix + "<mode>", content, suffix, add_special=True)


def test_with_tools_declared_it_follows_the_turns_own_head():
    fake = FakeContext(calls_tools=True)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake,
                    formatter=_Opening)
    specs = [ToolSpec("uhr", "", {})]
    w.begin("s", tools=specs)
    head, tail = w._template.system_turn_declaring(specs)
    own_head = w._template._plain_system_turn().partition("PROMPTMSG")[0]
    assert head.startswith(own_head)
    opened = own_head + "<mode>" + head[len(own_head):]
    assert w._table.messages[0].token_ids == fake.tokenize_fragment(
        opened, "s", tail, add_special=True)


def test_a_formatter_that_opens_the_system_turn_again_is_refused():
    class Twice(TemplateFormatter):
        def system_open(self):
            return self._fragments.system_prefix
    with pytest.raises(ValueError, match="system_open"):
        ChatWrapper(Config(), context=FakeContext(), formatter=Twice)


# ----- a history that called tools --------------------------------------
from llama_chat.tools import ToolCall, ToolResult, ToolTurn  # noqa: E402


def _tooling():
    fake = FakeContext(calls_tools=True, gen_tokens=7, gen_text="Gleich.",
                       tokens_spell_text=True)
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
