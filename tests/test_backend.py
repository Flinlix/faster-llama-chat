"""What the llama.cpp-facing layer settles with no model loaded.

The calls are stubbed, because what is pinned is this package's own criteria
rather than llama.cpp's answers: what counts as one of the model's own tags,
and which cache edits are performed while loading rather than at the first
eviction. The first is pinned here rather than through the fake context used
everywhere else, which answers that question itself and so cannot test it. See
design_history.md, "How a call's format is recovered".
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import llama_cpp
import pytest

from llama_chat.template import CallBoundary

import llama_chat

from llama_chat._backend import Backend
from llama_chat.config import KV_CACHE_TYPES
from llama_chat.tokenizing import split_at_tags, tokenize_pieces
from llama_chat.cache_context import CACHE_CONTEXT, CacheContext
from llama_chat.cache_context import GenerationAccumulator
from llama_chat.context import KVContext
from llama_chat.generation import generate
from llama_chat.config import Config, Sampling


def _backend(attr):
    """A Backend whose attribute lookup returns ``attr`` for every token."""
    backend = Backend.__new__(Backend)
    backend._get_attr = lambda vocab, token: attr
    return backend


def test_a_control_token_is_a_tag():
    assert _backend(llama_cpp.LLAMA_TOKEN_ATTR_CONTROL).is_special(None, 1)


def test_a_tag_a_finetune_added_is_a_tag_too():
    # The tags bracketing a tool call are added tokens rather than control
    # ones, so a criterion that accepts only control tokens turns tool calling
    # off without saying so.
    assert _backend(llama_cpp.LLAMA_TOKEN_ATTR_USER_DEFINED).is_special(None, 1)


def test_ordinary_text_is_not():
    assert not _backend(llama_cpp.LLAMA_TOKEN_ATTR_NORMAL).is_special(None, 1)


# ----- the cache edits eviction rests on ---------------------------------
@pytest.mark.parametrize("reports_shift", (True, False))
def test_every_cache_edit_is_performed_while_loading(reports_shift):
    """A build missing one of them says so at load, not at the first eviction.

    Removal and the shift that closes the gap are reached only once history
    crosses the threshold, a whole conversation after the model was loaded.
    Each is proven before the conversation begins and leaves the cache empty.
    """
    _context, performed = _probed_context(reports_shift, cuts_tail=True)
    assert ("seq_rm", 1, -1) in performed
    # A cache that cannot shift is never shifted, and is not asked to.
    assert [edit for edit in performed if edit[0] == "seq_add"] == (
        [("seq_add", 0, 0, 0)] if reports_shift else [])


def _probed_context(reports_shift, cuts_tail, saves=True):
    """A context over a backend reporting a shift, taking or refusing a cut at its end, and saving a state or not."""
    performed = []

    def save_state(ctx, seq):
        if not saves:
            raise RuntimeError("the cache did not save its state")
        performed.append(("save",))
        return "state"

    def kv_seq_rm(ctx, seq, p0, p1):
        # Every removal that stops before the end succeeds, as a running state's does,
        # and so does one from where a restored state ends.
        restored = bool(performed) and performed[-1][0] == "restore"
        performed.append(("seq_rm", p0, p1))
        return cuts_tail or p1 >= 0 or restored

    context = object.__new__(KVContext)
    context._engine_context, context._vocab, context._max_batch_tokens = "ctx", None, 8
    context._backend = SimpleNamespace(
        can_shift=lambda ctx: reports_shift,
        token_eos=lambda vocab: 2,
        decode=lambda ctx, tokens, pos, *rest: performed.append(("decode", tokens, pos)),
        kv_seq_rm=kv_seq_rm,
        kv_seq_add=lambda ctx, seq, p0, p1, d: performed.append(("seq_add", p0, p1, d)),
        kv_clear=lambda ctx: performed.append(("clear",)),
        save_state=save_state,
        restore_state=lambda ctx, seq, state: performed.append(("restore", state)),
    )
    context._settle_cache_edits()
    return context, performed


def test_a_cache_that_cannot_be_cut_saves_and_restores_a_state_while_loading():
    """It is brought back with saved states, so a build that cannot says so at load, not at a reply."""
    _context, performed = _probed_context(reports_shift=True, cuts_tail=False)
    saved = performed.index(("save",))
    assert ("restore", "state") in performed[saved:]
    assert performed[-1] == ("clear",)
    with pytest.raises(RuntimeError, match="cannot save and restore"):
        _probed_context(reports_shift=True, cuts_tail=False, saves=False)


def test_a_cache_that_can_be_cut_is_not_asked_for_a_state_while_loading():
    _context, performed = _probed_context(reports_shift=False, cuts_tail=True)
    assert ("save",) not in performed


def test_a_cache_that_refuses_a_cut_at_its_end_is_never_shifted_whatever_it_reports():
    """A hybrid cache reports the shift its attention part can do; its running state cannot."""
    context, performed = _probed_context(reports_shift=True, cuts_tail=False)
    assert not context.can_shift and not context.can_cut_tail
    assert not any(edit[0] == "seq_add" for edit in performed)    # not even at load
    assert performed[-1] == ("clear",)


def test_an_eviction_removes_its_stretch_and_moves_everything_after_it_down_by_its_length():
    context, performed = _probed_context(reports_shift=True, cuts_tail=True)
    performed.clear()
    context.apply_eviction(4, 10)
    assert performed == [("seq_rm", 4, 10), ("seq_add", 10, -1, -6)]    # -1: to the end


@pytest.mark.parametrize("reports_shift", (True, False))
def test_a_cache_that_takes_the_cut_shifts_as_it_reports(reports_shift):
    context, _performed = _probed_context(reports_shift, cuts_tail=True)
    assert context.can_shift == reports_shift and context.can_cut_tail


def test_the_cut_is_tried_inside_a_running_state_not_at_its_end():
    """At the last position a removal may clear a running state whole and succeed."""
    _context, performed = _probed_context(reports_shift=False, cuts_tail=True)
    probe = performed.index(("seq_rm", 1, -1))
    assert len(performed[probe - 1][1]) > 2
    assert performed[probe + 1] == ("clear",)


def _raises_type_error(*_args):
    raise TypeError("this build takes a different argument list")


def test_a_build_whose_penalty_signature_differs_is_warned_about_not_refused():
    """A model still generates without them; refusing to load would cost more.

    The penalties are the one sampler stage whose argument list has changed
    across releases, so it is the one whose absence has to be survivable.
    """
    added = []
    backend = Backend.__new__(Backend)
    backend._chain_add = lambda chain, stage: added.append(stage)
    backend._init_penalties = _raises_type_error
    backend._penalties_take_vocab_size = False

    with pytest.warns(RuntimeWarning, match="penalties"):
        backend.add_penalties(object(), "vocab", 64, 1.1, 0.0, 0.0)
    assert added == []


@pytest.mark.parametrize("takes_vocab_size, arguments", [
    (True, (32000, 64, 1.1, 0.5, 0.25)), (False, (64, 1.1, 0.5, 0.25))])
def test_the_penalties_get_the_argument_list_this_build_takes(takes_vocab_size, arguments):
    passed = []
    backend = Backend.__new__(Backend)
    backend._chain_add = lambda chain, stage: None
    backend._init_penalties = lambda *args: passed.append(args)
    backend._penalties_take_vocab_size = takes_vocab_size
    backend._vocab_n_tokens = lambda vocab: 32000
    backend.add_penalties(object(), "vocab", 64, 1.1, 0.5, 0.25)
    assert passed == [arguments]


def test_a_bindings_argument_count_is_read_from_its_types_or_its_signature():
    from types import SimpleNamespace

    from llama_chat._backend import _parameter_count
    assert _parameter_count(SimpleNamespace(argtypes=[1, 2, 3, 4, 5])) == 5
    assert _parameter_count(lambda a, b, c, d: None) == 4


def test_only_this_layer_names_llama_cpp():
    """The seam is what the rest of the package is written against.

    Every symbol llama.cpp has renamed is resolved in one file so the churn
    costs one edit; a second file reaching for the module puts the next rename
    somewhere nobody thinks to look. Read as source rather than imported,
    because an import that has not run yet would not be found.
    """
    package = Path(llama_chat.__file__).parent
    reaching = sorted(f.name for f in package.glob("*.py")
                      if f.name != "_backend.py" and "llama_cpp" in f.read_text())
    assert reaching == [], (
        f"{reaching} name llama_cpp directly; the handle belongs in _backend.py")


def test_every_cache_quantization_config_offers_resolves_to_a_type_id():
    """The names a caller may pass and this build's numbering have to agree.

    ``Config`` offers the names and llama.cpp numbers them, so nothing else
    would notice a name it cannot resolve until a model is loaded with it.
    """
    backend = Backend.__new__(Backend)
    type_ids = [backend.kv_cache_type_id(name) for name in KV_CACHE_TYPES]
    assert len(set(type_ids)) == len(KV_CACHE_TYPES)


def test_a_removal_the_cache_refuses_is_raised():
    context = object.__new__(KVContext)
    context._engine_context = "the context"
    context._backend = SimpleNamespace(kv_seq_rm=lambda *args: False)
    with pytest.raises(RuntimeError, match="refused"):
        context.remove_from(3)


@pytest.mark.parametrize("answer, taken", [(True, True), (False, False), (None, True)])
def test_a_removal_reports_the_caches_answer_and_takes_none_for_yes(answer, taken):
    backend = object.__new__(Backend)
    backend._get_memory = lambda ctx: "the memory"
    backend._mem_rm = lambda *args: answer
    assert backend.kv_seq_rm("the context", 0, 3, -1) is taken


def test_a_reply_ends_at_its_turn_end_token_which_is_neither_cached_nor_shown():
    """A model whose closing takes two tags writes the first itself, as text would be."""
    sampled = iter(["Pa", "ris", "<response_end>", "<end>"])
    decoded = []
    context = object.__new__(KVContext)
    context._engine_context, context._vocab, context._max_batch_tokens = "ctx", None, 8
    context._config = Config()
    context._build_sampler = lambda config, withheld=(): "sampler"
    context._backend = SimpleNamespace(
        free_sampler=lambda sampler: None,
        sample=lambda sampler, ctx: next(sampled),
        is_eog=lambda vocab, token: token == "<end>",
        decode=lambda ctx, tokens, pos, *rest: decoded.extend(tokens),
        token_to_piece_bytes=lambda vocab, token: token.encode())
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 10, out=out, ends_turn=lambda token: token == "<response_end>"))
    assert shown == out.text == "Paris"
    assert decoded == out.token_ids == ["Pa", "ris"]
    assert out.stop_reason == "end_of_turn"


def _merging_context():
    """A context whose vocabulary merges whitespace runs and words, with two tags."""
    import re
    def tokenize(vocab, text, add_special, parse_special):
        pattern = r"<t>|<e>|\s+|[^\s<]+|<" if parse_special else r"\s+|[^\s]+"
        found = re.findall(pattern, text)
        return (["BOS"] if add_special else []) + [
            f"TAG{tok}" if parse_special and tok in ("<t>", "<e>") else tok for tok in found]
    context = object.__new__(KVContext)
    context._vocab = None
    context._backend = SimpleNamespace(
        tokenize=tokenize,
        is_special=lambda vocab, token: token.startswith("TAG"),
        token_to_piece=lambda vocab, token: token[3:] if token.startswith("TAG") else token)
    return context


@pytest.mark.parametrize("prefix, content, suffix", [
    ("<t>user\n", " sys ", "\n\n<e>"),       # merges across both boundaries
    ("<t>", "word", "<e>"),                    # tags right next to the content
    ("plain: ", "word", " end"),               # no tags at all
])
def test_a_message_tokenizes_as_the_whole_render_does(prefix, content, suffix):
    context = _merging_context()
    assert tokenize_pieces(context, [(prefix, False), (content, True), (suffix, False)]) == context._backend.tokenize(
        None, prefix + content + suffix, False, True)


def test_a_tag_typed_inside_a_message_stays_text():
    tokens = tokenize_pieces(_merging_context(), [("<t>", False), ("a<e>b", True), ("<e>", False)])
    assert tokens.count("TAG<e>") == 1 and tokens[-1] == "TAG<e>"


def test_the_beginning_of_a_conversation_is_attached_to_the_first_message():
    assert tokenize_pieces(_merging_context(), [("lead ", False), ("word", True)], add_special=True)[0] == "BOS"


def _sampling_context(sampled):
    """A context that samples ``sampled`` in order and records what it decodes and removes."""
    sampled, decoded, removed = iter(sampled), [], []
    context = object.__new__(KVContext)
    context._engine_context, context._vocab, context._max_batch_tokens = "ctx", None, 8
    context._config = Config()
    context._build_sampler = lambda config, withheld=(): "sampler"
    context._backend = SimpleNamespace(
        free_sampler=lambda sampler: None,
        sample=lambda sampler, ctx: next(sampled),
        is_eog=lambda vocab, token: token == "<end>",
        decode=lambda ctx, tokens, pos, *rest: decoded.extend(tokens),
        kv_seq_rm=lambda ctx, seq, start, end: removed.append(start) or True,
        token_to_piece_bytes=lambda vocab, token: token.encode())
    return context, decoded, removed


_CALL_TEXTS = ('<close>resp<sep><open>tools<sep><open>call tool="', '<sep><close>call<sep>')


def test_a_call_without_a_tag_of_its_own_is_read_by_its_texts():
    context, decoded, _ = _sampling_context(
        ["Gleich.", "<close>", "resp", "<sep>", "<open>", "tools", "<sep>", "<open>",
         'call tool="', "clock", '" index="1"', "<sep>", "<close>", "call", "<sep>", "x"])
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(call_texts=_CALL_TEXTS)))
    assert shown == out.text == "Gleich."
    assert out.stop_reason == "tool_call" and out.tool_call == 'clock" index="1"'
    assert decoded == out.token_ids and decoded[-1] == "<sep>"


@pytest.mark.parametrize("sampled", [
    # Written as the template writes it, whitespace around the texts included.
    ["Gleich.", "\n", "<open>", "call:", "\n", "clock{}", "\n", "<close>", "x"],
    # Without that whitespace, as a model may write it.
    ["Gleich.", "<open>", "call:", "clock{}", "<close>", "x"],
    # With more of it.
    ["Gleich.", " \n\n", "<open>", "call:", "  ", "clock{}", " \n", "<close>", "x"],
], ids=["as-written", "without-whitespace", "more-whitespace"])
def test_a_calls_texts_are_matched_without_the_whitespace_around_them_as_a_thoughts_tags_are(
        sampled):
    context, decoded, _ = _sampling_context(sampled)
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(
        call_texts=("\n<open>call:\n", "\n<close>"))))
    assert shown == out.text == "Gleich."
    assert out.stop_reason == "tool_call" and out.tool_call == "clock{}"


def test_a_reply_ends_where_its_closing_parts_from_a_call():
    context, decoded, removed = _sampling_context(
        ["Pa", "ris", "<close>", "resp", "<sep>", "<close>", "msg", "<end>"])
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(call_texts=_CALL_TEXTS, turn_end_text="<close>resp<sep><close>")))
    assert shown == out.text == "Paris"
    assert out.stop_reason == "end_of_turn" and out.token_ids == ["Pa", "ris"]
    # The closing's first tokens leave the reply; the caller cuts them from the
    # cache, in the way that cache allows.
    assert out.cached_beyond == 4 and removed == []


def _plain_context(sampled):
    context, decoded, removed = _sampling_context(sampled)
    context._backend.is_special = lambda vocab, token: token.startswith("<|")
    context._backend.is_eog = lambda vocab, token: token in ("<end>", "<|call|>")
    return context, decoded


_PLAIN = ("assistant to=functions.", "<|call|>")


@pytest.mark.parametrize("sampled", [
    ["assistant", " to=", "functions.", "clock", "{}", "<|call|>"],                   # the start
    ["Denken.", "<|start|>", "assistant", " to=", "functions.", "clock", "{}", "<|call|>"],  # a tag
])
def test_a_plain_text_opening_opens_a_call_at_the_start_or_after_a_tag(sampled):
    context, decoded = _plain_context(sampled)
    out = GenerationAccumulator()
    "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(call_texts=_PLAIN, opens_as_text=True)))
    assert out.stop_reason == "tool_call" and out.tool_call == "clock{}"
    assert decoded[-1] == "<|call|>"        # the end-of-generation token that closes it is kept


def test_a_plain_text_opening_inside_a_sentence_is_text():
    context, _ = _plain_context(["Say", " assistant", " to=", "functions.", "x", "<end>"])
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(call_texts=_PLAIN, opens_as_text=True)))
    assert out.stop_reason == "end_of_turn" and shown == "Say assistant to=functions.x"


def test_a_turn_ended_inside_a_call_closes_the_call_as_the_template_does():
    """Llama 3 ends a call with one end-of-turn token where its template writes another."""
    context, decoded = _plain_context(['{"', 'name": "', 'clock', '"}', "<|eom|>"])
    context._backend.is_eog = lambda vocab, token: token in ("<|eom|>", "<|eot|>")
    context.tokenize = lambda text, add_special=False, parse_special=True: [text]
    out = GenerationAccumulator()
    "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(call_texts=('{"name": "', "<|eot|>"), opens_as_text=True)))
    assert out.stop_reason == "tool_call" and out.tool_call == 'clock"}'
    assert decoded[-1] == "<|eot|>" and "<|eom|>" not in decoded


def test_a_tag_a_call_follows_is_the_calls_and_never_shown():
    """Llama 3.2 writes a tag of its own before a call its template opens with plain text."""
    context, decoded = _plain_context(["Sure.", "<|python_tag|>", '{"name": "', "clock", '"}', "<|eot|>"])
    context._backend.is_eog = lambda vocab, token: token in ("<|eom|>", "<|eot|>")
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(
        call_texts=('{"name": "', "<|eot|>"), opens_as_text=True)))
    assert out.stop_reason == "tool_call" and shown == out.text == "Sure."


def test_a_tag_no_call_follows_stays_the_replys():
    context, _ = _plain_context(["<|python_tag|>", "hi", "<end>"])
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(
        call_texts=('{"name": "', "<|eot|>"), opens_as_text=True)))
    assert out.stop_reason == "end_of_turn" and shown == "<|python_tag|>hi"


def test_a_thoughts_closing_tag_a_call_follows_stays_the_replys():
    context, _ = _plain_context(["Hm.", "<|/think|>", '{"name": "', "clock", '"}', "<|eot|>"])
    context._backend.is_eog = lambda vocab, token: token in ("<|eom|>", "<|eot|>")
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(
        call_texts=('{"name": "', "<|eot|>"), opens_as_text=True, thought_tags=("<|/think|>",))))
    assert out.stop_reason == "tool_call" and shown == "Hm.<|/think|>"


def test_with_calls_off_a_tag_in_a_reply_that_ends_in_text_is_shown():
    context, _ = _plain_context(["Hi", "<|start|>", "there", "<|end|>", "<end>"])
    out = GenerationAccumulator()
    shown = "".join(generate(context, 0, 20, out=out, boundary=CallBoundary(
        opens_as_text=True, turn_end_text="<|end|>")))
    assert out.stop_reason == "end_of_turn" and shown == "Hi<|start|>there"


def test_a_text_splits_into_the_models_tags_and_the_plain_text_between():
    context = _merging_context()
    assert split_at_tags(context, "a <t>b c<e>") == [
        ("a ", False), ("<t>", True), ("b c", False), ("<e>", True)]
    assert split_at_tags(context, "plain") == [("plain", False)]


def test_the_models_tags_are_the_tokens_the_engine_marks_read_once():
    walked = []
    context = object.__new__(KVContext)
    context._vocab, context._special_tokens = None, None
    context._backend = SimpleNamespace(
        n_tokens=lambda vocab: 5,
        is_special=lambda vocab, token: walked.append(token) or token in (1, 3))
    assert context.special_tokens() == (1, 3)
    assert context.special_tokens() == (1, 3)
    assert walked == [0, 1, 2, 3, 4]
    assert context.is_special(3) and not context.is_special(2)


@pytest.mark.parametrize("max_batch_tokens, context_size, full", [(8, 32, 8), (64, 16, 16)])
def test_loading_ends_by_decoding_a_full_batch_and_a_single_token_and_leaves_the_cache_empty(
        tmp_path, monkeypatch, max_batch_tokens, context_size, full):
    performed = []
    backend = SimpleNamespace(
        load_model=lambda path, layers, progress: "model",
        metadata_value=lambda model, key: None,
        kv_cache_type_id=lambda name: 0,
        new_context=lambda *args, **kwargs: "ctx",
        vocab=lambda model: None,
        token_eos=lambda vocab: 2,
        can_shift=lambda ctx: True,
        kv_seq_rm=lambda ctx, seq, p0, p1: True,
        kv_seq_add=lambda ctx, seq, p0, p1, delta: None,
        decode=lambda ctx, tokens, pos, *rest: performed.append(("decode", len(tokens), pos)),
        kv_clear=lambda ctx: performed.append(("clear",)),
    )
    monkeypatch.setattr("llama_chat.context.Backend", lambda: backend)
    model = tmp_path / "model.gguf"
    model.touch()

    KVContext(Config(model_path=str(model), context_size=context_size,
                     max_batch_tokens=max_batch_tokens, min_reply_tokens=0))

    assert performed[-4:] == [("decode", full, 0), ("clear",), ("decode", 1, 0), ("clear",)]


def test_a_ggufs_template_for_tools_is_read_beside_its_default_one(tmp_path, monkeypatch):
    stored = {"tokenizer.chat_template": "default:{{ messages[0].content }}",
              "tokenizer.chat_template.tool_use": "tools:{{ messages[0].content }}"}
    backend = SimpleNamespace(
        load_model=lambda path, layers, progress: "model",
        metadata_value=lambda model, key: stored.get(key),
        kv_cache_type_id=lambda name: 0, new_context=lambda *args, **kwargs: "ctx",
        vocab=lambda model: None, token_eos=lambda vocab: 2, eos_text=lambda vocab: "",
        can_shift=lambda ctx: True, kv_seq_rm=lambda ctx, seq, p0, p1: True,
        kv_seq_add=lambda ctx, seq, p0, p1, delta: None,
        decode=lambda *args: None, kv_clear=lambda ctx: None)
    monkeypatch.setattr("llama_chat.context.Backend", lambda: backend)
    model = tmp_path / "model.gguf"
    model.touch()
    context = KVContext(Config(model_path=str(model), context_size=32, max_batch_tokens=8,
                               min_reply_tokens=0))
    question = [{"role": "user", "content": "hi"}]
    assert context.render_with_model_template(question) == "default:hi"
    assert context.render_with_model_template(question, template_name="tool_use") == "tools:hi"
    assert context.chat_template_text("tool_use") == stored["tokenizer.chat_template.tool_use"]
    assert context.chat_template_text("rag") is None


def test_an_empty_opening_still_carries_the_beginning_of_the_conversation():
    assert tokenize_pieces(_merging_context(), [], add_special=True) == ["BOS"]


def test_every_method_takes_the_arguments_the_written_boundary_lists():
    from tests.fake_context import FakeContext

    def parameters(owner, name):
        member = inspect.getattr_static(owner, name, None)
        return list(inspect.signature(member).parameters) if inspect.isfunction(member) else None

    for context in (KVContext, FakeContext):
        for name in CACHE_CONTEXT:
            assert parameters(context, name) == parameters(CacheContext, name), (context, name)


def _set_flash_attn(cparams, enable, llama_cpp=None):
    """Drive Backend._set_flash_attn with fakes, bypassing llama_cpp import."""
    from types import SimpleNamespace
    from llama_chat._backend import Backend

    Backend._set_flash_attn(SimpleNamespace(_llama_cpp=llama_cpp), cparams, enable)


def _set_kv_cache_on_gpu(cparams, enable):
    """Drive Backend._set_kv_cache_on_gpu with fakes, bypassing llama_cpp import."""
    from types import SimpleNamespace
    from llama_chat._backend import Backend

    Backend._set_kv_cache_on_gpu(SimpleNamespace(_llama_cpp=None), cparams, enable)


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
    )
    ctx._vocab, ctx._engine_context = None, None
    ctx._max_batch_tokens = 8
    ctx._config = Config()
    ctx._build_sampler = lambda config, withheld=(): ("own", config.temperature) + withheld
    return ctx


# ----- flash attention and offload on each build ---------------------------
@pytest.mark.parametrize("enable,expected", [(True, 1), (False, 0)])
def test_flash_attention_is_set_on_or_off(enable, expected):
    from types import SimpleNamespace

    cparams = SimpleNamespace(flash_attn_type=None)
    llama_cpp = SimpleNamespace(LLAMA_FLASH_ATTN_TYPE_ENABLED=1,
                         LLAMA_FLASH_ATTN_TYPE_DISABLED=0)
    _set_flash_attn(cparams, enable, llama_cpp=llama_cpp)
    assert cparams.flash_attn_type == expected


@pytest.mark.parametrize("enable", [True, False])
def test_set_kv_cache_on_gpu_bool_build(enable):
    from types import SimpleNamespace

    cparams = SimpleNamespace(offload_kqv=None)
    _set_kv_cache_on_gpu(cparams, enable)
    assert cparams.offload_kqv is enable


# ----- the sampler chain per generation ------------------------------------
def test_every_generation_draws_from_a_chain_of_its_own_freed_when_it_ends():
    """No chain outlives its generation, so penalties and a seed start afresh in each."""
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    own = ("own", Config().temperature)
    list(generate(ctx, 0, 1, out=GenerationAccumulator()))
    assert chains["sampled"] == chains["freed"] == [own]
    list(generate(ctx, 1, 1, out=GenerationAccumulator(), sampling=Sampling()))
    assert chains["sampled"] == chains["freed"] == [own, own]


def test_an_override_gets_its_own_chain_and_frees_it_even_when_abandoned():
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    stream = generate(ctx, 0, 3, out=GenerationAccumulator(), sampling=Sampling(temperature=0.3))
    next(stream)
    stream.close()
    assert chains["sampled"] == [("own", 0.3)]
    assert chains["freed"] == [("own", 0.3)]


def test_a_withheld_token_gets_a_chain_of_its_own_that_never_draws_it():
    chains = {"sampled": [], "freed": []}
    ctx = _stub_context(chains)
    list(generate(ctx, 0, 3, out=GenerationAccumulator(),
                      boundary=CallBoundary(withheld_tokens=(42,))))
    assert chains["sampled"][0] == ("own", Config().temperature, 42)
    assert chains["freed"] == [chains["sampled"][0]]


def test_a_chain_a_stage_fails_to_join_is_freed():
    def refuse(*args):
        raise RuntimeError("the stage was refused")

    freed = []
    ctx = object.__new__(KVContext)
    ctx._vocab = None
    ctx._backend = SimpleNamespace(sampler_chain=lambda: "chain", add_withheld=refuse,
                                   free_sampler=freed.append)
    with pytest.raises(RuntimeError, match="refused"):
        ctx._build_sampler(Config(), withheld=(42,))
    assert freed == ["chain"]


def test_the_withheld_stage_biases_each_token_to_never():
    added = []
    backend = Backend.__new__(Backend)
    backend._llama_cpp = llama_cpp
    backend._vocab_n_tokens = lambda vocab: 100
    backend._init_logit_bias = lambda n_vocab, n, biases: (n_vocab, [(b.token, b.bias) for b in biases[:n]])
    backend._chain_add = lambda chain, stage: added.append(stage)
    backend.add_withheld("chain", "vocab", (7, 9))
    assert added == [(100, [(7, float("-inf")), (9, float("-inf"))])]
