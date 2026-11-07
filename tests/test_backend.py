"""What the llama.cpp-facing layer settles with no model loaded.

The calls are stubbed, because what is pinned is this package's own criteria
rather than llama.cpp's answers: what counts as one of the model's own tags,
and which cache edits are performed while loading rather than at the first
eviction. The first is pinned here rather than through the fake context used
everywhere else, which answers that question itself and so cannot test it. See
the README, "Tool calling".
"""

from pathlib import Path
from types import SimpleNamespace

import llama_cpp
import pytest

import llama_chat

from llama_chat._backend import Backend
from llama_chat.config import KV_CACHE_TYPES
from llama_chat.context import SEQ, GenerationAccumulator, KVContext


def _backend(attr):
    """A Backend whose attribute lookup returns ``attr`` for every token."""
    backend = Backend.__new__(Backend)
    backend._get_attr = lambda vocab, token: attr
    backend._is_control = None
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


def test_without_the_attribute_lookup_it_falls_back_to_control():
    backend = Backend.__new__(Backend)
    backend._get_attr = None
    backend._is_control = lambda vocab, token: token == 7
    assert backend.is_special(None, 7)
    assert not backend.is_special(None, 8)


def test_without_either_symbol_nothing_is_a_tag():
    """A build offering neither answer leaves every token counting as text.

    Tool calling is then off, and a turn terminator no longer validates, which
    is the honest outcome: nothing here can tell a tag from a word.
    """
    backend = Backend.__new__(Backend)
    backend._get_attr = None
    backend._is_control = None
    assert not backend.is_special(None, 7)


# ----- the cache edits eviction rests on ---------------------------------
def test_every_cache_edit_is_performed_while_loading():
    """A build missing one of them says so at load, not at the first eviction.

    Removal and the shift that closes the gap are reached only once history
    crosses the threshold, a whole conversation after the model was loaded.
    Each is performed empty, so proving it costs nothing and changes nothing.
    """
    for can_shift in (True, False):
        performed = []
        context = object.__new__(KVContext)
        context._llama_context = "the context"
        context._can_shift = can_shift
        context._backend = SimpleNamespace(
            kv_seq_rm=lambda *args: performed.append(("seq_rm",) + args),
            kv_seq_add=lambda *args: performed.append(("seq_add",) + args),
            kv_clear=lambda *args: performed.append(("clear",) + args),
        )

        context._verify_cache_edits()

        # A cache that cannot shift is never shifted, and is not asked to.
        assert performed == [("seq_rm", "the context", SEQ, 0, 0)] \
            + ([("seq_add", "the context", SEQ, 0, 0, 0)] if can_shift else []) \
            + [("clear", "the context")]


def _probed_context(reports_shift, cuts_tail):
    """A context over a backend reporting a shift and taking or refusing a cut at its end."""
    performed = []
    context = object.__new__(KVContext)
    context._llama_context, context._vocab, context._max_batch_tokens = "ctx", None, 8
    context._backend = SimpleNamespace(
        can_shift=lambda ctx: reports_shift,
        token_eos=lambda vocab: 2,
        decode=lambda ctx, tokens, pos, *rest: performed.append(("decode", tokens, pos)),
        # Every removal that stops before the end succeeds, as a running state's does.
        kv_seq_rm=lambda ctx, seq, p0, p1: performed.append(("seq_rm", p0, p1)) or (
            cuts_tail or p1 >= 0),
        kv_seq_add=lambda ctx, seq, p0, p1, d: performed.append(("seq_add", p0, p1, d)),
        kv_clear=lambda ctx: performed.append(("clear",)),
    )
    context._settle_cache_edits()
    return context, performed


def test_a_cache_that_refuses_a_cut_at_its_end_is_never_shifted_whatever_it_reports():
    """A hybrid cache reports the shift its attention part can do; its running state cannot."""
    context, performed = _probed_context(reports_shift=True, cuts_tail=False)
    assert not context.can_shift and not context.can_cut_tail
    assert not any(edit[0] == "seq_add" for edit in performed)    # not even at load
    assert performed[-1] == ("clear",)


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


def test_a_cache_edit_counts_from_either_api():
    """What install.sh asks, having no model to prove the edits against.

    It used to reach into these attributes and repeat the pairing itself, which
    goes stale the next time llama.cpp renames one of them.
    """
    present = object()

    def backend(**offered):
        b = Backend.__new__(Backend)
        for attr in ("_get_memory", "_mem_clear", "_mem_rm", "_mem_add",
                     "_kv_clear", "_kv_rm", "_kv_add"):
            setattr(b, attr, offered.get(attr))
        return b

    memory_api = backend(_get_memory=present, _mem_clear=present,
                         _mem_rm=present, _mem_add=present)
    legacy_api = backend(_kv_clear=present, _kv_rm=present, _kv_add=present)
    no_shift = backend(_kv_clear=present, _kv_rm=present)
    no_handle = backend(_mem_clear=present, _mem_rm=present, _mem_add=present)

    assert memory_api.can_edit_cache
    assert legacy_api.can_edit_cache
    assert not no_shift.can_edit_cache      # closing the gap is an edit too
    assert not no_handle.can_edit_cache     # no way to reach the memory ops


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
    context._llama_context = "the context"
    context._backend = SimpleNamespace(kv_seq_rm=lambda *args: False)
    with pytest.raises(RuntimeError, match="refused"):
        context.remove_from(3)


@pytest.mark.parametrize("answer, taken", [(True, True), (False, False), (None, True)])
def test_a_removal_reports_the_caches_answer_and_takes_none_for_yes(answer, taken):
    backend = object.__new__(Backend)
    backend._mem = lambda ctx: "the memory"
    backend._mem_rm = lambda *args: answer
    assert backend.kv_seq_rm("the context", 0, 3, -1) is taken


def _tag_context(tags):
    """A context whose vocabulary splits text on spaces and holds ``tags`` as tags."""
    context = object.__new__(KVContext)
    context._vocab = None
    context._backend = SimpleNamespace(
        tokenize=lambda vocab, text, add_special, parse_special: text.split(),
        is_special=lambda vocab, token: token in tags)
    return context


@pytest.mark.parametrize("text, made_of_tags", [
    ("<end>", True), ("<response_end> <end>", True), ("<end> plain", False),
    ("plain", False), ("", False)])
def test_a_text_is_made_of_tags_where_every_token_is_one(text, made_of_tags):
    assert _tag_context({"<end>", "<response_end>"}).tokenizes_to_tags(text) is made_of_tags


def test_a_reply_ends_at_its_turn_end_token_which_is_neither_cached_nor_shown():
    """A model whose closing takes two tags writes the first itself, as text would be."""
    sampled = iter(["Pa", "ris", "<response_end>", "<end>"])
    decoded = []
    context = object.__new__(KVContext)
    context._llama_context, context._vocab, context._max_batch_tokens = "ctx", None, 8
    context._sampler = "sampler"
    context._backend = SimpleNamespace(
        sampler_reset=lambda sampler: None,
        sample=lambda sampler, ctx: next(sampled),
        is_eog=lambda vocab, token: token == "<end>",
        decode=lambda ctx, tokens, pos, *rest: decoded.extend(tokens),
        token_to_piece_bytes=lambda vocab, token: token.encode())
    out = GenerationAccumulator()
    shown = "".join(context.generate(
        0, 10, [], out=out, ends_turn=lambda token: token == "<response_end>"))
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
    assert context.tokenize_fragment(prefix, content, suffix) == context._backend.tokenize(
        None, prefix + content + suffix, False, True)


def test_a_tag_typed_inside_a_message_stays_text():
    tokens = _merging_context().tokenize_fragment("<t>", "a<e>b", "<e>")
    assert tokens.count("TAG<e>") == 1 and tokens[-1] == "TAG<e>"


def test_the_beginning_of_a_conversation_is_attached_to_the_first_message():
    assert _merging_context().tokenize_fragment("lead ", "word", "", add_special=True)[0] == "BOS"


def test_tokens_generated_in_other_pieces_come_back_as_their_text_tokenizes():
    """A word the model wrote as "Nut" + "zer" holds the one token its text gives."""
    context = object.__new__(KVContext)
    context._vocab = None
    context._backend = SimpleNamespace(
        is_special=lambda vocab, token: token.startswith("TAG"),
        token_to_piece_bytes=lambda vocab, token: token.encode(),
        tokenize=lambda vocab, text, add_special, parse_special: text.split(" "))
    retokenized = context.retokenized(["Nut", "zer", "TAG<e>", "a", " b"])
    assert retokenized == ["Nutzer", "TAG<e>", "a", "b"]


def test_text_cut_inside_a_character_keeps_the_tokens_it_was_generated_as():
    context = object.__new__(KVContext)
    context._vocab = None
    context._backend = SimpleNamespace(
        is_special=lambda vocab, token: False,
        token_to_piece_bytes=lambda vocab, token: token,
        tokenize=lambda vocab, text, add_special, parse_special: [text.encode()])
    assert context.retokenized([b"a", "ä".encode()[:1]]) == [b"a", "ä".encode()[:1]]
