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
from llama_chat.context import SEQ, KVContext


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
    performed = []
    context = object.__new__(KVContext)
    context._llama_context = "the context"
    context._backend = SimpleNamespace(
        kv_seq_rm=lambda *args: performed.append(("seq_rm",) + args),
        kv_seq_add=lambda *args: performed.append(("seq_add",) + args),
        kv_clear=lambda *args: performed.append(("clear",) + args),
    )

    context._verify_cache_edits()

    assert performed == [
        ("seq_rm", "the context", SEQ, 0, 0),
        ("seq_add", "the context", SEQ, 0, 0, 0),
        ("clear", "the context"),
    ]


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

    with pytest.warns(RuntimeWarning, match="penalties"):
        backend.add_penalties(object(), 64, 1.1, 0.0, 0.0)
    assert added == []


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
