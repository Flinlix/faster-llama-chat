"""What counts as one of the model's own tags, rather than as literal text.

No model is loaded: the attribute lookup is stubbed, because what is being
pinned is the criterion, not llama.cpp's answer. It is pinned here rather than
through the fake context used everywhere else, which answers this question
itself and so cannot test it. See the README, "Tool calling".
"""

import llama_cpp

from llama_chat._backend import Backend


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
