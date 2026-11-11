"""Every wrapper a test builds on a model-free cache ends the test with that cache as it holds it, as the template renders the conversation, and with every turn kept as generated in the place of the template's version of it.

A test that asks for ``clock`` sets the time a template that reads the clock is rendered at.
"""

import datetime

import pytest

from llama_chat.wrapper import ChatWrapper

from .internals import context_of, is_closed
from .wrappers import assert_cache_reads_as_rendered


@pytest.fixture(autouse=True)
def _the_cache_reads_as_the_template_renders_the_conversation(monkeypatch):
    built, contexts = [], []
    build = ChatWrapper.__init__

    def recording(self, *args, **kwargs):
        build(self, *args, **kwargs)
        context = context_of(self)
        if hasattr(context, "cache") and hasattr(context, "text_of"):
            built.append(self)
            contexts.append(context)

    monkeypatch.setattr(ChatWrapper, "__init__", recording)
    yield built
    for wrapper, context in zip(built, contexts):
        if not is_closed(wrapper):
            assert_cache_reads_as_rendered(wrapper, context)


@pytest.fixture
def clock(monkeypatch):
    """The time a template that reads the clock is rendered at, a minute before midnight until a test moves it."""
    import llama_cpp.llama_chat_format as chat_format
    today = [datetime.datetime(2026, 11, 7, 23, 59)]

    class _Clock:
        @staticmethod
        def now():
            return today[0]

    monkeypatch.setattr(chat_format, "datetime", _Clock)
    return today
