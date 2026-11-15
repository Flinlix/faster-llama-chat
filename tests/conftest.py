"""Every wrapper a test builds on a model-free cache ends the test with that cache as it holds it, as the template renders the conversation, and with every turn kept as generated in the place of the template's version of it; and with a conversation that is its two history forms, which begin takes back.

A test that asks for ``clock`` sets the time a template that reads the clock is rendered at.
"""

import datetime

import pytest

from llama_chat.wrapper import ChatWrapper

from .fake_context import RecordedTemplate
from .internals import context_of, is_closed
from .wrappers import assert_cache_reads_as_rendered, assert_conversation_is_its_forms


def pytest_configure(config):
    """Have the library render each template through :class:`RecordedTemplate`, compiled once per text and each render counted."""
    from llama_chat import resolved_template
    resolved_template.ModelTemplate = RecordedTemplate


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
            # The work after a reply, on its thread where a test asks.
            assert wrapper.wait_idle(timeout_s=5), "the work after a reply did not end"
            assert_cache_reads_as_rendered(wrapper, context)
            assert_conversation_is_its_forms(wrapper)


@pytest.fixture
def clock(monkeypatch):
    """The time a template that reads the clock is rendered at, a minute before midnight until a test moves it."""
    from llama_chat import model_template
    today = [datetime.datetime(2026, 11, 7, 23, 59)]

    class _Clock:
        @staticmethod
        def now():
            return today[0]

    monkeypatch.setattr(model_template, "datetime", _Clock)
    return today


@pytest.fixture(autouse=True)
def _the_work_after_a_reply_is_done_as_its_stream_ends(request, monkeypatch):
    """Done on the reader's thread, before the stream's end reaches it, so that a test reads the cache as that work left it, and its failures reach the reader; a test that asks for ``settles_in_the_background`` keeps the thread of its own."""
    if "settles_in_the_background" in request.fixturenames:
        return

    def at_once(wrapper):
        if wrapper._conversation.settle_due:
            with wrapper._lock:
                wrapper._conversation.settle()

    monkeypatch.setattr(ChatWrapper, "_start_settling", at_once)


@pytest.fixture
def settles_in_the_background():
    """The work after a reply runs on a thread of its own, as it does outside the tests."""
