"""Every wrapper a test builds on a model-free cache ends the test with that cache as it holds it, as the template renders the conversation, and with every turn kept as generated in the place of the template's version of it, spelling no result written after it; and with a conversation that is its two history forms, which begin takes back. Every error the library makes during a test says what went wrong.

A test that shows a fault of the library the user has yet to decide on is marked ``known_library_flaw``; the check must then still find that fault.

A test that asks for ``clock`` sets the time a template that reads the clock is rendered at.
Every test ends within its time limit, or the process stops and prints where
each thread stood.
"""

import datetime
import faulthandler
import os
import sys

import pytest

from llama_chat.errors import ArgumentTypeError, EngineError, InternalError, RefusalError
from llama_chat.wrapper import ChatWrapper

from llama_chat.testing.fake_context import RecordedTemplate
from .internals import context_of, is_closed
from .wrappers import (assert_cache_reads_as_rendered, assert_conversation_is_its_forms,
                       assert_kept_turns_hold_no_later_result)


# How long a test may run before it counts as hung; a test that needs longer says so with
# ``@pytest.mark.time_limit_s(n)``.
_TIME_LIMIT_S = 60


def pytest_configure(config):
    """Have the library render each template through :class:`RecordedTemplate`, compiled once per text and each render counted; with ``LLAMA_CHAT_CHECK_EVERY_ACTION=1``, check every action's cache against a fresh begin; and keep nothing of Hypothesis's on disk."""
    from hypothesis import configuration
    # A home it cannot write to: it then keeps neither examples nor what it read of the code.
    configuration.set_hypothesis_home_dir(os.devnull)
    from llama_chat import resolved_template
    resolved_template.ModelTemplate = RecordedTemplate
    if os.environ.get("LLAMA_CHAT_CHECK_EVERY_ACTION") == "1":
        from dataclasses import replace
        from llama_chat.rendered_cache import RenderedCache
        build = RenderedCache.__init__

        def checking(self, context, templates, settings):
            from llama_chat.cache_check import _ReadsTokensOnly
            checks = not isinstance(context, _ReadsTokensOnly)     # the fresh begin, not again
            build(self, context, templates, replace(settings, check_cache=checks))
        RenderedCache.__init__ = checking
    config.addinivalue_line("markers", "time_limit_s(seconds): the test may run this long")
    config.addinivalue_line("markers", "known_library_flaw(what): a fault of the library the "
                                       "check after the test must still find, until it is fixed")
    config.addinivalue_line("markers", "slow: a long run, left out unless asked for with -m slow")
    config.addinivalue_line("markers", "gpu: runs real model files on the GPU")
    config.addinivalue_line("markers", "counts_work: counts the library's own renders or calls, "
                                       "which the check after every action adds to")
    config.addinivalue_line("markers", "timed: bounds the processor time a test takes, which "
                                       "mutation testing slows")


def pytest_collection_modifyitems(config, items):
    """Leave out the tests that count the library's work where every action is checked, which adds work of its own."""
    if os.environ.get("LLAMA_CHAT_CHECK_EVERY_ACTION") != "1":
        return
    skip = pytest.mark.skip(reason="counts the library's work, which the check adds to")
    for item in items:
        if item.get_closest_marker("counts_work"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _within_its_time_limit(request):
    """Stop the process, printing every thread's stack, once the test runs past its limit, so that a hang is a failure that shows where it hung."""
    marker = request.node.get_closest_marker("time_limit_s")
    limit_s = marker.args[0] if marker else _TIME_LIMIT_S
    faulthandler.dump_traceback_later(limit_s, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _the_cache_reads_as_the_template_renders_the_conversation(request, monkeypatch):
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
    # A fault of the library the user has yet to decide on: the check must still find it.
    flaw, shown = request.node.get_closest_marker("known_library_flaw"), []
    for wrapper, context in zip(built, contexts):
        if not is_closed(wrapper):
            # The work after a reply, on its thread where a test asks.
            assert wrapper.wait_idle(timeout_s=5), "the work after a reply did not end"
            assert wrapper._conversation._found_fault is None, "check_cache found a fault"
            assert_cache_reads_as_rendered(wrapper, context)
            try:
                assert_kept_turns_hold_no_later_result(wrapper, context)
            except AssertionError:
                if flaw is None:
                    raise
                shown.append(wrapper)
            assert_conversation_is_its_forms(wrapper)
    if flaw is not None:
        assert shown, f"the known flaw no longer shows, so its mark can go: {flaw.args[0]}"


@pytest.fixture(autouse=True)
def _every_library_error_says_what_went_wrong(monkeypatch):
    """Rule 11: every error the library makes during the test carries a message of words, not an empty one or a placeholder such as ``None``; those the test's own stand-ins make are left out."""
    made = []
    for kind, build in ((RefusalError, RefusalError.__init__), (ArgumentTypeError, TypeError.__init__),
                        (EngineError, RuntimeError.__init__), (InternalError, RuntimeError.__init__)):
        def recording(self, *args, _build=build, **kwargs):
            _build(self, *args, **kwargs)
            maker = sys._getframe(1)
            while maker.f_code.co_name == "__init__":     # a subclass's own, which builds on this one
                maker = maker.f_back
            module = maker.f_globals.get("__name__", "")
            if module.startswith("llama_chat.") and not module.startswith("llama_chat.testing"):
                made.append(self)
        monkeypatch.setattr(kind, "__init__", recording)
    yield
    for error in made:
        message = str(error).strip()
        assert len(message.split()) >= 3 and message != "None", \
            f"a {type(error).__name__} says only {message!r}"


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
