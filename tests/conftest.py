"""Every wrapper a test builds on a model-free cache ends the test with that cache as it holds it, as the template renders the conversation, and with every turn kept as generated in the place of the template's version of it."""

import pytest

from llama_chat.wrapper import ChatWrapper

from .wrappers import assert_cache_reads_as_rendered


@pytest.fixture(autouse=True)
def _the_cache_reads_as_the_template_renders_the_conversation(monkeypatch):
    built = []
    build = ChatWrapper.__init__

    def recording(self, *args, **kwargs):
        build(self, *args, **kwargs)
        if hasattr(self._context, "cache") and hasattr(self._context, "text_of"):
            built.append(self)

    monkeypatch.setattr(ChatWrapper, "__init__", recording)
    yield built
    for wrapper in built:
        if not wrapper._closed:
            assert_cache_reads_as_rendered(wrapper)
