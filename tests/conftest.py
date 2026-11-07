"""Every wrapper a test builds on the model-free cache ends the test with that cache equal to its message table."""

import pytest

from llama_chat.wrapper import ChatWrapper

from .fake_context import FakeContext


@pytest.fixture(autouse=True)
def _the_cache_holds_the_table(monkeypatch):
    built = []
    build = ChatWrapper.__init__

    def recording(self, *args, **kwargs):
        build(self, *args, **kwargs)
        if isinstance(self._context, FakeContext):
            built.append(self)

    monkeypatch.setattr(ChatWrapper, "__init__", recording)
    yield
    for wrapper in built:
        tokens = [t for m in wrapper._table.messages for t in m.token_ids]
        assert wrapper._context.cache == tokens, "the cache no longer holds the message table"
