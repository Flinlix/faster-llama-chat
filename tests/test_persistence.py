"""Persistence layer tests using the model-free FakeContext."""

import pytest

from llama_chat import ChatWrapper, Config, InMemoryStore, PersistentChat
from tests.fake_context import FakeContext


def _cfg(**kw):
    return Config(**kw)


# ----- core on_message hook ----------------------------------------------
def test_on_message_fires_user_then_assistant_on_request():
    fake = FakeContext(gen_len=3, gen_text="xyz")
    seen = []
    w = ChatWrapper(_cfg(context_size=500, eviction_threshold=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    w.request("hi")
    assert seen == [("user", "hi"), ("assistant", "xyz")]


def test_on_message_captures_bargein_partial():
    fake = FakeContext(gen_len=8, gen_text="abcdefgh")
    seen = []
    w = ChatWrapper(_cfg(context_size=500, eviction_threshold=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    gen = w.stream("hi")
    next(gen)          # consume one delta
    gen.close()        # barge-in
    # User persisted, plus the partial assistant reply that reached the cache.
    assert seen[0] == ("user", "hi")
    assert seen[1][0] == "assistant"
    assert len(seen[1][1]) < 8  # partial, not the full "abcdefgh"


def test_on_message_fires_for_inject():
    fake = FakeContext(gen_len=2)
    seen = []
    w = ChatWrapper(_cfg(context_size=1000, eviction_threshold=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    w.inject("some retrieved context")
    assert seen == [("user", "some retrieved context")]


def test_on_message_not_fired_for_begin_replay():
    fake = FakeContext(gen_len=2)
    seen = []
    w = ChatWrapper(_cfg(context_size=1000, eviction_threshold=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s", [("user", "old"), ("assistant", "older")])  # replay -> no hook
    assert seen == []


def test_default_wrapper_has_no_hook():
    # on_message defaults to None; existing behavior is unchanged.
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(context_size=500, eviction_threshold=1.0), context=fake)
    w.begin("s")
    w.request("hi")  # must not raise
    assert w.snapshot()[-1]["role"] == "assistant"


# ----- PersistentChat ----------------------------------------------------
def test_persistent_chat_records_turns():
    fake = FakeContext(gen_len=3, gen_text="xyz")
    store = InMemoryStore()
    chat = PersistentChat(store, _cfg(context_size=500, eviction_threshold=1.0), context=fake)
    chat.begin("conv-1", "s")
    chat.request("hi")
    chat.request("again")
    assert store.load("conv-1") == [
        ("user", "hi"), ("assistant", "xyz"),
        ("user", "again"), ("assistant", "xyz"),
    ]


def test_persistent_chat_persists_inject():
    fake = FakeContext(gen_len=2, gen_text="rr")
    store = InMemoryStore()
    chat = PersistentChat(store, _cfg(context_size=1000, eviction_threshold=1.0), context=fake)
    chat.begin("conv-1", "s")
    chat.inject("doc: deadline is Friday")
    chat.request("when?")
    assert store.load("conv-1") == [
        ("user", "doc: deadline is Friday"),
        ("user", "when?"), ("assistant", "rr"),
    ]


def test_persistent_chat_reloads_history_on_begin():
    store = InMemoryStore()

    fake1 = FakeContext(gen_len=2, gen_text="ok")
    s1 = PersistentChat(store, _cfg(context_size=1000, eviction_threshold=1.0), context=fake1)
    s1.begin("conv-1", "s")
    s1.request("first question")

    # A fresh session on the same store + id reloads the prior turns.
    fake2 = FakeContext(gen_len=2, gen_text="ok")
    s2 = PersistentChat(store, _cfg(context_size=1000, eviction_threshold=1.0), context=fake2)
    s2.begin("conv-1", "s")
    roles = [m["role"] for m in s2.snapshot()]
    assert roles == ["system", "user", "assistant"]


def test_persistent_chat_isolates_conversations():
    fake = FakeContext(gen_len=2, gen_text="rr")
    store = InMemoryStore()
    chat = PersistentChat(store, _cfg(context_size=1000, eviction_threshold=1.0), context=fake)

    chat.begin("conv-a", "s")
    chat.request("a-question")
    chat.begin("conv-b", "s")
    chat.request("b-question")

    assert store.load("conv-a") == [("user", "a-question"), ("assistant", "rr")]
    assert store.load("conv-b") == [("user", "b-question"), ("assistant", "rr")]


def test_persistent_chat_requires_begin():
    fake = FakeContext(gen_len=2)
    chat = PersistentChat(InMemoryStore(), _cfg(context_size=500, eviction_threshold=1.0), context=fake)
    with pytest.raises(RuntimeError, match="begin"):
        chat.request("hi")


def test_persistent_chat_failed_begin_selects_no_conversation():
    # begin fails (system prompt over threshold) -> no conversation selected,
    # so a later request cannot persist into a never-loaded conversation.
    fake = FakeContext(gen_len=2)
    chat = PersistentChat(InMemoryStore(), _cfg(context_size=1000, eviction_threshold=0.05), context=fake)
    with pytest.raises(ValueError):
        chat.begin("conv-1", "x" * 100)
    with pytest.raises(RuntimeError, match="begin"):
        chat.request("hi")


def test_in_memory_store_roundtrip():
    store = InMemoryStore()
    assert store.load("x") == []
    store.append("x", "user", "hello")
    store.append("x", "assistant", "hi")
    assert store.load("x") == [("user", "hello"), ("assistant", "hi")]
