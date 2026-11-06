"""Persistence layer tests using the model-free FakeContext."""

import pytest

from llama_chat import ChatWrapper, Config, InMemoryStore, PersistentChat
from tests.fake_context import FakeContext


def _config(**kw):
    return Config(**kw)


# ----- core on_message hook ----------------------------------------------
def test_on_message_fires_user_then_assistant_on_a_request():
    fake = FakeContext(gen_tokens=3, gen_text="xyz")
    seen = []
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    w.reply_without_tools("hi")
    assert seen == [("user", "hi"), ("assistant", "xyz")]


def test_on_message_captures_the_partial_reply_of_an_early_close():
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    seen = []
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    stream = w.reply_without_tools_stream("hi")
    next(stream)          # consume one delta
    stream.close()        # stop early
    # User persisted, plus the partial assistant reply that reached the cache.
    assert seen[0] == ("user", "hi")
    assert seen[1][0] == "assistant"
    assert len(seen[1][1]) < 8  # partial, not the full "abcdefgh"


def test_on_message_fires_for_inject():
    fake = FakeContext(gen_tokens=2)
    seen = []
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    w.inject("some retrieved context")
    assert seen == [("user", "some retrieved context")]


def test_on_message_not_fired_for_begin_replay():
    fake = FakeContext(gen_tokens=2)
    seen = []
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s", [("user", "old"), ("assistant", "older")])  # replay -> no hook
    assert seen == []


def test_default_wrapper_has_no_hook():
    # on_message defaults to None; existing behavior is unchanged.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.reply_without_tools("hi")  # must not raise
    assert w.snapshot()[-1]["role"] == "assistant"


# ----- PersistentChat ----------------------------------------------------
def test_persistent_chat_records_turns():
    fake = FakeContext(gen_tokens=3, gen_text="xyz")
    store = InMemoryStore()
    chat = PersistentChat(store, _config(context_size=500, evict_above_fraction=1.0), context=fake)
    chat.begin("conv-1", "s")
    chat.reply_without_tools("hi")
    chat.reply_without_tools("again")
    assert store.load("conv-1") == [
        ("user", "hi"), ("assistant", "xyz"),
        ("user", "again"), ("assistant", "xyz"),
    ]


def test_persistent_chat_persists_inject():
    fake = FakeContext(gen_tokens=2, gen_text="rr")
    store = InMemoryStore()
    chat = PersistentChat(store, _config(context_size=1000, evict_above_fraction=1.0), context=fake)
    chat.begin("conv-1", "s")
    chat.inject("doc: deadline is Friday")
    chat.reply_without_tools("when?")
    assert store.load("conv-1") == [
        ("user", "doc: deadline is Friday"),
        ("user", "when?"), ("assistant", "rr"),
    ]


def test_persistent_chat_reloads_history_on_begin():
    store = InMemoryStore()

    fake1 = FakeContext(gen_tokens=2, gen_text="ok")
    s1 = PersistentChat(store, _config(context_size=1000, evict_above_fraction=1.0), context=fake1)
    s1.begin("conv-1", "s")
    s1.reply_without_tools("first question")

    # A fresh session on the same store + id reloads the prior turns.
    fake2 = FakeContext(gen_tokens=2, gen_text="ok")
    s2 = PersistentChat(store, _config(context_size=1000, evict_above_fraction=1.0), context=fake2)
    s2.begin("conv-1", "s")
    roles = [m["role"] for m in s2.snapshot()]
    assert roles == ["system", "user", "assistant"]


def test_persistent_chat_isolates_conversations():
    fake = FakeContext(gen_tokens=2, gen_text="rr")
    store = InMemoryStore()
    chat = PersistentChat(store, _config(context_size=1000, evict_above_fraction=1.0), context=fake)

    chat.begin("conv-a", "s")
    chat.reply_without_tools("a-question")
    chat.begin("conv-b", "s")
    chat.reply_without_tools("b-question")

    assert store.load("conv-a") == [("user", "a-question"), ("assistant", "rr")]
    assert store.load("conv-b") == [("user", "b-question"), ("assistant", "rr")]


def test_persistent_chat_requires_begin():
    """And refuses before prefilling, not on the way to the store.

    Noticed there, the request would already be in the cache, with the reply it
    never generated missing from both the cache and the store.
    """
    fake = FakeContext(gen_tokens=2)
    chat = PersistentChat(InMemoryStore(), _config(context_size=500, evict_above_fraction=1.0), context=fake)
    with pytest.raises(RuntimeError, match="begin"):
        chat.reply_without_tools("hi")
    assert fake.cache == []
    assert chat.snapshot() == []


def test_a_store_that_cannot_append_is_refused_where_it_is_adopted():
    """A store is written to mid-turn, which is far too late to find this out."""
    class ReadOnly:
        def load(self, conversation_id):
            return []

    with pytest.raises(TypeError, match="append"):
        PersistentChat(ReadOnly(), _config(context_size=500), context=FakeContext())


def test_a_store_that_cannot_replace_its_last_turn_is_refused_where_it_is_adopted():
    class AppendOnly:
        def load(self, conversation_id):
            return []

        def append(self, conversation_id, role, text):
            pass

    with pytest.raises(TypeError, match="replace_last"):
        PersistentChat(AppendOnly(), _config(context_size=500), context=FakeContext())


def test_persistent_chat_failed_begin_selects_no_conversation():
    # begin fails (system prompt over threshold) -> no conversation selected,
    # so a later request cannot persist into a never-loaded conversation.
    fake = FakeContext(gen_tokens=2)
    chat = PersistentChat(InMemoryStore(), _config(context_size=1000, evict_above_fraction=0.05), context=fake)
    with pytest.raises(ValueError):
        chat.begin("conv-1", "x" * 100)
    with pytest.raises(RuntimeError, match="begin"):
        chat.reply_without_tools("hi")


def test_in_memory_store_roundtrip():
    store = InMemoryStore()
    assert store.load("x") == []
    store.append("x", "user", "hello")
    store.append("x", "assistant", "hi")
    assert store.load("x") == [("user", "hello"), ("assistant", "hi")]


# ----- tools through the facade -------------------------------------------
def _tool_chat(store):
    fake = FakeContext(calls_tools=True, gen_text="18 Grad")
    return PersistentChat(store, _config(context_size=800), context=fake), fake


def _weather():
    from llama_chat import ToolRegistry
    registry = ToolRegistry()
    registry.register(lambda city: {"celsius": 18}, "Wetter.", name="weather",
                      parameters={"type": "object",
                                  "properties": {"city": {"type": "string"}}})
    return registry


def test_a_tool_turn_persists_only_what_a_reader_saw():
    """The call and the result are not messages, so the store never sees them."""
    store = InMemoryStore()
    chat, fake = _tool_chat(store)
    chat.begin("c1", "persona", tools=_weather())
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    reply = chat.reply_with_tools("wetter?").text

    assert store.load("c1") == [("user", "wetter?"), ("assistant", reply)]
    assert not chat.awaiting_tool_result


def test_a_tool_turn_reloads_as_an_ordinary_exchange():
    """What comes back on the next begin is the conversation, not its plumbing."""
    store = InMemoryStore()
    chat, fake = _tool_chat(store)
    chat.begin("c1", "persona", tools=_weather())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    chat.reply_with_tools("wetter?")

    chat.begin("c1", "persona", tools=_weather())     # same conversation again
    assert [m["role"] for m in chat.snapshot()] == ["system", "user", "assistant"]


def test_the_manual_loop_works_through_the_facade():
    store = InMemoryStore()
    chat, fake = _tool_chat(store)
    chat.begin("c1", "persona")
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    events = list(chat.reply_until_tool("wetter?"))
    assert chat.awaiting_tool_result
    chat.inject_tool_result(events[-1].name, {"celsius": 18})
    fake.call_script = []
    list(chat.resume_until_tool())

    assert not chat.awaiting_tool_result
    assert [r for r, _ in store.load("c1")] == ["user", "assistant"]


def test_every_delegation_reaches_the_wrapper():
    """A facade of one-line forwards fails by dropping an argument, silently.

    Each assertion is chosen so a wrong attribute or a lost parameter changes
    the answer: reaching the line is not the same as forwarding correctly.
    """
    import threading


    from llama_chat import ToolRegistry, ToolSpec

    ran = []
    registry = ToolRegistry()
    registry.register(lambda city: ran.append(city) or {"celsius": 18}, "Wetter.",
                      name="weather",
                      parameters={"type": "object",
                                  "properties": {"city": {"type": "string"}}})

    store = InMemoryStore()
    # can_shift False while can_call_tools is True, so confusing the two shows.
    fake = FakeContext(calls_tools=True, gen_text="18 Grad", can_shift=False)
    with PersistentChat(store, _config(context_size=800), context=fake) as chat:
        inner = chat._chat
        chat.begin("c1", "persona")

        assert chat.can_call_tools is True
        assert chat.can_shift is False
        assert chat.busy is False
        assert chat.awaiting_tool_result is False
        assert chat.reply_open is False
        assert chat.total_tokens == inner.total_tokens
        # Distinct values, so forwarding one under another's name shows.
        assert chat.context_size == 800
        assert chat.evict_above_tokens == 600
        assert chat.max_history_messages is None

        # A timeout that is actually forwarded gives up while the lock is held.
        held = threading.Event()
        def hold():
            with inner._lock:
                held.set()
                threading.Event().wait(0.3)
        keeper = threading.Thread(target=hold, daemon=True)
        keeper.start()
        held.wait(1)
        assert chat.wait_idle(0.05) is False
        keeper.join()
        assert chat.wait_idle(0.5) is True

        assert "".join(chat.reply_without_tools_stream("hallo")) == "18 Gr"

        # The registry has to arrive, or the call is refused instead of run.
        fake.call_script = ["fn:weather{city:'Bonn'}"]
        assert "".join(chat.reply_with_tools_stream("wetter?", registry)) == "18 Gr"
        assert ran == ["Bonn"]

        before = chat.total_tokens
        assert chat.declare_unknown_tools(
            [ToolSpec("clock", "Zeit.", {"type": "object", "properties": {}})]) == 0
        assert chat.total_tokens > before        # the declaration was prefilled

    assert [r for r, _ in store.load("c1")] == ["user", "assistant"] * 2


def test_a_cut_reply_is_stored_as_it_was_cut_and_comes_back_so():
    store = InMemoryStore()
    config = _config(context_size=500, evict_above_fraction=1.0)
    chat = PersistentChat(store, config, context=FakeContext(gen_tokens=8,
                                                             gen_text="Eins. Zw"))
    chat.begin("conv-1", "s")
    chat.reply_without_tools("hi")
    chat.truncate_last_reply("Eins.")
    assert store.load("conv-1") == [("user", "hi"), ("assistant", "Eins.")]
    again = PersistentChat(store, config, context=FakeContext())
    again.begin("conv-1", "s")
    assert again.snapshot()[-1]["text"] == "Eins."


def test_a_refused_cut_leaves_the_stored_turn_alone():
    store = InMemoryStore()
    chat = PersistentChat(store, _config(context_size=500, evict_above_fraction=1.0),
                          context=FakeContext(gen_tokens=5, gen_text="Hallo"))
    chat.begin("conv-1", "s")
    chat.reply_without_tools("hi")
    with pytest.raises(ValueError):
        chat.truncate_last_reply("Tschüss")
    assert store.load("conv-1")[-1] == ("assistant", "Hallo")


def test_a_reply_is_stored_without_its_thought_and_comes_back_so():
    store = InMemoryStore()
    config = _config(context_size=500, evict_above_fraction=1.0)
    thoughtful = "<thought>Denken</thought>Antwort."
    chat = PersistentChat(store, config, context=FakeContext(
        can_reason=True, calls_tools=True, gen_tokens=len(thoughtful), gen_text=thoughtful))
    chat.begin("conv-1", "s", reasoning=True)
    chat.reply_without_tools("hi")
    assert store.load("conv-1") == [("user", "hi"), ("assistant", "Antwort.")]
    again = PersistentChat(store, config, context=FakeContext())
    again.begin("conv-1", "s")
    assert again.snapshot()[-1]["text"] == "Antwort."
