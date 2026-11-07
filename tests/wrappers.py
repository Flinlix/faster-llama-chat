"""Wrappers on the model-free cache that several test files build, and their consistency check."""

import dataclasses

from llama_chat.config import Config
from llama_chat.template import CallBody
from llama_chat.tools import ToolRegistry
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import DEFAULT_FRAGMENTS, FakeContext


# The call_body the fake model's renders yield, spelled out so the parser tests
# read without a round trip through extraction.
FAKE_CALL_BODY = CallBody(name_close="{", key_open="", key_close=":", pair_sep=",",
                          body_close="}", empty_close="}", quote_open="'", quote_close="'")


def tool_wrapper(**kw):
    fake = FakeContext(calls_tools=True, **kw)
    return ChatWrapper(Config(context_size=512, evict_above_fraction=0.9),
                       context=fake), fake


def cache_tokens(wrapper):
    toks = []
    for m in wrapper._table.messages:
        toks.extend(m.token_ids)
    return toks


def weather_registry():
    registry = ToolRegistry()
    registry.register(lambda city: {"city": city, "celsius": 18},
                      "Wetter.", name="weather",
                      parameters={"type": "object",
                                  "properties": {"city": {"type": "string"}}})
    return registry


def fragments_with(**kw):
    """The default fragments with the given fields overridden."""
    return dataclasses.replace(DEFAULT_FRAGMENTS, **kw)


def begun_wrapper(**fake_kw):
    fake = FakeContext(**fake_kw)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    return w, fake


def assert_consistent(wrapper, fake):
    """The physical cache equals the concatenation of the message table."""
    assert fake.cache == cache_tokens(wrapper)
    snap = wrapper.snapshot()
    if snap:
        assert snap[-1]["pos_end"] == wrapper.total_tokens == len(fake.cache)


THOUGHTFUL = "<thought>Denken</thought>Antwort."


def reasoning_wrapper(reasoning=True, config=None, **fake_kw):
    fake_kw = {"gen_tokens": len(THOUGHTFUL), "gen_text": THOUGHTFUL, **fake_kw}
    fake = FakeContext(can_reason=True, calls_tools=True, **fake_kw)
    w = ChatWrapper(config or Config(context_size=500, evict_above_fraction=1.0),
                    context=fake)
    w.begin("s", reasoning=reasoning)
    return w, fake


def unbegun_tool_wrapper():
    fake = FakeContext(calls_tools=True, gen_tokens=7, gen_text="Gleich.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    return w, fake


# ----- what the fake template writes, read from its render ----------------------
_QUESTION = [{"role": "user", "content": "x"}]


def rendered_prompt(fake, reasoning=None) -> str:
    """The generation prompt as the fake template renders one after a question."""
    asked = fake.render_with_model_template(_QUESTION, reasoning=reasoning)
    return fake.render_with_model_template(_QUESTION, generation_prompt=True,
                                           reasoning=reasoning)[len(asked):]


def rendered_reply(fake, text, reasoning=None) -> str:
    """A past reply as the fake template renders one after a question, its tags included."""
    asked = fake.render_with_model_template(_QUESTION, reasoning=reasoning)
    return fake.render_with_model_template(
        _QUESTION + [{"role": "assistant", "content": text}], reasoning=reasoning)[len(asked):]


def rendered_close(fake, reasoning=None) -> str:
    """What the fake template writes after a reply's text to close its turn."""
    return rendered_reply(fake, "TEXT", reasoning).split("TEXT", 1)[1]
