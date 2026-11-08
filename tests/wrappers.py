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


def tool_wrapper(context_size=512, **kw):
    fake = FakeContext(calls_tools=True, **kw)
    return ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.9),
                       context=fake), fake


def cache_tokens(wrapper):
    """What the wrapper holds the cache to be, token by token."""
    return list(wrapper._conversation._memory.tokens)


def held_text(wrapper) -> str:
    """What the fake cache spells: the characters, the call tags and the generated text."""
    return wrapper._context.text_of(wrapper._context.cache)


def newest_turn(wrapper) -> str:
    """The cache's text from the newest user turn on."""
    text = held_text(wrapper)
    return text[text.rfind("<[turn>user"):]


def system_turn(wrapper) -> str:
    """The cache's text up to the end of the system turn."""
    text = held_text(wrapper)
    return text[:text.find("<turn]>") + len("<turn]>")]


def held_renders(template, messages, tools=None, **options) -> list[str]:
    """What a cache holding ``messages`` may read as: their render with the generation prompt or without, or decoded ahead of the next question up to where it goes."""
    from llama_chat.render_checks import STAND_IN as _STAND_IN
    render = lambda m, prompt: template.render(       # noqa: E731
        m, tools, generation_prompt=prompt, **options)
    found = [render(messages, False), render(messages, True)]
    ahead = render(messages + [{"role": "user", "content": _STAND_IN}], True)
    if ahead is not None and _STAND_IN in ahead:
        before = ahead[:ahead.index(_STAND_IN)]
        cut = len(before)
        for piece, is_tag in reversed(template._context.split_at_tags(before) or []):
            if is_tag:
                break
            cut -= len(piece)
        found.append(before[:cut])
    return [text for text in found if text is not None]


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
    """The physical cache holds what the wrapper holds it to be."""
    assert fake.cache == cache_tokens(wrapper)
    assert wrapper.total_tokens == len(fake.cache)


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


def rows(wrapper) -> list[dict]:
    """The conversation as ``{"role", "text"}`` rows, the system prompt first, read from what the wrapper publishes."""
    from llama_chat.tools import ToolTurn
    found = [{"role": "system", "text": wrapper.system_prompt}] if wrapper.system_prompt else []
    for item in wrapper.history():
        role, text = ("assistant", item.text) if isinstance(item, ToolTurn) else item
        found.append({"role": role, "text": text})
    return found
