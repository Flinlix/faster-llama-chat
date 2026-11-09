"""Wrappers on the model-free cache that several test files build, and their consistency check."""

import dataclasses

from llama_chat.config import Config
from llama_chat.template import CallBody
from llama_chat.tokenizing import split_at_tags
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


def held_renders(context, messages, tools=None, reasoning=False) -> list[str]:
    """What a cache holding ``messages`` may read as: their render with the generation prompt or without, or decoded ahead of the next question up to where it goes.

    Rendered as the context states its template renders, with nothing the
    library read off the template.
    """
    from llama_chat.render_checks import STAND_IN as _STAND_IN
    render = lambda m, prompt: context.stated_render(       # noqa: E731
        m, tools, generation_prompt=prompt, reasoning=reasoning)
    found = [render(messages, False), render(messages, True)]
    ahead = render(messages + [{"role": "user", "content": _STAND_IN}], True)
    if ahead is not None and _STAND_IN in ahead:
        before = ahead[:ahead.index(_STAND_IN)]
        cut = len(before)
        for piece, is_tag in reversed(split_at_tags(context, before) or []):
            if is_tag:
                break
            cut -= len(piece)
        found.append(before[:cut])
    return [text for text in found if text is not None]


def assert_cache_reads_as_rendered(wrapper):
    """The cache holds what the wrapper holds it to be, and reads as the template renders the conversation around each kept turn.

    The cache is compared as it stands, unless a decode, an edit or a
    generation failed since it was last brought up: then it is brought back
    first, as the next action would bring it back.
    """
    conversation = wrapper._conversation
    memory = conversation._memory
    if not memory.is_known:
        memory.repair()
    assert wrapper._context.cache == memory.tokens, \
        "the cache no longer holds what the wrapper holds it to be"
    if not conversation.begun:
        return
    messages, text_of = conversation.messages, wrapper._context.text_of
    tools = [spec.as_declaration() for spec in conversation.tools] or None
    renders = held_renders(wrapper._context, messages, tools,
                           reasoning=conversation._state.reasoning)
    held = text_of(wrapper._context.cache)
    kept = [(text_of(k.tokens), text_of(k.tokens[:1]), text_of(k.tokens[-1:]),
             (k.last.get("content") or "").strip()) for k in conversation._state.kept]
    assert any(_reads_as(held, render, kept) for render in renders), \
        "the cache does not read as the template renders the conversation"
    for render in (False, True):
        split = conversation._rendering.split(messages, render)
        if split is not None:
            for kept_turn in conversation._state.kept:
                placed = conversation._rendering.place(kept_turn, messages, split)
                assert placed is not None, "a turn kept as generated was lost"
                assert placed[0] < placed[1], \
                    "a turn kept as generated stands in for none of the template's text"


def _reads_as(held: str, render: str, kept: list[tuple[str, str, str, str]]) -> bool:
    """Whether ``held`` is ``render`` with each kept turn in place of the template's version of it, everything around them exactly as rendered.

    ``kept`` holds each turn's text, the text of its first and of its last
    token, and the text of its last message. The template's version of a turn
    begins with the same token, and ends with the same token, with that
    message's text, or, for a turn whose call the template does not write, is
    where the kept turn begins; so it takes in none of the template's text
    beside it.
    """
    pieces, at = [], 0
    for turn, *_ in kept:
        found = held.find(turn, at)
        if found < 0:
            return False
        pieces.append(held[at:found])
        at = found + len(turn)
    pieces.append(held[at:])
    if not kept:
        return held == render
    if not render.startswith(pieces[0]):
        return False
    at = len(pieces[0])
    for n, ((turn, opening, closing, text), piece) in enumerate(zip(kept, pieces[1:])):
        if not render.startswith(opening, at):
            return False
        last = n == len(kept) - 1
        ends = [len(render) - len(piece)] if last else _occurrences(render, piece, at)
        ends = [end for end in ends if end >= at + len(opening) and render.startswith(piece, end)
                and (render[at:end].endswith(closing) or turn.startswith(render[at:end])
                     or bool(text) and render[at:end].rstrip().endswith(text))]
        if not ends:
            return False
        at = ends[0] + len(piece)
    return True


def _occurrences(text: str, piece: str, start: int) -> list[int]:
    """Every position from ``start`` on where ``piece`` begins in ``text``."""
    found, at = [], text.find(piece, start)
    while at >= 0:
        found.append(at)
        at = text.find(piece, at + 1)
    return found


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
    system, history = wrapper.conversation()
    found = [{"role": "system", "text": system}] if system else []
    for item in history:
        role, text = ("assistant", item.text) if isinstance(item, ToolTurn) else item
        found.append({"role": role, "text": text})
    return found
