"""Wrappers on the model-free cache that several test files build, and their consistency check."""

import dataclasses

from llama_chat.config import Config
from llama_chat.tokenizing import split_at_tags
from llama_chat.tools import ToolRegistry
from llama_chat.wrapper import ChatWrapper
from tests import internals
from tests.chat_templates import in_each_thought_form
from tests.fake_context import FakeContext
from tests.invented_templates import PLAIN, REASONING, TOOLS


def tool_wrapper(context_size=512, **kw):
    fake = FakeContext(TOOLS, **kw)
    return ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.9),
                       context=fake), fake


def held_text(fake) -> str:
    """What the cache spells: the characters, the tags and the generated text."""
    return fake.text_of(fake.cache)


def newest_turn(fake) -> str:
    """The cache's text from the newest user turn on, in the invented format."""
    text = held_text(fake)
    return text[text.rfind("<[turn>user"):]


def system_turn(fake) -> str:
    """The cache's text up to the end of the system turn, in the invented format."""
    text = held_text(fake)
    return text[:text.find("<turn]>") + len("<turn]>")]


def held_renders(context, messages, tools=None, reasoning=False,
                 for_tools: bool | None = None) -> list[str]:
    """What a cache holding ``messages`` may read as: their render with the generation prompt or without, or decoded ahead of the next question up to where it goes.

    Rendered as the context states its template renders, with nothing the
    library read off the template, each reply's thought in each form a
    template may read it in.
    """
    from llama_chat.render_checks import STAND_IN as _STAND_IN
    render = lambda m, prompt: context.stated_render(       # noqa: E731
        m, tools, generation_prompt=prompt, reasoning=reasoning, for_tools=for_tools)
    found = []
    for written in in_each_thought_form(messages):
        found += [render(written, False), render(written, True)]
        ahead = render(written + [{"role": "user", "content": _STAND_IN}], True)
        if ahead is not None and _STAND_IN in ahead:
            before = ahead[:ahead.index(_STAND_IN)]
            cut = len(before)
            for piece, is_tag in reversed(split_at_tags(context, before) or []):
                if is_tag:
                    break
                cut -= len(piece)
            found.append(before[:cut])
    return [text for text in found if text is not None]


def assert_cache_reads_as_rendered(wrapper, fake):
    """``fake``, the wrapper's cache, holds what the wrapper holds it to be, and reads as the template renders the conversation around each kept turn.

    The cache is compared as it stands, unless a decode, an edit or a
    generation failed since it was last brought up: then it is brought back
    first, as the next action would bring it back.
    """
    internals.bring_cache_back(wrapper)
    assert fake.cache == internals.held_tokens(wrapper), \
        "the cache no longer holds what the wrapper holds it to be"
    if not internals.is_begun(wrapper):
        return
    renders = held_renders(fake, internals.rendered_messages(wrapper),
                           internals.declarations(wrapper), reasoning=internals.reasons(wrapper),
                           for_tools=internals.uses_tool_template(wrapper))
    text_of = fake.text_of
    kept = [(text_of(tokens), text_of(tokens[:1]), text_of(tokens[-1:]), text)
            for tokens, text in internals.kept_turns(wrapper)]
    assert any(_reads_as(text_of(fake.cache), render, kept) for render in renders), \
        "the cache does not read as the template renders the conversation"
    for placed in internals.kept_turn_places(wrapper):
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


def without_ids(calls) -> list:
    """``calls`` with their ids left out, to compare with calls built by hand."""
    return [dataclasses.replace(call, id="") for call in calls]


def weather_registry():
    registry = ToolRegistry()
    registry.tool(lambda city: {"city": city, "celsius": 18},
                  description="Wetter.", name="weather",
                  parameters={"type": "object",
                              "properties": {"city": {"type": "string"}}})
    return registry


def begun_wrapper(template=PLAIN, **fake_kw):
    fake = FakeContext(template, **fake_kw)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    return w, fake


def assert_consistent(wrapper, fake):
    """The physical cache holds what the wrapper holds it to be."""
    assert fake.cache == internals.held_tokens(wrapper)
    assert wrapper.total_tokens == len(fake.cache)


THOUGHTFUL = "<thought>Denken</thought>Antwort."


def reasoning_wrapper(reasoning=True, config=None, template=REASONING, **fake_kw):
    fake_kw = {"gen_tokens": len(THOUGHTFUL), "gen_text": THOUGHTFUL, **fake_kw}
    fake = FakeContext(template, **fake_kw)
    w = ChatWrapper(config or Config(context_size=500, evict_above_fraction=1.0),
                    context=fake)
    w.begin("s", reasoning=reasoning)
    return w, fake


def unbegun_tool_wrapper(template=TOOLS):
    fake = FakeContext(template, gen_tokens=7, gen_text="Gleich.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    return w, fake


# ----- what the template writes, as its fixture states it ------------------------
_QUESTION = [{"role": "user", "content": "x"}]


def rendered_prompt(fake, reasoning=False) -> str:
    """The generation prompt as the template renders one after a question."""
    asked = fake.stated_render(_QUESTION, reasoning=reasoning)
    return fake.stated_render(_QUESTION, generation_prompt=True,
                              reasoning=reasoning)[len(asked):]


def rendered_reply(fake, text, reasoning=False) -> str:
    """A past reply as the template renders one after a question, its tags included."""
    asked = fake.stated_render(_QUESTION, reasoning=reasoning)
    return fake.stated_render(
        _QUESTION + [{"role": "assistant", "content": text}], reasoning=reasoning)[len(asked):]


def rendered_close(fake, reasoning=False) -> str:
    """What the template writes after a reply's text to close its turn."""
    return rendered_reply(fake, "TEXT", reasoning).split("TEXT", 1)[1]


def rows(wrapper) -> list[dict]:
    """The conversation as ``{"role", "text"}`` rows, the system prompt first, read from what the wrapper publishes."""
    from llama_chat.tools import ToolTurn
    saved = wrapper.conversation()
    system = saved["system_prompt"]
    found = [{"role": "system", "text": system}] if system else []
    for item in saved["messages"]:
        if isinstance(item, ToolTurn):
            role, text = "assistant", item.text
        elif isinstance(item, dict):
            role, text = item["role"], item["content"]
        else:
            role, text = item
        found.append({"role": role, "text": text})
    return found
