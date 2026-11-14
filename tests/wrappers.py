"""Wrappers on the model-free cache that several test files build, and their consistency check."""

import dataclasses
import json
import os
import re

from llama_chat.config import Config
from llama_chat.messages import CALL_AS_WRITTEN, result_message
from llama_chat.tokenizing import split_at_tags
from llama_chat.tools import ToolRegistry
from llama_chat.wrapper import ChatWrapper
from tests import internals
from tests.chat_templates import in_each_thought_form
from tests.fake_context import FakeContext
from tests.invented_templates import PLAIN, REASONING, TOOLS


def tool_wrapper(context_size=512, template=None, **kw):
    fake = FakeContext(template or TOOLS, **kw)
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
                 for_tools: bool | None = None, typed: bool = False) -> list[str]:
    """What a cache holding ``messages`` may read as: their render with the generation prompt or without, or decoded ahead of the next question up to where it goes.

    Rendered as the context states its template renders, with nothing the
    library read off the template, each reply's thought in each form a
    template may read it in; with ``typed``, each spelled as :func:`typed_text`
    spells the tokens the cache must hold for it.
    """
    from llama_chat.render_checks import STAND_IN as _STAND_IN

    def render(messages, prompt):
        if not typed:
            return context.stated_render(messages, tools, generation_prompt=prompt,
                                         reasoning=reasoning, for_tools=for_tools)
        pieces = rendered_pieces(context, messages, tools, generation_prompt=prompt,
                                 reasoning=reasoning, for_tools=for_tools)
        return None if pieces is None else "".join(
            typed_text(context, context.tokenize(text, parse_special=not own))
            for text, own in pieces)

    found = []
    for written in in_each_thought_form(messages):
        found += [render(written, False), render(written, True)]
        ahead = render(written + [{"role": "user", "content": _STAND_IN}], True)
        if ahead is not None and _STAND_IN in ahead:
            before = ahead[:ahead.index(_STAND_IN)]
            cut = len(before)
            if typed:
                cut = before.rfind(_TAG_MARK) + 1
            for piece, is_tag in ([] if typed else reversed(split_at_tags(context, before) or [])):
                if is_tag:
                    break
                cut -= len(piece)
            found.append(before[:cut])
    return [text for text in found if text is not None]


# What sets a tag apart in :func:`typed_text`: a character no render here holds.
_TAG_MARK = "\x00"
# Spells a token only a generation gives, which no tokenization of the same text gives.
_GENERATED_MARK = "\u2061"


def typed_text(context, tokens: list[int]) -> str:
    """What ``tokens`` spell, each of the model's tags between two :data:`_TAG_MARK` and each token only a generation gives between two :data:`_GENERATED_MARK`, so that neither reads as the characters a tokenization gives."""
    generated_only = getattr(context, "is_generated_only", lambda token: False)
    return "".join(_TAG_MARK + context.token_text(token) + _TAG_MARK if context.is_special(token)
                   else _GENERATED_MARK + context.token_text(token) + _GENERATED_MARK
                   if generated_only(token) else context.token_text(token)
                   for token in tokens)


def rendered_pieces(context, messages, declarations=None, *, generation_prompt: bool,
                    reasoning: bool = False,
                    for_tools: bool | None = None) -> list[tuple[str, bool]] | None:
    """The render of ``messages`` in pieces, each marked whether it is a text of the caller's own, worked out here on its own; ``None`` where the template refuses them or writes them otherwise than their marked render.

    The caller's own texts are a message's content and thought, a call's name
    and its string arguments, and each description of a declaration.
    """
    texts = {}

    def mark(text):
        marker = f"ZQ{len(texts)}QZ"
        texts[marker] = text
        return marker

    def marked(value, key=None):
        if isinstance(value, str):
            return mark(value) if value and key in _OWN_KEYS else value
        if isinstance(value, dict):
            return {k: (marked(v, k) if k != "arguments" else _all_marked(v, mark))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [marked(item, key) for item in value]
        return value

    def render(messages, declared):
        return context.stated_render(messages, declared, generation_prompt=generation_prompt,
                                     reasoning=reasoning, for_tools=for_tools)

    # A call the model wrote unreadably is written as it wrote it, not as the template writes one.
    as_written = [call.get(CALL_AS_WRITTEN) for m in messages for call in m.get("tool_calls") or ()]
    messages = [{**m, "tool_calls": [{k: v for k, v in call.items() if k != CALL_AS_WRITTEN}
                                     for call in m["tool_calls"]]} if m.get("tool_calls") else m
                for m in messages]
    real = render(messages, declarations)
    rendered = render(marked(messages), marked(declarations))
    if real is None or rendered is None:
        return None
    # Template text and own texts alternate; own texts with no template text between them are one.
    parts = [part for part in re.split(r"(ZQ\d+QZ)", rendered) if part]
    pieces, at = [], 0
    for k, part in enumerate(parts):
        if not part.startswith("ZQ"):
            assert real.startswith(part, at)
            pieces.append((part, False))
            at += len(part)
        elif k + 1 == len(parts) or not parts[k + 1].startswith("ZQ"):
            following = parts[k + 1] if k + 1 < len(parts) else ""
            alone = k == 0 or not parts[k - 1].startswith("ZQ")
            # The text as given or in a form a template writes it in, where what follows matches.
            written = next((form for form in _written_forms(texts[part]) if alone and following
                            and real.startswith(form, at)
                            and real.startswith(following, at + len(form))), None)
            end = (at + len(written) if written is not None
                   else real.find(following, at) if following else len(real))
            if end < 0:
                return None                 # the template writes these messages otherwise than marked
            pieces.append((real[at:end], True))
            at = end
    assert at == len(real), "the render does not match its marked render"
    written_calls = [written for written in as_written if written is not None]
    if written_calls:
        pieces = _written_calls(context, pieces, written_calls)
    return [(text, own) for text, own in pieces if text]


def _written_calls(context, pieces, written_calls):
    """``pieces`` with each call to the unknown function written between its opening and closing as the model wrote it, in order."""
    from llama_chat.messages import UNKNOWN_FUNCTION
    from llama_chat.resolved_template import ResolvedTemplates

    from tests.fake_context import call_marks, shipped_addon_call

    reader = FakeContext(context._fixture)
    template = ResolvedTemplates(reader, shipped_addon_call(context._template)).for_conversation(True)
    opening, closing = (text.strip() for text in call_marks(template, reader))
    text = "".join(piece for piece, _ in pieces)
    regions, at = [], 0
    while (start := text.find(opening, at)) >= 0 and len(regions) < len(written_calls):
        end = text.find(closing, start + len(opening))
        if end < 0:
            break
        if UNKNOWN_FUNCTION in text[start + len(opening):end]:
            regions.append((start + len(opening), end))
        at = start + len(opening)
    if len(regions) != len(written_calls):
        return pieces       # the template writes a call in another frame than the model: its own way
    for (start, end), written in reversed(list(zip(regions, written_calls))):
        pieces = _written_in(pieces, start, end, written)
    return pieces


def _written_in(pieces, start, end, written):
    """``pieces`` with their text from ``start`` to ``end`` replaced by ``written``, the template's own."""
    rebuilt, at = [], 0
    for piece, own in pieces:
        before, after = piece[:max(0, start - at)], piece[max(0, end - at):]
        rebuilt += [(before, own)] if at < start else []
        if at <= start < at + len(piece):
            rebuilt.append((written, False))
        rebuilt += [(after, own)] if at + len(piece) > end else []
        at += len(piece)
    return rebuilt


def _written_forms(text: str) -> list[str]:
    """How a template may write ``text``: as it is, trimmed, or inside a JSON string."""
    return [text, text.strip(), text.rstrip(), text.lstrip(), json.dumps(text)[1:-1],
            json.dumps(text, ensure_ascii=False)[1:-1]]


# The keys whose text is the caller's own, wherever they stand in a message or a declaration.
_OWN_KEYS = ("content", "reasoning_content", "thinking", "text", "name", "description")


def _all_marked(value, mark):
    """A call's arguments, each string in them marked."""
    if isinstance(value, str):
        return mark(value) if value else value
    if isinstance(value, dict):
        return {key: _all_marked(item, mark) for key, item in value.items()}
    if isinstance(value, list):
        return [_all_marked(item, mark) for item in value]
    return value


def assert_cache_reads_as_rendered(wrapper, fake):
    """``fake``, the wrapper's cache, holds what the wrapper holds it to be, and reads as the template renders the conversation around each kept turn.

    The cache is compared as it stands, unless a decode, an edit or a
    generation failed since it was last brought up: then it is brought back
    first, as the next action would bring it back. While a call is owed a
    result, what the template writes after the newest kept turn is decoded
    with the results, so a cache that ends with that turn reads as the render
    of the conversation with its results up to it; one that holds no kept
    turn reads as far as that render and the render without the results
    agree, to the last whole token.
    """
    internals.bring_cache_back(wrapper)
    assert fake.cache == internals.held_tokens(wrapper), \
        "the cache no longer holds what the wrapper holds it to be"
    if not internals.is_begun(wrapper):
        return
    messages = internals.rendered_messages(wrapper)
    # A kept turn's tokens hold the text of each of its calling messages, so the template
    # is given those without it.
    for first, last in internals.kept_turn_spans(wrapper):
        for i in range(first, last + 1):
            if messages[i].get("tool_calls"):
                messages[i] = {**messages[i], "content": ""}
    owed = wrapper.pending_calls
    answered = messages + [result_message(call.name, call.id, "stand-in") for call in owed]
    standing, completed = (held_renders(fake, written, internals.declarations(wrapper),
                                        reasoning=internals.reasons(wrapper),
                                        for_tools=internals.uses_tool_template(wrapper),
                                        typed=True)
                           for written in (messages, answered))
    if owed and not internals.kept_turns(wrapper):
        assert typed_text(fake, fake.cache) in [_shared_whole_tokens(one, other)
                                                for one in standing for other in completed], \
            "the cache does not read as the template renders the conversation while a call is owed"
        return
    renders = standing + (completed if owed else [])
    text_of = fake.text_of
    closing = _reply_closing(fake, internals.reasons(wrapper))
    kept = [(typed_text(fake, tokens), text_of(tokens[:1]), text_of(tokens[-1:]), text,
             closing if not text and not on_call else "")
            for tokens, text, on_call in internals.kept_turns(wrapper)]
    assert any(_reads_as(typed_text(fake, fake.cache), render, kept, bool(owed))
               for render in renders), \
        "the cache does not read as the template renders the conversation"
    for placed in internals.kept_turn_places(wrapper):
        assert placed is not None, "a turn kept as generated was lost"
        assert placed[0] < placed[1], \
            "a turn kept as generated stands in for none of the template's text"


def _shared_whole_tokens(one: str, other: str) -> str:
    """The start two typed renders share, ending before a tag it would cut in two."""
    shared = os.path.commonprefix([one, other])
    return shared[:shared.rfind(_TAG_MARK)] if shared.count(_TAG_MARK) % 2 else shared


def _reply_closing(fake, reasoning: bool) -> str:
    """What the template writes after a reply's text to close its turn, from its first character that is no space."""
    marker = "QZQ7"
    reply = fake.stated_render(_QUESTION + [{"role": "assistant", "content": marker}],
                               reasoning=reasoning)
    return reply.split(marker, 1)[1].lstrip()


def _reads_as(held: str, render: str, kept: list[tuple[str, str, str, str, str]],
              owed: bool = False) -> bool:
    """Whether ``held`` is ``render`` with each kept turn in place of the template's version of it, everything around them exactly as rendered.

    ``held``, ``render`` and each turn's text are spelled as :func:`typed_text`
    spells tokens, so that everything around the kept turns compares token for
    token. ``kept`` holds each turn's text, the text of its first and of its
    last token, the text of its last message, and, for a turn that ends on a
    reply without text, the template's closing after a reply. The template's
    version of a turn begins with the same token, and ends with the same
    token, with that message's text, where that closing begins, or, for a turn
    whose call the template does not write, is where the kept turn begins; so
    it takes in none of the template's text beside it. Where a call is
    ``owed`` and ``held`` ends with the newest kept turn, the rest of the
    render is not decoded yet. A turn's bounds are found by its characters,
    since a model may write a tag of its turn as text.
    """
    plain = lambda text: text.replace(_TAG_MARK, "")       # noqa: E731
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
    for n, ((turn, opening, closing, text, reply_closing), piece) in enumerate(
            zip(kept, pieces[1:])):
        if not plain(render[at:]).startswith(opening):
            return False
        last = n == len(kept) - 1
        if last and owed and not piece:
            return True             # the rest of the render is decoded with the results
        ends = [len(render) - len(piece)] if last else _occurrences(render, piece, at)
        ends = [end for end in ends if len(plain(render[at:end])) >= len(opening)
                and render.startswith(piece, end)
                and (plain(render[at:end]).endswith(closing)
                     or plain(turn).startswith(plain(render[at:end]))
                     or bool(text) and plain(render[at:end]).rstrip().endswith(text)
                     or bool(reply_closing) and plain(render[end:]).startswith(reply_closing))]
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
        else:
            role, text = item["role"], item["content"]
        found.append({"role": role, "text": text})
    return found
