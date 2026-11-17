"""Wrappers on the model-free cache that several test files build, and their consistency check."""

import dataclasses
import json
import os
import re

from llama_chat.config import Config
from llama_chat.history_forms import (answer_in_order, ensure_generated_turns, read_messages,
                                      with_generated_turns, without_unreadable)
from llama_chat.messages import GENERATED_TURN, UNREADABLE_CALL, result_message
from llama_chat.rendering import generated_turns_in
from llama_chat.tokenizing import split_at_tags
from llama_chat.tool_registry import ToolRegistry
from tests.chat_messages import opened
from llama_chat.wrapper import ChatWrapper
from tests import internals
from tests.chat_templates import in_each_thought_form
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN, REASONING, TOOLS


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
                 for_tools: bool | None = None, typed: bool = False,
                 owned: bool = False) -> list:
    """What a cache holding ``messages`` may read as: their render with the generation prompt or without, or decoded ahead of the next question up to where it goes.

    Rendered as the context states its template renders, with nothing the
    library read off the template, each reply's thought in each form a
    template may read it in; with ``typed``, each spelled as :func:`typed_text`
    spells the tokens the cache must hold for it. With ``owned`` as well, each
    comes as a pair: the text, and where in it each message's own texts stand,
    as ``(start, end, indexes)``.
    """
    from llama_chat.render_checks import STAND_IN as _STAND_IN

    def render(messages, prompt):
        if not typed:
            return context.stated_render(messages, tools, generation_prompt=prompt,
                                         reasoning=reasoning, for_tools=for_tools)
        pieces = rendered_pieces(context, messages, tools, generation_prompt=prompt,
                                 reasoning=reasoning, for_tools=for_tools, owned=True)
        if pieces is None:
            return None
        text, spans = "", []
        for piece, own, owners in pieces:
            typed_piece = typed_text(context, context.tokenize(piece, parse_special=not own))
            if owners:
                spans.append((len(text), len(text) + len(typed_piece), owners))
            text += typed_piece
        return text, spans

    def text_of(rendered):
        return rendered[0] if typed else rendered

    found = []
    for written in in_each_thought_form(messages):
        found += [render(written, False), render(written, True)]
        ahead = render(written + [{"role": "user", "content": _STAND_IN}], True)
        if ahead is not None and _STAND_IN in text_of(ahead):
            before = text_of(ahead)[:text_of(ahead).index(_STAND_IN)]
            cut = len(before)
            if typed:
                cut = before.rfind(_TAG_MARK) + 1
            for piece, is_tag in ([] if typed else reversed(split_at_tags(context, before) or [])):
                if is_tag:
                    break
                cut -= len(piece)
            found.append((before[:cut], [(start, min(end, cut), owners)
                                         for start, end, owners in ahead[1] if start < cut])
                         if typed else before[:cut])
    found = [rendered for rendered in found if rendered is not None]
    return found if owned or not typed else [text for text, _spans in found]


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
                    reasoning: bool = False, for_tools: bool | None = None,
                    owned: bool = False) -> list[tuple] | None:
    """The render of ``messages`` in pieces, each marked whether it is a text of the caller's own, worked out here on its own; ``None`` where the template refuses them or writes them otherwise than their marked render.

    The caller's own texts are a message's content and thought, a call's name
    and its string arguments, and each description of a declaration. With
    ``owned``, each piece also names the indexes of the messages its own texts
    come from, empty for the template's text and the declarations. Where the
    template writes equal texts otherwise than their unequal markers, as one
    that compares texts does, equal texts are marked alike, and each is placed
    in the order of the messages it comes from.
    """
    try:
        return _rendered_pieces(context, messages, declarations, generation_prompt, reasoning,
                                for_tools, owned, shares_equal_texts=False)
    except AssertionError:
        return _rendered_pieces(context, messages, declarations, generation_prompt, reasoning,
                                for_tools, owned, shares_equal_texts=True)


def _rendered_pieces(context, messages, declarations, generation_prompt, reasoning, for_tools,
                     owned, shares_equal_texts):
    """:func:`rendered_pieces`, equal texts marked alike where ``shares_equal_texts``."""
    texts, owner_of, by_text = {}, {}, {}

    def mark(text, owner=None):
        marker = by_text.setdefault(text, f"ZQ{len(texts)}QZ") if shares_equal_texts \
            else f"ZQ{len(texts)}QZ"
        texts[marker] = text
        owner_of.setdefault(marker, []).append(owner)
        return marker

    def marked(value, key=None, owner=None):
        if isinstance(value, str):
            return mark(value, owner) if value and key in _OWN_KEYS else value
        if isinstance(value, dict):
            return {k: (marked(v, k, owner) if k != "arguments"
                        else _all_marked(v, lambda text: mark(text, owner)))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [marked(item, key, owner) for item in value]
        return value

    def render(messages, declared):
        return context.stated_render(messages, declared, generation_prompt=generation_prompt,
                                     reasoning=reasoning, for_tools=for_tools)

    # The mark of a call that could not be read is the conversation's, not the template's.
    messages = [{**m, "tool_calls": [{k: v for k, v in call.items() if k != UNREADABLE_CALL}
                                     for call in m["tool_calls"]]} if m.get("tool_calls") else m
                for m in messages]
    real = render(messages, declarations)
    rendered = render([marked(m, owner=i) for i, m in enumerate(messages)], marked(declarations))
    if real is None or rendered is None:
        return None
    # Template text and own texts alternate; own texts with no template text between them are one.
    parts = [part for part in re.split(r"(ZQ\d+QZ)", rendered) if part]
    pieces, at, run = [], 0, set()
    for k, part in enumerate(parts):
        if not part.startswith("ZQ"):
            assert real.startswith(part, at)
            pieces.append((part, False, frozenset()))
            at += len(part)
        else:
            owners = owner_of[part]
            owner = owners.pop(0) if len(owners) > 1 else owners[0]
            if owner is not None:
                run.add(owner)
            if k + 1 < len(parts) and parts[k + 1].startswith("ZQ"):
                continue                # own texts with no template text between them are one
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
            pieces.append((real[at:end], True, frozenset(run)))
            at, run = end, set()
    assert at == len(real), "the render does not match its marked render"
    return [piece if owned else piece[:2] for piece in pieces if piece[0]]


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
    """``fake``, the wrapper's cache, holds what the wrapper holds it to be, and reads as the template renders the conversation around each generated turn.

    The cache is compared as it stands, unless a decode, an edit or a
    generation failed since it was last brought up: then it is brought back
    first, as the next action would bring it back. While a call is owed a
    result, what the template writes after the newest generated turn is decoded
    with the results, so a cache that ends with that turn reads as the render
    of the conversation with its results up to it; one that holds no kept
    turn, or none yet where the two renders part before the first, reads as
    far as that render and the render without the results agree, to the last
    whole token. A cache that ends with that turn may also read as the render
    the turn was generated after, up to its question, where the template
    moves its declarations once a call follows.
    """
    internals.bring_cache_back(wrapper)
    assert fake.cache == internals.held_tokens(wrapper), \
        "the cache no longer holds what the wrapper holds it to be"
    if not internals.is_begun(wrapper):
        return
    messages = internals.rendered_messages(wrapper)
    # A generated turn's tokens hold the text of each of its calling messages, so the template
    # is given those without it.
    for first, last in internals.generated_turn_spans(wrapper):
        for i in range(first, last + 1):
            if messages[i].get("tool_calls"):
                messages[i] = {**messages[i], "content": ""}
    owed = wrapper.pending_calls
    answered = messages + [result_message(call.name, call.id, "stand-in") for call in owed]
    standing, completed = (held_renders(fake, written, internals.declarations(wrapper),
                                        reasoning=internals.reasons(wrapper),
                                        for_tools=internals.uses_tool_template(wrapper),
                                        typed=True, owned=True)
                           for written in (messages, answered))
    shared = [_shared_whole_tokens(one, other) for one, _ in standing for other, _ in completed]
    if owed and not internals.generated_turns_held(wrapper):
        assert typed_text(fake, fake.cache) in shared, \
            "the cache does not read as the template renders the conversation while a call is owed"
        return
    asked = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
    generated_after = (held_renders(fake, messages[:asked + 1], internals.declarations(wrapper),
                                    reasoning=internals.reasons(wrapper),
                                    for_tools=internals.uses_tool_template(wrapper),
                                    typed=True, owned=True)
                       if owed and asked is not None else [])
    renders = standing + (completed + generated_after if owed else [])
    text_of = fake.text_of
    closing = _reply_closing(fake, internals.reasons(wrapper), internals.uses_tool_template(wrapper))
    turns = internals.generated_turns_held(wrapper)
    kept = [(typed_text(fake, tokens), text_of(tokens[:1]), text_of(tokens[-1:]), text,
             closing if not text and not on_call else "")
            for tokens, text, on_call in turns]
    if owed:
        # Decoded as far as the renders with and without the results agree, the generated turns
        # that stand there in their places.
        held = typed_text(fake, fake.cache)
        if any(_reads_as(held, _shared_whole_tokens(one, other), kept[:n])
               for one, _ in standing for other, _ in completed for n in range(len(kept) + 1)):
            return
    # The messages each generated turn stands for; the results still owed come after every message.
    stands_for = [set(range(first, last + 1)) for first, last in internals.generated_turn_spans(wrapper)]
    if stands_for:
        stands_for[-1] |= set(range(len(messages), len(answered)))
    assert any(_reads_as(typed_text(fake, fake.cache), render, kept, bool(owed),
                         _owned_by_others(spans, stands_for))
               for render, spans in renders), \
        "the cache does not read as the template renders the conversation"
    for placed in internals.generated_turn_places(wrapper):
        assert placed is not None, "a turn kept as generated was lost"
        assert placed[0] < placed[1], \
            "a turn kept as generated stands in for none of the template's text"


def _shared_whole_tokens(one: str, other: str) -> str:
    """The start two typed renders share, ending before a tag it would cut in two."""
    shared = os.path.commonprefix([one, other])
    return shared[:shared.rfind(_TAG_MARK)] if shared.count(_TAG_MARK) % 2 else shared


def _reply_closing(fake, reasoning: bool, for_tools: bool) -> str:
    """What the template the conversation is written with writes after a reply's text to close its turn, from its first character that is no space."""
    marker = "QZQ7"
    reply = fake.stated_render(_QUESTION + [{"role": "assistant", "content": marker}],
                               reasoning=reasoning, for_tools=for_tools)
    return reply.split(marker, 1)[1].lstrip()


def _owned_by_others(spans, stands_for: list[set[int]]) -> list[list[tuple[int, int]]]:
    """For each generated turn, where the render holds own texts of messages that turn does not stand for.

    Own texts with no template text between them read as one, so such a run
    counts as the turn's where any of its messages is.
    """
    return [[(start, end) for start, end, owners in spans if owners.isdisjoint(messages)]
            for messages in stands_for]


def _reads_as(held: str, render: str, kept: list[tuple[str, str, str, str, str]],
              owed: bool = False, others: list[list[tuple[int, int]]] | None = None) -> bool:
    """Whether ``held`` is ``render`` with each generated turn in place of the template's version of it, everything around them exactly as rendered.

    ``held``, ``render`` and each turn's text are spelled as :func:`typed_text`
    spells tokens, so that everything around the generated turns compares token for
    token. ``kept`` holds each turn's text, the text of its first and of its
    last token, the text of its last message, and, for a turn that ends on a
    reply without text, the template's closing after a reply. The template's
    version of a turn begins with the same token, and ends with the same
    token, with that message's text, where that closing begins, or, for a turn
    whose call the template does not write, is where the generated turn begins; so
    it takes in none of the template's text beside it. Where a call is
    ``owed`` and ``held`` ends with the newest generated turn, the rest of the
    render is not decoded yet. A turn's bounds are found by its characters,
    since a model may write a tag of its turn as text. ``others`` holds, for each
    generated turn, where the render holds texts of messages it does not stand for:
    its version takes in none of them, so a whole exchange missing after it is
    caught however the replies repeat their text.
    """
    plain = lambda text: text.replace(_TAG_MARK, "")       # noqa: E731
    if not kept:
        return held == render

    def placements(n: int, at: int):
        """Each way the generated turns from the ``n``-th on may stand in ``held`` from ``at`` on."""
        if n == len(kept):
            yield []
            return
        for found in _occurrences(held, kept[n][0], at):
            for rest in placements(n + 1, found + len(kept[n][0])):
                yield [found] + rest

    def reads(pieces: list[str], n: int, at: int) -> bool:
        """Whether the render from ``at`` on reads as the generated turns from the ``n``-th on and ``pieces`` around them."""
        if n == len(kept):
            return True
        turn, opening, closing, text, reply_closing = kept[n]
        piece = pieces[n + 1]
        if not plain(render[at:]).startswith(opening):
            return False
        last = n == len(kept) - 1
        if last and owed and not piece:
            return True             # the rest of the render is decoded with the results
        foreign = others[n] if others else []
        ends = [len(render) - len(piece)] if last else _occurrences(render, piece, at)
        ends = [end for end in ends if len(plain(render[at:end])) >= len(opening)
                and render.startswith(piece, end)
                and not any(start < end and at < stop for start, stop in foreign)
                and (plain(render[at:end]).endswith(closing)
                     or plain(turn).startswith(plain(render[at:end]))
                     or bool(text) and plain(render[at:end]).rstrip().endswith(text)
                     or bool(reply_closing) and plain(render[end:]).startswith(reply_closing))]
        return any(reads(pieces, n + 1, end + len(piece)) for end in ends)

    for places in placements(0, 0):
        bounds = [0] + [edge for found, (turn, *_) in zip(places, kept)
                        for edge in (found, found + len(turn))] + [len(held)]
        pieces = [held[bounds[k]:bounds[k + 1]] for k in range(0, len(bounds), 2)]
        if render.startswith(pieces[0]) and reads(pieces, 0, len(pieces[0])):
            return True
    return False


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


# The settings that let a reply end without an answer, for a test about what such a reply leaves.
UNGUARDED = {"min_answer_tokens": None, "answer_room_tokens": None}


def begun_wrapper(template=PLAIN, guarded=True, **fake_kw):
    fake = FakeContext(template, **fake_kw)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0,
                           **({} if guarded else UNGUARDED)), context=fake)
    w.begin(opened("s"))
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
    w.begin(opened("s"), reasoning=reasoning)
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
    return [{"role": m["role"], "text": m["content"]} for m in wrapper.conversation()["messages"]]


def add_tools(wrapper, specs) -> int:
    """Declare ``specs`` beside the tools the conversation declares already, as one that adds tools does; how many history messages went."""
    held = wrapper.conversation()["tools"] or []
    known = {spec.name for spec in held}
    return wrapper.begin(**{**wrapper.conversation(),
                            "tools": held + [s for s in specs if s.name not in known]}
                         ).n_evicted_messages


def assert_conversation_is_its_forms(wrapper):
    """What ``conversation()`` gives, begin takes, and the conversation the wrapper holds is its Chat Completions messages with each generated turn put back, nothing else needed to go on with it."""
    if not internals.is_begun(wrapper):
        return
    saved = wrapper.conversation()
    ensure_generated_turns(saved["generated_turns"])
    joins = internals.joins_questions(wrapper)
    turns = (saved["generated_turns"] or {}).get("turns", [])
    messages, n_found = with_generated_turns(read_messages(saved["messages"]), turns, joins)
    held = {i for first, last, _ in generated_turns_in(messages) for i in range(first, last + 1)}
    messages, _gone = without_unreadable(messages, held, joins)
    rebuilt = answer_in_order(messages, internals.unreadable_result_text(wrapper))
    assert n_found == sum(len(record["kept"]) for record in turns), \
        "a generated turn is not found among the messages it was saved with"
    assert [_comparable(m) for m in rebuilt] \
        == [_comparable(m) for m in internals.rendered_messages(wrapper)], \
        "the conversation is not its messages with each generated turn put back"


def _comparable(message: dict) -> dict:
    """``message`` with the turn it may end as its tokens and span, to compare by value."""
    turn = message.get(GENERATED_TURN)
    return {**message, GENERATED_TURN: (turn.tokens, turn.span, turn.on_call)} if turn else message
