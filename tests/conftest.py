"""Every wrapper a test builds on a model-free cache ends the test with that cache as it holds it, as the template renders the conversation, and with every turn kept as generated in the place of the template's version of it."""

import pytest

from llama_chat.wrapper import ChatWrapper

from .wrappers import held_renders


@pytest.fixture(autouse=True)
def _the_cache_reads_as_the_template_renders_the_conversation(monkeypatch):
    built = []
    build = ChatWrapper.__init__

    def recording(self, *args, **kwargs):
        build(self, *args, **kwargs)
        if hasattr(self._context, "cache") and hasattr(self._context, "text_of"):
            built.append(self)

    monkeypatch.setattr(ChatWrapper, "__init__", recording)
    yield
    for wrapper in built:
        if wrapper._closed:
            continue
        conversation = wrapper._conversation
        conversation._memory.repair()
        assert wrapper._context.cache == conversation._memory.tokens, \
            "the cache no longer holds what the wrapper holds it to be"
        if not conversation.begun:
            continue
        messages, text_of = conversation.messages, wrapper._context.text_of
        tools = [spec.as_declaration() for spec in conversation.tools] or None
        renders = held_renders(wrapper._template, messages, tools,
                               reasoning=conversation._reasoning)
        held = text_of(wrapper._context.cache)
        kept = [text_of(k.tokens) for k in conversation._kept]
        assert any(_reads_as(held, render, kept) for render in renders), \
            "the cache does not read as the template renders the conversation"
        for render in (False, True):
            split = conversation._split(messages, render)
            if split is not None:
                for kept_turn in conversation._kept:
                    placed = conversation._place(kept_turn, messages, split)
                    assert placed is not None, "a turn kept as generated was lost"
                    assert placed[0] < placed[1], \
                        "a turn kept as generated stands in for none of the template's text"


def _reads_as(held: str, render: str, kept: list[str]) -> bool:
    """Whether ``held`` is ``render`` with a stretch in place of each kept turn, in order: what the cache holds around them reads as the template writes it."""
    pieces = [held]
    for turn in kept:
        if turn not in pieces[-1]:
            return False
        before, after = pieces[-1].split(turn, 1)
        pieces[-1:] = [before, after]
    at = 0
    for n, piece in enumerate(pieces):
        found = render.find(piece, at) if n else (0 if render.startswith(piece) else -1)
        if found < 0:
            return False
        at = found + len(piece)
    return render.endswith(pieces[-1]) or held == render
