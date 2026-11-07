"""Whole conversations through real chat templates: the cache always holds the template's render.

Before every reply, the cache must read exactly as the template renders the
messages that are still in it, with the generation prompt after them. A
conversation the template refuses to render must be refused, never encoded.
"""

import json
from pathlib import Path

import pytest

from llama_chat import ChatWrapper, Config

from .template_contexts import ConversationContext

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_QUESTIONS = ["How are you?", "Name a colour.", "And a number?", "Why that one?",
              "Tell me more.", "Shorter, please.", "Thanks!", "One more thing?"]
_REPLIES = [f"Answer number {i}, kept short." for i in range(len(_QUESTIONS) + 1)]


def _conversation(path, context_size):
    fixture = json.loads(path.read_text())
    context = ConversationContext(fixture, _REPLIES)
    chat = ChatWrapper(Config(context_size=context_size, evict_above_fraction=0.75,
                              min_reply_tokens=8), context=context)
    seen = []
    context.before_reply = lambda text: seen.append((text, chat.snapshot()))
    return context, chat, seen


def _rendered(context, rows):
    messages = [{"role": row["role"], "content": row["text"]} for row in rows
                if not (row["role"] == "system" and not row["text"])]
    return context.render_with_model_template(messages, generation_prompt=True,
                                              reasoning=False)


def _system_prompt(chat):
    return "Be brief." if chat._template.has_system_turn else ""


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_every_reply_starts_from_the_templates_render_of_what_the_cache_holds(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    for question in _QUESTIONS[:3]:
        chat.reply_without_tools(question)
    for cached, rows in seen:
        assert cached == _rendered(context, rows)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_after_evictions_the_cache_still_holds_the_templates_render(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    # Small enough that the oldest messages are evicted after a few turns.
    budget = int((len(context.cache) + 200) / 0.75)
    context, chat, seen = _conversation(path, context_size=budget)
    chat.begin(_system_prompt(chat))
    for question in _QUESTIONS:
        chat.reply_without_tools(question)
    assert any(len(rows) < 2 * len(_QUESTIONS) for _, rows in seen[-1:])
    for cached, rows in seen:
        rendered = _rendered(context, rows)
        assert rendered is not None, [row["role"] for row in rows]
        assert cached == rendered


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_restart_keeps_only_history_the_template_renders(path):
    context, chat, seen = _conversation(path, context_size=4096)
    chat.begin(_system_prompt(chat))
    budget = int((len(context.cache) + 150) / 0.75)
    context, chat, seen = _conversation(path, context_size=budget)
    history = []
    for question, answer in zip(_QUESTIONS, _REPLIES):
        history += [("user", question), ("assistant", answer)]
    chat.begin(_system_prompt(chat), history)
    chat.reply_without_tools("And now?")
    cached, rows = seen[-1]
    rendered = _rendered(context, rows)
    assert rendered is not None, [row["role"] for row in rows]
    assert cached == rendered
