"""Random sequences of actions, from fixed seeds, on templates that call tools: every state they reach begins again from conversation() unchanged, and from its messages alone.

The checks after every test (see conftest.py) also run on each state reached:
the cache is the template's render, and the conversation is its two forms.
"""

import random

import pytest

from llama_chat import ChatWrapper, Config
from llama_chat.tool_registry import ToolRegistry
from tests import invented_templates as invented
from tests.fake_context import FakeContext, ScriptedCall

_TEMPLATES = {"tools": invented.TOOLS, "json tools": invented.JSON_TOOLS,
              "two questions refused": invented.TOOLS_REFUSING_TWO_USER_TURNS,
              "thoughts kept": invented.REASONING_KEEPING_THOUGHTS,
              "open turns closed": invented.TOOLS_CLOSING_OPEN_TURNS,
              "result lines": invented.RESULT_LINES}
_PLAIN_CALLS = ["fn:clock{}", "fn:nonsense with no name"]
_JSON_CALLS = ['{"name": "clock", "arguments": {}}', '{"no": "call"}',
               '[{"no": "call"}, {"name": "clock", "arguments": {}}]',
               '[{"name": "clock", "arguments": {}}, {"no": "call"}]']


def _sequence(seed: int):
    """A wrapper after the actions ``seed`` draws, and its template."""
    draw = random.Random(seed)
    name = draw.choice(sorted(_TEMPLATES))
    template = _TEMPLATES[name]
    calls = _JSON_CALLS if "json" in name else _PLAIN_CALLS
    fake = FakeContext(template)
    size = draw.choice([400, 700, 2048])
    w = ChatWrapper(Config(context_size=size, min_reply_tokens=16), context=fake)
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Zeit", name="clock",
                  parameters={"type": "object", "properties": {}})
    w.begin([{"role": "system", "content": "s"}] if draw.random() < 0.8 else [],
            tools=registry.specs(), reasoning="thoughts" in name and draw.random() < 0.5)
    for step in range(draw.randint(2, 8)):
        action = draw.random()
        try:
            if action < 0.45:
                fake.replies = [ScriptedCall(draw.choice(calls), before=draw.choice(["", "Ich schaue."]))
                                for _ in range(draw.randint(0, 3))] + [draw.choice(["Fertig.", ""])]
                fails_at = draw.randint(1, 3) if draw.random() < 0.2 else None
                counted = [0]

                def fail(_text):
                    counted[0] += 1
                    if counted[0] == fails_at:
                        raise RuntimeError("the decode failed")
                fake.before_reply = fail
                text = draw.choice(["frage", f"frage {step}"] + ([None] if w.awaits_reply else []))
                policy = draw.choice(["yield", registry, "refuse"])
                w.reply(text, tool_call_policy=policy).finish()
            elif action < 0.7 and w.pending_calls:
                w.inject_tool_result(w.pending_calls[0].id,
                                     draw.choice(["12:00", "x" * draw.choice([10, 3000])]))
            elif action < 0.8:
                w.inject(f"info {step}")
            else:
                w.reply(f"neu {step}").finish()
        except RuntimeError:
            pass                        # a failed decode, which leaves everything as it was
        except ValueError:
            pass                        # a refusal, which leaves everything as it was
        fake.before_reply = None
    return w, template, size


@pytest.mark.parametrize("seed", range(150))
def test_every_state_a_sequence_reaches_begins_again_the_same(seed):
    w, template, size = _sequence(seed)
    saved = w.conversation()
    _assert_begins_with_its_newest_part(w, saved)          # the cache holds it already
    again = ChatWrapper(Config(context_size=size, min_reply_tokens=16), context=FakeContext(template))
    _assert_begins_with_its_newest_part(again, saved)
    alone = ChatWrapper(Config(context_size=size, min_reply_tokens=16), context=FakeContext(template))
    alone.begin(saved["messages"], tools=saved["tools"], reasoning=saved["reasoning"])


def _assert_begins_with_its_newest_part(w, saved):
    """``begin(**saved)`` on ``w`` holds the newest part of the conversation a begin keeps, each message unchanged and in its place; all of it where it keeps every message."""
    kept = w.begin(**saved).n_kept_messages
    system = [m for m in saved["messages"][:1] if m["role"] == "system"]
    history = saved["messages"][len(system):]
    again = w.conversation()
    assert again["messages"] == system + history[len(history) - kept:]
    if kept == len(history):
        assert again == saved
