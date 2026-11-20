"""Every order in which an action can meet the work after a reply, at each point where they hand over, keeps the rules.

The work after a reply runs on a thread of its own. It hands over at three
kinds of points: before it takes the wrapper, between the parts of a long
decode, and once it has ended. Each test here holds that work at one of
those points, runs one kind of action from another thread there, and lets
the work go on. Every order ends, leaves the cache the render, gives the
conversation the same action gives when the work is waited for, and decodes
no more than that.
"""

import threading

import pytest

from llama_cpp_conversation import ChatWrapper, Config
from llama_cpp_conversation import wrapper as wrapper_module
from tests.chat_messages import opened
from llama_cpp_conversation.testing.fake_context import FakeContext
from llama_cpp_conversation.testing.invented_templates import PLAIN
from tests import internals
from tests.internals import held_tokens
from tests.wrappers import assert_consistent

_CACHES = {"shifts": dict(can_shift=True), "cuts": dict(can_shift=False, can_cut_tail=True),
           "saves states": dict(can_shift=False)}
# Where the action meets the work: before it takes the wrapper, during its first or second
# decode part, or after it has ended.
_POINTS = ("before", "part 1", "part 2", "after")
_ACTIONS = ("question", "inject and reply", "begin the same", "begin another", "truncate",
            "wait", "close", "a reply cancelled at once")
_THREAD = "llama_cpp_conversation after a reply"
_WAIT_S = 10


def _history():
    return [m for n in range(60) for m in (
        {"role": "user", "content": f"Frage {n} " + "x" * 60},
        {"role": "assistant", "content": "Antwort " + "y" * 60})]


def _act(wrapper: ChatWrapper, action: str, given: list) -> None:
    """Run ``action``; what a begin of the same conversation was given goes into ``given``."""
    if action == "question":
        wrapper.reply("Und weiter?").finish()
    elif action == "inject and reply":
        wrapper.inject("Gleich weiter.")
        wrapper.reply().finish()
    elif action == "begin the same":
        given.append(wrapper.conversation())
        wrapper.begin(**given[-1])
    elif action == "begin another":
        wrapper.begin(opened("ein ganz anderer Anfang", [{"role": "user", "content": "Hallo?"}]))
    elif action == "truncate":
        wrapper.truncate_last_reply("Gu")
    elif action == "wait":
        assert wrapper.wait_idle(timeout_s=_WAIT_S)
    elif action == "close":
        wrapper.close()
    else:
        cancel = threading.Event()
        cancel.set()
        wrapper.reply("Abgebrochen?", cancel=cancel).finish()


class _HeldWork:
    """Holds the work after a reply at one point until it is let go."""

    def __init__(self, wrapper: ChatWrapper, fake: FakeContext, point: str, monkeypatch) -> None:
        self.reached, self.go_on = threading.Event(), threading.Event()
        self.parts = 0
        if point == "before":
            pause = wrapper_module.time.sleep

            def held_before_taking_the_wrapper(seconds):
                if threading.current_thread().name == _THREAD:
                    self.reached.set()
                    assert self.go_on.wait(_WAIT_S), "the work was never let go"
                pause(seconds)
            monkeypatch.setattr(wrapper_module.time, "sleep", held_before_taking_the_wrapper)
        elif point.startswith("part"):
            at_part = int(point.split()[1])
            decode = fake.prefill

            def held_in_a_part(token_ids, start_pos, want_logits):
                if threading.current_thread().name == _THREAD:
                    self.parts += 1
                    if self.parts == at_part:
                        self.reached.set()
                        assert self.go_on.wait(_WAIT_S), "the work was never let go"
                decode(token_ids, start_pos, want_logits)
            fake.prefill = held_in_a_part


def _run(cache: str, point: str, action: str, monkeypatch):
    """The conversation after ``action`` met the work at ``point``, and the tokens decoded from the reply on."""
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="Gut.", **_CACHES[cache])
    wrapper = ChatWrapper(Config(context_size=8192, evict_above_fraction=0.5, min_reply_tokens=8),
                          context=fake)
    wrapper.begin(opened("s", _history()))
    held = _HeldWork(wrapper, fake, point, monkeypatch)
    before = fake.total_prefilled()
    wrapper.reply("frage " + "z" * 200).finish()
    if point == "after":
        assert wrapper.wait_idle(timeout_s=_WAIT_S)
    else:
        assert held.reached.wait(_WAIT_S), f"the work after the reply never reached {point}"
    given = []
    acting = threading.Thread(target=_act, args=(wrapper, action, given), daemon=True)
    acting.start()
    if point == "before":
        acting.join(_WAIT_S)                # it takes the wrapper first, and does the work itself
        held.go_on.set()
    else:
        # It waits for the work, which gives way at its next part.
        wrapper._waiting.wait(timeout=1)
        held.go_on.set()
        acting.join(_WAIT_S)
    assert not acting.is_alive(), f"{action} did not end, met at {point}"
    if action == "close":
        return None, None
    assert wrapper.wait_idle(timeout_s=_WAIT_S)
    internals.bring_cache_back(wrapper)     # a cancel leaves the rest of a long decode to the next action
    assert_consistent(wrapper, fake)
    if given:
        # Rule 10, on the conversation as the begin was given it, which the work after the reply
        # may not have shortened yet: its newest part, each message as it was.
        saved, held = given[0]["messages"], wrapper.conversation()["messages"]
        assert held[:1] == saved[:1] and saved[len(saved) - len(held) + 1:] == held[1:]
        return None, None
    return wrapper.conversation(), fake.total_prefilled() - before


# On a cache that shifts the work after this reply decodes in one part, so it has no second.
_ORDERS = [(cache, point) for cache in _CACHES for point in _POINTS
           if not (cache == "shifts" and point == "part 2")]


@pytest.mark.parametrize("action", _ACTIONS)
@pytest.mark.parametrize("cache, point", _ORDERS, ids=[f"{c}-{p}" for c, p in _ORDERS])
def test_an_action_met_at_any_point_of_the_work_after_a_reply_ends_as_after_it(
        settles_in_the_background, monkeypatch, cache, point, action):
    met, decoded = _run(cache, point, action, monkeypatch)
    if met is None:
        return                              # closed, or checked against what it was given
    waited, decoded_waiting = _run(cache, "after", action, monkeypatch)
    assert met == waited
    assert decoded <= decoded_waiting + 8


@pytest.mark.parametrize("cache", ["cuts", "saves states"])
def test_a_refused_begin_that_met_the_work_after_a_reply_leaves_the_cache_as_it_was(
        settles_in_the_background, monkeypatch, cache):
    """Rule 6: what the interrupted work left undecoded is put back with the rest."""
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="Gut.", **_CACHES[cache])
    wrapper = ChatWrapper(Config(context_size=8192, evict_above_fraction=0.5, min_reply_tokens=8),
                          context=fake)
    wrapper.begin(opened("s", _history()))
    held = _HeldWork(wrapper, fake, "part 1", monkeypatch)
    wrapper.reply("frage " + "z" * 200).finish()
    assert held.reached.wait(_WAIT_S)
    refused = []

    def refused_begin():
        # Refused once it holds the wrapper: a system prompt that leaves no room.
        try:
            wrapper.begin(opened("s" * 9000))
        except ValueError as error:
            refused.append(error)
    acting = threading.Thread(target=refused_begin, daemon=True)
    acting.start()
    wrapper._waiting.wait(timeout=1)
    held.go_on.set()
    acting.join(_WAIT_S)
    assert refused and wrapper.wait_idle(timeout_s=_WAIT_S)
    # As the work left it: every token of the conversation held, those it did not decode
    # decoded by the next action.
    assert wrapper.total_tokens == len(held_tokens(wrapper)) > len(fake.cache)
    before = fake.total_prefilled()
    wrapper.reply("weiter?").finish()
    assert fake.total_prefilled() - before < 1500
    assert_consistent(wrapper, fake)
