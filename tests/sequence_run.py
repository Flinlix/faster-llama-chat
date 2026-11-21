"""One random sequence of actions on one template, cache kind and tokenizer, and the checks every state it reaches must pass.

Every choice is drawn from ``draw``, which answers ``random()``, ``choice()``
and ``randint()``, so that a test framework that shrinks its draws shrinks
the sequence. The checks raise ``AssertionError`` naming the rule a state
breaks.
"""

from __future__ import annotations

import json

from llama_cpp_conversation import (ChatWrapper, Config, ContextOverflowError, OutOfOrderError,
                        TemplateRefusedError, ToolRegistry, ToolSpec, UnsupportedError)
from llama_cpp_conversation.errors import LIBRARY_ERRORS, EngineError, InternalError, InvalidArgumentError
from tests import internals
from llama_cpp_conversation.testing import invented_templates as invented
from tests.chat_templates import STEMS, fixture
from llama_cpp_conversation.testing.fake_context import FakeContext, ScriptedCall, shipped_addon_call, written_call
from tests.test_failures import _Flaky
from tests.wrappers import assert_cache_reads_as_rendered, assert_conversation_is_its_forms

INVENTED = {"tools": invented.TOOLS, "json tools": invented.JSON_TOOLS,
            "two questions refused": invented.TOOLS_REFUSING_TWO_USER_TURNS,
            "reasoning": invented.REASONING, "thoughts kept": invented.REASONING_KEEPING_THOUGHTS,
            "open turns closed": invented.TOOLS_CLOSING_OPEN_TURNS,
            "result lines": invented.RESULT_LINES, "json result lines": invented.JSON_RESULT_LINES,
            "declared in the newest turn": invented.TOOLS_DECLARED_IN_THE_NEWEST_TURN,
            "a template of their own": invented.TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN}
REAL = sorted(STEMS)
TEMPLATES = sorted(INVENTED) + REAL
CACHES = [dict(can_shift=True), dict(can_shift=False, can_cut_tail=True), dict(can_shift=False),
          dict(can_shift=True, decode_blocks=True),
          dict(can_shift=False, can_cut_tail=True, window_tokens=24)]
KINDS = ("question", "reply", "result", "inject", "truncate", "begin", "wait")

REGISTRY = ToolRegistry()


@REGISTRY.tool
def get_weather(city: str) -> dict:
    """Weather for a city."""
    if city == "Fail":
        raise RuntimeError("tool broke")
    if city == "Err":
        return {"error": "no such city"}
    return {"city": city, "celsius": 18}


@REGISTRY.tool
def clock() -> str:
    """The time."""
    return "12:00"


CALENDAR = ToolSpec("kalender", "Die Termine.", {"type": "object", "properties": {}})

# Texts no model can read, and texts every model must take.
LONE_SURROGATE = "halb \ud800 Zeichen"
ODD_TEXTS = ["\u200b", "🙂 ü ß \u202e", "\x00\x1b[0m", "a\u0301\u0301\u0301", " " * 30]


class _BrokenMapping(dict):
    """A result whose own methods fail when JSON writes it."""

    def items(self):
        raise KeyError("broken")

    def __iter__(self):
        raise KeyError("broken")


def _deep(levels: int):
    value: list = []
    for _ in range(levels):
        value = [value]
    return value


def _known(name: str) -> dict:
    """The template file of ``name``, invented or real."""
    return INVENTED[name] if name in INVENTED else fixture(name)


def _bodies(name: str, fake) -> list[str]:
    """Call bodies the scripted model may write: read, an unknown name, a failing tool, unreadable."""
    call = shipped_addon_call(_known(name)["chat_template"]) if name in REAL else None
    written = []
    for function, arguments in (("get_weather", {"city": "Bonn"}), ("clock", {}),
                                ("nosuch", {"a": "b"}), ("get_weather", {"city": "Err"}),
                                ("get_weather", {"city": "Fail"})):
        if call:
            written.append(function + call["name_close"] + json.dumps(arguments))
        else:
            try:
                written.append(written_call(None, fake, function, arguments)[1])
            except Exception:
                pass                        # a template that writes no such call
    return written + ["@@ not a call at all"]


class SequenceRun:
    """A wrapper on one drawn setup, the actions drawn for it, and the checks after each."""

    def __init__(self, draw, note) -> None:
        self.draw, self.note = draw, note
        self.name = draw.choice(TEMPLATES)
        self.cache_kind = draw.choice(CACHES)
        self.merges = draw.random() < 0.5
        self.writes_tags = draw.random() < 0.3
        self.size = (draw.choice([700, 1200, 2000, 4000, 8192]) if self.name in REAL
                     else draw.choice([250, 400, 700, 2048]))
        self.evict = draw.choice([0.5, 0.75, 0.9, 1.0])
        self.max_exchanges = draw.choice([None, None, None, 2, 4])
        note(f"setup: {self.name}, cache {self.cache_kind}, merges {self.merges}, "
             f"tags written {self.writes_tags}, context {self.size}, evict {self.evict}, "
             f"exchanges {self.max_exchanges}")
        self.fake = self._new_fake(flaky=True)
        self.wrapper = ChatWrapper(self.config(), context=self.fake)
        self.bodies = _bodies(self.name, self.fake)
        self.client_ids = self.questions = 0
        self.reply_failed = False
        self.shown = ""

    def config(self) -> Config:
        return Config(context_size=self.size, min_reply_tokens=16, evict_above_fraction=self.evict,
                      max_history_exchanges=self.max_exchanges)

    def _new_fake(self, flaky: bool = False):
        return (_Flaky if flaky else FakeContext)(_known(self.name), merges=self.merges,
                                                  writes_tags=self.writes_tags, **self.cache_kind)

    # ----- the checks -------------------------------------------------------------
    def _state(self):
        wrapper = self.wrapper
        return (wrapper.conversation(), list(wrapper.pending_calls), wrapper.turn_open,
                wrapper.total_tokens, wrapper.awaits_reply)

    def check(self, where: str) -> None:
        """Rule 1, the cache is the render; the conversation is its two forms; rule 10, a fresh begin gives the same conversation."""
        wrapper, fake = self.wrapper, self.fake
        fake.fail_at = fake.fail = None
        assert wrapper.wait_idle(timeout_s=10), f"the work after a reply did not end after {where}"
        try:
            assert_cache_reads_as_rendered(wrapper, fake)
        except AssertionError:
            # A failed reply leaves the cache for the next action, which a question is.
            assert self.reply_failed, f"rule 1: the cache is not the render after {where}"
            self.reply_failed = False
            if self.question_answered(f"{where}, after a failed reply"):
                self.check(f"a question after a failed reply at {where}")
            return
        self.reply_failed = False
        assert_conversation_is_its_forms(wrapper)
        if not internals.is_begun(wrapper):
            return
        saved = wrapper.conversation()
        again = ChatWrapper(self.config(), context=self._new_fake())
        try:
            kept = again.begin(**saved).n_kept_messages
            system = [m for m in saved["messages"][:1] if m["role"] == "system"]
            history = saved["messages"][len(system):]
            got = again.conversation()
            assert got["messages"] == system + history[len(history) - kept:], \
                f"rule 10: a fresh begin kept other messages after {where}"
            if kept == len(history):
                assert got == saved, f"rule 10: a fresh begin gives another conversation after {where}"
            assert_cache_reads_as_rendered(again, again._context)
            assert_conversation_is_its_forms(again)
            # From its messages alone, as another model or a client without the turns has it.
            again.begin(saved["messages"], tools=saved["tools"], reasoning=saved["reasoning"])
        except ContextOverflowError:
            raise AssertionError(f"rule 10: a fresh begin refused the conversation for its length "
                                 f"after {where}") from None
        finally:
            again.close()

    def question_answered(self, where: str) -> bool:
        """Rule 11: a question asked now is answered, or refused as one the template does not write, which changes nothing."""
        wrapper, fake = self.wrapper, self.fake
        fake.call_script, fake.replies = [], []
        before = self._state()
        self.questions += 1
        text = f"Und jetzt bitte {self.questions}?"
        try:
            reply = wrapper.reply(text).finish()
        except ContextOverflowError as refused:
            raise AssertionError(f"rule 11: a question was refused for its length after {where}: "
                                 f"{refused}") from None
        except TemplateRefusedError:
            self._assert_refused_alone(text, where)
            self._assert_unchanged(before, f"a refused question after {where}")
            return False
        assert not wrapper.turn_open or reply.stop_reason == "tool_call", \
            f"rule 11: a turn stays open after an answered question after {where}"
        assert reply.text.strip() or reply.tool_calls, \
            f"rule 11: a question got an empty reply after {where}"
        return True

    def _assert_refused_alone(self, text: str, where: str) -> None:
        """A question refused as one the template does not write is refused beside the system prompt alone too."""
        saved = self.wrapper.conversation()
        system = [m for m in saved["messages"][:1] if m["role"] == "system"]
        alone = ChatWrapper(self.config(), context=self._new_fake())
        try:
            alone.begin(system, tools=saved["tools"], reasoning=saved["reasoning"])
            alone._context.replies = []
            alone.reply(text).finish()
        except TemplateRefusedError:
            return
        finally:
            alone.close()
        raise AssertionError(f"rule 11: a question the template writes was refused after {where}")

    def _assert_unchanged(self, before, where: str) -> None:
        self.wrapper.wait_idle(timeout_s=10)
        after = self._state()
        assert after == before, f"rule 6: {where} changed the conversation"

    # ----- the actions --------------------------------------------------------------
    def first_begin(self) -> bool:
        """Begin the conversation as drawn; whether it fits."""
        draw, wrapper = self.draw, self.wrapper
        # A template without tools is begun without them, and its replies call nothing it reads.
        calls = wrapper.capabilities(tools=True).can_call_tools
        can = wrapper.capabilities(tools=calls)
        system = ([{"role": "system", "content": draw.choice(["Be brief.", "", "Du bist hilfreich. " * 5])}]
                  if can.can_take_system_prompt and draw.random() < 0.85 else [])
        # One draw decides the reasoning and its level, so that sequences kept from before replay alike.
        chance = draw.random() if can.can_reason else 1.0
        levels = (None,) + can.reasoning_levels
        reasoning, level = chance < 0.4, levels[min(int(chance / 0.4 * len(levels)), len(levels) - 1)]
        level = level if reasoning else None
        history = ([{"role": "user", "content": "alt"}, {"role": "assistant", "content": "älter"}]
                   if draw.random() < 0.4 else [])
        try:
            wrapper.begin(system + history, tools=REGISTRY.specs() if calls else None,
                          reasoning=reasoning, reasoning_level=level)
        except ContextOverflowError:
            return False
        return True

    def _script(self) -> None:
        draw, fake = self.draw, self.fake
        replies = [ScriptedCall(draw.choice(self.bodies),
                                before=draw.choice(["", "", "Moment.", "Ich schaue"]),
                                closes=draw.random() < 0.85)
                   for _ in range(draw.choice([0, 0, 1, 1, 2, 3]))]
        if not self.wrapper.capabilities(tools=True).can_call_tools:
            replies = []                # a template without calls is never scripted one
        replies.append(draw.choice(["Fertig.", "", "Gut so", " spaced\n", "Ein Satz. " * draw.randint(3, 40),
                                    "Antwort <call> mit Tag.", "Denken</thought>Antwort.",
                                    "<thought>Denken</thought>Antwort."]))
        if draw.random() < 0.3:
            replies += ["Weiter.", "Danach."]
        fake.replies, fake.call_script = replies, []

    def _reply(self, text: str | None) -> None:
        draw, wrapper = self.draw, self.wrapper
        self._script()
        policy = draw.choice(["yield", "refuse", REGISTRY, "withhold", None])
        options = {}
        if draw.random() < 0.3:
            options["max_reply_tokens"] = draw.choice([1, 5, 10, 40, 200])
        if draw.random() < 0.2:
            options["stop"] = draw.choice(["Satz", "Fertig", "\n"])
        cancel_after = close_after = None
        if draw.random() < 0.15:
            import threading
            options["cancel"] = threading.Event()
            if draw.random() < 0.4:
                options["cancel"].set()
            else:
                cancel_after = draw.randint(0, 5)
        if draw.random() < 0.15:
            close_after = draw.randint(0, 4)
        self.note(f"reply {text!r}, policy {policy.__class__.__name__ if policy is REGISTRY else policy}, "
                  f"{options}, cancel after {cancel_after}, close after {close_after}")
        stream = wrapper.reply(text, tool_call_policy=policy, **options)
        shown = []
        try:
            for n, event in enumerate(stream, 1):
                if isinstance(event, str):
                    shown.append(event)
                if cancel_after is not None and n > cancel_after:
                    options["cancel"].set()
                if close_after is not None and n > close_after:
                    break
        finally:
            stream.close()
        self.shown = "".join(shown)

    def act(self, kind: str, step: int) -> None:
        """Run one action of ``kind``, a failure injected into it where drawn, and check what a refusal or failure left."""
        draw, wrapper, fake = self.draw, self.wrapper, self.fake
        fails = draw.random() < 0.15
        if fails:
            fake.operations, fake.fail_at = 0, draw.randint(0, 60)
            self.note(f"a failure at operation {fake.fail_at}")
        try:
            if kind == "question":
                text = draw.choice([f"Frage {step}?", "frage", f" spaced {step}\n",
                                    "x" * draw.choice([5, 200, 900, 20000]), "", LONE_SURROGATE,
                                    *ODD_TEXTS])
                before = None if fails else self._state()
                try:
                    self._reply(text)
                except UnsupportedError as refused:
                    # A tool runner given to a conversation that cannot call tools: the
                    # argument is refused, not the question.
                    assert refused.field == "tool_call_policy", f"rule 11: {refused}"
                    if before is not None:
                        self._assert_unchanged(before, f"a refused tool runner at step {step}")
                except ContextOverflowError as refused:
                    raise AssertionError(f"rule 11: a question was refused for its length at step "
                                         f"{step}: {refused}") from None
                except InvalidArgumentError as refused:
                    assert text == LONE_SURROGATE and refused.field == "text", f"rule 11: {refused}"
                    if before is not None:
                        self._assert_unchanged(before, f"an unreadable question at step {step}")
                except TemplateRefusedError:
                    if before is not None:
                        self._assert_unchanged(before, f"a refused question at step {step}")
            elif kind == "reply":
                before = None if fails or wrapper.awaits_reply else self._state()
                try:
                    self._reply(None)
                except (OutOfOrderError, UnsupportedError):
                    if before is not None:
                        self._assert_unchanged(before, f"a refused reply at step {step}")
            elif kind == "result":
                pending = wrapper.pending_calls
                call_id = (pending[0].id if pending and draw.random() < 0.9
                           else draw.choice(["nope", "call_x"] + [c.id for c in pending]))
                result = draw.choice(["sunny", "x" * draw.choice([10, 500, 3000, 20000]),
                                      {"error": "kaputt"}, {"t": 1, "u": [1, 2]}, "", float("nan"),
                                      LONE_SURROGATE, _BrokenMapping(a=1), _deep(5000),
                                      *ODD_TEXTS])
                self.note(f"result for {call_id}: {repr(result)[:40]}")
                self._guarded(lambda: wrapper.inject_tool_result(call_id, result), fails, step, "result")
            elif kind == "inject":
                role = draw.choice(["user", "user", "assistant"])
                text = draw.choice([f"Notiz {step}", "", "y" * draw.choice([10, 400, 20000]),
                                    LONE_SURROGATE, *ODD_TEXTS])
                self.note(f"inject {role} {text[:20]!r}")
                self._guarded(lambda: wrapper.inject(text, role), fails, step, "inject")
            elif kind == "truncate":
                keep = draw.choice(self._cuts())
                self.note(f"truncate to {keep[:40]!r}")
                self._guarded(lambda: wrapper.truncate_last_reply(keep), fails, step, "truncate")
            elif kind == "begin":
                arguments = self._edited_begin()
                self.note(f"begin with {len(arguments['messages'])} messages, tools "
                          f"{None if arguments['tools'] is None else len(arguments['tools'])}, "
                          f"reasoning {arguments['reasoning']}, turns {bool(arguments['generated_turns'])}")
                self._guarded(lambda: wrapper.begin(**arguments), fails, step, "begin")
            else:
                wrapper.wait_idle(timeout_s=10)
        except InternalError:
            raise
        except EngineError:
            assert fails, f"an engine failure at step {step} with no failure injected"
            if kind in ("question", "reply"):
                self.reply_failed = True
        except (TemplateRefusedError, ContextOverflowError, OutOfOrderError):
            pass
        finally:
            if fails and kind in ("question", "reply") and fake.fail_at is None:
                self.reply_failed = True        # the failure may have hit the work after the reply
            fake.fail_at = fake.fail = None

    def _guarded(self, action, fails: bool, step: int, what: str) -> None:
        """Run an action that is no reply; a refusal or a failure leaves the conversation as it was (rule 6)."""
        saved_fail, self.fake.fail_at = self.fake.fail_at, None
        self.wrapper.wait_idle(timeout_s=10)
        before = self._state()
        self.fake.fail_at = saved_fail
        try:
            action()
        except InternalError:
            raise
        except LIBRARY_ERRORS as error:
            self.fake.fail_at = None
            if isinstance(error, EngineError):
                assert fails, f"{what} raised {error!r} at step {step} with no failure injected"
            self._assert_unchanged(before, f"a refused or failed {what} at step {step}")

    def _cuts(self) -> list[str]:
        messages = self.wrapper.conversation()["messages"]
        last = messages[-1] if messages else None
        cuts = ["", "nicht vorn"]
        if last and last["role"] == "assistant" and last.get("content"):
            text = last["content"]
            cuts += [text[:self.draw.randint(0, len(text))], text, text[:1]]
        if self.shown:
            cuts += [self.shown[:self.draw.randint(0, len(self.shown))], self.shown,
                     "\n\n" + ((last or {}).get("content") or "")[:3]]
        return cuts

    def _edited_begin(self) -> dict:
        draw, wrapper = self.draw, self.wrapper
        saved = wrapper.conversation()
        messages = [dict(m) for m in saved["messages"]]
        arguments = dict(saved)
        system = messages[:1] if messages and messages[0]["role"] == "system" else []
        history = messages[len(system):]
        edit = draw.choice(["as it is", "last reply dropped", "oldest exchange dropped",
                            "a question added", "a client's call added", "other tools",
                            "other reasoning"])
        if edit == "last reply dropped" and history and history[-1]["role"] == "assistant":
            history = history[:-1]
        elif edit == "oldest exchange dropped":
            asked = [i for i, m in enumerate(history) if m["role"] == "user"]
            if len(asked) >= 2:
                history = history[asked[1]:]
        elif edit == "a question added":
            history = history + [{"role": "user", "content": draw.choice(["Neue Frage?", "frage"])}]
        elif edit == "a client's call added":
            self.client_ids += 1
            call_id = f"client{self.client_ids:05d}"
            history = history + [{"role": "user", "content": "Wetter?"}] * (draw.random() < 0.5) + [
                {"role": "assistant", "content": draw.choice([None, "", "Ich schaue."]),
                 "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {"name": "get_weather", "arguments": '{"city": "Rom"}'}}]}]
            if draw.random() < 0.7:
                history.append({"role": "tool", "tool_call_id": call_id, "content": "warm"})
                if draw.random() < 0.5:
                    history.append({"role": "assistant", "content": "Es ist warm."})
        elif edit == "other tools":
            arguments["tools"] = draw.choice([None, [], [CALENDAR], REGISTRY.specs() + [CALENDAR],
                                              REGISTRY.specs()])
        elif edit == "other reasoning":
            if wrapper.capabilities(tools=saved["tools"] is not None).can_reason:
                arguments["reasoning"] = not saved["reasoning"]
                arguments["reasoning_level"] = None
        if draw.random() < 0.25:
            arguments["generated_turns"] = None
        arguments["messages"] = system + history
        self.note(f"  edited: {edit}")
        return arguments
