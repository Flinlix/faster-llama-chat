"""Every loop of the library ends: each pass makes its variant smaller, or the loop is one of the forms that end by construction."""

import re
import llama_chat.call_parsing
import ast
from pathlib import Path

import pytest

from llama_chat import InternalError, ChatWrapper
from llama_chat.loop_variant import LoopVariant
from llama_chat.call_parsing import parse_calls
from llama_chat.testing.invented_templates import CALL_BODY, PLAIN
from tests.test_call_parsing import _ELEMENTS
from llama_chat.reply_text import ReplyText, reply_header
from llama_chat.template import Reasoning
from tests.test_reply_text import _fed_as_tokens
from llama_chat.config import Config
from llama_chat.testing.fake_context import FakeContext
from tests.chat_messages import opened

_PACKAGE = Path(__file__).parent.parent / "llama_chat"


def _loops():
    """Every ``while`` loop of the library, with the file and line it stands at."""
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.While):
                yield path.name, node.lineno, node


def _takes_each_step(statements) -> bool:
    """Whether ``statements`` take the next step of an iterator, ``x = next(it)``, which raises once it ends, outside a function of their own."""
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "next" \
                and len(node.value.args) == 1 and not node.value.keywords:
            return True
        pending += ast.iter_child_nodes(node)
    return False


def _ends_by_construction(loop: ast.While) -> bool:
    """Whether ``loop`` is a form that ends without a variant of its own.

    A loop that takes each step of an iterator, with no default, ends when the iterator does; an
    endless generator whose every pass yields is stopped by its reader; a work
    list a pass pops from ends once the finite value it was filled from is
    read.
    """
    body = loop.body
    if _takes_each_step(body):
        return True
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Yield):
        return True
    tested = loop.test.values[0] if isinstance(loop.test, ast.BoolOp) else loop.test
    return isinstance(tested, ast.Name) and any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop" and isinstance(node.func.value, ast.Name)
        and node.func.value.id == tested.id for stmt in body for node in ast.walk(stmt))


def _passes_first(loop: ast.While) -> bool:
    """Whether the first statement of every pass of ``loop`` gives a variant its measure."""
    first = loop.body[0]
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) \
        and isinstance(first.value.func, ast.Attribute) and first.value.func.attr == "passes"


@pytest.mark.parametrize("where, line, loop", list(_loops()),
                         ids=[f"{where}:{line}" for where, line, _ in _loops()])
def test_every_loop_makes_progress_on_each_pass_or_ends_by_construction(where, line, loop):
    assert _passes_first(loop) or _ends_by_construction(loop), \
        f"{where}:{line}: give the loop a LoopVariant, its measure as the first statement of each pass"


def test_a_variant_that_does_not_shrink_stops_the_loop_naming_it():
    variant = LoopVariant("a test loop")
    variant.passes(3)
    variant.passes(2)
    with pytest.raises(InternalError, match="a test loop made no progress"):
        variant.passes(2)


def test_a_tuple_variant_shrinks_in_order_of_its_parts():
    variant = LoopVariant("a test loop")
    for measure in [(2, 0), (1, 9), (1, 3), (0, 50)]:
        variant.passes(measure)
    with pytest.raises(InternalError):
        variant.passes((0, 50))


def test_a_variant_below_zero_stops_the_loop():
    with pytest.raises(InternalError):
        LoopVariant("a test loop").passes(-1)


def test_a_reply_whose_turn_never_closes_stops_instead_of_running_on():
    """A generation loop that makes no progress is stopped by its variant, the conversation left usable as after any failed reply."""
    from llama_chat.wrapper import ChatWrapper
    from llama_chat import Config
    from llama_chat.testing.fake_context import FakeContext, ScriptedCall
    from llama_chat.testing.invented_templates import TOOLS
    from tests.chat_messages import opened
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=2048), context=fake)
    wrapper.begin(opened("s"))
    # Every generation an unreadable call the turn goes on after, each costing no token.
    fake.replies = [ScriptedCall("fn:nonsense with no name")] * 50
    real = wrapper._rounds.read_generation

    def free_tokens(deltas, after):
        reply = yield from real(deltas, after)
        from dataclasses import replace
        return replace(reply, n_generated_tokens=0)

    wrapper._rounds.read_generation = free_tokens
    with pytest.raises(InternalError, match="made no progress"):
        wrapper.reply("wie spät?", tool_call_policy="refuse").finish()
    del wrapper._rounds.read_generation
    fake.replies = ["Zwölf."]
    assert wrapper.reply("Und jetzt?").finish().text == "Zwölf."


@pytest.mark.parametrize("body, call_body", [("fn:f{a:1}", CALL_BODY),
                                             ('f">|<a>|<a>1|</a>|</a>|', _ELEMENTS)],
                         ids=["splitting", "matching-close"])
def test_a_loop_of_the_call_reader_that_makes_no_progress_is_named_in_its_error(
        body, call_body, monkeypatch):
    class Stalling(LoopVariant):
        """A variant whose measure never shrinks, as a loop with a fault would give it."""

        def passes(self, measure):
            super().passes(0)

    monkeypatch.setattr(llama_chat.call_parsing, "LoopVariant", Stalling)
    with pytest.raises(InternalError) as raised:
        parse_calls(body, call_body, "fn:")
    loop = re.match(r"(.*) made no progress", str(raised.value)).group(1)
    assert loop != "None" and len(loop.split()) >= 2


@pytest.mark.parametrize("run", [
    lambda: _fed_as_tokens(ReplyText(Reasoning("<t>", "</t>", ""), "", None, "", [], []),
                           [("Hi", False), (" <t> and <t> as text", False)], []),
    lambda: reply_header("<|a|>user<|a|>", Reasoning("<|a|>self<|a|>", "<|b|>", ""), ("<|a|>",)),
], ids=["thought-opening", "tag-around-a-place"])
def test_a_loop_of_the_reply_text_that_makes_no_progress_is_named_in_its_error(run, monkeypatch):
    class StallingLoop(LoopVariant):
        """A variant whose measure never shrinks, as a loop with a fault would give it."""

        def passes(self, measure):
            super().passes(0)

    monkeypatch.setattr(llama_chat.reply_text, "LoopVariant", StallingLoop)
    with pytest.raises(InternalError) as raised:
        run()
    loop = re.match(r"(.*) made no progress", str(raised.value)).group(1)
    assert loop != "None" and len(loop.split()) >= 2


def test_a_loop_of_the_thought_filter_that_makes_no_progress_is_named_in_its_error(monkeypatch):
    class Stalling(LoopVariant):
        """A variant whose measure never shrinks, as a loop with a fault would give it."""

        def passes(self, measure):
            super().passes(0)

    monkeypatch.setattr(llama_chat.reply_text, "LoopVariant", Stalling)
    reply = ReplyText(Reasoning("<thought>", "</thought>", "", ""), "", None, "", [])
    with pytest.raises(InternalError) as raised:
        reply.feed("<thought>x</thought>y")
    loop = re.match(r"(.*) made no progress", str(raised.value)).group(1)
    assert loop != "None" and len(loop.split()) >= 2


def test_a_loop_dropping_exchanges_in_windows_that_makes_no_progress_is_named_in_its_error(monkeypatch):
    class StallingDrop(LoopVariant):
        """A variant whose measure never shrinks, as a loop with a fault would give it."""

        def passes(self, measure):
            super().passes(0)

    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="Gern.")
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.5, min_reply_tokens=8),
                    context=fake)
    w.begin(opened("s"))
    for i in range(10):
        w.reply(f"Frage {i}?").finish()
    w.wait_idle()
    monkeypatch.setattr(llama_chat.windows, "LoopVariant", StallingDrop)
    # Long enough that more than one exchange goes at once.
    with pytest.raises(InternalError) as raised:
        w.reply("lang " * 80).finish()
    loop = re.match(r"(.*) made no progress", str(raised.value)).group(1)
    assert loop != "None" and len(loop.split()) >= 2
    monkeypatch.undo()
    assert w.reply("Und jetzt?").finish().text == "Gern."


def test_a_loop_finding_a_tag_before_a_place_that_makes_no_progress_is_named_in_its_error(monkeypatch):
    class Stalling(LoopVariant):
        """A variant whose measure never shrinks, as a loop with a fault would give it."""

        def passes(self, measure):
            super().passes(0)

    monkeypatch.setattr(llama_chat.template, "LoopVariant", Stalling)
    with pytest.raises(InternalError) as raised:
        llama_chat.template.shared_tags_length("<a><a>", "<a><a>x", ("<a>",))
    loop = re.match(r"(.*) made no progress", str(raised.value)).group(1)
    assert loop != "None" and len(loop.split()) >= 2
