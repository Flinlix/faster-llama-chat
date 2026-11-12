"""Fragment extraction over real chat templates, and over renderers written for a case no real template shows.

A real format is read from its template in ``tests/templates/``. A renderer
written here reproduces no model's format and carries no model's name.
"""

import ast
import functools
import itertools
import re
import time
from pathlib import Path

import pytest

import llama_chat
from llama_chat.call_format import extract_call_body, extract_tool_fragments
from llama_chat.model_template import names_read
from llama_chat.template import CallBody
from llama_chat.template_extract import extract_content_rewrites, extract_fragments, reasoning_switch

from .chat_templates import STEMS, expected, fixture
from .fake_context import FakeContext


def _tool_fragments(render):
    """The call tags a render yields, through both steps of the recovery."""
    return extract_tool_fragments(render, extract_fragments(render))


@functools.cache
def _real(stem):
    """The renderer the test template ``stem`` is loaded with."""
    return FakeContext(fixture(stem)).render_with_model_template


_TOOL_STEMS = [stem for stem in STEMS if fixture(stem)["calls_tools"]]


def _stems_with(key):
    """The test templates whose file states ``key``."""
    return [stem for stem in STEMS if expected(fixture(stem), key)]


@pytest.mark.parametrize("stem", _stems_with("fragments"))
def test_a_templates_turn_fragments_are_the_tags_it_writes(stem):
    known = expected(fixture(stem), "fragments")
    f = extract_fragments(_real(stem))
    assert {name: getattr(f, name) for name in known} == known


def test_no_template_returns_none():
    assert extract_fragments(lambda messages, **_options: None) is None


def _inconsistent_render(messages, **_options):
    """Degenerate template whose user turn changes shape in a longer render.

    The single-message user fragment therefore never reappears.
    """
    if len(messages) == 1:
        return f"<u>{messages[0]['content']}</u>"
    return "".join(f"[{m['role']}]{m['content']}" for m in messages)


def test_inconsistent_multi_message_render_falls_back_to_user_tags():
    # The user fragment recovered from the single-message probe does not occur
    # in the two-message render; the extractor must fall back instead of
    # crashing on the missing split point.
    f = extract_fragments(_inconsistent_render)
    assert f.user_prefix == "<u>"
    assert f.assistant_prefix == f.user_prefix
    assert f.assistant_suffix == f.user_suffix


def _prefix_only_render(messages, **_options):
    """Degenerate template whose tags only open a turn (empty suffixes)."""
    return "".join(f"<[{m['role']}>{m['content']}" for m in messages)


def test_empty_suffix_template_extracts_without_crashing():
    f = extract_fragments(_prefix_only_render)
    assert f.user_prefix == "<[user>"
    assert f.user_suffix == ""
    assert f.assistant_prefix == "<[assistant>"
    assert f.assistant_suffix == ""


@pytest.mark.parametrize("stem", _stems_with("tool_fragments"))
def test_a_tool_calling_template_yields_its_call_tags(stem):
    tools = _tool_fragments(_real(stem))
    assert tools is not None
    # The argument's own text is gone from the closing; what is left is the
    # call's closing tag, with what the template writes around it.
    known = expected(fixture(stem), "tool_fragments")
    assert {name: getattr(tools, name) for name in known} == known


@pytest.mark.parametrize("stem", [stem for stem in STEMS if not fixture(stem)["calls_tools"]])
def test_a_template_that_cannot_call_tools_has_no_tool_fragments(stem):
    assert _tool_fragments(_real(stem)) is None


def _untagged_call_render(messages, **_options):
    """Degenerate template with no role tags at all, yet it renders tool calls."""
    out = []
    for m in messages:
        for call in m.get("tool_calls", ()):
            out.append(f"[call {call['function']['name']}]")
        out.append(m["content"])
    return "".join(out)


def test_a_call_with_no_locatable_turn_yields_no_tool_fragments():
    # Nothing marks where the turn holding the call begins, so the call's own
    # opening cannot be told from the conversation before it.
    assert _tool_fragments(_untagged_call_render) is None


def _argless_call_render(messages, **_options):
    """Degenerate template that renders a call's name but drops its arguments."""
    out = []
    for m in messages:
        out.append(f"<{m['role']}>")
        for call in m.get("tool_calls", ()):
            out.append(f"<c>{call['function']['name']}</c>")
        out.append(m["content"])
        out.append(f"</{m['role']}>")
    return "".join(out)


def test_a_template_that_drops_arguments_yields_no_tool_fragments():
    # Both probes render identically, so their shared ending runs back past the
    # call and describes the conversation rather than the call's closing tag.
    assert _tool_fragments(_argless_call_render) is None


@pytest.mark.parametrize("stem", _stems_with("call_body"))
def test_the_call_body_is_spelled_as_the_template_writes_it(stem):
    known = fixture(stem)
    render = _real(stem)
    body = extract_call_body(render, _tool_fragments(render).call_open,
                             expected(known, "call_closing_tag"))
    assert body == CallBody(**expected(known, "call_body"))


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_a_model_that_renders_no_call_has_no_body_either(stem):
    render = _real(stem)
    tools = _tool_fragments(render)
    if tools is None:
        return                          # no call tags to measure from at all
    # A closing tag this template never emits leaves nothing to measure.
    assert extract_call_body(render, tools.call_open, "<nope>") is None


# ----- giving up rather than guessing -------------------------------------
# Each renderer below is a template the probes cannot measure some delimiter
# from. The extractor has to answer ``None``, because a guess would be written
# into every prefill for the rest of the conversation.
def _turn_tags(m, body):
    """One turn of a delimited model, wrapping ``body`` in the usual tags."""
    role = "model" if m["role"] == "assistant" else m["role"]
    return f"<[turn>{role}\n{body}{m['content'].strip()}<turn]>\n"


def _no_call_without_arguments_render(messages, **_options):
    """Renders a call only when it has arguments, and nothing when it has none."""
    out = []
    for m in messages:
        body = ""
        for call in m.get("tool_calls", ()):
            fn = call["function"]
            if not fn["arguments"]:
                continue
            pairs = ",".join(f"{k}:'{v}'" for k, v in fn["arguments"].items())
            body += f"<call>fn:{fn['name']}{{{pairs}}}</call>"
        out.append(_turn_tags(m, body))
    return "".join(out)


def test_a_call_shape_that_vanishes_without_arguments_has_no_body():
    tools = _tool_fragments(_no_call_without_arguments_render)
    assert tools is not None            # the tags are still recoverable
    # But the no-argument probe renders no call, so `empty_close` cannot be
    # measured and the whole call_body is refused.
    assert extract_call_body(_no_call_without_arguments_render,
                             tools.call_open, "</call>") is None


def _nothing_after_the_name_render(messages, **_options):
    """Writes the arguments straight after the name, with no delimiter at all."""
    out = []
    for m in messages:
        body = ""
        for call in m.get("tool_calls", ()):
            fn = call["function"]
            pairs = ",".join(f"{k}={v}" for k, v in fn["arguments"].items())
            body += f"<call>fn:{fn['name']}{pairs}</call>"
        out.append(_turn_tags(m, body))
    return "".join(out)


def test_a_name_that_runs_straight_into_its_arguments_has_no_body():
    tools = _tool_fragments(_nothing_after_the_name_render)
    # The one-argument and no-argument renders share nothing after the name, so
    # there is no `name_close` to find and nothing downstream can be trusted.
    assert extract_call_body(_nothing_after_the_name_render,
                             tools.call_open, "</call>") is None


def _constant_body_render(messages, **_options):
    """Renders a fixed ``{}`` whatever the arguments, so no key ever appears.

    What a template does when it reads the arguments from a field the caller
    did not populate: the braces survive, the keys never do.
    """
    out = []
    for m in messages:
        body = ""
        for call in m.get("tool_calls", ()):
            body += f"<call>fn:{call['function']['name']}{{}}</call>"
        out.append(_turn_tags(m, body))
    return "".join(out)


def test_a_body_that_never_shows_a_key_has_no_body():
    """The body probes refuse a template that writes no key, whatever tags they are given.

    ``extract_tool_fragments`` refuses this template itself, so here the body
    probes run against a template they were never screened for.
    """
    assert _tool_fragments(_constant_body_render) is None
    # `{}` is common to both probes, so `name_close` is found and its guard
    # passes; the keys the splits below index on are still absent. Refused
    # rather than indexed off the end.
    assert extract_call_body(_constant_body_render, "<call>fn:", "</call>") is None


def _typed_key_render(messages, **_options):
    """Spells the argument's type into its key, so no two probes agree on it."""
    out = []
    for m in messages:
        body = ""
        for call in m.get("tool_calls", ()):
            fn = call["function"]
            pairs = ",".join(
                f"{k}:str='{v}'" if isinstance(v, str) else f"{k}:int={v}"
                for k, v in fn["arguments"].items())
            body += f"<call>fn:{fn['name']}{{{pairs}}}</call>"
        out.append(_turn_tags(m, body))
    return "".join(out)


def test_a_key_that_changes_with_the_value_type_has_no_body():
    tools = _tool_fragments(_typed_key_render)
    # The string probe writes `:str=` and the number probe `:int=`, so what
    # stands between a key and its value cannot be told from the value itself.
    assert extract_call_body(_typed_key_render, tools.call_open, "</call>") is None


def _last_value_only_quoted_render(messages, **_options):
    """Quotes only the final argument, so the quoting is not a fixed delimiter."""
    out = []
    for m in messages:
        body = ""
        for call in m.get("tool_calls", ()):
            fn = call["function"]
            items = list(fn["arguments"].items())
            pairs = ",".join(
                f"{k}:'{v}'" if at == len(items) - 1 and isinstance(v, str)
                else f"{k}:{v}"
                for at, (k, v) in enumerate(items))
            body += f"<call>fn:{fn['name']}{{{pairs}}}</call>"
        out.append(_turn_tags(m, body))
    return "".join(out)


def test_quoting_that_depends_on_position_has_no_body():
    tools = _tool_fragments(_last_value_only_quoted_render)
    # One argument is quoted and the first of two is not, so what follows a
    # value does not match what separates one value from the next key.
    assert extract_call_body(_last_value_only_quoted_render,
                             tools.call_open, "</call>") is None


def _no_tool_render(render):
    """``render``, refusing a message carrying a tool call outright."""
    def refusing(messages, **options):
        if any(m.get("tool_calls") for m in messages):
            return None
        return render(messages, **options)
    return refusing


@pytest.mark.parametrize("stem", _TOOL_STEMS)
def test_a_template_that_refuses_tool_calls_has_no_tool_fragments(stem):
    assert _tool_fragments(_no_tool_render(_real(stem))) is None


def _dropped_assistant_render(messages, **_options):
    """Drops an assistant turn's content, yet still renders its tool calls.

    The assistant fragment cannot be recovered from a render its sentinel never
    reaches, so extraction falls back to the user tags, while a call probe
    still produces a call.
    """
    out = []
    for m in messages:
        if m["role"] != "assistant":
            out.append(f"<[turn>{m['role']}\n{m['content'].strip()}<turn]>\n")
            continue
        for call in m.get("tool_calls", ()):
            fn = call["function"]
            pairs = ",".join(f"{k}:'{v}'" for k, v in fn["arguments"].items())
            out.append(f"<call>fn:{fn['name']}{{{pairs}}}</call>")
    return "".join(out)


def test_a_fallen_back_assistant_tag_is_not_used_to_locate_a_call():
    """The user tags standing in for a missing assistant tag locate nothing.

    Splitting on them lands in the middle of the conversation, and what came
    back would open a call with a turn tag rather than with the call's own.
    """
    fragments = extract_fragments(_dropped_assistant_render)
    assert fragments.assistant_prefix == fragments.user_prefix   # it fell back
    assert extract_tool_fragments(_dropped_assistant_render, fragments) is None


def _degenerate_render(dropped, constant_body):
    """A delimited render with each text of ``dropped`` deleted from every call."""
    def render(messages, tools=None, **_options):
        out = []
        for m in messages:
            body = ""
            for call in m.get("tool_calls", ()):
                fn = call["function"]
                pairs = "" if constant_body else ",".join(
                    f"{k}:{v!r}" if isinstance(v, str) else f"{k}:{v}"
                    for k, v in fn["arguments"].items())
                text = f"<call>fn:{fn['name']}{{{pairs}}}</call>"
                for written in dropped:
                    text = text.replace(written, "")
                body += text
            out.append(_turn_tags(m, body))
        return "".join(out)
    return render


def _leaves(value):
    """The keys and the plain values in ``value``, nested ones included, as text."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield str(value)


def _texts_the_call_probes_write() -> list[str]:
    """Every name, key and value the call probes ask a render to write, recorded as they ask."""
    written, sound = set(), _degenerate_render((), constant_body=False)

    def recording(messages, tools=None, **options):
        for message in messages:
            for call in message.get("tool_calls", ()):
                written.add(call["function"]["name"])
                written.update(_leaves(call["function"]["arguments"]))
        return sound(messages, tools, **options)

    extract_call_body(recording, "<call>fn:", "</call>")
    return sorted(written)


def test_no_render_makes_body_extraction_raise():
    """Refusing is the only failure; a template cannot crash construction.

    Every subset of the texts the call probes write that a template might
    drop, against both a rendered and a constant body. Before the guard on
    them, some of these indexed off the end of a split instead of returning None.
    """
    written = _texts_the_call_probes_write()
    for size in range(len(written) + 1):
        for dropped in itertools.combinations(written, size):
            for constant_body in (False, True):
                render = _degenerate_render(dropped, constant_body)
                body = extract_call_body(render, "<call>fn:", "</call>")
                assert body is None or isinstance(body, CallBody)


def _rendering(open_thought):
    """A render of the probe conversation whose ending ``open_thought(variables)`` decides."""
    def render(messages, generation_prompt=False, variables=None):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + (open_thought(variables or {}) if generation_prompt else "")
    return render


_SHARED_FLAG = ({"enable_thinking": True}, {"enable_thinking": False})


@pytest.mark.parametrize("ending, names, switch_on, switch_off", [
    # Reads the shared variable: it stays the switch.
    (lambda v: "<reason>" if v.get("enable_thinking") else "<reason></reason>", (), *_SHARED_FLAG),
    # Reads only an effort, and that opens the thought: the effort is the switch.
    (lambda v: "<reason>" if v.get("reasoning_effort") == "high" else "<reason></reason>", (),
     {"reasoning_effort": "high"}, {}),
    # Reasons unless the effort is none: none is off, and the template's own default is on.
    (lambda v: "<reason></reason>" if v.get("reasoning_effort") == "none" else "<reason>", (),
     {}, {"reasoning_effort": "none"}),
    # Reasons by default, writes more for high, and stops only for none: the default
    # stays on rather than the higher effort.
    (lambda v: {"none": "<reason></reason>", "high": "<more><reason>"}.get(
        v.get("reasoning_effort"), "<reason>"), (),
     {}, {"reasoning_effort": "none"}),
    # An effort that changes nothing after the user turn is not a switch.
    (lambda v: "", (), *_SHARED_FLAG),
    # A flag of another name that opens the thought is the switch.
    (lambda v: "<reason>" if v.get("thinking") is True else "<reason></reason>",
     ("messages", "thinking"), {"thinking": True}, {"thinking": False}),
    # An effort of another name is read by the effort's rule.
    (lambda v: "<reason></reason>" if v.get("reasoning_strength") == "none" else "<reason>",
     ("reasoning_strength",), {}, {"reasoning_strength": "none"}),
    # The shared variable is tried first, where the template reads another as well.
    (lambda v: "<reason>" if v.get("enable_thinking") or v.get("thinking") else "<reason></reason>",
     ("thinking",), *_SHARED_FLAG),
    # A variable whose name speaks of neither thinking nor reasoning is no switch,
    # however it changes what follows the user turn.
    (lambda v: "<reason>" if v.get("low_effort") is False else "<reason></reason>",
     ("low_effort",), *_SHARED_FLAG),
    # Nor is one that speaks of thinking and changes nothing after the user turn.
    (lambda v: "", ("keep_thinking",), *_SHARED_FLAG),
])
def test_the_reasoning_switch_is_the_variable_that_opens_the_thought(ending, names, switch_on,
                                                                     switch_off):
    switch = reasoning_switch(_rendering(ending), names)
    assert switch(True) == switch_on
    assert switch(False) == switch_off
    assert switch(None) == {}


@pytest.mark.parametrize("stem", STEMS)
def test_the_names_a_template_reads_are_read_as_the_renderer_parses_it(stem):
    """A tag only the renderer knows, such as a loop's ``break``, leaves the names readable."""
    names = names_read(fixture(stem)["chat_template"])
    assert "messages" in names


def test_a_variable_named_as_a_render_argument_reaches_the_template():
    context = FakeContext({"chat_template": "{{ reasoning }}/{{ generation_prompt }}",
                           "eos_token": "", "render_variables": {"off": {}}})
    rendered = context.render_with_model_template(
        [], generation_prompt=True, reasoning=False,
        variables={"reasoning": "deep", "generation_prompt": "given"})
    assert rendered == "deep/given"


# ----- what a template writes in a message's content -----------------------
def _escaping_render(messages, **options):
    """A user turn whose content has ``<`` escaped and the tag ``<drop>`` left out."""
    return "<u>" + messages[-1]["content"].replace("<drop>", "").replace("<", "&lt;") + "</u>"


def test_each_text_a_template_changes_in_content_is_read_as_the_template_writes_it():
    assert extract_content_rewrites(_escaping_render, ["<drop>", "<keep>", "<", ">", "plain"]) \
        == (("<drop>", ""), ("<keep>", "&lt;keep>"), ("<", "&lt;"))


def test_a_render_that_loses_a_texts_place_reads_no_rewrites():
    def truncating(messages, **options):
        return messages[-1]["content"][:-1]
    assert extract_content_rewrites(truncating, ["<a>", "<b>"]) == ()


def test_the_rewrites_of_many_texts_are_read_in_one_pass_over_the_render():
    texts = [f"<tag{n}>" for n in range(20_000)]
    started = time.perf_counter()
    rewrites = extract_content_rewrites(_escaping_render, texts)
    assert time.perf_counter() - started < 0.5
    assert len(rewrites) == len(texts)


# ----- probe sentinels -----------------------------------------------------
# A probe's sentinel: a text of capital letters and digits, which trimming,
# escaping and quoting leave whole.
_SENTINEL = re.compile(r"[A-Z][A-Z0-9]{3,}")


def _sentinels() -> dict[str, str]:
    """Every sentinel the library defines, by the module and name it is defined under."""
    found = {}
    for path in sorted(Path(llama_chat.__file__).parent.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) and _SENTINEL.fullmatch(node.value.value):
                found.update({f"{path.stem}.{target.id}": node.value.value
                              for target in node.targets})
    return found


def test_no_render_sentinel_contains_another():
    """A split on one has to land on that one, whatever else the render holds.

    ``PROBE`` used to sit inside ``RESULTPROBE`` and ``TAILPROBE``, so a render
    carrying all three could be cut six characters into the wrong sentinel.
    """
    sentinels = _sentinels()
    assert {text for text in _texts_the_call_probes_write() if _SENTINEL.fullmatch(text)} \
        <= set(sentinels.values())
    assert len(set(sentinels.values())) == len(sentinels)
    for name, value in sentinels.items():
        others = [v for n, v in sentinels.items() if n != name]
        assert not any(value in other for other in others), \
            f"{name}={value!r} occurs inside another sentinel"
