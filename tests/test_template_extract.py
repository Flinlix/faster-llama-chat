"""Fragment extraction over real chat templates, and over renderers written for a case no real template shows.

A real format is read from its template in ``tests/templates/``. A renderer
written here reproduces no model's format and carries no model's name.
"""

import ast
import functools
import itertools
import re
from pathlib import Path

import pytest

import llama_chat
from llama_chat.call_format import extract_call_body, extract_tool_fragments
from llama_chat.model_template import names_read
from llama_chat.template import CallBody
from llama_chat.template_extract import extract_fragments, reasoning_switch

from .chat_templates import STEMS, fixture
from .fake_context import FakeContext


def _tool_fragments(render):
    """The call tags a render yields, through both steps of the recovery."""
    return extract_tool_fragments(render, extract_fragments(render))


@functools.cache
def _real(stem):
    """The renderer the test template ``stem`` is loaded with."""
    return FakeContext(fixture(stem)).render_with_model_template


def test_gemma_user_assistant_fragments():
    f = extract_fragments(_real("gemma-3"))
    assert f.user_prefix == "<start_of_turn>user\n"
    assert f.user_suffix == "<end_of_turn>\n"
    assert f.assistant_prefix == "<start_of_turn>model\n"
    assert f.assistant_suffix == "<end_of_turn>\n"


def test_chatml_extracts_its_tags():
    f = extract_fragments(_real("qwen-3"))
    assert f.user_prefix == "<|im_start|>user\n"
    assert f.assistant_prefix == "<|im_start|>assistant\n"


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
    return "".join(f"<{m['role']}>{m['content']}" for m in messages)


def test_empty_suffix_template_extracts_without_crashing():
    f = extract_fragments(_prefix_only_render)
    assert f.user_prefix == "<user>"
    assert f.user_suffix == ""
    assert f.assistant_prefix == "<assistant>"
    assert f.assistant_suffix == ""


def test_a_tool_calling_template_yields_its_call_tags():
    tools = _tool_fragments(_real("gemma-4"))
    assert tools is not None
    assert tools.call_open == "<|tool_call>call:"
    # The argument's own text is gone; what is left is the call's closing tag
    # plus the punctuation every body ends with.
    assert tools.call_close == "}<tool_call|><|tool_response>"


@pytest.mark.parametrize("stem", [stem for stem in STEMS if not fixture(stem)["calls_tools"]])
def test_a_template_that_cannot_call_tools_has_no_tool_fragments(stem):
    assert _tool_fragments(_real(stem)) is None


def test_tool_extraction_does_not_disturb_the_role_fragments():
    f = extract_fragments(_real("gemma-4"))
    assert f.user_prefix == "<|turn>user\n"
    assert f.user_suffix == "<turn|>\n"
    assert f.assistant_prefix == "<|turn>model\n"


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


def test_the_call_body_of_a_delimited_model():
    render = _real("gemma-4")
    body = extract_call_body(render, _tool_fragments(render).call_open, "<tool_call|>")
    assert body == CallBody(name_close="{", key_open="", key_close=":",
                            pair_sep=",", body_close="}", empty_close="}",
                            quote_open='<|"|>', quote_close='<|"|>')


def test_the_call_body_of_an_element_shaped_model():
    render = _real("qwen-3-coder")
    body = extract_call_body(render, _tool_fragments(render).call_open, "</tool_call>")
    # Every field differs from the delimited model's, and nothing is quoted.
    assert body == CallBody(name_close=">\n<", key_open="parameter=",
                            key_close=">\n",
                            pair_sep="\n</parameter>\n<parameter=",
                            body_close="\n</parameter>\n</function>\n",
                            empty_close="/function>\n",
                            quote_open="", quote_close="")


def test_a_model_that_renders_no_call_has_no_body_either():
    render = _real("gemma-4")
    # A closing tag this template never emits leaves nothing to measure.
    assert extract_call_body(render, _tool_fragments(render).call_open, "<nope>") is None


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


def _no_tool_render(messages, **_options):
    """A template that refuses a message carrying a tool call outright."""
    if any(m.get("tool_calls") for m in messages):
        return None
    return _real("qwen-3")(messages)


def test_a_template_that_refuses_tool_calls_has_no_tool_fragments():
    assert _tool_fragments(_no_tool_render) is None


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


@pytest.mark.parametrize("ending, switch_on, switch_off", [
    # Reads the shared variable: it stays the switch.
    (lambda v: "<think>" if v.get("enable_thinking") else "<think></think>",
     {"enable_thinking": True}, {"enable_thinking": False}),
    # Reads only an effort, and that opens the thought: the effort is the switch.
    (lambda v: "<think>" if v.get("reasoning_effort") == "high" else "<think></think>",
     {"reasoning_effort": "high"}, {}),
    # Reasons unless the effort is none: none is off, and the template's own default is on.
    (lambda v: "<think></think>" if v.get("reasoning_effort") == "none" else "<think>",
     {}, {"reasoning_effort": "none"}),
    # Reasons by default, writes more for high, and stops only for none: the default
    # stays on rather than the higher effort.
    (lambda v: {"none": "<think></think>", "high": "<effort><think>"}.get(
        v.get("reasoning_effort"), "<think>"),
     {}, {"reasoning_effort": "none"}),
    # An effort that changes nothing after the user turn is not a switch.
    (lambda v: "", {"enable_thinking": True}, {"enable_thinking": False}),
])
def test_the_reasoning_switch_is_the_variable_that_opens_the_thought(ending, switch_on, switch_off):
    switch = reasoning_switch(_rendering(ending))
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
