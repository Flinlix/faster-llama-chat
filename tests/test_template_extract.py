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
from llama_chat.tokenizing import tag_texts
from llama_chat.template import CallBody
from llama_chat.template_extract import (as_probes_read, extract_content_rewrites,
                                         extract_fragments, reasoning_switch)

from .chat_templates import STEMS, expected, fixture
from .fake_context import FakeContext
from .invented_templates import TAGS as INVENTED_TAGS


def _tool_fragments(render, tags=INVENTED_TAGS):
    """The call tags a render yields, through both steps of the recovery, the model's own tags ``tags``."""
    return extract_tool_fragments(render, extract_fragments(render, tags), tags)


def _tags(stem):
    """The tags of the test template ``stem``, as its tokenizer gives them."""
    return tag_texts(FakeContext(fixture(stem)))


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
    f = extract_fragments(_real(stem), tag_texts(FakeContext(fixture(stem))))
    assert {name: getattr(f, name) for name in known} == known


def test_no_template_returns_none():
    assert extract_fragments(lambda messages, **_options: None, INVENTED_TAGS) is None


def _inconsistent_render(messages, **_options):
    """Degenerate template whose user turn changes shape in a longer render.

    The single-message user fragment therefore never reappears.
    """
    if len(messages) == 1:
        return f"<u>{messages[0]['content']}</u>"
    return "".join(f"[{m['role']}]{m['content']}" for m in messages)


def test_a_render_whose_reply_cannot_be_split_off_gives_no_fragments_rather_than_a_guess():
    # The user fragment recovered from the single-message probe does not occur
    # in the two-message render, so the assistant's tags cannot be read: the
    # template is refused, not read with the user's tags.
    assert extract_fragments(_inconsistent_render, INVENTED_TAGS) is None


def _prefix_only_render(messages, **_options):
    """Degenerate template whose tags only open a turn (empty suffixes)."""
    return "".join(f"<[{m['role']}>{m['content']}" for m in messages)


def test_empty_suffix_template_extracts_without_crashing():
    f = extract_fragments(_prefix_only_render, INVENTED_TAGS)
    assert f.user_prefix == "<[user>"
    assert f.user_suffix == ""
    assert f.assistant_prefix == "<[assistant>"
    assert f.assistant_suffix == ""


@pytest.mark.parametrize("stem", _stems_with("tool_fragments"))
def test_a_tool_calling_template_yields_its_call_tags(stem):
    tools = _tool_fragments(_real(stem), _tags(stem))
    assert tools is not None
    # The argument's own text is gone from the closing; what is left is the
    # call's closing tag, with what the template writes around it.
    known = expected(fixture(stem), "tool_fragments")
    assert {name: getattr(tools, name) for name in known} == known


@pytest.mark.parametrize("stem", [stem for stem in STEMS if not fixture(stem)["calls_tools"]])
def test_a_template_that_cannot_call_tools_has_no_tool_fragments(stem):
    assert _tool_fragments(_real(stem), _tags(stem)) is None


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


def test_a_template_that_drops_the_reply_it_is_given_is_refused_rather_than_read_with_the_user_tags():
    """Read with the user tags, a call would be found in the middle of the conversation."""
    assert extract_fragments(_dropped_assistant_render, INVENTED_TAGS) is None


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


@pytest.mark.parametrize("ending, switch_on, switch_off", [
    # Reads the shared variable: it stays the switch.
    (lambda v: "<reason>" if v.get("enable_thinking") else "<reason></reason>", *_SHARED_FLAG),
    # Reasons unless the effort is none: none is off, and the template's own default is on.
    (lambda v: "<reason></reason>" if v.get("reasoning_effort") == "none" else "<reason>",
     {}, {"reasoning_effort": "none"}),
    # Reasons by default, writes more for high, and stops only for none: the default
    # stays on rather than the higher effort.
    (lambda v: {"none": "<reason></reason>", "high": "<more><reason>"}.get(
        v.get("reasoning_effort"), "<reason>"),
     {}, {"reasoning_effort": "none"}),
    # An effort that opens the thought only at some other value is no switch the
    # template states; the add-on file's reasoning_on names it.
    (lambda v: "<reason>" if v.get("reasoning_effort") == "high" else "<reason></reason>",
     *_SHARED_FLAG),
    # An effort that changes nothing after the user turn is not a switch.
    (lambda v: "", *_SHARED_FLAG),
    # Nor is a variable of another name, however it changes what follows the user turn.
    (lambda v: "<reason>" if v.get("thinking") is True else "<reason></reason>", *_SHARED_FLAG),
])
def test_the_reasoning_switch_is_the_shared_variable_or_an_effort_that_none_turns_off(
        ending, switch_on, switch_off):
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


# ----- what a template writes in a message's content -----------------------
def _escaping_render(messages, **options):
    """A user turn whose content has ``<`` escaped and the tag ``<drop>`` left out."""
    return "<u>" + messages[-1]["content"].replace("<drop>", "").replace("<", "&lt;") + "</u>"


def test_each_text_a_template_changes_in_content_is_read_as_the_template_writes_it():
    assert extract_content_rewrites(_escaping_render, ["<drop>", "<keep>", "<", ">", "plain"], ()) \
        == (("<drop>", ""), ("<keep>", "&lt;keep>"), ("<", "&lt;"))


def test_a_render_that_loses_a_texts_place_reads_no_rewrites():
    def truncating(messages, **options):
        return messages[-1]["content"][:-1]
    assert extract_content_rewrites(truncating, ["<a>", "<b>"], ()) == ()


def test_the_rewrites_of_many_texts_are_read_in_one_pass_over_the_render():
    texts = [f"<tag{n}>" for n in range(20_000)]
    started = time.perf_counter()
    rewrites = extract_content_rewrites(_escaping_render, texts, ())
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


def test_a_user_tag_after_a_past_replys_own_closing_is_found_by_the_models_tags_of_any_shape():
    from llama_chat.template_extract import _longest_shared_end
    tags = ("{{user}}", "{{bos}}")
    assert _longest_shared_end("</a>{{user}}", "{{bos}}{{user}}", tags) == "{{user}}"
    assert _longest_shared_end("</a>{{user}}", "{{bos}}{{user}}", ()) == ""


def test_a_shared_beginning_ends_where_one_of_the_models_tags_ends_whatever_its_shape():
    from llama_chat.template import shared_tags_length
    assert shared_tags_length("«turn»model", "«turn»user", ("«turn»",)) == len("«turn»")
    assert shared_tags_length("«turn»model", "«turn»user", ()) == 0
    assert shared_tags_length("say model", "say user", ()) == len("say ")


def test_a_shared_beginning_never_ends_inside_a_tag_that_a_shorter_tag_begins():
    from llama_chat.template import shared_tags_length
    assert shared_tags_length("<a><ab>x", "<a><ac>y", ("<a", "<ab>", "<ac>")) == 2


def test_an_empty_tag_text_is_no_tag():
    from llama_chat.tokenizing import tag_texts

    class _Reader:
        def special_tokens(self):
            return (1, 2)

        def token_text(self, token):
            return {1: "", 2: "<x>"}[token]

    assert tag_texts(_Reader()) == ("<x>",)


def test_what_a_template_writes_once_after_the_last_message_is_read_by_tags_of_any_shape():
    def render(messages, generation_prompt=False, **_options):
        turns = "".join(f"«{m['role']}»{m['content']}«end»" for m in messages)
        return turns + ("«assistant»" if generation_prompt else "«close»")

    tags = ("«user»", "«assistant»", "«end»", "«close»")
    alone = [{"role": "user", "content": "U"}]
    assert as_probes_read(render, tags)(alone, reasoning=False) == "«user»U«end»"
    # Without the model's tags the end is read wider, taking the turn's own closing.
    assert as_probes_read(render, ())(alone, reasoning=False) == "«user»U"
