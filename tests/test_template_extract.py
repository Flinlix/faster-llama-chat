"""Fragment extraction over real chat templates, and over renderers written for a case no real template shows.

A real format is read from its template in ``tests/templates/``. A renderer
written here reproduces no model's format and carries no model's name.
"""

import json
import ast
import functools
import itertools
import re
from pathlib import Path

import pytest

import llama_chat
from llama_chat.call_format import extract_call_body, extract_tool_fragments, extract_call_close_text, extract_json_call_id_key, extract_named_twice_opening
from llama_chat.model_template import ModelTemplate, names_read
from llama_chat.tokenizing import tag_texts
from llama_chat.template import CallBody
from llama_chat.template_extract import (as_probes_read, extract_content_rewrites,
                                         extract_fragments, reasoning_switch, extract_reasoning, extract_removed_thought)

from .chat_templates import STEMS, expected, fixture
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import TAGS as INVENTED_TAGS
from llama_chat.messages import call_id


def _tool_fragments(render, tags=INVENTED_TAGS):
    """The call tags a render yields, through both steps of the recovery, the model's own tags ``tags``."""
    return extract_tool_fragments(render, extract_fragments(render, tags), tags)


def _tags(stem):
    """The tags of the test template ``stem``, as its tokenizer gives them."""
    return tag_texts(FakeContext(fixture(stem)))


@functools.cache
def _real(stem):
    """The renderer the test template ``stem`` is loaded with."""
    known = fixture(stem)
    return ModelTemplate(known["chat_template"], known["eos_token"]).render


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
    rendered = ModelTemplate("{{ reasoning }}/{{ generation_prompt }}", "").render(
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
    def renders(n):
        """How many renders reading the rewrites of ``n`` texts takes."""
        made = []

        def counted(messages, **options):
            made.append(messages)
            return _escaping_render(messages, **options)
        texts = [f"<tag{k}>" for k in range(n)]
        assert len(extract_content_rewrites(counted, texts, ())) == n
        return len(made)
    assert renders(20_000) == renders(20)


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


@pytest.mark.parametrize("where, readable", [("[ID]{{c.id}}{ ", True), ("{ ", False)])
def test_a_call_id_is_read_between_the_name_and_the_arguments_and_refused_elsewhere(where, readable):
    """Rule 2: an id the template writes where it cannot be told from the text around it leaves no call format, rather than one that misreads every call."""
    import jinja2
    from llama_chat.call_format import holds_the_probe_id
    after = "#{{c.id}}" if not readable else ""
    template = jinja2.Environment().from_string(
        "{%- for m in messages -%}<{{m.role}}>{{m.content}}{%- for c in m.tool_calls or [] -%}"
        "<call>{{c.function.name}}" + where + "{%- for k,v in c.function.arguments.items() -%}"
        "{{k}}:'{{v}}'{{',' if not loop.last}}{%- endfor -%} }" + after + "</call>"
        "{%- endfor -%}</{{m.role}}>{%- endfor -%}")
    body = extract_call_body(lambda messages, reasoning=False, **_: template.render(messages=messages),
                             "<call>", "</call>")
    assert body is not None and holds_the_probe_id(body) == (not readable)
    if readable:
        assert body.id_open == "[ID]"


def _restating_the_question(render):
    """``render``, writing the last message's text once more after the generation prompt."""
    def restating(messages, generation_prompt=False, variables=None):
        again = f"<again>{messages[-1]['content']}" if generation_prompt else ""
        return render(messages, generation_prompt, variables) + again
    return restating


def test_a_template_that_writes_the_question_again_after_the_user_turn_has_no_effort_switch_for_it():
    """What follows the user turn is read from its first end, so the question written again is no change."""
    switch = reasoning_switch(_restating_the_question(_rendering(lambda v: "")))
    assert switch(True) == _SHARED_FLAG[0]
    assert switch(False) == _SHARED_FLAG[1]
    assert switch(None) == {}


# The tags of the renders below, a closing tag and one that is its own end among them.
_PROMPT_TAGS = ("<u>", "<a>", "<r>", "</r>", "r>")


def _prompt_ending_render(off, on):
    """A render that writes no thought in any reply and ends its generation prompt in ``off`` without reasoning and ``on`` with it, raising where one is ``None``."""
    def render(messages, generation_prompt=False, reasoning=None, **_options):
        text = "".join(f"<{m['role'][0]}>{m['content']}" for m in messages)
        if not generation_prompt:
            return text
        ending = on if reasoning else off
        if ending is None:
            raise ValueError("the template refuses this reasoning setting")
        return text + "<a>" + ending
    return render


@pytest.mark.parametrize("off, on, thought_tags", [
    # Ends in an opening and its closing tag, the closing tag being the longest that ends it.
    ("<r></r>", "<r>", ("<r>", "</r>")),
    # Ends in no tag at all, whatever the length of the text after the last one.
    ("WXYZ", "<r>", None),
    ("W", "<r>", None),
    # Ends in a closing tag with no opening tag before it.
    ("x</r>", "<r>", None),
    # Refuses one of the two settings.
    ("<r></r>", None, None),
    (None, "<r>", None),
])
def test_an_empty_thought_is_read_only_from_a_prompt_that_ends_in_an_opening_and_a_closing_tag(
        off, on, thought_tags):
    render = _prompt_ending_render(off, on)
    found = extract_reasoning(render, extract_fragments(render, _PROMPT_TAGS), None, _PROMPT_TAGS)
    assert (found and (found.thought_open, found.thought_close)) == (thought_tags or None)


def _thought_removing_render(close, *, removes_only_without_reasoning=False):
    """A render that writes no thought and leaves out of a past reply all it holds up to ``close``."""
    def render(messages, generation_prompt=False, reasoning=None, **_options):
        removes = reasoning is False or not removes_only_without_reasoning
        text = "".join(
            f"<{m['role'][0]}>"
            + (m["content"].split(close)[-1] if removes and m["role"] == "assistant" else m["content"])
            for m in messages)
        return text + ("<a>" if generation_prompt else "")
    return render


def _removed_thought_tags(close, tags, **options):
    """The thought tags ``extract_removed_thought`` reads from :func:`_thought_removing_render`."""
    render = _thought_removing_render(close, **options)
    found = extract_removed_thought(render, extract_fragments(render, tags), tags)
    return found and (found.thought_open, found.thought_close)


def test_a_removed_thoughts_opening_tag_is_its_closing_tag_without_the_first_slash():
    tags = ("<u>", "<a>", "<q/s>", "</q/s>")
    assert _removed_thought_tags("</q/s>", tags) == ("<q/s>", "</q/s>")


@pytest.mark.parametrize("close, tags", [
    # A closing tag whose opening is none of the model's tags.
    ("</q>", ("<u>", "<a>", "</q>")),
    # A tag without a slash, which opens nothing.
    ("<q>", ("<u>", "<a>", "<q>")),
])
def test_a_removed_thought_is_read_only_between_two_of_the_models_tags(close, tags):
    assert _removed_thought_tags(close, tags) is None


def test_a_removed_thought_is_read_from_a_past_reply_written_with_reasoning_off():
    tags = ("<u>", "<a>", "<q>", "</q>")
    assert _removed_thought_tags("</q>", tags, removes_only_without_reasoning=True) == ("<q>", "</q>")


def _one_exchange_render(messages, generation_prompt=False, reasoning=None, **_options):
    """A render that writes one question and its reply at most, marking the reply unless reasoning is off."""
    if len(messages) > 2:
        raise ValueError("the template writes one exchange at most")
    mark = "" if reasoning is False else "<r/>"
    text = "".join(f"<{m['role'][0]}>" + (mark if m["role"] == "assistant" else "") + m["content"]
                   for m in messages)
    return text + ("<a>" if generation_prompt else "")


def test_the_tags_of_a_template_that_writes_one_exchange_are_read_with_reasoning_off():
    found = extract_fragments(_one_exchange_render, ("<u>", "<a>", "<r/>"))
    assert (found.user_prefix, found.assistant_prefix, found.assistant_suffix) == ("<u>", "<a>", "")


def _thought_after_reply_tag_render(reply_tag):
    """A render that writes a reply's thought straight after ``reply_tag``, with no tag of its own before it, and closes it with ``</r>``."""
    def written(m):
        if m["role"] != "assistant":
            return f"<{m['role'][0]}>{m['content']}"
        thought = m.get("reasoning_content")
        return reply_tag + (f"{thought}</r>" if thought else "") + f"{m['content']}"

    def render(messages, generation_prompt=False, reasoning=None, **_options):
        return "".join(written(m) for m in messages) + (reply_tag if generation_prompt else "")
    return render


@pytest.mark.parametrize("reply_tag, tags, thought_tags", [
    # The reply's tag is the only tag before the thought, at the turn's very start.
    ("<a>", ("<u>", "<a>", "</r>"), ("<a>", "</r>")),
    # A tag written twice before the thought opens it from its last place.
    ("<n>reply<n>", ("<u>", "<n>", "</r>"), ("<n>", "</r>")),
    # A model with no tags of its own has no tag to open a thought with.
    ("<a>\n", (), None),
])
def test_a_thought_written_straight_after_the_replys_tag_opens_with_the_last_of_the_models_tags_before_it(
        reply_tag, tags, thought_tags):
    render = _thought_after_reply_tag_render(reply_tag)
    found = extract_reasoning(render, extract_fragments(render, tags), None, tags)
    assert (found and (found.thought_open, found.thought_close)) == thought_tags


def test_a_template_drops_a_block_tags_own_line_end_and_the_indentation_before_it():
    text = ("{% for m in messages %}\n    {% if m.content %}\n{{ m.content }}\n"
            "    {% endif %}\n{% endfor %}")
    messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert ModelTemplate(text, "").render(messages) == "a\nb\n"


def test_a_model_template_keeps_its_text_and_writes_the_end_of_sequence_text_it_was_given():
    text = "{{ messages[0].content }}{{ eos_token }}"
    template = ModelTemplate(text, "~end~")
    assert template.text == text
    assert template.render([{"role": "user", "content": "hi"}]) == "hi~end~"


# A template that refuses by what it is given besides the messages: the tools, the
# reasoning switch and a further variable, each with a reason of its own.
_REFUSES_BY_ITS_INPUTS = (
    "{% if tools %}{{ raise_exception('declared ' ~ (tools | length)) }}{% endif %}"
    "{% if enable_thinking %}{{ raise_exception('thinking') }}{% endif %}"
    "{% if word is defined %}{{ raise_exception('word ' ~ word) }}{% endif %}"
    "{% if enable_thinking is defined %}off{% endif %}written")


def test_a_templates_refusal_is_read_with_the_tools_reasoning_and_variables_its_render_is_given():
    template = ModelTemplate(_REFUSES_BY_ITS_INPUTS, "")
    declared = [{"type": "function", "function": {"name": "clock", "parameters": {}}}]
    for tools, reasoning, variables in itertools.product(
            (None, declared), (None, True, False), (None, {"word": "given"})):
        options = dict(generation_prompt=True, reasoning=reasoning, variables=variables)
        reason = template.refusal_reason([], tools, **options)
        assert reason == ("declared 1" if tools else "thinking" if reasoning
                          else "word given" if variables else None)
        assert (reason is None) == (template.render([], tools, **options) is not None)


def _turns_render(write_call):
    """A renderer writing each message between tags naming its role, each call of a reply as ``write_call`` gives it from the call and its place."""
    def render(messages, **_options):
        out = []
        for m in messages:
            calls = "".join(write_call(call, place) for place, call in enumerate(m.get("tool_calls") or ()))
            out.append(f"<{m['role']}>{m['content']}{calls}</{m['role']}>")
        return "".join(out)
    return render


def _named_three_times(call, _place):
    """A call naming its function, then again after an opening that holds ``to=`` once more, then a third time in its closing."""
    name = call["function"]["name"]
    return f"to={name}|via to={name}{json.dumps(call['function']['arguments'])}</call {name}>"


def test_where_a_call_names_its_function_twice_the_opening_is_all_the_text_between_its_first_two_names():
    assert extract_named_twice_opening(_turns_render(_named_three_times), "to=") == "|via to="


def test_a_call_not_opened_with_the_given_text_or_not_rendered_at_all_has_no_opening_named_twice():
    assert extract_named_twice_opening(_turns_render(_named_three_times), "@call ") is None
    assert extract_named_twice_opening(lambda messages, **_options: None, "to=") is None


def _call_in_a_turn_of_its_own(call, _place):
    """Each call in an assistant turn of its own, ``</call></turn>`` ending it."""
    return f"<turn>to={call['function']['name']}{json.dumps(call['function']['arguments'])}</call></turn>"


def test_the_text_that_ends_a_call_is_all_of_the_tags_it_ends_with_whatever_their_order():
    assert extract_call_close_text(_turns_render(_call_in_a_turn_of_its_own), "to=") == "</call></turn>"


def _json_call_with_id(id_of):
    """A JSON call under ``<call>``, its ``id`` written as ``id_of`` gives it from the call and its place."""
    def write(call, place):
        written = {"name": call["function"]["name"], "arguments": call["function"]["arguments"],
                   "id": id_of(call, place)}
        return f"<call>{json.dumps(written)}</call>"
    return write


def test_a_json_call_holds_its_id_under_a_key_only_where_the_template_writes_the_calls_own_id():
    keys = ("name", "arguments", "id")
    own = _turns_render(_json_call_with_id(lambda call, _place: call["id"]))
    numbered_by_place = _turns_render(_json_call_with_id(lambda _call, place: call_id(place)))
    assert extract_json_call_id_key(own, keys) == "id"
    assert extract_json_call_id_key(numbered_by_place, keys) is None
    assert extract_json_call_id_key(lambda messages, **_options: None, keys) is None


@pytest.mark.parametrize("tags", [("aa",), ("ab", "b"), ("a", "ab"), ("aba",), ("a", "aa", "aab")])
def test_a_shared_beginning_is_the_longest_that_ends_a_tag_without_cutting_another(tags):
    from llama_chat.template import shared_tags_length

    def ends_cleanly(text, at):
        if not text[:at].endswith(tags):
            return False
        return not any(start < at < start + len(tag)
                       for tag in tags for start in range(len(text))
                       if text.startswith(tag, start))

    for length in range(1, 8):
        for letters in itertools.product("ab", repeat=length):
            text = "".join(letters)
            for common in range(len(text) + 1):
                other = text[:common] + "x"
                expected = max((at for at in range(1, common + 1) if ends_cleanly(text, at)),
                               default=0)
                assert shared_tags_length(text, other, tags) == expected, (text, common)


def _turns_whose_calls_end_with(closing, writes_two_calls=True):
    """A render with no whitespace whose every call ends with ``closing``, refusing a turn of two calls unless ``writes_two_calls``."""
    def render(messages, **_options):
        out = []
        for m in messages:
            calls = m.get("tool_calls") or ()
            if len(calls) > 1 and not writes_two_calls:
                raise ValueError("one call per turn")
            body = "".join(f"<c>{call['function']['name']}"
                           f"{json.dumps(call['function']['arguments'], separators=(',', ':'))}{closing}"
                           for call in calls)
            role = m["role"][0]
            out.append(f"<{role}>{m['content'] or ''}{body}</{role}>")
        return "".join(out)
    return render


@pytest.mark.parametrize("closing, writes_two_calls, close_text", [
    ("<c>", False, "<c></a>"),
    ("X</c>", False, "</c></a>"),
    ("X</c>", True, "</c>"),
    ("</c>EX", True, "</c>"),
], ids=["opening-again", "letter-before-the-tag", "letter-before-the-tag-two-calls",
        "letters-after-the-tag-two-calls"])
def test_a_calls_closing_text_runs_from_where_a_tag_starts_to_where_one_ends_in_what_every_call_ends_with(
        closing, writes_two_calls, close_text):
    assert extract_call_close_text(_turns_whose_calls_end_with(closing, writes_two_calls), "<c>") \
        == close_text


def _call_written_only_with(arguments_given):
    """A render that writes a call only where ``arguments_given(arguments)`` is true, and raises where it is ``"refuse"``."""
    sound = _turns_whose_calls_end_with("</c>")

    def render(messages, **options):
        written = []
        for m in messages:
            calls = []
            for call in m.get("tool_calls") or ():
                kept = arguments_given(call["function"]["arguments"])
                if kept == "refuse":
                    raise ValueError("no such call")
                if kept:
                    calls.append(call)
            written.append({**m, "tool_calls": calls})
        return sound(written, **options)
    return render


@pytest.mark.parametrize("arguments_given", [
    lambda arguments: bool(arguments),
    lambda arguments: True if arguments else "refuse",
    lambda arguments: "refuse" if arguments else True,
], ids=["no-call-without-arguments", "refuses-a-call-without-arguments", "refuses-a-call-with-arguments"])
def test_a_template_that_writes_only_one_of_the_two_call_probes_has_no_closing_text(arguments_given):
    assert extract_call_close_text(_call_written_only_with(arguments_given), "<c>") is None


def _removing_render(*, refuses_other_roles: bool):
    """A render that writes no thought and leaves the thought out of a reply a question follows, refusing a role it does not know where ``refuses_other_roles``."""
    def render(messages, generation_prompt=False, reasoning=None, **_options):
        if refuses_other_roles and any(m["role"] not in ("user", "assistant") for m in messages):
            raise ValueError("only questions and replies are written")
        questions = [at for at, m in enumerate(messages) if m["role"] == "user"]
        text = "".join(
            f"<{m['role'][0]}>" + (m["content"].split("</q>")[-1]
                                   if m["role"] == "assistant" and questions and at < questions[-1]
                                   else m["content"])
            for at, m in enumerate(messages))
        return text + ("<a>" if generation_prompt else "")
    return render


@pytest.mark.parametrize("refuses_other_roles", [True, False], ids=["refusing-other-roles", "writing-other-roles"])
def test_a_removed_thought_is_read_from_a_template_that_removes_it_only_from_a_reply_a_question_follows(
        refuses_other_roles):
    tags = ("<u>", "<a>", "<q>", "</q>")
    render = _removing_render(refuses_other_roles=refuses_other_roles)
    found = extract_removed_thought(render, extract_fragments(render, tags), tags)
    assert (found and (found.thought_open, found.thought_close)) == ("<q>", "</q>")


def _string_marked_before(messages, **_options):
    """A delimited model that marks a string by text before it and by nothing after it."""
    def written(fn):
        return ",".join(f"{k}:s~{v}" if isinstance(v, str) else f"{k}:{v}"
                        for k, v in fn["arguments"].items())
    return "".join(_turn_tags(m, "".join(f"<call>fn:{c['function']['name']}{{{written(c['function'])}}}</call>"
                                         for c in m.get("tool_calls", ())))
                   for m in messages)


def _member_opened_by_its_key_alone(messages, **_options):
    """A delimited model that opens a nested object's member with its bare key."""
    def value(v):
        if isinstance(v, dict):
            return "".join(f"{k}{value(x)}|</{k}>" for k, x in v.items())
        if isinstance(v, list):
            return "".join(f"|<entry>{value(x)}|</entry>" for x in v)
        return str(v)
    return "".join(_turn_tags(m, "".join(
        f"<call>fn:{c['function']['name']}\">|"
        + "".join(f"<{k}>{value(v)}|</{k}>|" for k, v in c["function"]["arguments"].items())
        + "</call>" for c in m.get("tool_calls", ()))) for m in messages)


@pytest.mark.parametrize("render", [_string_marked_before, _member_opened_by_its_key_alone],
                         ids=["string-marked-on-one-side", "member-opened-by-its-key"])
def test_a_call_format_whose_delimiters_cannot_tell_where_a_value_ends_is_not_recovered(render):
    assert extract_call_body(render, "<call>", "</call>") is None


@pytest.mark.parametrize("fields", [
    {"quote_open": "s~"}, {"quote_close": "~"},
    {"member_open": "{key}", "member_close": "|</{key}>", "item_open": "[", "item_close": "]"},
    {"member_open": "<{key}>", "member_close": "{key}", "item_open": "[", "item_close": "]"}],
    ids=["quote-opened-only", "quote-closed-only", "member-opened-by-its-key", "member-closed-by-its-key"])
def test_a_call_format_that_cannot_tell_where_a_value_ends_is_refused_where_it_is_made(fields):
    with pytest.raises(ValueError, match="could not be told"):
        CallBody(name_close=">", key_open="<", key_close=">", pair_sep=",", body_close="",
                 empty_close="", **fields)
