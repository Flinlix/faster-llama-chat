"""Fragment-extraction tests driven by canned chat-template renderers (no model).

A renderer that reproduces a real format carries that format's name, because the
name is what says which shape is covered. One built to exercise a code path
carries no model's name, because it reproduces none.
"""

import itertools

import pytest

from llama_chat.call_format import (_KEY, _KEY2, _NUMBER, _STRING, _STRING2, extract_call_body,
                                    extract_tool_fragments)
from llama_chat.template import CallBody
from llama_chat.template_extract import TOOL_SENTINEL, extract_fragments, reasoning_switch


def _tool_fragments(render):
    """The call tags a render yields, through both steps of the recovery."""
    return extract_tool_fragments(render, extract_fragments(render))


def _gemma_render(messages, **_options):
    """Gemma-style template: trims content, rejects a system role, uses ``model``."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            return None  # Gemma has no system role
        role = "model" if role == "assistant" else role
        out.append(f"<start_of_turn>{role}\n{m['content'].strip()}<end_of_turn>\n")
    return "".join(out)


def _chatml_render(messages, **_options):
    """ChatML-style template: verbatim content, explicit system role."""
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def test_gemma_user_assistant_fragments():
    f = extract_fragments(_gemma_render)
    assert f.user_prefix == "<start_of_turn>user\n"
    assert f.user_suffix == "<end_of_turn>\n"
    assert f.assistant_prefix == "<start_of_turn>model\n"
    assert f.assistant_suffix == "<end_of_turn>\n"


def test_chatml_extracts_its_tags():
    f = extract_fragments(_chatml_render)
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


def _quoted_value(value):
    """Quotes a string argument and emits anything else bare."""
    return f'<|"|>{value}<|"|>' if isinstance(value, str) else str(value)


def _tool_calling_render(messages, **_options):
    """A template with a real system role that puts tool calls on a model turn."""
    out = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        out.append(f"<[turn>{role}\n")
        calls = m.get("tool_calls", ())
        for call in calls:
            fn = call["function"]
            body = ",".join(f"{k}:{_quoted_value(v)}"
                            for k, v in fn["arguments"].items())
            out.append(f"<|tool_call>call:{fn['name']}{{{body}}}<tool_call|>")
        out.append(m["content"].strip())
        # A call still awaiting its result opens the result block instead of
        # closing the turn.
        out.append("<|tool_response>" if calls else "<turn]>\n")
    return "".join(out)


def test_a_tool_calling_template_yields_its_call_tags():
    tools = _tool_fragments(_tool_calling_render)
    assert tools is not None
    assert tools.call_open == "<|tool_call>call:"
    # The argument's own text is gone; what is left is the call's closing tag
    # plus the punctuation every body ends with.
    assert tools.call_close == "}<tool_call|><|tool_response>"


def test_a_template_that_ignores_tool_calls_has_no_tool_fragments():
    # The canned renderers above build a turn out of `content` alone, which is
    # exactly what a model without tool support looks like from here.
    assert _tool_fragments(_gemma_render) is None
    assert _tool_fragments(_chatml_render) is None


def test_tool_extraction_does_not_disturb_the_role_fragments():
    f = extract_fragments(_tool_calling_render)
    assert f.user_prefix == "<[turn>user\n"
    assert f.user_suffix == "<turn]>\n"
    assert f.assistant_prefix == "<[turn>model\n"


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


def _element_render(messages, **_options):
    """A model that wraps each argument in an element of its own.

    Deliberately unlike the delimited shape above: the name and every key sit
    inside a tag, and nothing is quoted. What the extractor recovers from this
    has to differ in every field.
    """
    out = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        out.append(f"<[turn>{role}\n")
        calls = m.get("tool_calls", ())
        for call in calls:
            fn = call["function"]
            out.append(f"<tool_call>\n<function={fn['name']}>\n")
            for key, value in fn["arguments"].items():
                out.append(f"<parameter={key}>\n{value}\n</parameter>\n")
            out.append("</function>\n</tool_call>")
        out.append(m["content"].strip())
        if not calls:
            out.append("<turn]>\n")
    return "".join(out)


def test_the_call_body_of_a_delimited_model():
    tools = _tool_fragments(_tool_calling_render)
    body = extract_call_body(_tool_calling_render, tools.call_open, "<tool_call|>")
    assert body == CallBody(name_close="{", key_open="", key_close=":",
                            pair_sep=",", body_close="}", empty_close="}",
                            quote_open='<|"|>', quote_close='<|"|>')


def test_the_call_body_of_an_element_shaped_model():
    tools = _tool_fragments(_element_render)
    body = extract_call_body(_element_render, tools.call_open, "</tool_call>")
    # Every field differs from the delimited model's, and nothing is quoted.
    assert body == CallBody(name_close=">\n<", key_open="parameter=",
                            key_close=">\n",
                            pair_sep="\n</parameter>\n<parameter=",
                            body_close="\n</parameter>\n</function>\n",
                            empty_close="/function>\n",
                            quote_open="", quote_close="")


def test_a_model_that_renders_no_call_has_no_body_either():
    tools = _tool_fragments(_tool_calling_render)
    # A closing tag this template never emits leaves nothing to measure.
    assert extract_call_body(_tool_calling_render, tools.call_open, "<nope>") is None


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
    """Reached when the tags are supplied rather than extracted.

    ``extract_tool_fragments`` refuses this template itself, so the tags come
    from ``ChatWrapper(tool_calls=...)`` and the body probes run against a
    template they were never screened for.
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
    return _chatml_render(messages)


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


# The sentinels a render can drop, and the text each appears as.
_SENTINELS = {"name": TOOL_SENTINEL, "key": _KEY, "key2": _KEY2,
              "string": _STRING, "string2": _STRING2, "number": str(_NUMBER)}


def _degenerate_render(dropped, constant_body):
    """A delimited render with ``dropped`` sentinels deleted from every call."""
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
                for sentinel in dropped:
                    text = text.replace(_SENTINELS[sentinel], "")
                body += text
            out.append(_turn_tags(m, body))
        return "".join(out)
    return render


def test_no_render_makes_body_extraction_raise():
    """Refusing is the only failure; a template cannot crash construction.

    Every subset of the sentinels a template might drop, against both a
    rendered and a constant body. Before the guard on them, some of these
    indexed off the end of a split instead of returning None.
    """
    for size in range(len(_SENTINELS) + 1):
        for dropped in itertools.combinations(_SENTINELS, size):
            for constant_body in (False, True):
                render = _degenerate_render(dropped, constant_body)
                body = extract_call_body(render, "<call>fn:", "</call>")
                assert body is None or isinstance(body, CallBody)


def _rendering(open_thought):
    """A render of the probe conversation whose ending ``open_thought(variables)`` decides."""
    def render(messages, generation_prompt=False, **variables):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + (open_thought(variables) if generation_prompt else "")
    return render


@pytest.mark.parametrize("ending, switch_on", [
    # Reads the shared variable: it stays the switch.
    (lambda v: "<think>" if v.get("enable_thinking") else "<think></think>",
     {"enable_thinking": True}),
    # Reads only an effort, and that opens the thought: the effort is the switch.
    (lambda v: "<think>" if v.get("reasoning_effort") == "high" else "<think></think>",
     {"reasoning_effort": "high"}),
    # An effort that changes nothing after the user turn is not a switch.
    (lambda v: "", {"enable_thinking": True}),
])
def test_the_reasoning_switch_is_the_variable_that_opens_the_thought(ending, switch_on):
    switch = reasoning_switch(_rendering(ending))
    assert switch(True) == switch_on
    assert switch(None) == {}


# ----- probe sentinels -----------------------------------------------------
def test_no_render_sentinel_contains_another():
    """A split on one has to land on that one, whatever else the render holds.

    ``PROBE`` used to sit inside ``RESULTPROBE`` and ``TAILPROBE``, so a render
    carrying all three could be cut six characters into the wrong sentinel.
    """
    from llama_chat import resolved_template, template_extract

    sentinels = {name: getattr(module, name)
                 for module, names in (
                     (resolved_template, ("_PROMPT", "_RESULT", "_NEXT")),
                     (template_extract, ("_SYS", "USER_SENTINEL", "_USER2",
                                         "ASSISTANT_SENTINEL", "_THOUGHT",
                                         "TOOL_SENTINEL")))
                 for name in names}
    assert len(set(sentinels.values())) == len(sentinels)
    for name, value in sentinels.items():
        others = [v for n, v in sentinels.items() if n != name]
        assert not any(value in other for other in others), \
            f"{name}={value!r} occurs inside another sentinel"
