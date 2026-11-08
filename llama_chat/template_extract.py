"""Recover a model's own chat template by rendering probes and comparing them.

Probes carry sentinel content, and splitting a render on its sentinels recovers
the template that produced it. Four things come out:

* :func:`extract_fragments`: the tags wrapped around each role's content.
* :func:`extract_tool_fragments`: the tags a generated tool call sits between.
* :func:`extract_call_body`: how a call spells its name and arguments.
* :func:`extract_reasoning`: how a thought is marked and reasoning turned on.

No model's format is written down here, and where a comparison proves nothing
the answer is ``None`` rather than a guess. See the README, "Tool calling".

A pure function of the render callable, so it is testable without a model.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Callable

from .messages import calling_message
from .template import (CallBody, Reasoning, TemplateFragments, ToolCallFragments,
                       common_prefix_length, common_suffix_length, shared_tags_length)
from .tools import ToolCall

# Plain alphanumerics: they survive Jinja `| trim` and are never HTML-escaped.

# The fields chat templates read a reply's thought from.
_THOUGHT_FIELDS = ("reasoning_content", "thinking")

# Message content, one per role, a second user turn, and a thought.
_SYS = "SYSMSG"
_USER = "USERMSG"
_USER2 = "USERTWO"
_ASSISTANT = "ASSTMSG"
_THOUGHT = "THOUGHTMSG"

# `AKEYONE` sorts before `AKEYTWO`, so the two-argument probe reads the keys in
# the order it wrote them even under a template that sorts them.
_TOOL = "TOOLNAME"
_KEY = "AKEYONE"
_KEY2 = "AKEYTWO"

# The string and the number end in different characters, so a shared ending
# cannot run back into the value itself.
_STRING = "VALONE"
_NUMBER = 1234
_STRING2 = "VALTWO"
_MARK = "QQMARK"

# Renders messages through the model's template, or returns None where the
# template rejects their shape, as a template that raises does too. Keywords:
# ``generation_prompt`` appends the generation prompt, and ``reasoning`` sets
# the template's reasoning switch.
Render = Callable[..., "str | None"]


def reasoning_switch(render_variables: Callable[..., str]) -> Callable[[bool | None], dict]:
    """The template variables that turn reasoning on or off, as a render reads them.

    ``enable_thinking``, the variable chat templates share for it, or, where
    that changes nothing, ``reasoning_effort`` where its setting changes what
    follows the user turn: ``"high"`` for on, the template's own default for
    off.

    Args:
        render_variables: Renders ``messages`` with ``generation_prompt`` and
            template variables as keywords, raising where the template does.
    """
    probe = [{"role": "system", "content": _SYS}, {"role": "user", "content": _USER}]
    render = refusing(render_variables)

    def rendered(**variables):
        return render(probe, generation_prompt=True, **variables)

    thinking = {True: {"enable_thinking": True}, False: {"enable_thinking": False}}
    if rendered(**thinking[True]) == rendered(**thinking[False]):
        on, off = rendered(reasoning_effort="high"), rendered()
        if None not in (on, off) and on.split(_USER, 1)[-1] != off.split(_USER, 1)[-1]:
            thinking = {True: {"reasoning_effort": "high"}, False: {}}
    return lambda reasoning: {} if reasoning is None else thinking[reasoning]


def names_read(template: str) -> set[str]:
    """Every name a template reads, a variable it gives a default when undefined included."""
    import jinja2
    try:
        parsed = jinja2.Environment().parse(template)
    except jinja2.TemplateError:
        return set()
    return {node.name for node in parsed.find_all(jinja2.nodes.Name) if node.ctx == "load"}


def refusing(render: Render) -> Render:
    """``render``, answering ``None`` where it raises, as a template refusing a shape does."""
    def rendered(*args, **kwargs):
        try:
            return render(*args, **kwargs)
        except Exception:
            return None
    return rendered


def _as_probes_read(render: Render) -> Render:
    """``render`` as probes read it.

    It answers ``None`` where it raises, a template refusing a shape. A render
    without a generation prompt comes back without what the template writes
    once after a conversation's last message, which belongs to no turn.
    """
    refused = refusing(render)
    end = _conversation_end(refused)

    def rendered(*args, **kwargs):
        text = refused(*args, **kwargs)
        if end and text is not None and not kwargs.get("generation_prompt") \
                and text.endswith(end):
            return text[:-len(end)]
        return text
    return rendered


def _conversation_end(render: Render) -> str:
    """What a render without a generation prompt writes after its last turn.

    Read as what follows a lone user turn but not the same turn followed by a
    reply, and also ends the render with the reply.
    """
    alone = render([{"role": "user", "content": _USER}], reasoning=False)
    followed = render([{"role": "user", "content": _USER},
                       {"role": "assistant", "content": _ASSISTANT}], reasoning=False)
    if alone is None or followed is None or _USER not in alone or _USER not in followed:
        return ""
    after_alone, after_followed = alone.split(_USER, 1)[1], followed.split(_USER, 1)[1]
    end = after_alone[shared_tags_length(after_alone, after_followed):]
    return end if followed.endswith(end) else ""


def extract_fragments(render: Render) -> TemplateFragments | None:
    """Recover template fragments by probing ``render`` with sentinel content.

    The user and assistant tags are read from a conversation of two user turns,
    where the first reply is a past one: a first turn alone may carry text the
    template writes once at the start, and a newest reply may carry what only
    the newest one does. Every probe is rendered with reasoning off, which is
    what the fragments describe; what reasoning changes is
    :func:`extract_reasoning`'s.

    Args:
        render: As ``Render`` above.

    Returns:
        The fragments, or ``None`` where the model ships no usable template.
    """
    render = _as_probes_read(render)
    user = render([{"role": "user", "content": _USER}], reasoning=False)
    if user is None or _USER not in user:
        return None
    first_prefix, _, user_suffix = user.partition(_USER)
    tags = (_past_exchange(render, first_prefix, user_suffix)
            or _newest_exchange(render, first_prefix, user_suffix))
    user_prefix, assistant_prefix, assistant_suffix = tags
    return TemplateFragments(
        user_prefix=user_prefix, user_suffix=user_suffix,
        assistant_prefix=assistant_prefix, assistant_suffix=assistant_suffix,
        trim_content=_trims_content(render),
        reply_strip=_reply_strip(render, user_suffix, assistant_prefix),
        system_strip=_system_strip(render),
        generation_prompt=_whole_prompt(render, assistant_prefix),
        **_extract_system(render, user_prefix, user_suffix))


def extract_reasoning(render: Render, fragments: TemplateFragments,
                      tool_calls: ToolCallFragments | None) -> Reasoning | None:
    """Recover what turns this template's reasoning on, and how a thought is marked.

    Args:
        render: As ``Render`` above.
        fragments: The recovered turn tags.
        tool_calls: The recovered call tags, for a template that writes a
            thought only in a turn that calls a tool.

    Returns:
        The reasoning, or ``None`` where the template marks no thought that can
        be recovered. A template without a switch that marks one reasons, or
        may, in every reply.
    """
    render = _as_probes_read(render)
    found = _thought_tags(render, fragments, tool_calls) or _empty_thought_tags(render)
    if found is None:
        return None
    tags, reply_with_thought = found
    prompt = fragments.generation_prompt or fragments.assistant_prefix
    messages = [{"role": "system", "content": _SYS}, {"role": "user", "content": _USER}]
    on = render(messages, generation_prompt=True, reasoning=True)
    off = render(messages, generation_prompt=True, reasoning=False)
    if on is not None and off is not None and on != off:
        prompt = _prompt_added(render, True)
        if prompt is None:
            return None
    return Reasoning(thought_open=tags[0], thought_close=tags[1], generation_prompt=prompt,
                     past_keeps_thought=_past_keeps_thought(render, reply_with_thought))


def extract_removed_thought(render: Render, fragments: TemplateFragments,
                            tags: tuple[str, ...]) -> Reasoning | None:
    """The thought of a template that writes none but removes one from a past reply.

    Tried with each of the model's tags: the closing tag is the one after which
    a past reply keeps only its answer, and the opening tag the one of the same
    name without the slash. Such a template has no switch, so the model may
    reason in every reply.
    """
    render = _as_probes_read(render)
    known = set(tags)
    for close in tags:
        opening = "<" + close[2:]
        if not close.startswith("</") or opening not in known:
            continue
        rendered = render([{"role": "user", "content": _USER},
                           {"role": "assistant", "content": _THOUGHT + close + _ASSISTANT},
                           {"role": "user", "content": _USER2}], reasoning=False)
        if rendered is not None and _ASSISTANT in rendered and _THOUGHT not in rendered:
            return Reasoning(thought_open=opening, thought_close=close,
                             generation_prompt=fragments.generation_prompt
                             or fragments.assistant_prefix)
    return None


def _empty_thought_tags(render: Render) -> tuple[tuple[str, str], None] | None:
    """The tags of the empty thought only the generation prompt without reasoning ends in.

    Read where a template writes no thought into a reply: two tags with nothing
    but whitespace between them, which the prompt with reasoning lacks.
    """
    off, on = _prompt_added(render, False), _prompt_added(render, True)
    if off is None or on is None or off == on:
        return None
    found = re.search(r"(<[^<>\s]+>)\s*(<[^<>\s]+>)\s*$", off)
    if found is None or found.group(1) == found.group(2) or on.endswith(found.group(0)):
        return None
    return found.groups(), None


def _past_keeps_thought(render: Render, reply: dict | None) -> bool:
    """Whether ``reply``, a reply holding a thought, keeps it once a later turn follows."""
    if reply is None:
        return False
    rendered = render([
        {"role": "user", "content": _USER},
        {key: value for key, value in reply.items() if key != "tool_calls"}
        | ({"content": _ASSISTANT} if "tool_calls" in reply else {}),
        {"role": "user", "content": _USER2},
    ], reasoning=True)
    return rendered is not None and _THOUGHT in rendered


def extract_tool_fragments(render: Render,
                           fragments: TemplateFragments) -> ToolCallFragments | None:
    """Recover what a template wraps a generated tool call in.

    Two probes differing only in their argument's type, so what the renders
    share around the call is structure rather than content.

    Args:
        render: As :func:`extract_fragments`.
        fragments: The recovered turn tags. The turn holding a call begins
            after the user turn and the assistant tag, so a template whose
            assistant tag cannot be told from its user tag cannot locate that
            turn at all, and what came back would open a call with a turn tag.
            An assistant tag that is empty is no such case: the reply then
            begins where the user turn ends.

    Returns:
        The tags, or ``None`` where nothing usable can be recovered, which is
        what a model that cannot call tools looks like from here.
    """
    render = _as_probes_read(render)
    if fragments.assistant_prefix == fragments.user_prefix:
        return None
    text = _render_call(render, {_KEY: _STRING})
    number = _render_call(render, {_KEY: _NUMBER})
    if text is None or number is None:
        return None
    if _TOOL not in text or _TOOL not in number:
        return None            # the template dropped the call instead of rendering it
    head = text.split(_TOOL, 1)[0]
    # The turn holding the call has to be locatable, or there is no telling
    # where the conversation before it ends and the call's own opening begins.
    # It may open without the content tag a reply in words opens with, and
    # then begins after the tags the two share.
    after_user = _USER + fragments.user_suffix
    if after_user not in head:
        return None
    turn = head.split(after_user, 1)[1]
    shared = (len(fragments.assistant_prefix) if turn.startswith(fragments.assistant_prefix)
              else shared_tags_length(turn, fragments.assistant_prefix))
    if not shared and fragments.assistant_prefix:
        return None
    call_open = turn[shared:]
    # A newest reply may open with more than a past one, an empty thought for
    # instance, which is not part of the call.
    call_open = call_open.removeprefix(_newest_opening(render, fragments))
    call_close = _common_suffix(text, number)
    # A template that ignored the arguments renders the two probes identically,
    # and then the shared ending runs back past the call into the conversation.
    if not call_open or not call_close or _TOOL in call_close:
        return None
    return ToolCallFragments(call_open=call_open, call_close=call_close)


def extract_call_separator(render: Render, call_open: str) -> str | None:
    """What a template writes between one call's argument and the next call's opening.

    Read from a turn that makes two calls, so it holds the first call's own
    ending but not what the template writes after the last call. Where the
    opening begins with a tag written once for all the calls of a turn, each
    call is found by the shortest end of the opening that starts with a tag.

    Returns:
        The text, or ``None`` where the template does not open each call with
        ``call_open`` or such an end of it.
    """
    rendered = _as_probes_read(render)([
        {"role": "user", "content": _USER},
        calling_message("", [ToolCall(_TOOL, {_KEY: value}) for value in (_STRING, _STRING2)], 0),
    ], reasoning=False)
    if rendered is None:
        return None
    openings = [call_open] + [call_open[i:] for i in range(len(call_open) - 1, 0, -1)
                              if call_open[i] == "<"]
    each_call = next((opening for opening in openings if rendered.count(opening) >= 2), None)
    if each_call is None:
        return None
    first_call = rendered.split(each_call)[1]
    return first_call.split(_STRING, 1)[1] if _STRING in first_call else None


def extract_call_body(render: Render, call_open: str,
                      close_tag: str) -> CallBody | None:
    """Recover how this model spells a name and its arguments between the tags.

    Four renders: no arguments, one string, one number, and two strings. Every
    delimiter falls out of comparing them, and three more recover how nested
    values are written. See the README, "Tool calling".

    Args:
        render: As :func:`extract_fragments`.
        call_open: The recovered opening, tag and lead-in together.
        close_tag: The control token that ends a call, on its own. The recovered
            ``call_close`` cannot stand in for it, because it carries the body's
            own ending in front of the tag.

    Returns:
        The grammar, or ``None`` where the probes prove nothing about some
        delimiter. Such a model's calls are still detected, because the tags are
        tokens, but what is inside one cannot be read.
    """
    render = _as_probes_read(render)
    def rendered_body(arguments) -> str | None:
        rendered = _render_call(render, arguments)
        if rendered is None or call_open not in rendered:
            return None
        after = rendered.split(call_open, 1)[1]
        return after.split(close_tag, 1)[0] if close_tag in after else None

    empty, text = rendered_body({}), rendered_body({_KEY: _STRING})
    number, pair = rendered_body({_KEY: _NUMBER}), rendered_body({_KEY: _STRING,
                                                                  _KEY2: _STRING2})
    if any(r is None or _TOOL not in r for r in (empty, text, number, pair)):
        return None
    # The sentinels the splits below index on. A template that drops one proves
    # nothing about the delimiter that would have surrounded it.
    if (_KEY not in text or _STRING not in text
            or _KEY not in number or str(_NUMBER) not in number
            or _STRING not in pair or _KEY2 not in pair):
        return None

    # After the name: as far as the one-argument and the no-argument renders
    # agree. Past that they diverge into the first argument and into the end.
    opens_args = text.split(_TOOL, 1)[1].split(_KEY, 1)[0]
    no_arguments = empty.split(_TOOL, 1)[1]
    name_close = opens_args[:common_prefix_length(opens_args, no_arguments)]
    # Nothing between the name and the arguments is readable only where the
    # first key's own tag marks where they begin, and nothing follows a name
    # without them.
    if not name_close and (not opens_args or no_arguments):
        return None

    before_value = text.split(_KEY, 1)[1].split(_STRING, 1)[0]
    bare_before = number.split(_KEY, 1)[1].split(str(_NUMBER), 1)[0]
    after_value = text.split(_STRING, 1)[1]
    bare_after = number.split(str(_NUMBER), 1)[1]
    # A type written between the key and its value differs where the values do.
    type_close = string_type = ""
    if not before_value.startswith(bare_before):
        type_close = _common_suffix(before_value, bare_before)
        key_close = before_value[:common_prefix_length(before_value, bare_before)]
        string_type = before_value[len(key_close):len(before_value) - len(type_close)]
        number_type = bare_before[len(key_close):len(bare_before) - len(type_close)]
        if not (type_close and string_type and number_type):
            return None
        before_value = bare_before = key_close
    if not after_value.endswith(bare_after):
        return None

    # Between the first value and the second key, the quoting excluded.
    between = pair.split(_STRING, 1)[1].split(_KEY2, 1)[0]
    quote_close = after_value[:len(after_value) - len(bare_after)]
    if not between.startswith(quote_close):
        return None

    pair_sep, body_close = between[len(quote_close):], bare_after
    # A value closed by a tag naming its own key: the key is the closing's, not a delimiter.
    key_in_close = _KEY in pair_sep and _KEY in body_close
    if key_in_close:
        pair_sep, body_close = pair_sep.replace(_KEY, "{key}"), body_close.replace(_KEY, "{key}")
    body = CallBody(
        name_close=name_close,
        key_open=opens_args[len(name_close):],
        key_close=bare_before,
        pair_sep=pair_sep,
        body_close=body_close,
        empty_close=no_arguments[len(name_close):],
        quote_open=before_value[len(bare_before):],
        quote_close=quote_close,
        type_close=type_close,
        string_type=string_type,
        key_in_close=key_in_close,
        **_nested_forms(rendered_body, text, before_value[len(bare_before):], quote_close),
    )
    # A delimiter still holding a probe's own text is no fixed delimiter at all.
    sentinels = (_TOOL, _KEY, _KEY2, _STRING, _STRING2, str(_NUMBER))
    if any(sentinel in delimiter for sentinel in sentinels
           for delimiter in dataclasses.astuple(body) if isinstance(delimiter, str)):
        return None
    return body


def _nested_forms(rendered_body, text: str, quote_open: str, quote_close: str) -> dict:
    """How a template writes an object's members and a list's items in their own delimiters.

    Read from a call whose one argument is an object with one member, one with
    two, and a list of two strings, each compared with the call whose argument
    is a string. Empty where the template writes nested values as JSON or a
    literal, which the values are read as anyway, or in no fixed form.
    """
    value_at = text.index(_STRING)
    before, after = text[:value_at - len(quote_open)], text[value_at + len(_STRING) + len(quote_close):]

    def nested_value(argument) -> str | None:
        body = rendered_body({_KEY: argument})
        if body is None or not body.startswith(before) or not body.endswith(after) \
                or len(body) < len(before) + len(after):
            return None
        return body[len(before):len(body) - len(after)]

    member, items = nested_value({_KEY2: _STRING}), nested_value([_STRING, _STRING2])
    if not member or not items or member[0] in "{[" or member.count(_STRING) != 1:
        return {}
    opening, closing = member.split(_STRING)
    if opening.count(_KEY2) != 1 or closing.count(_KEY2) != 1:
        return {}
    member_open, member_close = opening.replace(_KEY2, "{key}"), closing.replace(_KEY2, "{key}")
    # Two members follow each other with nothing between them.
    pair = nested_value({_KEY2: _STRING, _KEY: _STRING2})
    expected = (opening + _STRING + closing
                + member_open.replace("{key}", _KEY) + _STRING2 + member_close.replace("{key}", _KEY))
    item_open, found, rest = items.partition(_STRING)
    between, found_second, item_close = rest.partition(_STRING2)
    if pair != expected or not (found and found_second and item_open and item_close) \
            or between != item_close + item_open:
        return {}
    return {"member_open": member_open, "member_close": member_close,
            "item_open": item_open, "item_close": item_close}


def extract_call_close_text(render: Render, call_open: str) -> str | None:
    """The text that ends one call, for a template whose call has no closing tag of its own.

    The longest text, from where a tag starts to where one ends, found both in
    what follows the first of two calls and in what a call with an argument
    and one without both end with. Where a template writes no two calls in one
    turn, the longest such end of the latter.
    """
    render = _as_probes_read(render)
    empty, text = _render_call(render, {}), _render_call(render, {_KEY: _STRING})
    if empty is None or text is None or call_open not in empty or call_open not in text:
        return None
    rest = _common_suffix(empty.split(call_open, 1)[1], text.split(call_open, 1)[1])
    separator = extract_call_separator(render, call_open)
    if separator is None:
        found = [rest[start:] for start in range(len(rest)) if rest[start] in "<["]
    else:
        found = [separator[start:end] for start in range(len(separator))
                 if separator[start] in "<["
                 for end in range(len(separator), start, -1) if separator[start:end] in rest]
    found = [candidate for candidate in found if candidate[-1] in ">]"]
    return max(found, key=len) if found else None


def extract_named_twice_opening(render: Render, call_open: str) -> str | None:
    """Where a call names its function twice, the text between the two names.

    Such a template writes the name in the turn's header, right after
    ``call_open``, and again in the call's body, which is what opens the call
    there. ``None`` where the name does not follow ``call_open`` and recur.
    """
    rendered = _render_call(_as_probes_read(render), {_KEY: _STRING})
    if rendered is None or call_open not in rendered:
        return None
    after = rendered.split(call_open, 1)[1]
    if not after.startswith(_TOOL) or _TOOL not in after[len(_TOOL):]:
        return None
    return after[len(_TOOL):after.index(_TOOL, len(_TOOL))]


def _common_suffix(one: str, other: str) -> str:
    """The longest ending the two share."""
    return one[len(one) - common_suffix_length(one, other):]


def _render_call(render: Render, arguments: dict) -> str | None:
    """Render one assistant turn calling ``TOOLNAME`` with ``arguments``."""
    return render([{"role": "user", "content": _USER},
                   calling_message("", [ToolCall(_TOOL, arguments)], 0)], reasoning=False)


def _past_exchange(render: Render, first_prefix: str,
                   user_suffix: str) -> tuple[str, str, str] | None:
    """The user and assistant tags, read from a reply that a later turn follows."""
    rendered = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": _ASSISTANT},
        {"role": "user", "content": _USER2},
    ], reasoning=False)
    newest = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": _ASSISTANT},
    ], reasoning=False)
    opened = _USER + user_suffix
    if rendered is None or newest is None or _ASSISTANT not in newest \
            or opened not in rendered or _USER2 not in rendered:
        return None
    exchange = rendered.split(opened, 1)[1]
    if _ASSISTANT not in exchange:
        return None
    assistant_prefix, _, between = exchange.partition(_ASSISTANT)
    between = between.split(_USER2, 1)[0]
    # Empty where turns have no closing tag, the next turn's tag ending them.
    assistant_suffix = newest.split(_ASSISTANT, 1)[1]
    if not between.startswith(assistant_suffix):
        # A past reply closes with another tag than a newest one; the user
        # tag then starts where the role tags' shared opening last occurs.
        opener = assistant_prefix[:shared_tags_length(assistant_prefix, first_prefix)]
        start = between.rfind(opener) if opener else -1
        if start <= 0:
            return None
        assistant_suffix = between[:start]
    user_prefix = between[len(assistant_suffix):]
    if user_prefix and not first_prefix.endswith(user_prefix):
        # A past reply closes with a tag a newest one lacks; the user tag is
        # then the longest end of what follows that a first turn's opening shares.
        user_prefix = _longest_shared_end(user_prefix, first_prefix)
        assistant_suffix = between[:len(between) - len(user_prefix)]
    # What a first turn carries beyond the tag is written once, before it.
    if not user_prefix or not first_prefix.endswith(user_prefix):
        return None
    return user_prefix, assistant_prefix, assistant_suffix


def _longest_shared_end(text: str, other: str) -> str:
    """The longest end of ``text`` that ``other`` ends with, starting where a tag starts."""
    for start in range(len(text)):
        if text[start] in "<[" and other.endswith(text[start:]):
            return text[start:]
    return ""


def _newest_exchange(render: Render, user_prefix: str,
                     user_suffix: str) -> tuple[str, str, str]:
    """The tags read from one user turn and its reply, or the user tags reused."""
    both = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": _ASSISTANT},
    ], reasoning=False)
    if both is None or _ASSISTANT not in both or _USER + user_suffix not in both:
        # No assistant turn to split off; the user tags still give generation
        # an open and close pair.
        return user_prefix, user_prefix, user_suffix
    tail = both.split(_USER + user_suffix, 1)[1]
    assistant_prefix, _, assistant_suffix = tail.partition(_ASSISTANT)
    return user_prefix, assistant_prefix, assistant_suffix


def _newest_opening(render: Render, fragments: TemplateFragments) -> str:
    """What a newest reply opens with after the assistant tag, beyond a past one's."""
    newest = render([{"role": "user", "content": _USER},
                     {"role": "assistant", "content": _ASSISTANT}], reasoning=False)
    opened = _USER + fragments.user_suffix + fragments.assistant_prefix
    if newest is None or opened not in newest:
        return ""
    return newest.split(opened, 1)[1].split(_ASSISTANT, 1)[0]


def _prompt_added(render: Render, reasoning: bool) -> str | None:
    """The whole generation prompt after a user turn, or ``None``."""
    messages = [{"role": "user", "content": _USER}]
    plain = render(messages, reasoning=reasoning)
    prompted = render(messages, generation_prompt=True, reasoning=reasoning)
    if plain is None or prompted is None or not prompted.startswith(plain):
        return None
    return prompted[len(plain):]


def _whole_prompt(render: Render, assistant_prefix: str) -> str:
    """The generation prompt without reasoning, empty where it is the tag alone."""
    prompt = _prompt_added(render, False)
    return "" if prompt in (None, assistant_prefix) else prompt


def _reply_strip(render: Render, user_suffix: str, assistant_prefix: str) -> str | None:
    """Which whitespace a past reply loses, where it differs from a user turn's."""
    rendered = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": " " + _ASSISTANT + " "},
        {"role": "user", "content": _USER2},
    ], reasoning=False)
    opened = _USER + user_suffix + assistant_prefix
    if rendered is None or opened not in rendered or _ASSISTANT not in rendered:
        return None
    before, _, after = rendered.split(opened, 1)[1].partition(_ASSISTANT)
    return _strip_unlike_user(render, before, after)


def _system_strip(render: Render) -> str | None:
    """Which whitespace a system prompt loses, where it differs from a user turn's."""
    rendered = render([{"role": "system", "content": " " + _SYS + " "},
                       {"role": "user", "content": _USER}], reasoning=False)
    if rendered is None or _SYS not in rendered:
        return None
    before, _, after = rendered.partition(_SYS)
    return _strip_unlike_user(render, before, after)


def _strip_unlike_user(render: Render, before: str, after: str) -> str | None:
    """The strip a content padded with spaces shows, ``None`` where a user turn's matches."""
    keeps_start, keeps_end = before.endswith(" "), after.startswith(" ")
    strip = {(False, False): "both", (True, False): "end",
             (False, True): "start", (True, True): "none"}[keeps_start, keeps_end]
    user_strip = "both" if _trims_content(render) else "none"
    return None if strip == user_strip else strip


def _thought_tags(render: Render, fragments: TemplateFragments,
                  tool_calls: ToolCallFragments | None
                  ) -> tuple[tuple[str, str], dict] | None:
    """The tags around a reply's thought, read from a newest reply that holds one.

    A template may write a thought only in a turn that calls a tool, so that
    turn is the second probe, and templates read the thought from either of two
    fields. The opening tag is what the model writes after the generation
    prompt before the thought, or else what the turn holds before it beyond a
    past reply's tag; where a past reply's tag already holds it, as an empty
    thought, it is the last tag before the thought.
    """
    opened = _USER + fragments.user_suffix
    prompt = fragments.generation_prompt or fragments.assistant_prefix
    probes = []
    replies = [{"role": "assistant", "content": _ASSISTANT, field: _THOUGHT}
               for field in _THOUGHT_FIELDS]
    # A template may read the thought from a content block of its own.
    replies.append({"role": "assistant", "content": [
        {"type": "thinking", "thinking": _THOUGHT},
        {"type": "text", "text": _ASSISTANT}]})
    for reply in replies:
        probes.append(([{"role": "user", "content": _USER}, reply], _ASSISTANT))
        if tool_calls is not None:
            probes.append(([{"role": "user", "content": _USER},
                            {**reply, **calling_message("", [ToolCall(_TOOL, {})], 0)}],
                           tool_calls.call_open))
    for messages, after in probes:
        rendered = render(messages, reasoning=True)
        if rendered is None or opened not in rendered:
            continue
        turn = rendered.split(opened, 1)[1]
        if _THOUGHT not in turn or _prints_blocks(turn, messages[-1]["content"]):
            continue
        before, _, rest = turn.partition(_THOUGHT)
        if before.startswith(prompt) and before[len(prompt):].strip():
            # What the model writes itself after the generation prompt.
            thought_open = before[len(prompt):]
        else:
            thought_open = before[shared_tags_length(before, fragments.assistant_prefix):]
        if not thought_open.strip():
            thought_open = _last_tag(before)
        thought_close = rest.split(after, 1)[0] if after in rest else ""
        if thought_open.strip() and thought_close.strip():
            return (thought_open, thought_close), messages[-1]
    return None


def _prints_blocks(turn: str, content) -> bool:
    """Whether ``turn`` holds ``content``, a list of content blocks, printed rather than read."""
    return not isinstance(content, str) and any(
        printed in turn for printed in (str(content), json.dumps(content)))


def _last_tag(text: str) -> str:
    """The last tag in ``text`` and what follows it.

    A tag holds no whitespace, and of a run of tags written together the last
    is the one that starts at the run's last ``<``.
    """
    stripped = text.rstrip()
    if not stripped:
        return ""
    run = stripped.rsplit(None, 1)[-1]
    start = len(stripped) - len(run) + max(run.rfind("<"), 0)
    return text[start:]


def _extract_system(render: Render, user_prefix: str, user_suffix: str) -> dict:
    """Split off the system fragment, or fall back to the user tags.

    There is no portable ground truth for a system role, so a sentinel that does
    not survive verbatim ahead of the user turn means the user tags are reused.
    """
    rendered = render([
        {"role": "system", "content": _SYS},
        {"role": "user", "content": _USER},
    ], reasoning=False)
    if rendered is None or _SYS not in rendered or _USER not in rendered:
        return {"system_prefix": user_prefix, "system_suffix": user_suffix}
    # The system fragment is everything before the user turn begins. Templates
    # that fold the system message *into* the first user turn have no standalone
    # user turn to split on; ``head`` then still contains the user sentinel, so
    # no clean system fragment exists, so fall back to the user tags.
    head = rendered.split(user_prefix + _USER, 1)[0]
    if _USER in head:
        return {"system_prefix": user_prefix, "system_suffix": user_suffix}
    sys_prefix, _, sys_suffix = head.partition(_SYS)
    return {"system_prefix": sys_prefix, "system_suffix": sys_suffix}


# Characters a template may write otherwise in content, as an escaping filter does.
ESCAPABLE = ("<", ">", "&", '"', "'")


def extract_content_rewrites(render: Render, texts) -> tuple[tuple[str, str], ...]:
    """What the template writes in a user message's content for each of ``texts`` it changes.

    One render, each text between markers of its own, so a template that
    escapes or drops a tag, or escapes a character, is read as it does so.
    ``()`` where the render cannot be read that way.
    """
    texts = [text for text in dict.fromkeys(texts) if text and _MARK not in text]
    marks = [f"{_MARK}{n}{_MARK}" for n in range(len(texts) + 1)]
    content = marks[0] + "".join(text + mark for text, mark in zip(texts, marks[1:]))
    rendered = _as_probes_read(render)([{"role": "user", "content": content}], reasoning=False)
    if rendered is None or not all(mark in rendered for mark in marks):
        return ()
    rewrites = []
    for n, text in enumerate(texts):
        written = rendered.split(marks[n], 1)[1].split(marks[n + 1], 1)[0]
        if written != text:
            rewrites.append((text, written))
    return tuple(rewrites)


def _trims_content(render: Render) -> bool:
    """Whether the template strips surrounding whitespace from content."""
    padded = render([{"role": "user", "content": " " + _USER + " "}], reasoning=False)
    if padded is None or _USER not in padded:
        return True
    before, _, after = padded.partition(_USER)
    return not (before.endswith(" ") and after.startswith(" "))
