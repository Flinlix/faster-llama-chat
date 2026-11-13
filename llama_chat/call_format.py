"""Recover how a model's own chat template writes a tool call, by rendering probes and comparing them.

The tags a generated call sits between, what separates and closes calls, and
how a call spells its name and arguments. Where a comparison proves nothing the
answer is ``None`` rather than a guess. See design_history.md, "How a call's
format is recovered".
"""

from __future__ import annotations

import dataclasses
import json

from .messages import calling_message
from .template import (CallBody, TemplateFragments, ToolCallFragments, common_prefix_length,
                       common_suffix_length, json_arguments, shared_tags_length)
from .template_extract import (ASSISTANT_SENTINEL, TOOL_SENTINEL, USER_SENTINEL, Render,
                               as_probes_read)
from .tools import ToolCall

# Probe arguments, plain alphanumerics like every probe text. `AKEYONE` sorts
# before `AKEYTWO`, so the two-argument probe reads the keys in the order it
# wrote them even under a template that sorts them.
_KEY = "AKEYONE"
_KEY2 = "AKEYTWO"

# The string and the number end in different characters, so a shared ending
# cannot run back into the value itself.
_STRING = "VALONE"
_NUMBER = 1234
_STRING2 = "VALTWO"


def extract_tool_fragments(render: Render, fragments: TemplateFragments,
                           tags: tuple[str, ...]) -> ToolCallFragments | None:
    """Recover what a template wraps a generated tool call in.

    Two probes differing only in their argument's type, so what the renders
    share around the call is structure rather than content.

    Args:
        render: As :func:`~llama_chat.template_extract.extract_fragments`.
        fragments: The recovered turn tags. The turn holding a call begins
            after the user turn and the assistant tag, so a template whose
            assistant tag cannot be told from its user tag cannot locate that
            turn at all, and what came back would open a call with a turn tag.
            An assistant tag that is empty is no such case: the reply then
            begins where the user turn ends.
        tags: The texts of the model's own tags, where a tag ends.

    Returns:
        The tags, or ``None`` where nothing usable can be recovered, which is
        what a model that cannot call tools looks like from here.
    """
    render = as_probes_read(render, tags)
    if fragments.assistant_prefix == fragments.user_prefix:
        return None
    text = _render_call(render, {_KEY: _STRING})
    number = _render_call(render, {_KEY: _NUMBER})
    if text is None or number is None:
        return None
    if TOOL_SENTINEL not in text or TOOL_SENTINEL not in number:
        return None            # the template dropped the call instead of rendering it
    head = text.split(TOOL_SENTINEL, 1)[0]
    # The turn holding the call has to be locatable, or there is no telling
    # where the conversation before it ends and the call's own opening begins.
    # It may open without the content tag a reply in words opens with, and
    # then begins after the tags the two share.
    after_user = USER_SENTINEL + fragments.user_suffix
    if after_user not in head:
        return None
    turn = head.split(after_user, 1)[1]
    shared = (len(fragments.assistant_prefix) if turn.startswith(fragments.assistant_prefix)
              else shared_tags_length(turn, fragments.assistant_prefix, tags))
    if not shared and fragments.assistant_prefix:
        return None
    call_open = turn[shared:]
    # A newest reply may open with more than a past one, an empty thought for
    # instance, which is not part of the call.
    call_open = call_open.removeprefix(_newest_opening(render, fragments))
    call_close = _common_suffix(text, number)
    # A template that ignored the arguments renders the two probes identically,
    # and then the shared ending runs back past the call into the conversation.
    if not call_open or not call_close or TOOL_SENTINEL in call_close:
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
    rendered = as_probes_read(render)([
        {"role": "user", "content": USER_SENTINEL},
        calling_message("", [ToolCall(TOOL_SENTINEL, {_KEY: value})
                             for value in (_STRING, _STRING2)], 0),
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
    values are written. See design_history.md, "How a call's format is
    recovered" and "Nested values in a model's own elements, and tests by
    module".

    Args:
        render: As :func:`~llama_chat.template_extract.extract_fragments`.
        call_open: The recovered opening, tag and lead-in together.
        close_tag: The control token that ends a call, on its own. The recovered
            ``call_close`` cannot stand in for it, because it carries the body's
            own ending in front of the tag.

    Returns:
        The grammar, or ``None`` where the probes prove nothing about some
        delimiter. Such a model's calls are still detected, because the tags are
        tokens, but what is inside one cannot be read.
    """
    render = as_probes_read(render)
    def rendered_body(arguments) -> str | None:
        rendered = _render_call(render, arguments)
        if rendered is None or call_open not in rendered:
            return None
        after = rendered.split(call_open, 1)[1]
        return after.split(close_tag, 1)[0] if close_tag in after else None

    empty, text = rendered_body({}), rendered_body({_KEY: _STRING})
    number, pair = rendered_body({_KEY: _NUMBER}), rendered_body({_KEY: _STRING,
                                                                  _KEY2: _STRING2})
    if any(r is None or TOOL_SENTINEL not in r for r in (empty, text, number, pair)):
        return None
    # The sentinels the splits below index on. A template that drops one proves
    # nothing about the delimiter that would have surrounded it.
    if (_KEY not in text or _STRING not in text
            or _KEY not in number or str(_NUMBER) not in number
            or _STRING not in pair or _KEY2 not in pair):
        return None

    # After the name: as far as the one-argument and the no-argument renders
    # agree. Past that they diverge into the first argument and into the end.
    opens_args = text.split(TOOL_SENTINEL, 1)[1].split(_KEY, 1)[0]
    no_arguments = empty.split(TOOL_SENTINEL, 1)[1]
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
        arguments_json=all(_reads_as_json(json_arguments(
                               name_close, body.split(TOOL_SENTINEL, 1)[1][len(name_close):]),
                                          arguments)
                           for body, arguments in ((text, {_KEY: _STRING}),
                                                   (pair, {_KEY: _STRING, _KEY2: _STRING2}))),
        **_nested_forms(rendered_body, text, before_value[len(bare_before):], quote_close),
    )
    # A delimiter still holding a probe's own text is no fixed delimiter at all.
    sentinels = (TOOL_SENTINEL, _KEY, _KEY2, _STRING, _STRING2, str(_NUMBER))
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
    render = as_probes_read(render)
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
    rendered = _render_call(as_probes_read(render), {_KEY: _STRING})
    if rendered is None or call_open not in rendered:
        return None
    after = rendered.split(call_open, 1)[1]
    if not after.startswith(TOOL_SENTINEL) or TOOL_SENTINEL not in after[len(TOOL_SENTINEL):]:
        return None
    return after[len(TOOL_SENTINEL):after.index(TOOL_SENTINEL, len(TOOL_SENTINEL))]


def _common_suffix(one: str, other: str) -> str:
    """The longest ending the two share."""
    return one[len(one) - common_suffix_length(one, other):]


def extract_json_call_keys(render: Render) -> tuple[str, ...] | None:
    """Where the template writes a call as one JSON object, its keys: the one naming the function, the one holding the arguments, then the others; ``None`` where it does not.

    Read from a call to ``TOOLNAME`` with one text argument: the JSON object
    or array around the name is decoded, and the keys whose values are the
    name and the arguments are taken.
    """
    arguments = {_KEY: _STRING}
    rendered = _render_call(as_probes_read(render), arguments)
    at = -1 if rendered is None else rendered.find(TOOL_SENTINEL)
    decoder = json.JSONDecoder()
    for start in (i for i in range(at, -1, -1) if rendered[i] in "{["):
        try:
            decoded, _end = decoder.raw_decode(rendered, start)
        except ValueError:
            continue
        call = decoded[0] if isinstance(decoded, list) and decoded else decoded
        if not isinstance(call, dict):
            continue
        name_key = next((key for key, value in call.items() if value == TOOL_SENTINEL), None)
        arguments_key = next((key for key, value in call.items() if value == arguments), None)
        if name_key is not None and arguments_key is not None:
            return (name_key, arguments_key,
                    *sorted(set(call) - {name_key, arguments_key}))
    return None


def _reads_as_json(text: str, arguments: dict) -> bool:
    """Whether ``text``, whatever follows it aside, is ``arguments`` written as one JSON object."""
    try:
        decoded, _end = json.JSONDecoder().raw_decode(text.lstrip())
    except ValueError:
        return False
    return decoded == arguments


def _render_call(render: Render, arguments: dict) -> str | None:
    """Render one assistant turn calling ``TOOLNAME`` with ``arguments``."""
    return render([{"role": "user", "content": USER_SENTINEL},
                   calling_message("", [ToolCall(TOOL_SENTINEL, arguments)], 0)], reasoning=False)


def _newest_opening(render: Render, fragments: TemplateFragments) -> str:
    """What a newest reply opens with after the assistant tag, beyond a past one's."""
    newest = render([{"role": "user", "content": USER_SENTINEL},
                     {"role": "assistant", "content": ASSISTANT_SENTINEL}], reasoning=False)
    opened = USER_SENTINEL + fragments.user_suffix + fragments.assistant_prefix
    if newest is None or opened not in newest:
        return ""
    return newest.split(opened, 1)[1].split(ASSISTANT_SENTINEL, 1)[0]
