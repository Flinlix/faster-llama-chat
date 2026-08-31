"""Recover a model's own chat template by rendering probes and comparing them.

Probes carry sentinel content, and splitting a render on its sentinels recovers
the template that produced it. Three things come out:

* :func:`extract_fragments`: the tags wrapped around each role's content.
* :func:`extract_tool_fragments`: the tags a generated tool call sits between.
* :func:`extract_call_body`: how a call spells its name and arguments.

No model's format is written down here, and where a comparison proves nothing
the answer is ``None`` rather than a guess. See the README, "Tool calling".

A pure function of the render callable, so it is testable without a model.
"""

from __future__ import annotations

from typing import Callable

from .template import CallBody, TemplateFragments, ToolCallFragments

# Plain alphanumerics: they survive Jinja `| trim` and are never HTML-escaped.

# Message content, one per role.
_SYS = "SYSMSG"
_USER = "USERMSG"
_ASSISTANT = "ASSTMSG"

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

Render = Callable[[list[dict]], "str | None"]


def extract_fragments(render: Render) -> TemplateFragments | None:
    """Recover template fragments by probing ``render`` with sentinel content.

    Args:
        render: Renders ``{"role", "content"}`` messages through the model's
            chat template, or returns ``None`` where it rejects their shape.

    Returns:
        The fragments, or ``None`` where the model ships no usable template.
    """
    user = render([{"role": "user", "content": _USER}])
    if user is None or _USER not in user:
        return None
    user_prefix, _, user_suffix = user.partition(_USER)

    fragments = {
        "user_prefix": user_prefix,
        "user_suffix": user_suffix,
        "trim_content": _trims_content(render),
    }
    assistant, assistant_recovered = _extract_assistant(
        render, user_prefix, user_suffix)
    fragments.update(assistant)
    fragments.update(_extract_system(render, user_prefix, user_suffix))
    # Only a tag the template really produced locates the turn a call sits in.
    # Splitting on the user tags standing in for it lands mid-conversation, and
    # what came back would open a call with a turn tag.
    fragments["tools"] = (
        extract_tool_fragments(render, fragments["assistant_prefix"])
        if assistant_recovered else None)
    return TemplateFragments(**fragments)


def extract_tool_fragments(render: Render,
                           assistant_prefix: str) -> ToolCallFragments | None:
    """Recover what a template wraps a generated tool call in.

    Two probes differing only in their argument's type, so what the renders
    share around the call is structure rather than content.

    Args:
        render: As :func:`extract_fragments`.
        assistant_prefix: The recovered assistant-open tag, where the turn
            holding the call begins.

    Returns:
        The tags, or ``None`` where nothing usable can be recovered, which is
        what a model that cannot call tools looks like from here.
    """
    text = _render_call(render, {_KEY: _STRING})
    number = _render_call(render, {_KEY: _NUMBER})
    if text is None or number is None:
        return None
    if _TOOL not in text or _TOOL not in number:
        return None            # the template dropped the call instead of rendering it
    head = text.split(_TOOL, 1)[0]
    # The turn holding the call has to be locatable, or there is no telling
    # where the conversation before it ends and the call's own opening begins.
    if not assistant_prefix or assistant_prefix not in head:
        return None
    call_open = head.rsplit(assistant_prefix, 1)[1]
    call_close = _common_suffix(text, number)
    # A template that ignored the arguments renders the two probes identically,
    # and then the shared ending runs back past the call into the conversation.
    if not call_open or not call_close or _TOOL in call_close:
        return None
    return ToolCallFragments(call_open=call_open, call_close=call_close)


def extract_call_body(render: Render, call_open: str,
                      close_tag: str) -> CallBody | None:
    """Recover how this model spells a name and its arguments between the tags.

    Four renders: no arguments, one string, one number, and two strings. Every
    delimiter falls out of comparing them. See the README, "Tool calling".

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
    name_close = _common_prefix(opens_args, no_arguments)
    if not name_close:
        return None

    before_value = text.split(_KEY, 1)[1].split(_STRING, 1)[0]
    bare_before = number.split(_KEY, 1)[1].split(str(_NUMBER), 1)[0]
    after_value = text.split(_STRING, 1)[1]
    bare_after = number.split(str(_NUMBER), 1)[1]
    if not before_value.startswith(bare_before) \
            or not after_value.endswith(bare_after):
        return None

    # Between the first value and the second key, the quoting excluded.
    between = pair.split(_STRING, 1)[1].split(_KEY2, 1)[0]
    quote_close = after_value[:len(after_value) - len(bare_after)]
    if not between.startswith(quote_close):
        return None

    return CallBody(
        name_close=name_close,
        key_open=opens_args[len(name_close):],
        key_close=bare_before,
        pair_sep=between[len(quote_close):],
        body_close=bare_after,
        empty_close=no_arguments[len(name_close):],
        quote_open=before_value[len(bare_before):],
        quote_close=quote_close,
    )


def _common_prefix(one: str, other: str) -> str:
    """The longest beginning the two share."""
    kept = 0
    while kept < len(one) and kept < len(other) and one[kept] == other[kept]:
        kept += 1
    return one[:kept]


def _common_suffix(one: str, other: str) -> str:
    """The longest ending the two share."""
    kept = 0
    while (kept < len(one) and kept < len(other)
           and one[-1 - kept] == other[-1 - kept]):
        kept += 1
    return one[len(one) - kept:]


def _render_call(render: Render, arguments: dict) -> str | None:
    """Render one assistant turn calling ``TOOLNAME`` with ``arguments``."""
    return render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": _TOOL, "arguments": arguments}}]},
    ])


def _extract_assistant(render: Render, user_prefix: str,
                       user_suffix: str) -> tuple[dict, bool]:
    """Split off the assistant fragment from a user and assistant render.

    Returns:
        The fragments, and whether they were recovered rather than fallen back
        to the user tags.
    """
    both = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": _ASSISTANT},
    ])
    fallback = {"assistant_prefix": user_prefix, "assistant_suffix": user_suffix}
    if both is None or _ASSISTANT not in both or _USER + user_suffix not in both:
        # The two-message render does not contain the expected user turn (or no
        # assistant sentinel at all); fall back to the user tags so generation
        # still has an open/close pair.
        return fallback, False
    tail = both.split(_USER + user_suffix, 1)[1]
    asst_prefix, _, asst_suffix = tail.partition(_ASSISTANT)
    return {"assistant_prefix": asst_prefix,
            "assistant_suffix": asst_suffix}, True


def _extract_system(render: Render, user_prefix: str, user_suffix: str) -> dict:
    """Split off the system fragment, or fall back to the user tags.

    There is no portable ground truth for a system role, so a sentinel that does
    not survive verbatim ahead of the user turn means the user tags are reused.
    """
    rendered = render([
        {"role": "system", "content": _SYS},
        {"role": "user", "content": _USER},
    ])
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


def _trims_content(render: Render) -> bool:
    """Whether the template strips surrounding whitespace from content."""
    padded = render([{"role": "user", "content": " " + _USER + " "}])
    if padded is None or _USER not in padded:
        return True
    before, _, after = padded.partition(_USER)
    return not (before.endswith(" ") and after.startswith(" "))
