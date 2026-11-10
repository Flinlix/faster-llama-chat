"""Recover a model's own chat template by rendering probes and comparing them.

Probes carry sentinel content, and splitting a render on its sentinels recovers
the template that produced it. Two things come out:

* :func:`extract_fragments`: the tags wrapped around each role's content.
* :func:`extract_reasoning`: how a thought is marked and reasoning turned on.

How a call is written is recovered by :mod:`llama_chat.call_format`. No model's
format is written down here, and where a comparison proves nothing the answer
is ``None`` rather than a guess.

A pure function of the render callable, so it is testable without a model.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .messages import calling_message
from .template import Reasoning, TemplateFragments, ToolCallFragments, shared_tags_length
from .tools import ToolCall

# The fields chat templates read a reply's thought from.
_THOUGHT_FIELDS = ("reasoning_content", "thinking")

# Message content, one per role, a second user turn, and a thought. These and the
# probe texts below are plain alphanumerics: they survive Jinja `| trim` and are
# never HTML-escaped.
_SYS = "SYSMSG"
USER_SENTINEL = "USERMSG"
_USER2 = "USERTWO"
ASSISTANT_SENTINEL = "ASSTMSG"
_THOUGHT = "THOUGHTMSG"
# The function a probe calls, and what brackets each text the rewrite probe reads.
TOOL_SENTINEL = "TOOLNAME"
_MARK = "QQMARK"

# Renders messages through the model's template, or returns None where the
# template rejects their shape, as a template that raises does too. Keywords:
# ``generation_prompt`` appends the generation prompt, and ``reasoning`` sets
# the template's reasoning switch.
Render = Callable[..., "str | None"]


def reasoning_switch(render_variables: Callable[..., str]) -> Callable[[bool | None], dict]:
    """The template variables that turn reasoning on or off, as a render reads them.

    ``enable_thinking``, the variable chat templates share for it, or, where
    that changes nothing, ``reasoning_effort``. Off is ``"none"`` where that
    changes what follows the user turn compared with the template's own
    default, and the default is then on; otherwise the default is off, and on
    is ``"high"`` where that differs there from it. Where on and off are the
    same, there is no switch.

    Args:
        render_variables: Renders ``messages`` with ``generation_prompt`` and the
            template variables in ``variables``, raising where the template does.
    """
    probe = [{"role": "system", "content": _SYS}, {"role": "user", "content": USER_SENTINEL}]
    render = refusing(render_variables)

    def rendered(variables: dict):
        return render(probe, generation_prompt=True, variables=variables)

    def after_user_turn_differs(on: dict, off: dict) -> bool:
        on_render, off_render = rendered(on), rendered(off)
        return None not in (on_render, off_render) \
            and on_render.split(USER_SENTINEL, 1)[-1] != off_render.split(USER_SENTINEL, 1)[-1]

    thinking = {True: {"enable_thinking": True}, False: {"enable_thinking": False}}
    if rendered(thinking[True]) == rendered(thinking[False]):
        none, high = {"reasoning_effort": "none"}, {"reasoning_effort": "high"}
        off = none if after_user_turn_differs(none, {}) else {}
        on = {} if off else high if after_user_turn_differs(high, {}) else {}
        if on != off:
            thinking = {True: on, False: off}
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


def as_probes_read(render: Render) -> Render:
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
    alone = render([{"role": "user", "content": USER_SENTINEL}], reasoning=False)
    followed = render([{"role": "user", "content": USER_SENTINEL},
                       {"role": "assistant", "content": ASSISTANT_SENTINEL}], reasoning=False)
    if alone is None or followed is None \
            or USER_SENTINEL not in alone or USER_SENTINEL not in followed:
        return ""
    after_alone = alone.split(USER_SENTINEL, 1)[1]
    after_followed = followed.split(USER_SENTINEL, 1)[1]
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
    render = as_probes_read(render)
    user = render([{"role": "user", "content": USER_SENTINEL}], reasoning=False)
    if user is None or USER_SENTINEL not in user:
        return None
    first_prefix, _, user_suffix = user.partition(USER_SENTINEL)
    tags = (_past_exchange(render, first_prefix, user_suffix)
            or _newest_exchange(render, first_prefix, user_suffix))
    user_prefix, assistant_prefix, assistant_suffix = tags
    return TemplateFragments(
        user_prefix=user_prefix, user_suffix=user_suffix,
        assistant_prefix=assistant_prefix, assistant_suffix=assistant_suffix,
        generation_prompt=_whole_prompt(render, assistant_prefix))


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
    render = as_probes_read(render)
    found = _thought_tags(render, fragments, tool_calls) or _empty_thought_tags(render)
    if found is None:
        return None
    tags, reply_with_thought = found
    prompt = fragments.generation_prompt_without_reasoning
    messages = [{"role": "system", "content": _SYS}, {"role": "user", "content": USER_SENTINEL}]
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
    render = as_probes_read(render)
    known = set(tags)
    for close in tags:
        opening = "<" + close[2:]
        if not close.startswith("</") or opening not in known:
            continue
        rendered = render([{"role": "user", "content": USER_SENTINEL},
                           {"role": "assistant", "content": _THOUGHT + close + ASSISTANT_SENTINEL},
                           {"role": "user", "content": _USER2}], reasoning=False)
        if rendered is not None and ASSISTANT_SENTINEL in rendered and _THOUGHT not in rendered:
            return Reasoning(thought_open=opening, thought_close=close,
                             generation_prompt=fragments.generation_prompt_without_reasoning)
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
        {"role": "user", "content": USER_SENTINEL},
        {key: value for key, value in reply.items() if key != "tool_calls"}
        | ({"content": ASSISTANT_SENTINEL} if "tool_calls" in reply else {}),
        {"role": "user", "content": _USER2},
    ], reasoning=True)
    return rendered is not None and _THOUGHT in rendered


def _past_exchange(render: Render, first_prefix: str,
                   user_suffix: str) -> tuple[str, str, str] | None:
    """The user and assistant tags, read from a reply that a later turn follows."""
    rendered = render([
        {"role": "user", "content": USER_SENTINEL},
        {"role": "assistant", "content": ASSISTANT_SENTINEL},
        {"role": "user", "content": _USER2},
    ], reasoning=False)
    newest = render([
        {"role": "user", "content": USER_SENTINEL},
        {"role": "assistant", "content": ASSISTANT_SENTINEL},
    ], reasoning=False)
    opened = USER_SENTINEL + user_suffix
    if rendered is None or newest is None or ASSISTANT_SENTINEL not in newest \
            or opened not in rendered or _USER2 not in rendered:
        return None
    exchange = rendered.split(opened, 1)[1]
    if ASSISTANT_SENTINEL not in exchange:
        return None
    assistant_prefix, _, between = exchange.partition(ASSISTANT_SENTINEL)
    between = between.split(_USER2, 1)[0]
    # Empty where turns have no closing tag, the next turn's tag ending them.
    assistant_suffix = newest.split(ASSISTANT_SENTINEL, 1)[1]
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
        {"role": "user", "content": USER_SENTINEL},
        {"role": "assistant", "content": ASSISTANT_SENTINEL},
    ], reasoning=False)
    if both is None or ASSISTANT_SENTINEL not in both or USER_SENTINEL + user_suffix not in both:
        # No assistant turn to split off; the user tags still give generation
        # an open and close pair.
        return user_prefix, user_prefix, user_suffix
    tail = both.split(USER_SENTINEL + user_suffix, 1)[1]
    assistant_prefix, _, assistant_suffix = tail.partition(ASSISTANT_SENTINEL)
    return user_prefix, assistant_prefix, assistant_suffix


def _prompt_added(render: Render, reasoning: bool) -> str | None:
    """The whole generation prompt after a user turn, or ``None``."""
    messages = [{"role": "user", "content": USER_SENTINEL}]
    plain = render(messages, reasoning=reasoning)
    prompted = render(messages, generation_prompt=True, reasoning=reasoning)
    if plain is None or prompted is None or not prompted.startswith(plain):
        return None
    return prompted[len(plain):]


def _whole_prompt(render: Render, assistant_prefix: str) -> str:
    """The generation prompt without reasoning, empty where it is the tag alone."""
    prompt = _prompt_added(render, False)
    return "" if prompt in (None, assistant_prefix) else prompt


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
    opened = USER_SENTINEL + fragments.user_suffix
    prompt = fragments.generation_prompt_without_reasoning
    probes = []
    replies = [{"role": "assistant", "content": ASSISTANT_SENTINEL, field: _THOUGHT}
               for field in _THOUGHT_FIELDS]
    # A template may read the thought from a content block of its own.
    replies.append({"role": "assistant", "content": [
        {"type": "thinking", "thinking": _THOUGHT},
        {"type": "text", "text": ASSISTANT_SENTINEL}]})
    for reply in replies:
        probes.append(([{"role": "user", "content": USER_SENTINEL}, reply], ASSISTANT_SENTINEL))
        if tool_calls is not None:
            probes.append(([{"role": "user", "content": USER_SENTINEL},
                            {**reply, **calling_message("", [ToolCall(TOOL_SENTINEL, {})], 0)}],
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
    rendered = as_probes_read(render)([{"role": "user", "content": content}], reasoning=False)
    if rendered is None or not all(mark in rendered for mark in marks):
        return ()
    rewrites = []
    for n, text in enumerate(texts):
        written = rendered.split(marks[n], 1)[1].split(marks[n + 1], 1)[0]
        if written != text:
            rewrites.append((text, written))
    return tuple(rewrites)
