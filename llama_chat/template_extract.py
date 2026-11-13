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
from typing import Callable

from .messages import THOUGHT_FORMS, calling_message, with_thought_as
from .template import Reasoning, TemplateFragments, ToolCallFragments, shared_tags_length
from .tools import ToolCall

# The variables chat templates share for the reasoning switch, a flag and an effort.
_SWITCH_NAMES = ("enable_thinking", "reasoning_effort")

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

    ``enable_thinking``, the variable chat templates share for it, where it
    changes the render; else ``reasoning_effort``, where ``"none"`` changes
    what follows the user turn compared with the template's own default: then
    ``"none"`` is off and the default is on. A template that turns reasoning
    on otherwise names it in its add-on file's ``reasoning_on``.

    Args:
        render_variables: Renders ``messages`` with ``generation_prompt`` and the
            template variables in ``variables``, raising where the template does.
    """
    probe = [{"role": "system", "content": _SYS}, {"role": "user", "content": USER_SENTINEL}]
    render = refusing(render_variables)

    def rendered(variables: dict):
        return render(probe, generation_prompt=True, variables=variables)

    flag_on, flag_off = {_SWITCH_NAMES[0]: True}, {_SWITCH_NAMES[0]: False}
    effort_off = {_SWITCH_NAMES[1]: "none"}
    found = None
    if rendered(flag_on) != rendered(flag_off):
        found = flag_on, flag_off
    else:
        none, default = rendered(effort_off), rendered({})
        if None not in (none, default) \
                and none.split(USER_SENTINEL, 1)[-1] != default.split(USER_SENTINEL, 1)[-1]:
            found = {}, effort_off
    on, off = found or (flag_on, flag_off)
    return lambda reasoning: {} if reasoning is None else on if reasoning else off


def refusing(render: Render) -> Render:
    """``render``, answering ``None`` where it raises, as a template refusing a shape does."""
    def rendered(*args, **kwargs):
        try:
            return render(*args, **kwargs)
        except Exception:
            return None
    return rendered


def as_probes_read(render: Render, tags: tuple[str, ...] = ()) -> Render:
    """``render`` as probes read it.

    It answers ``None`` where it raises, a template refusing a shape. A render
    without a generation prompt comes back without what the template writes
    once after a conversation's last message, which belongs to no turn, read
    up to where one of the model's ``tags`` or a word ends. A render this
    made already comes back as it is.
    """
    if getattr(render, "reads_as_probes", False):
        return render
    refused = refusing(render)
    end = _conversation_end(refused, tags)

    def rendered(*args, **kwargs):
        text = refused(*args, **kwargs)
        if end and text is not None and not kwargs.get("generation_prompt") \
                and text.endswith(end):
            return text[:-len(end)]
        return text
    rendered.reads_as_probes = True
    return rendered


def _conversation_end(render: Render, tags: tuple[str, ...]) -> str:
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
    end = after_alone[shared_tags_length(after_alone, after_followed, tags):]
    return end if followed.endswith(end) else ""


def extract_fragments(render: Render, tags: tuple[str, ...] = ()) -> TemplateFragments | None:
    """Recover template fragments by probing ``render`` with sentinel content.

    The user and assistant tags are read from a conversation of two user turns,
    where the first reply is a past one: a first turn alone may carry text the
    template writes once at the start, and a newest reply may carry what only
    the newest one does. Every probe is rendered with reasoning off, which is
    what the fragments describe; what reasoning changes is
    :func:`extract_reasoning`'s.

    Args:
        render: As ``Render`` above.
        tags: The texts of the model's own tags, where a user tag may begin.

    Returns:
        The fragments, or ``None`` where the model ships no usable template.
    """
    render = as_probes_read(render, tags)
    user = render([{"role": "user", "content": USER_SENTINEL}], reasoning=False)
    if user is None or USER_SENTINEL not in user:
        return None
    first_prefix, _, user_suffix = user.partition(USER_SENTINEL)
    found = (_past_exchange(render, first_prefix, user_suffix, tags)
             or _newest_exchange(render, first_prefix, user_suffix))
    if found is None:
        return None
    user_prefix, assistant_prefix, assistant_suffix = found
    return TemplateFragments(
        user_prefix=user_prefix, user_suffix=user_suffix,
        assistant_prefix=assistant_prefix, assistant_suffix=assistant_suffix,
        generation_prompt=_whole_prompt(render, assistant_prefix))


def extract_reasoning(render: Render, fragments: TemplateFragments,
                      tool_calls: ToolCallFragments | None,
                      tags: tuple[str, ...]) -> Reasoning | None:
    """Recover what turns this template's reasoning on, and how a thought is marked.

    Args:
        render: As ``Render`` above.
        fragments: The recovered turn tags.
        tool_calls: The recovered call tags, for a template that writes a
            thought only in a turn that calls a tool.
        tags: The texts of the model's own tags.

    Returns:
        The reasoning, or ``None`` where the template marks no thought that can
        be recovered. A template without a switch that marks one reasons, or
        may, in every reply.
    """
    render = as_probes_read(render, tags)
    found = (_thought_tags(render, fragments, tool_calls, tags)
             or _empty_thought_tags(render, tags))
    if found is None:
        return None
    (thought_open, thought_close), form = found
    prompt = fragments.generation_prompt_without_reasoning
    messages = [{"role": "system", "content": _SYS}, {"role": "user", "content": USER_SENTINEL}]
    on = render(messages, generation_prompt=True, reasoning=True)
    off = render(messages, generation_prompt=True, reasoning=False)
    switches = on is not None and off is not None and on != off
    if switches:
        prompt = _prompt_added(render, True)
        if prompt is None:
            return None
    return Reasoning(thought_open=thought_open, thought_close=thought_close,
                     generation_prompt=prompt, can_turn_off=switches,
                     **_past_thought(render, form))


def extract_removed_thought(render: Render, fragments: TemplateFragments,
                            tags: tuple[str, ...]) -> Reasoning | None:
    """The thought of a template that writes none but removes one from a past reply.

    Tried with each of the model's tags: the closing tag is the one after which
    a past reply keeps only its answer, and the opening tag the one that reads
    as the closing without its first slash. Such a template has no switch, so
    the model may reason in every reply.
    """
    render = as_probes_read(render, tags)
    known = set(tags)
    for close in tags:
        opening = close.replace("/", "", 1)
        if opening == close or opening not in known:
            continue
        rendered = render([{"role": "user", "content": USER_SENTINEL},
                           {"role": "assistant", "content": _THOUGHT + close + ASSISTANT_SENTINEL},
                           {"role": "user", "content": _USER2}], reasoning=False)
        if rendered is not None and ASSISTANT_SENTINEL in rendered and _THOUGHT not in rendered:
            return Reasoning(thought_open=opening, thought_close=close,
                             generation_prompt=fragments.generation_prompt_without_reasoning)
    return None


def _empty_thought_tags(render: Render, tags: tuple[str, ...]
                        ) -> tuple[tuple[str, str], None] | None:
    """The tags of the empty thought only the generation prompt without reasoning ends in.

    Read where a template writes no thought into a reply: two of the model's
    ``tags`` with nothing but whitespace between them, which the prompt with
    reasoning lacks.
    """
    off, on = _prompt_added(render, False), _prompt_added(render, True)
    if off is None or on is None or off == on:
        return None
    close = _ending_tag(off, tags)
    before = off.rstrip()[:-len(close)] if close else ""
    opening = _ending_tag(before, tags)
    if not close or not opening or opening == close:
        return None
    if on.endswith(off[len(before.rstrip()) - len(opening):]):
        return None
    return (opening, close), None


def _ending_tag(text: str, tags: tuple[str, ...]) -> str:
    """The longest of ``tags`` that ``text`` ends with, whitespace after it aside; empty where none."""
    stripped = text.rstrip()
    return max((tag for tag in tags if tag and stripped.endswith(tag)), key=len, default="")


def _past_thought(render: Render, found_in: str | None) -> dict:
    """Whether a past reply keeps its thought once a later turn follows, and the form it is read in.

    Tried in ``found_in``, the form the thought's tags were read from, first,
    then in each other of :data:`~llama_chat.messages.THOUGHT_FORMS`; not at
    all where the tags were read from no reply.

    Returns:
        The fields of :class:`~llama_chat.template.Reasoning` that say so.
    """
    if found_in is None:
        return {}
    for form in dict.fromkeys((found_in, *THOUGHT_FORMS)):
        reply = with_thought_as(_reply_with_thought(), form)
        rendered = render([{"role": "user", "content": USER_SENTINEL}, reply,
                           {"role": "user", "content": _USER2}], reasoning=True)
        if rendered is not None and _THOUGHT in rendered \
                and not _prints_blocks(rendered, reply["content"]):
            return {"past_keeps_thought": True, "past_thought_form": form}
    return {}


def _reply_with_thought() -> dict:
    """A probe reply that holds a thought, under ``reasoning_content``."""
    return {"role": "assistant", "content": ASSISTANT_SENTINEL, "reasoning_content": _THOUGHT}


def _past_exchange(render: Render, first_prefix: str,
                   user_suffix: str, tags: tuple[str, ...]) -> tuple[str, str, str] | None:
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
        opener = assistant_prefix[:shared_tags_length(assistant_prefix, first_prefix, tags)]
        start = between.rfind(opener) if opener else -1
        if start <= 0:
            return None
        assistant_suffix = between[:start]
    user_prefix = between[len(assistant_suffix):]
    if user_prefix and not first_prefix.endswith(user_prefix):
        # A past reply closes with a tag a newest one lacks; the user tag is
        # then the longest end of what follows that a first turn's opening shares.
        user_prefix = _longest_shared_end(user_prefix, first_prefix, tags)
        assistant_suffix = between[:len(between) - len(user_prefix)]
    # What a first turn carries beyond the tag is written once, before it.
    if not user_prefix or not first_prefix.endswith(user_prefix):
        return None
    return user_prefix, assistant_prefix, assistant_suffix


def _longest_shared_end(text: str, other: str, tags: tuple[str, ...]) -> str:
    """The longest end of ``text`` that ``other`` ends with, starting where one of the model's ``tags`` starts."""
    for start in range(len(text)):
        if text.startswith(tags, start) and other.endswith(text[start:]):
            return text[start:]
    return ""


def _newest_exchange(render: Render, user_prefix: str,
                     user_suffix: str) -> tuple[str, str, str] | None:
    """The tags read from one user turn and its reply, ``None`` where the reply cannot be split off."""
    both = render([
        {"role": "user", "content": USER_SENTINEL},
        {"role": "assistant", "content": ASSISTANT_SENTINEL},
    ], reasoning=False)
    if both is None or ASSISTANT_SENTINEL not in both or USER_SENTINEL + user_suffix not in both:
        return None
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
                  tool_calls: ToolCallFragments | None, tags: tuple[str, ...]
                  ) -> tuple[tuple[str, str], str] | None:
    """The tags around a reply's thought, read from a newest reply that holds one, and the form it was read in.

    A template may write a thought only in a turn that calls a tool, so that
    turn is the second probe, and templates read the thought in any of
    :data:`~llama_chat.messages.THOUGHT_FORMS`. The opening tag is what the
    model writes after the generation prompt before the thought, or else what
    the turn holds before it beyond a past reply's tag; where a past reply's
    tag already holds it, as an empty thought, it is the last tag before the
    thought.
    """
    opened = USER_SENTINEL + fragments.user_suffix
    prompt = fragments.generation_prompt_without_reasoning
    probes = []
    for form in THOUGHT_FORMS:
        reply = with_thought_as(_reply_with_thought(), form)
        probes.append(([{"role": "user", "content": USER_SENTINEL}, reply], ASSISTANT_SENTINEL,
                       form))
        if tool_calls is not None:
            probes.append(([{"role": "user", "content": USER_SENTINEL},
                            {**reply, **calling_message("", [ToolCall(TOOL_SENTINEL, {})], 0)}],
                           tool_calls.call_open, form))
    for messages, after, form in probes:
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
            thought_open = before[shared_tags_length(before, fragments.assistant_prefix, tags):]
        if not thought_open.strip():
            thought_open = _last_tag(before, tags)
        thought_close = rest.split(after, 1)[0] if after in rest else ""
        if thought_open.strip() and thought_close.strip():
            return (thought_open, thought_close), form
    return None


def _prints_blocks(turn: str, content) -> bool:
    """Whether ``turn`` holds ``content``, a list of content blocks, or one of its blocks, printed rather than read."""
    return not isinstance(content, str) and any(
        printed in turn for value in (content, *content)
        for printed in (str(value), json.dumps(value)))


def _last_tag(text: str, tags: tuple[str, ...]) -> str:
    """The last of the model's ``tags`` in ``text``, and what follows it; empty where none is."""
    starts = [text.rfind(tag) for tag in tags if tag]
    start = max(starts, default=-1)
    return text[start:] if start >= 0 else ""


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
    start = -1 if rendered is None else rendered.find(marks[0])
    if start < 0:
        return ()
    end = start + len(marks[0])
    rewrites = []
    for text, mark in zip(texts, marks[1:]):
        start = rendered.find(mark, end)
        if start < 0:
            return ()
        if rendered[end:start] != text:
            rewrites.append((text, rendered[end:start]))
        end = start + len(mark)
    return tuple(rewrites)
