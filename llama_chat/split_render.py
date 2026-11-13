"""A render of a conversation split into the template's own text and the caller's, each message's text placed exactly.

The conversation is rendered twice: once as it is, and once with a marker in
place of every text of the caller's own (a content, a thought, a call's name
and its string arguments, and each description of a declaration). The markers
say where each text goes; the render as
it is must then be the marked one with each marker replaced by its text as the
template writes it, which may wrap text of its own around it where it is placed
in only one way. Otherwise the template wrote more of a text than the text
itself, and the conversation is refused. See design_history.md, "Every text
placed by a marker render".
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

# A marker: letters and digits only, so that trimming, escaping and quoting leave it whole.
_MARKER = re.compile(r"(QQ\d+MQQ)")


@dataclass(frozen=True)
class SplitRender:
    """One render, and where the caller's own text sits in it.

    Attributes:
        text: The render.
        own: Where each piece of the caller's own text sits, in order.
        contents: For each message, where its content is written last, or
            ``None`` where the template writes none of it.
        call_ends: For each message, where its last call's text ends as far
            as it is the caller's own: after its last string argument, else
            after its name where the template first writes it; ``None`` where
            it makes no call the template writes.
    """

    text: str
    own: tuple[tuple[int, int], ...]
    contents: tuple[tuple[int, int] | None, ...]
    call_ends: tuple[int | None, ...]

    def pieces(self, start: int = 0, end: int | None = None) -> list[tuple[str, bool]]:
        """The render from ``start`` to ``end`` in pieces, each marked whether it is the caller's own."""
        end = len(self.text) if end is None else end
        pieces, at = [], start
        for own_start, own_end in self.own:
            if own_end <= start or own_start >= end:
                continue
            own_start, own_end = max(own_start, start), min(own_end, end)
            pieces += [(self.text[at:own_start], False), (self.text[own_start:own_end], True)]
            at = own_end
        pieces.append((self.text[at:end], False))
        return [(text, own) for text, own in pieces if text]


def split_render(render: Callable[[list[dict], list[dict] | None], str | None],
                 messages: list[dict], written: Callable[[str], str],
                 declarations: list[dict] | None = None,
                 reads_clock: bool = True) -> SplitRender | None:
    """``messages`` rendered with ``declarations`` and split; ``None`` where the template refuses them.

    Args:
        render: The template's render of a list of messages and of declarations.
        written: A content's text with the template's rewrites applied, tried
            on the text itself and on each of its stripped forms.
        reads_clock: Whether the template reads the clock, so that the two
            renders may differ by a moment and are made again once.

    Raises:
        ValueError: If the render is not the marked render with each marker
            replaced by its text.
    """
    for attempt in range(2 if reads_clock else 1):
        rendered = render(messages, declarations)
        if rendered is None:
            return None
        marked_messages, marked_declarations, values = _marked(messages, declarations)
        marked = render(marked_messages, marked_declarations)
        split = (_placed(rendered, marked, values, len(messages), written)
                 if marked is not None else None)
        if split is not None:
            return split
        # Once more, for a render that read the clock as it moved on between the two.
    raise ValueError("this model's template writes a message's text otherwise than as "
                     "text in its place, so it could not be kept apart from the "
                     "template's own tags; the conversation cannot be written")


@dataclass(frozen=True)
class _Value:
    """One text of the caller's own, and where it belongs."""

    text: str
    message: int                    # -1 for a declaration's
    kind: str                       # "content", "thought", "name", "argument" or "description"
    call: int = -1                  # which of the message's calls a name or an argument is of


def _marked(messages: list[dict], declarations: list[dict] | None
            ) -> tuple[list[dict], list[dict] | None, dict[str, _Value]]:
    """The messages and the declarations with a marker in place of each text of the caller's own, and what each marker stands for."""
    values: dict[str, _Value] = {}

    def mark(text: str, index: int, kind: str, call: int = -1) -> str:
        marker = f"QQ{len(values)}MQQ"
        values[marker] = _Value(text, index, kind, call)
        return marker

    def mark_arguments(value, index: int, call: int):
        if isinstance(value, str):
            return mark(value, index, "argument", call)
        if isinstance(value, dict):
            return {key: mark_arguments(item, index, call) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [mark_arguments(item, index, call) for item in value]
        return value

    def mark_descriptions(value):
        if isinstance(value, dict):
            return {key: mark(item, -1, "description")
                    if key == "description" and isinstance(item, str) and item
                    else mark_descriptions(item) for key, item in value.items()}
        if isinstance(value, list):
            return [mark_descriptions(item) for item in value]
        return value

    marked = []
    for index, message in enumerate(messages):
        role, copy = message["role"], dict(message)
        content = message.get("content")
        # An empty reply is left as it is: a template may write a reply only where it has text.
        if isinstance(content, str) and (content or role in ("user", "tool")):
            copy["content"] = mark(content, index, "content")
        if message.get("reasoning_content"):
            copy["reasoning_content"] = mark(message["reasoning_content"], index, "thought")
        if message.get("tool_calls"):
            copy["tool_calls"] = [
                {**call, "function": {
                    "name": mark(call["function"]["name"], index, "name", n),
                    "arguments": mark_arguments(call["function"]["arguments"], index, n)}}
                for n, call in enumerate(message["tool_calls"])]
        marked.append(copy)
    return marked, mark_descriptions(declarations), values


def _placed(rendered: str, marked: str, values: dict[str, _Value], n_messages: int,
            written) -> SplitRender | None:
    """Where each marker's text sits in ``rendered``; ``None`` where it is not the marked render with the texts in place."""
    placed = _walk(rendered, _MARKER.split(marked), 0, 0, values, written, ())
    if placed is None:
        return None
    own, contents = [], [None] * n_messages
    last_call = {}
    for value in values.values():
        last_call[value.message] = max(last_call.get(value.message, -1), value.call)
    # A template may write a call's name again elsewhere, in its result say; its
    # arguments, and the name where first written, are the call's own.
    argument_ends, name_ends = {}, {}
    for value, span, own_span in placed:
        if own_span[1] > own_span[0]:
            own.append(own_span)
        if value.kind == "content":
            contents[value.message] = span
        elif value.call == last_call[value.message] >= 0:
            if value.kind == "argument":
                argument_ends[value.message] = max(argument_ends.get(value.message, 0), span[1])
            else:
                name_ends.setdefault(value.message, span[1])
    call_ends = tuple(argument_ends.get(m, name_ends.get(m)) for m in range(n_messages))
    return SplitRender(rendered, tuple(own), tuple(contents), call_ends)


def _walk(rendered: str, parts: list[str], k: int, at: int, values, written, placed: tuple):
    """The marked render's parts from ``k`` on matched in ``rendered`` from ``at``, each marker's text placed; ``None`` where they do not match.

    A text the template writes otherwise than in any of its forms, wrapped in
    text of the template's own say, is what lies before the template's next
    text; it is placed only where exactly one place for it leaves the rest of
    the render matching.
    """
    while k < len(parts):
        part = parts[k]
        if k % 2 == 0:
            if not rendered.startswith(part, at):
                return None
            at, k = at + len(part), k + 1
            continue
        value = values.get(part)
        if value is None:
            return None
        following = parts[k + 1] if k + 1 < len(parts) else ""
        text = next((form for form in _forms(value, written)
                     if rendered.startswith(form, at)
                     and rendered.startswith(following, at + len(form))), None)
        if text is not None:
            span = (at, at + len(text))
            placed += ((value, span, span),)
            at, k = at + len(text), k + 1
            continue
        if not following:
            return None
        found, start = [], at
        while (end := rendered.find(following, start)) >= 0 and len(found) < 2:
            gap = rendered[at:end]
            inner = gap.find(value.text) if value.text and gap.count(value.text) == 1 else -1
            own = (at + inner, at + inner + len(value.text)) if inner >= 0 else (at, end)
            rest = _walk(rendered, parts, k + 1, end, values, written,
                         placed + ((value, (at, end), own),))
            if rest is not None:
                found.append(rest)
            start = end + 1
        return found[0] if len(found) == 1 else None
    return placed if at == len(rendered) else None


def _forms(value: _Value, written) -> list[str]:
    """How the template may write ``value``: as it is, trimmed, with a content's rewrites, or inside a JSON string."""
    text = value.text
    forms = [text]
    if value.kind in ("content", "thought"):
        forms += [text.strip(), text.strip("\n"), text.rstrip(), text.lstrip()]
    if value.kind == "content":
        forms += [written(form) for form in forms]
    if value.kind != "name":
        quoted = json.dumps(text)[1:-1]
        forms += [quoted, json.dumps(text, ensure_ascii=False)[1:-1],
                  quoted.replace("<", "\\u003c").replace(">", "\\u003e")
                  .replace("&", "\\u0026").replace("'", "\\u0027")]
    return list(dict.fromkeys(forms))
