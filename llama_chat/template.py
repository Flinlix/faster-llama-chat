"""What a chat template is read for: where a call and a reply end, the tags around a turn and a thought."""

from __future__ import annotations

import re
from collections.abc import Sequence

from dataclasses import dataclass

from .messages import THOUGHT_FORMS


@dataclass(frozen=True)
class CallBoundary:
    """Where a generated call opens and closes, and where a reply ends in text.

    Attributes:
        tool_tokens: The ``(open, close)`` tokens bracketing a call, where a
            call has tags of its own.
        call_texts: The ``(open, close)`` texts around a call, where it has
            none, matched in a generation without the whitespace around them.
        opens_as_text: Whether the opening of ``call_texts`` is plain text,
            which opens a call only at the reply's start or right after a tag,
            unless ``follows_text``.
        follows_text: Whether the template writes a call right after a
            reply's text, so that a plain-text opening opens a call after
            text too.
        turn_end_text: Text that ends a reply where generated, removed from the
            reply again and never yielded.
        withheld_tokens: Tokens never drawn, such as the one that opens a
            call where calls are withheld.
        thought_tags: The tags around a thought, which stay the reply's where
            a call follows them.
        ends_at_call: Whether a call that opens with plain text ends the reply
            where it opens instead of being read, where calls are withheld.
    """

    tool_tokens: tuple[int, int] | None = None
    call_texts: tuple[str, str] | None = None
    opens_as_text: bool = False
    turn_end_text: str | None = None
    follows_text: bool = False
    withheld_tokens: tuple[int, ...] = ()
    thought_tags: tuple[str, ...] = ()
    ends_at_call: bool = False


def json_arguments(name_close: str, rest: str) -> str:
    """The arguments after a call's name, ``rest``, as one JSON object: with the brace that opens it where ``name_close`` holds it."""
    brace = name_close.rfind("{")
    return name_close[brace:] + rest if brace >= 0 else rest


@dataclass(frozen=True)
class CallBody:
    """How a model spells a function name and its arguments between the tags.

    Recovered from the model's own template, or built from its add-on file's
    call. See design_history.md, "How a call's format is recovered", for the
    shapes these fields take on real models.

    Attributes:
        name_close: After the function name, before the arguments. Empty
            where the first key's tag follows the name directly.
        key_open: Before each key, including the first.
        key_close: Between a key and its value.
        pair_sep: Between one value and the next key.
        body_close: After the last value, when there is at least one.
        empty_close: After the name, when there are no arguments at all.
        quote_open, quote_close: Around a string value, empty where the model
            marks strings no differently from anything else.
        type_close: Between the type a model writes after ``key_close`` and
            the value, empty where it writes no type.
        string_type: The type a string value is written with, read as the
            text it stands as rather than as JSON.
        key_in_close: Whether ``pair_sep`` and ``body_close`` hold the key of
            the value they close, written there as ``{key}``.
        member_open, member_close: Around each member of a nested object,
            with its key written as ``{key}``; empty where the model writes
            nested values as JSON or a literal instead.
        item_open, item_close: Around each item of a nested list, empty where
            the model writes nested values as JSON or a literal instead.
        arguments_json: Whether the arguments after the name are one JSON
            object, as the template writes them or an add-on file's call says;
            such arguments are read as JSON first, however they are spaced.
    """

    name_close: str
    key_open: str
    key_close: str
    pair_sep: str
    body_close: str
    empty_close: str
    quote_open: str = ""
    quote_close: str = ""
    type_close: str = ""
    string_type: str = ""
    key_in_close: bool = False
    member_open: str = ""
    member_close: str = ""
    item_open: str = ""
    item_close: str = ""
    arguments_json: bool = False

    def __post_init__(self) -> None:
        if not self.name_close and not self.key_open:
            raise ValueError("a call body needs a name_close or a key_open: one of "
                             "them is where the name ends")
        nested = (self.member_open, self.member_close, self.item_open, self.item_close)
        if any(nested) and not (all(nested) and self.member_open.count("{key}") == 1
                                and self.member_close.count("{key}") == 1):
            raise ValueError("nested delimiters come as all four, member_open and "
                             "member_close each holding {key} once, or not at all")


@dataclass(frozen=True)
class ToolCallFragments:
    """How a model writes a tool call: the tags around it.

    The two tags are taken verbatim from a render, so either may carry fixed
    text past the tag itself. Narrowing one to the single tag it holds needs a
    vocabulary, which this module has none of.

    Attributes:
        call_open: What stands between the assistant turn's opening tag and the
            name of the function being called.
        call_close: What stands between a call's last argument and whatever
            follows the call.
    """

    call_open: str
    call_close: str


@dataclass(frozen=True)
class TemplateFragments:
    """The template tags a turn is wrapped in.

    A turn renders as ``prefix + content + suffix``. Recovered from the model's
    own GGUF chat template, and read for what a generation holds: a thought,
    the end of a reply. What a tool call is wrapped in is
    :class:`ToolCallFragments`.

    Attributes:
        user_prefix, user_suffix: Wrap a user turn.
        assistant_prefix, assistant_suffix: Wrap an assistant turn of the
            history, as the template writes a past reply.
        generation_prompt: The whole generation prompt of a reply that does
            not reason, the assistant tag included, where it is more than the
            tag or differs from it, such as by an empty thought.
        rewrites: Text the template writes otherwise where a message holds
            it, each with what it writes instead: a tag it escapes or drops,
            a character it escapes.
    """

    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    generation_prompt: str = ""
    rewrites: tuple[tuple[str, str], ...] = ()

    @property
    def generation_prompt_without_reasoning(self) -> str:
        """The generation prompt of a reply that does not reason: ``generation_prompt``, or else the assistant tag."""
        return self.generation_prompt or self.assistant_prefix

    def content(self, text: str) -> str:
        """``text`` with ``rewrites`` applied: a content as the template writes it, apart from whitespace it strips."""
        if self.rewrites:
            written = dict(self.rewrites)
            # One pass, longest first, so nothing rewritten is rewritten again.
            pattern = "|".join(re.escape(old) for old in sorted(written, key=len, reverse=True))
            text = re.sub(pattern, lambda found: written[found.group()], text)
        return text


@dataclass(frozen=True)
class Reasoning:
    """How a model's template marks a thought, and what reasoning opens a reply with.

    A template without a switch writes the same conversation either way, so
    its forms with reasoning are its forms without.

    Attributes:
        thought_open, thought_close: Around the thought a reply opens with.
        generation_prompt: The whole generation prompt of a reply that
            reasons, the assistant tag included.
        past_keeps_thought: Whether the template writes a past reply's thought
            into the conversation, rather than leaving it out.
        past_thought_form: Which of
            :data:`~llama_chat.messages.THOUGHT_FORMS` the template reads a
            past reply's thought in, where it keeps one.
        can_turn_off: Whether the template's switch writes a reply without
            reasoning otherwise than one with it.
    """

    thought_open: str
    thought_close: str
    generation_prompt: str
    past_keeps_thought: bool = False
    past_thought_form: str = THOUGHT_FORMS[0]
    can_turn_off: bool = False


def common_prefix_length(one: Sequence, other: Sequence) -> int:
    """How long a beginning two sequences of the same type share."""
    low, high = 0, min(len(one), len(other))
    while low < high:
        middle = (low + high + 1) // 2
        if one[:middle] == other[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def common_suffix_length(one: Sequence, other: Sequence) -> int:
    """How long an ending two sequences of the same type share."""
    return common_prefix_length(one[::-1], other[::-1])


def _ends_a_tag(text: str, at: int, tags: tuple[str, ...]) -> bool:
    """Whether one of ``tags`` ends at ``at`` in ``text`` and none of them runs across it."""
    if not text[:at].endswith(tags):
        return False
    for tag in tags:
        found = text.find(tag)
        while 0 <= found < at:
            if found + len(tag) > at:
                return False
            found = text.find(tag, found + 1)
    return True


def shared_tags_length(text: str, other: str, tags: tuple[str, ...]) -> int:
    """How long a beginning the two share, ending where one of the model's ``tags`` or a word ends.

    A word ends at whitespace, so the shared part never stops inside a tag or
    a word the two spell differently.
    """
    shared = common_prefix_length(text, other)
    while shared and not (text[shared - 1].isspace() or _ends_a_tag(text, shared, tags)):
        shared -= 1
    return shared
