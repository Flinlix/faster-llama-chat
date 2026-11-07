"""Per-message chat-template formatting, separate from tokenizing it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallBody:
    """How a model spells a function name and its arguments between the tags.

    Recovered from the model's own template, or supplied as the ``body`` of
    ``ChatWrapper(tool_calls=...)``. See the README, "Tool calling", for the
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

    def __post_init__(self) -> None:
        if not self.name_close and not self.key_open:
            raise ValueError("a call body needs a name_close or a key_open: one of "
                             "them is where the name ends")


@dataclass(frozen=True)
class ToolCallFragments:
    """How a model writes a tool call: the tags around it, and what is inside.

    The two tags are taken verbatim from a render, so either may carry fixed
    text past the tag itself. Narrowing one to the single tag it holds needs a
    vocabulary, which this module has none of.

    Attributes:
        call_open: What stands between the assistant turn's opening tag and the
            name of the function being called.
        call_close: What stands between a call's last argument and whatever
            follows the call.
        body: How the name and the arguments are spelled between the tags, or
            ``None`` to have it recovered from the model.
    """

    call_open: str
    call_close: str
    body: CallBody | None = None


@dataclass(frozen=True)
class TemplateFragments:
    """The per-role template tags a turn is wrapped in.

    A turn renders as ``prefix + content + suffix``. Recovered from the model's
    own GGUF chat template, or supplied via ``ChatWrapper(fragments=...)``. What
    a tool call is wrapped in is :class:`ToolCallFragments`, its own argument.

    Attributes:
        system_prefix, system_suffix: Wrap a system turn.
        user_prefix, user_suffix: Wrap a user turn.
        assistant_prefix, assistant_suffix: Wrap an assistant turn of the
            history, as the template writes a past reply.
        trim_content: Strip surrounding whitespace from each message's content,
            matching templates that apply Jinja ``| trim``. ``False`` emits it
            verbatim.
        reply_strip, system_strip: Which whitespace the template strips from
            a reply's or a system prompt's content where that differs from a
            user turn's: ``"both"``, ``"end"``, ``"start"`` or ``"none"``.
            ``None`` follows ``trim_content``.
        generation_prompt: The whole generation prompt of a reply that does
            not reason, the assistant tag included, where it is more than the
            tag or differs from it, such as by an empty thought.
    """

    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    trim_content: bool = True
    reply_strip: str | None = None
    system_strip: str | None = None
    generation_prompt: str = ""

    def __post_init__(self) -> None:
        for name in ("reply_strip", "system_strip"):
            if getattr(self, name) not in (None, *_STRIPS):
                raise ValueError(f"{name} must be one of {sorted(_STRIPS)} or None, "
                                 f"not {getattr(self, name)!r}")

    def content(self, role: str, text: str) -> str:
        """``text`` as the template writes it as ``role``'s content."""
        strip = {"assistant": self.reply_strip, "system": self.system_strip}.get(role)
        if strip is not None:
            return _STRIPS[strip](text)
        return text.strip() if self.trim_content else text


# What each `reply_strip` and `system_strip` does to a content's text.
_STRIPS = {"both": str.strip, "end": str.rstrip, "start": str.lstrip,
           "none": lambda text: text}


@dataclass(frozen=True)
class Reasoning:
    """How a model's template marks a thought, and what turns reasoning on.

    A template without a switch writes the same conversation either way, so
    its forms with reasoning are its forms without.

    Attributes:
        thought_open, thought_close: Around the thought a reply opens with.
        system_prefix: Opens the system turn of a conversation that reasons,
            in place of the fragments' own. ``None`` where the template's
            switch changes more of the conversation than its system turn, so
            reasoning cannot be turned on as the template does.
        system_suffix: Closes that system turn, in place of the fragments'
            own, where the switch changes it too. ``None`` where it does not.
        generation_prompt: The whole generation prompt of a reply that
            reasons, the assistant tag included.
        past_keeps_thought: Whether the template writes a past reply's thought
            into the conversation, rather than leaving it out.
        assistant_prefix: Opens a past reply of a conversation that reasons,
            in place of the fragments' own, where the template writes it
            otherwise, such as with an empty thought. ``None`` where it does not.
    """

    thought_open: str
    thought_close: str
    system_prefix: str | None
    generation_prompt: str
    system_suffix: str | None = None
    past_keeps_thought: bool = False
    assistant_prefix: str | None = None


class TemplateFormatter:
    """Renders one message as ``prefix + text + suffix`` from its role's tags.

    A conversation is the concatenation of its per-message fragments, so
    prefilling message by message yields the same tokens as prefilling the
    whole conversation at once. What is decoded around sampling follows the
    conversation's reasoning: ``reasoning`` is set for one that reasons.
    """

    def __init__(self, fragments: TemplateFragments,
                 reasoning: Reasoning | None = None) -> None:
        self._fragments = fragments
        self._reasoning = reasoning

    def _parts(self, role: str) -> tuple[str, str]:
        """Return the ``(prefix, suffix)`` tags for ``role``.

        Raises:
            ValueError: If ``role`` is not one this template has tags for.
        """
        fragments, reasoning = self._fragments, self._reasoning
        if role == "system" and reasoning:
            return reasoning.system_prefix, (reasoning.system_suffix if reasoning.system_suffix
                                             is not None else fragments.system_suffix)
        if role == "system":
            return fragments.system_prefix, fragments.system_suffix
        if role == "user":
            return fragments.user_prefix, fragments.user_suffix
        if role == "assistant":
            return fragments.assistant_prefix, fragments.assistant_suffix
        raise ValueError(
            f"unknown role: {role!r}; expected 'system', 'user' or 'assistant'")

    def parts(self, role: str, text: str) -> tuple[str, str, str]:
        """Return ``(prefix, content, suffix)`` for a complete message.

        Callers tokenize ``content`` with special-token parsing off and the
        tags with it on. See the README, "Special tokens in message content".
        """
        prefix, suffix = self._parts(role)
        if role == "assistant" and self._reasoning and self._reasoning.assistant_prefix is not None:
            prefix = self._reasoning.assistant_prefix
        return prefix, self._fragments.content(role, text), suffix

    def fragment(self, role: str, text: str) -> str:
        """Return the templated fragment string for a complete message."""
        prefix, content, suffix = self.parts(role, text)
        return f"{prefix}{content}{suffix}"

    def suffix(self, role: str) -> str:
        """Return the turn-terminator tag for ``role`` (kept when truncating)."""
        return self._parts(role)[1]

    def generation_prompt(self) -> str:
        """Generation prompt decoded immediately before sampling begins."""
        if self._reasoning:
            return self._reasoning.generation_prompt
        return self._fragments.generation_prompt or self._fragments.assistant_prefix

    def assistant_close(self) -> str:
        """Tokens decoded after generation to terminate the assistant turn."""
        return self._fragments.assistant_suffix

    def answer_opening(self) -> str:
        """What a past reply's tag holds that the generation prompt leaves to the model.

        A template that routes replies through channels ends its generation
        prompt before the channel's own header, which the model writes and
        the answer does not include.
        """
        prompt, tag = self.generation_prompt(), self._fragments.assistant_prefix
        return tag[len(prompt):] if tag.startswith(prompt) else ""

    def prompt_beyond_tag(self) -> str:
        """What the generation prompt holds beyond what it shares with a past reply's tag."""
        prompt = self.generation_prompt()
        return prompt[shared_tags_length(prompt, self._fragments.assistant_prefix):]


def shared_tags_length(text: str, other: str) -> int:
    """How long a beginning the two share, ending where a tag or a word ends.

    A tag ends at ``>`` or ``]`` and a word at whitespace, so the shared part
    never stops inside a tag the two spell differently.
    """
    shared = 0
    while shared < min(len(text), len(other)) and text[shared] == other[shared]:
        shared += 1
    while shared and not (text[shared - 1].isspace() or text[shared - 1] in ">]"):
        shared -= 1
    return shared
