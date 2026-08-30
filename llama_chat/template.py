"""Per-message chat-template formatting, separate from tokenizing it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallBody:
    """How a model spells a function name and its arguments between the tags.

    Recovered from the model's own template, or supplied via
    ``ChatWrapper(call_body=...)``. See the README, "Tool calling", for the
    shapes these fields take on real models.

    Attributes:
        name_close: After the function name, before the arguments.
        key_open: Before each key, including the first.
        key_close: Between a key and its value.
        pair_sep: Between one value and the next key.
        body_close: After the last value, when there is at least one.
        empty_close: After the name, when there are no arguments at all.
        quote_open, quote_close: Around a string value, empty where the model
            marks strings no differently from anything else.
    """

    name_close: str
    key_open: str
    key_close: str
    pair_sep: str
    body_close: str
    empty_close: str
    quote_open: str = ""
    quote_close: str = ""


@dataclass(frozen=True)
class ToolCallFragments:
    """What a model's template puts around a tool call it generates.

    Both fields are taken verbatim from a render, so either may carry fixed
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
    """The per-role template tags a turn is wrapped in.

    A turn renders as ``prefix + content + suffix``. Recovered from the model's
    own GGUF chat template, or supplied via ``ChatWrapper(fragments=...)``.

    Attributes:
        system_prefix, system_suffix: Wrap a system turn.
        user_prefix, user_suffix: Wrap a user turn.
        assistant_prefix, assistant_suffix: Wrap an assistant turn.
        trim_content: Strip surrounding whitespace from each message's content,
            matching templates that apply Jinja ``| trim``. ``False`` emits it
            verbatim.
        tools: How this model brackets a tool call, or ``None`` where it does
            not call tools. The brackets only; what a call spells between them
            is :class:`CallBody`.
    """

    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    trim_content: bool = True
    tools: ToolCallFragments | None = None


class TemplateFormatter:
    """Renders one message as ``prefix + text + suffix`` from its role's tags.

    A conversation is the concatenation of its per-message fragments, so
    prefilling message by message yields the same tokens as prefilling the
    whole conversation at once.

    Only the methods decoded around sampling may be overridden:
    :meth:`assistant_open`, :meth:`assistant_close` and :meth:`resume_open`.
    :meth:`parts` and :meth:`fragment` must reproduce what the model saw in
    training, and the wrapper checks them against the model's own template at
    construction. See the README, "A resumed turn needs its generation prompt
    back".
    """

    def __init__(self, fragments: TemplateFragments) -> None:
        self._fragments = fragments

    def _parts(self, role: str) -> tuple[str, str]:
        """Return the ``(prefix, suffix)`` tags for ``role``.

        Raises:
            ValueError: If ``role`` is not one this template has tags for.
        """
        fragments = self._fragments
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
        if self._fragments.trim_content:
            text = text.strip()
        return prefix, text, suffix

    def fragment(self, role: str, text: str) -> str:
        """Return the templated fragment string for a complete message."""
        prefix, content, suffix = self.parts(role, text)
        return f"{prefix}{content}{suffix}"

    def suffix(self, role: str) -> str:
        """Return the turn-terminator tag for ``role`` (kept when truncating)."""
        return self._parts(role)[1]

    def assistant_open(self) -> str:
        """Generation prompt decoded immediately before sampling begins."""
        return self._fragments.assistant_prefix

    def assistant_close(self) -> str:
        """Tokens decoded after generation to terminate the assistant turn."""
        return self._fragments.assistant_suffix

    def resume_open(self) -> str:
        """Decoded before generation resumes inside a turn already open.

        Return whatever :meth:`assistant_open` adds after the turn tag, never
        the tag itself. Empty by default. See the README, "A resumed turn needs
        its generation prompt back".
        """
        return ""
