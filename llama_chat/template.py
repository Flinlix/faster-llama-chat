"""Per-message chat-template formatting.

Each message is wrapped in the model's role tags before it reaches the cache.
Building the templated *string* for one message needs no model - only
tokenizing does. This module keeps the two concerns separate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fragments:
    """The per-role template tags a turn is wrapped in.

    A turn renders as ``prefix + content + suffix``. These are normally recovered
    from the model's own GGUF chat template
    (:func:`~llama_chat.template_extract.extract_fragments`); they can also be
    supplied explicitly via ``ChatWrapper(fragments=...)`` for a model that ships
    no embedded template.

    Attributes:
        system_prefix, system_suffix: Wrap a system turn.
        user_prefix, user_suffix: Wrap a user turn.
        assistant_prefix, assistant_suffix: Wrap an assistant turn.
        trim_content: Strip leading/trailing whitespace from each message's
            content before wrapping it, matching templates that apply Jinja
            ``| trim`` (e.g. Gemma, Llama-3). ``False`` emits content verbatim.
    """

    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str
    trim_content: bool = True


class TemplateFormatter:
    """Combines ``(role, text)`` with the template fragments into one message
    string.

    Each turn renders as ``prefix + text + suffix`` using the role's
    :class:`Fragments`. Because a conversation is just the concatenation of its
    per-message fragments, prefilling message-by-message yields the same tokens as
    prefilling the whole conversation at once (this is asserted by the
    template-equivalence check in ``begin``).
    """

    def __init__(self, fragments: Fragments) -> None:
        self._f = fragments

    def fragment(self, role: str, text: str) -> str:
        """Return the templated fragment for a complete message."""
        f = self._f
        if f.trim_content:
            text = text.strip()  # match templates that apply Jinja `| trim`
        if role == "system":
            return f"{f.system_prefix}{text}{f.system_suffix}"
        if role == "user":
            return f"{f.user_prefix}{text}{f.user_suffix}"
        if role == "assistant":
            return f"{f.assistant_prefix}{text}{f.assistant_suffix}"
        raise ValueError(f"unknown role: {role!r}")

    def assistant_open(self) -> str:
        """Generation prompt decoded immediately before sampling begins."""
        return self._f.assistant_prefix

    def assistant_close(self) -> str:
        """Tokens decoded after generation to terminate the assistant turn."""
        return self._f.assistant_suffix

    def full_conversation(
        self, system: str, messages: list[tuple[str, str]]
    ) -> str:
        """Render an entire conversation in one string (for the equivalence check)."""
        parts = [self.fragment("system", system)]
        parts.extend(self.fragment(role, text) for role, text in messages)
        return "".join(parts)
