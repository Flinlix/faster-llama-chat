"""Per-message chat-template formatting.

Each message is wrapped in the model's role tags before it reaches the cache.
Building the templated *string* for one message needs no model — only
tokenizing it does — so this module keeps the two concerns separate and stays
unit-testable on its own.
"""

from __future__ import annotations

from .config import TemplateConfig


class TemplateFormatter:
    """Turns ``(role, text)`` into the templated string for a single message.

    The wrapper tokenizes these fragments. Because a conversation is just the
    concatenation of its per-message fragments, prefilling message-by-message
    yields the same tokens as prefilling the whole conversation at once (this is
    asserted by the template-equivalence check in ``begin``).
    """

    def __init__(self, template: TemplateConfig) -> None:
        self._t = template

    def fragment(self, role: str, text: str) -> str:
        """Return the templated fragment for a complete message."""
        if role == "system":
            return self._t.system.format(text=text)
        if role == "user":
            return self._t.user.format(text=text)
        if role == "assistant":
            return self._t.assistant.format(text=text)
        raise ValueError(f"unknown role: {role!r}")

    def assistant_open(self) -> str:
        """Generation prompt decoded immediately before sampling begins."""
        return self._t.assistant_open

    def assistant_close(self) -> str:
        """Tokens decoded after generation to terminate the assistant turn."""
        return self._t.assistant_close

    def full_conversation(
        self, system: str, messages: list[tuple[str, str]]
    ) -> str:
        """Render an entire conversation in one string (for the equivalence check)."""
        parts = [self.fragment("system", system)]
        parts.extend(self.fragment(role, text) for role, text in messages)
        return "".join(parts)
