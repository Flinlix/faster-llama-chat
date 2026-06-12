"""Per-message chat-template formatting.

Each message is wrapped in the model's role tags before it reaches the cache.
Building the templated *string* for one message needs no model - only
tokenizing does. This module keeps the two concerns separate.
"""

from __future__ import annotations

from .config import Config


class TemplateFormatter:
    """Combines ``(role, text)`` with the config's template fragments into one
    message string.

    Each turn renders as ``prefix + text + suffix`` using the role's fragments on
    :class:`~llama_chat.config.Config`. Because a conversation is just the
    concatenation of its per-message fragments, prefilling message-by-message
    yields the same tokens as prefilling the whole conversation at once (this is
    asserted by the template-equivalence check in ``begin``).
    """

    def __init__(self, config: Config) -> None:
        self._c = config

    def fragment(self, role: str, text: str) -> str:
        """Return the templated fragment for a complete message."""
        c = self._c
        if c.trim_content:
            text = text.strip()  # match templates that apply Jinja `| trim`
        if role == "system":
            return f"{c.system_prefix}{text}{c.system_suffix}"
        if role == "user":
            return f"{c.user_prefix}{text}{c.user_suffix}"
        if role == "assistant":
            return f"{c.assistant_prefix}{text}{c.assistant_suffix}"
        raise ValueError(f"unknown role: {role!r}")

    def assistant_open(self) -> str:
        """Generation prompt decoded immediately before sampling begins."""
        return self._c.assistant_prefix

    def assistant_close(self) -> str:
        """Tokens decoded after generation to terminate the assistant turn."""
        return self._c.assistant_suffix

    def full_conversation(
        self, system: str, messages: list[tuple[str, str]]
    ) -> str:
        """Render an entire conversation in one string (for the equivalence check)."""
        parts = [self.fragment("system", system)]
        parts.extend(self.fragment(role, text) for role, text in messages)
        return "".join(parts)
