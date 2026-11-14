"""What an action refuses, by what the caller does about it, and the engine's failures.

Every refusal is a ``ValueError`` and is raised before the action changes
anything; ``field`` names the argument or setting it is about, where it is
about one. A wrong type is a plain ``TypeError`` that names the argument.
"""

from __future__ import annotations


class RefusalError(ValueError):
    """An action refused before it changed anything.

    Attributes:
        field: The argument or setting the refusal is about, as the call or
            :class:`~llama_chat.config.Config` spells it; ``None`` where it
            is about none.
    """

    def __init__(self, field: str | None, message: str) -> None:
        super().__init__(message)
        self.field = field


class InvalidSettingError(RefusalError):
    """A setting of :class:`~llama_chat.config.Config` or :class:`~llama_chat.config.Sampling`, or a request's own, holds a value that is refused."""


class InvalidArgumentError(RefusalError):
    """An argument of a call holds a value that is refused."""


class UnsupportedError(RefusalError):
    """What this model cannot do: call tools, reason, or keep a system prompt in place."""


class TemplateRefusedError(RefusalError):
    """The model's template does not write what was asked for; the message carries the template's own reason where it gave one."""


class OutOfOrderError(RefusalError):
    """An action that needs another one first; the message names it."""


class ContextOverflowError(RefusalError):
    """What was asked for does not fit the context: a message, a history, a declaration, or room for the reply.

    Attributes:
        needed_tokens: How many tokens it needs, where that was counted.
        available_tokens: How many it may take, where that was counted.
    """

    def __init__(self, message: str, needed_tokens: int | None = None,
                 available_tokens: int | None = None) -> None:
        super().__init__(None, message)
        self.needed_tokens = needed_tokens
        self.available_tokens = available_tokens


class EngineError(RuntimeError):
    """The engine failed while it decoded, edited or generated; the next action brings the cache back first."""


def with_template_reason(refusal: str, reason: str | None) -> str:
    """``refusal``, followed by what the template itself said where it said something."""
    return refusal if not reason else f"{refusal} (the template says: {reason})"
