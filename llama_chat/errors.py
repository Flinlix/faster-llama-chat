"""What an action refuses, by what the caller does about it, and the engine's failures.

Every refusal is a ``ValueError`` and is raised before the action changes
anything; ``field`` names the argument or setting it is about, where it is
about one. A wrong type is an :class:`ArgumentTypeError`, a ``TypeError``
that names the argument. Every public action of a loaded wrapper raises only
these; anything else raised inside one is raised as :class:`InternalError`.
"""

from __future__ import annotations

import functools
import inspect


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


class ArgumentTypeError(TypeError):
    """An argument of a call is of a type that is refused; the message names it and the type it must be. A setting of the wrong type is an :class:`InvalidSettingError`."""


class EngineError(RuntimeError):
    """The engine failed while it decoded, edited or generated; the next action brings the cache back first."""


class InternalError(RuntimeError):
    """A fault of this library, found before it could spoil the conversation, such as a loop that made no progress; the conversation stays usable, as after any failed action."""


def with_template_reason(refusal: str, reason: str | None) -> str:
    """``refusal``, followed by what the template itself said where it said something."""
    return refusal if not reason else f"{refusal} (the template says: {reason})"


# What a public action may raise as it is; anything else becomes an InternalError.
LIBRARY_ERRORS = (RefusalError, ArgumentTypeError, EngineError, InternalError)


def raising_library_errors_only(cls: type) -> type:
    """``cls`` with every public method and property raising only :data:`LIBRARY_ERRORS`, another exception raised as :class:`InternalError` naming the action and the cause.

    What reaches the caller of a generator a method returns is not covered;
    a class that streams wraps its own steps.
    """
    for name, member in list(vars(cls).items()):
        # A name holding a double underscore is machinery around a method, not the interface.
        if name.startswith("_") or "__" in name:
            continue
        if isinstance(member, property) and member.fget is not None:
            setattr(cls, name, property(_raising_library_errors(member.fget, cls.__name__),
                                        member.fset, member.fdel, member.__doc__))
        elif callable(member):
            setattr(cls, name, _raising_library_errors(member, cls.__name__))
    return cls


def _raising_library_errors(action, owner: str):
    """``action``, raising another exception than :data:`LIBRARY_ERRORS` as :class:`InternalError`, and arguments its signature does not take as :class:`ArgumentTypeError`."""
    signature = inspect.signature(action)

    @functools.wraps(action)
    def guarded(*args, **kwargs):
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            raise ArgumentTypeError(f"{owner}.{action.__name__}: {error}") from None
        try:
            return action(*args, **kwargs)
        except LIBRARY_ERRORS:
            raise
        except Exception as error:
            raise internal_error(f"{owner}.{action.__name__}", error) from error
    setattr(guarded, "raises_library_errors_only", True)  # noqa: B010 (read by the tests)
    return guarded


def internal_error(action: str, error: Exception) -> InternalError:
    """The :class:`InternalError` that reports ``error``, raised inside ``action``."""
    return InternalError(f"{action} failed inside the library ({type(error).__name__}: {error}); "
                         f"the conversation stays usable, as after any failed action")


# The one wording of what makes more room, at the end of every ContextOverflowError.
RAISE_THE_ROOM = "raise context_size or evict_above_fraction, or lower min_reply_tokens"
