"""What resolving a template and running a conversation call on a cache context.

Two protocols: the model as it is read (its tokenizer, tags, chat template and
the sampling it recommends), and that plus the cache a conversation is decoded
into. Each is checked once, where a context is adopted, rather than at the
first call that finds a member missing or taking other arguments.
"""

from __future__ import annotations

import inspect
from collections.abc import Generator, Mapping
from typing import Protocol

from .config import Sampling

# The name of the further chat template a GGUF may ship for a conversation that
# declares tools, beside its default one.
TOOL_USE_TEMPLATE = "tool_use"


class ModelReader(Protocol):
    """The model's tokenizer, its own tags, its chat template and the sampling it recommends."""

    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        """With ``parse_special``, each of the model's tags in ``text`` is its one token."""

    def token_text(self, token: int) -> str:
        """The text a single token stands for, a tag spelled as a text holds it."""

    def is_special(self, token: int) -> bool:
        """Whether ``token`` is one of the model's own tags."""

    def special_tokens(self) -> tuple[int, ...]:
        """Every token that is one of the model's own tags."""

    def ends_generation(self, token: int) -> bool:
        """Whether drawing ``token`` ends a generation, neither decoded nor shown."""

    def chat_template_text(self, template_name: str | None = None) -> str | None:
        """The source of the model's chat template of that name, ``None`` where it ships none.

        Args:
            template_name: One of the further templates a model may ship, such as
                :data:`TOOL_USE_TEMPLATE`; ``None`` for its default template.
        """

    def eos_token_text(self) -> str:
        """The text of the model's end-of-sequence token, which a template may write as ``eos_token``."""

    def model_fingerprint(self) -> str:
        """What tells this model's tokens apart from another model's: the same on every load of the same model, and different where its vocabulary, its chat templates or its weights' description differ."""

    def recommended_sampling(self) -> Mapping[str, float]:
        """The sampling values the model states for itself, by the names :class:`Sampling` gives them; empty where it states none.

        Only the fields an add-on file may set too are taken
        (:data:`~llama_chat.addon_files.SAMPLING_FIELDS`); another name is
        refused where the wrapper is built.
        """


class CacheContext(ModelReader, Protocol):
    """The same, with the cache a conversation is decoded into and the steps a generation takes in it."""

    @property
    def context_size(self) -> int:
        """How many tokens the cache holds at most."""

    @property
    def can_shift(self) -> bool:
        """Whether :meth:`apply_eviction` may be called, and :meth:`remove_from` with it."""

    @property
    def can_cut_tail(self) -> bool:
        """Whether :meth:`remove_from` may be called on a cache that cannot shift."""

    def reset(self) -> None:
        """Empty the cache."""

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        """Decode ``token_ids`` from ``start_pos``, keeping the last one's logits for a draw where ``want_logits``."""

    def sampled_tokens(self, sampling: Sampling,
                       withheld_tokens: tuple[int, ...]) -> Generator[int, None, None]:
        """Tokens drawn one at a time, each from the logits of the newest decode.

        Drawn with every value of ``sampling``, each of them set, but its
        ``seed``, whose ``None`` draws a fresh one; no token of
        ``withheld_tokens`` is drawn. Closing it frees what drawing took.
        """

    def decode_generated(self, token_ids: list[int], start_pos: int) -> None:
        """Decode the tokens a generation adds at ``start_pos``, logits kept for the last."""

    def token_bytes(self, token: int) -> bytes:
        """The bytes ``token`` stands for, which may hold part of a character."""

    def apply_eviction(self, start: int, end: int) -> None:
        """Remove the tokens from ``start`` to just before ``end``, those after them moving down."""

    def remove_from(self, pos: int) -> None:
        """Remove every cached token from ``pos`` to the end."""

    def save_state(self):
        """The cache's state as it is, for :meth:`restore_state`; ``None`` where the cache can never save one, which is read once where the context is adopted.

        It may raise ``RuntimeError`` where saving fails this time.
        """

    def restore_state(self, state, pos: int) -> None:
        """Put the cache back to ``state``, which :meth:`save_state` gave while it held ``pos`` tokens."""

    def close(self) -> None:
        """Free what the context holds; nothing is called on it afterwards."""


def _members(protocol) -> tuple[str, ...]:
    return tuple(name for name in vars(protocol)
                 if not name.startswith("_"))


MODEL_READER = _members(ModelReader)
CACHE_CONTEXT = MODEL_READER + _members(CacheContext)


def ensure_answers(context, members: tuple[str, ...]) -> None:
    """Refuse a context missing any of ``members``, or whose methods among them take other arguments than the protocol gives.

    Raises:
        TypeError: Naming every member that is missing, else every method
            that takes other arguments, with the arguments it is given.
    """
    missing = [name for name in members if not hasattr(context, name)]
    if missing:
        raise TypeError(f"{type(context).__name__} does not answer {missing}, "
                        "which this library calls on a cache context")
    other = [name for name in members if not _takes_arguments_of(getattr(CacheContext, name),
                                                                  getattr(context, name))]
    if other:
        given = "; ".join(f"{name}{_arguments(getattr(CacheContext, name))}" for name in other)
        raise TypeError(f"{type(context).__name__} takes other arguments than this library "
                        f"gives a cache context: {given}")


def _takes_arguments_of(expected, member) -> bool:
    """Whether ``member`` takes every argument the protocol method ``expected`` is given, positionally and by name."""
    if isinstance(expected, property):
        return True
    if not callable(member):
        return False
    try:
        signature = inspect.signature(member)
    except (TypeError, ValueError):
        return True                     # one whose arguments cannot be read
    parameters = list(inspect.signature(expected).parameters.values())[1:]
    positional = [None for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD]
    keywords = {p.name: None for p in parameters if p.kind is p.KEYWORD_ONLY}
    try:
        signature.bind(*positional, **keywords)
        signature.bind(**{p.name: None for p in parameters})
    except TypeError:
        return False
    return True


def _arguments(expected) -> str:
    """The arguments the protocol method ``expected`` is given, as its signature spells them."""
    parameters = list(inspect.signature(expected).parameters.values())[1:]
    return str(inspect.Signature([p.replace(annotation=p.empty) for p in parameters]))


def ensure_can_go_back(context) -> None:
    """Refuse a context whose cache can neither shift, cut its end nor save a state.

    Raises:
        ValueError: If ``save_state`` gives ``None`` or raises
            ``RuntimeError`` where the cache can neither shift nor cut.
    """
    if context.can_shift or context.can_cut_tail:
        return
    try:
        state = context.save_state()
    except RuntimeError:
        state = None
    if state is None:
        raise ValueError(
            f"{type(context).__name__} can neither shift its cache, cut its end nor save a "
            "state, so it could go back to an earlier point only by decoding the whole "
            "conversation again; let it do one of the three")
