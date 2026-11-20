"""The ``check_cache`` setting's check: the cache holds what a fresh begin of its conversation holds.

A fresh begin renders the whole conversation and reaches every place in the
cache from nothing, so it is the measure for every faster path an action took.
The begin runs on a context of its own that reads the model's tokens and
decodes nothing.
"""

from __future__ import annotations

from dataclasses import replace

from .conversation_state import reasoning_switch
from .errors import InternalError

# How many times the conversation's context the fresh begin is given, so that it keeps the
# whole conversation without evicting or cutting anything: a begin keeps what fits in half
# of its context, and a conversation fills at most its context.
_ROOM_FACTOR = 4
# How many tokens, from the first difference on, the error shows of the cache and of the render.
_SHOWN_TOKENS = 12


class _ReadsTokensOnly:
    """A cache context that reads the model's tokens as ``context`` does, and decodes nothing."""

    def __init__(self, context, context_size: int) -> None:
        self._context, self.context_size = context, context_size
        self.can_shift, self.can_cut_tail = context.can_shift, context.can_cut_tail
        self.decode_blocks = False      # it decodes nothing

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = False):
        return self._context.tokenize(text, add_special, parse_special)

    def token_text(self, token: int) -> str:
        return self._context.token_text(token)

    def token_bytes(self, token: int) -> bytes:
        return self._context.token_bytes(token)

    def is_special(self, token: int) -> bool:
        return self._context.is_special(token)

    def special_tokens(self) -> tuple[int, ...]:
        return self._context.special_tokens()

    def ends_generation(self, token: int) -> bool:
        return self._context.ends_generation(token)

    def chat_template_text(self, template_name: str | None = None) -> str | None:
        return self._context.chat_template_text(template_name)

    def eos_token_text(self) -> str:
        return self._context.eos_token_text()

    @property
    def model_fingerprint(self) -> str:
        return self._context.model_fingerprint

    def recommended_sampling(self):
        return self._context.recommended_sampling()

    def reset(self) -> None:
        pass

    def prefill(self, token_ids, start_pos, want_logits) -> None:
        pass

    def decode_generated(self, token_ids, start_pos) -> None:
        pass

    def apply_eviction(self, start: int, end: int) -> None:
        pass

    def remove_from(self, pos: int) -> None:
        pass

    def save_state(self):
        return ()

    def restore_state(self, state, pos: int) -> None:
        pass

    def save_whole_state(self):
        return b""

    def restore_whole_state(self, state) -> None:
        pass

    def sampled_tokens(self, sampling, withheld_tokens):
        raise InternalError("the cache check generates nothing")

    def close(self) -> None:
        pass


def check_cache(held: list[int], conversation: dict, unreadable_result: dict | None, context, templates,
                config, build, *, prompted: bool) -> None:
    """Raise :class:`InternalError` where ``held``, the cache's tokens, disagree with a fresh begin of ``conversation``.

    Once the work after a reply has ended, the cache may hold less than the
    fresh begin or more, as far as each decodes ahead of the next question,
    but no token of its own where the two overlap. Where the model is about
    to read it, ``prompted``, it holds exactly the fresh render with the
    generation prompt.

    Args:
        build: Makes a conversation's cache from a context, its templates and
            a config, as the wrapper makes its own.
    """
    room = config.context_size * _ROOM_FACTOR
    fresh_config = replace(config, context_size=room, evict_above_fraction=1.0,
                           max_history_exchanges=None, check_cache=False)
    fresh = build(_ReadsTokensOnly(context, room), templates, fresh_config)
    fresh.start(conversation["messages"], conversation["tools"],
                reasoning_switch(conversation["reasoning"], conversation["reasoning_level"]),
                conversation["generated_turns"], unreadable_result)
    wanted = fresh.prompted_tokens() if prompted else fresh.held_tokens()
    shared = min(len(held), len(wanted))
    if held == wanted or not prompted and held[:shared] == wanted[:shared]:
        return
    at = next((i for i, (one, other) in enumerate(zip(held, wanted)) if one != other), shared)
    shown = lambda tokens: "".join(context.token_text(t) for t in tokens[at:at + _SHOWN_TOKENS])  # noqa: E731
    where = "where the model reads it" if prompted else "after the work after a reply"
    raise InternalError(
        f"the cache differs from a fresh render of its conversation {where}, from token {at} on: "
        f"it holds {shown(held)!r} ({len(held)} tokens) where the render has {shown(wanted)!r} "
        f"({len(wanted)} tokens)")
