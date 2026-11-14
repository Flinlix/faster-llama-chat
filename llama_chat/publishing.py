"""What a conversation publishes after each change, for readers that hold no lock: a copy, worked out from its state and holding nothing of its own."""

from __future__ import annotations

from collections.abc import Callable

from .conversation_state import ConversationState, Published
from .errors import InvalidArgumentError
from .messages import as_history, portable_digest, system_part
from .tools import ToolCall


def published(state: ConversationState, pending_calls: list[ToolCall], turn_open: bool,
              total_tokens: int, model_fingerprint: Callable[[], str]) -> Published:
    """The copy of ``state`` readers without the lock see, the calls it owes, the cache's length and, where it keeps a turn as generated, the model's fingerprint beside it."""
    kept = generated_turns(state)
    system = system_part(state.messages)
    return Published(
        system_prompt=system[0]["content"] if system else "",
        history=tuple(as_history(state.messages)),
        tools=tuple(state.tools) if state.uses_tool_template else None,
        # A call written unreadably is answered by the library, so the caller owes it nothing.
        reasoning=state.reasoning,
        pending_calls=tuple(c for c in pending_calls if c.status != "unreadable"),
        begun=state.begun, turn_open=turn_open,
        newest_role=state.messages[-1]["role"] if state.messages else None,
        total_tokens=total_tokens, kept=kept, model=model_fingerprint() if kept else None)


def generated_turns(state: ConversationState) -> tuple:
    """Each turn ``state`` keeps as generated: its tokens, how many messages it covers, and their digest."""
    messages = state.messages
    index = {id(m): i for i, m in enumerate(messages)}
    turns = []
    for turn in state.kept:
        if id(turn.first) not in index or id(turn.last) not in index:
            continue
        first, last = index[id(turn.first)], index[id(turn.last)]
        turns.append((turn.tokens, last - first + 1, portable_digest(messages[first:last + 1])))
    return tuple(turns)


def ensure_generated_turns(given) -> None:
    """Refuse generated turns that are not what :func:`generated_turns` and the model's fingerprint make, as a conversation gives them.

    Raises:
        TypeError: If ``given`` is neither ``None`` nor a ``dict``.
        ValueError: If it lacks the model or a turn's tokens, message count
            or digest, or holds one of another type.
    """
    if given is None:
        return
    if not isinstance(given, dict):
        raise TypeError(f"generated_turns must be what conversation() gives, a dict or None, "
                        f"not {type(given).__name__}")
    turns = given.get("turns")
    if not isinstance(given.get("model"), str) or not isinstance(turns, list) or not all(
            isinstance(turn, dict) and isinstance(turn.get("tokens"), list)
            and all(type(token) is int for token in turn["tokens"])
            and type(turn.get("message_count")) is int and turn["message_count"] > 0
            and isinstance(turn.get("digest"), str) for turn in turns):
        raise InvalidArgumentError("generated_turns", "generated_turns is not what "
                                   "conversation() gives; pass that, or None to write every "
                                   "turn as the template writes it")
