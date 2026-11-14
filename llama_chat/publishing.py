"""What a conversation publishes after each change, for readers that hold no lock: a copy, worked out from its state and holding nothing of its own."""

from __future__ import annotations

from .conversation_state import ConversationState, Published
from .messages import as_history, portable_digest, system_part
from .tools import ToolCall


def published(state: ConversationState, pending_calls: list[ToolCall], turn_open: bool,
              total_tokens: int) -> Published:
    """The copy of ``state`` readers without the lock see, the calls it owes and the cache's length beside it."""
    system = system_part(state.messages)
    return Published(
        system_prompt=system[0]["content"] if system else "",
        history=tuple(as_history(state.messages)),
        tools=tuple(state.tools) if state.uses_tool_template else None,
        reasoning=state.reasoning, pending_calls=tuple(pending_calls),
        begun=state.begun, turn_open=turn_open,
        newest_role=state.messages[-1]["role"] if state.messages else None,
        total_tokens=total_tokens, kept=generated_turns(state))


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
