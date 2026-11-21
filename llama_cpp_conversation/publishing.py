"""What a conversation publishes after each change, for readers that hold no lock: a copy, worked out from its state and holding nothing of its own, and the history forms read from it."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import cast

from .conversation_types import ChatMessage, Conversation, GeneratedTurns
from .conversation_state import ConversationState, Published, reasoning_asked
from .history_forms import completions_messages, generated_turn_records
from .rendering import generated_turns_in
from .tools import ToolCall


def published(state: ConversationState, pending_calls: list[ToolCall], turn_open: bool,
              total_tokens: int, joins_questions: bool,
              model_fingerprint: Callable[[], str]) -> Published:
    """The copy of ``state`` readers without the lock see, the calls it owes, the cache's length and, where it keeps a turn as generated, the model's fingerprint beside it."""
    return Published(
        messages=tuple(state.messages), joins_questions=joins_questions,
        tools=tuple(state.tools) if state.uses_tool_template else None,
        reasoning=state.reasoning,
        # A call written unreadably is answered by the library, so the caller owes it nothing.
        pending_calls=tuple(c for c in pending_calls if c.status != "unreadable"),
        begun=state.begun, turn_open=turn_open,
        newest_role=state.messages[-1]["role"] if state.messages else None,
        total_tokens=total_tokens,
        model=model_fingerprint() if generated_turns_in(state.messages) else None,
        turn_policy=state.turn_policy)


def conversation_of(published: Published) -> Conversation:
    """Everything ``begin`` takes to begin the conversation ``published`` holds again, by its parameter names, the caller's to change."""
    # The messages are Chat Completions messages by construction; the type is stated here, where they leave.
    return {"messages": cast(list[ChatMessage], completions_messages(list(published.messages),
                                                                      published.joins_questions)),
            "tools": None if published.tools is None else copy.deepcopy(list(published.tools)),
            "reasoning": reasoning_asked(published.reasoning)[0],
            "reasoning_level": reasoning_asked(published.reasoning)[1],
            "generated_turns": generated_turns_of(published)}


def generated_turns_of(published: Published) -> GeneratedTurns | None:
    """The turns ``published`` keeps as generated, as ``begin`` takes them back; ``None`` where none is."""
    if published.model is None:
        return None
    return {"model": published.model,
            "turns": generated_turn_records(list(published.messages), published.joins_questions)}
