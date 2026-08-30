"""Durable conversation history for :class:`~llama_chat.wrapper.ChatWrapper`.

The store outlives KV eviction: it keeps every message but the system prompt,
while the cache holds only the recent window that fits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from typing import Protocol

from .config import Config
from .tools import ToolCall, ToolRegistry, ToolSpec
from .wrapper import ChatWrapper, Turn


class ConversationStore(Protocol):
    """Durable store of conversation turns, keyed by conversation id."""

    def load(self, conversation_id: str) -> list[tuple[str, str]]:
        """Return the conversation's ``(role, text)`` turns, oldest first."""
        ...

    def append(self, conversation_id: str, role: str, text: str) -> None:
        """Append one turn to the conversation's durable log."""
        ...


class InMemoryStore:
    """Reference :class:`ConversationStore` backed by a dict. Not durable."""

    def __init__(self) -> None:
        self._convos: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def load(self, conversation_id: str) -> list[tuple[str, str]]:
        return list(self._convos[conversation_id])

    def append(self, conversation_id: str, role: str, text: str) -> None:
        self._convos[conversation_id].append((role, text))


class PersistentChat:
    """A ``ChatWrapper`` whose messages are persisted to a store.

    Mirrors the wrapper method for method. Everything but the system prompt is
    persisted, that being supplied fresh on each ``begin``. A tool call and its
    result are not messages, so a tool turn is persisted like any other.
    """

    # ----- setting up ----------------------------------------------------------
    def __init__(self, store: ConversationStore, config: Config | None = None, **kwargs) -> None:
        self._store = store
        self._conversation_id: str | None = None
        self._chat = ChatWrapper(config, on_message=self._persist, **kwargs)

    def begin(self, conversation_id: str, system_prompt: str,
              tools: ToolRegistry | list[ToolSpec] | None = None) -> None:
        """Select the conversation and prefill its recent history.

        A failed begin selects nothing, so later turns cannot persist into a
        conversation whose history was never loaded.

        Args:
            conversation_id: Which conversation to load and persist into.
            system_prompt: Supplied fresh each time; never persisted.
            tools: As :meth:`~llama_chat.wrapper.ChatWrapper.begin`, and not
                persisted either.
        """
        self._chat.begin(system_prompt, self._store.load(conversation_id), tools)
        self._conversation_id = conversation_id

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        return self._chat.inject(text, role)

    # ----- asking: one generation ----------------------------------------------
    def stream_turn(self, text: str, **kwargs) -> Generator[str, None, Turn]:
        return self._chat.stream_turn(text, **kwargs)

    def turn(self, text: str, **kwargs) -> Turn:
        return self._chat.turn(text, **kwargs)

    # ----- asking: until it answers --------------------------------------------
    def stream_answer(self, text: str, registry: ToolRegistry | None = None,
                      **kwargs) -> Generator[str, None, Turn]:
        return self._chat.stream_answer(text, registry, **kwargs)

    def answer(self, text: str, registry: ToolRegistry | None = None,
               **kwargs) -> Turn:
        return self._chat.answer(text, registry, **kwargs)

    # ----- tools: driving the loop yourself ------------------------------------
    def stream_events(self, text: str,
                      **kwargs) -> Generator[str | ToolCall, None, Turn]:
        return self._chat.stream_events(text, **kwargs)

    def continue_stream(self, **kwargs) -> Generator[str | ToolCall, None, Turn]:
        return self._chat.continue_stream(**kwargs)

    def inject_tool_result(self, name: str, result) -> None:
        self._chat.inject_tool_result(name, result)

    def declare_tools(self, specs: list[ToolSpec]) -> int:
        return self._chat.declare_tools(specs)

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self._chat.total_tokens

    @property
    def can_shift(self) -> bool:
        return self._chat.can_shift

    @property
    def busy(self) -> bool:
        return self._chat.busy

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        return self._chat.wait_idle(timeout_s)

    def snapshot(self) -> list[dict]:
        return self._chat.snapshot()

    @property
    def can_call_tools(self) -> bool:
        return self._chat.can_call_tools

    @property
    def awaiting_tool_result(self) -> bool:
        return self._chat.awaiting_tool_result

    # ----- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._chat.close()

    def __enter__(self) -> PersistentChat:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- internals -----------------------------------------------------------
    def _persist(self, role: str, text: str) -> None:
        if self._conversation_id is None:
            raise RuntimeError("call begin() before sending turns")
        self._store.append(self._conversation_id, role, text)
