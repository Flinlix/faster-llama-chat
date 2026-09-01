"""Durable conversation history for :class:`~llama_chat.wrapper.ChatWrapper`.

The store outlives KV eviction: it keeps every message but the system prompt,
while the cache holds only the recent window that fits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .config import Config
from .tools import ToolCall, ToolRegistry, ToolSpec
from .wrapper import ChatWrapper, Reply, ReplyStream


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
        """Build the chat and adopt the store every message is mirrored to.

        Args:
            store: Where turns are persisted. Checked here rather than at the
                first message, which is written mid-turn.
            config, **kwargs: As :class:`~llama_chat.wrapper.ChatWrapper`.

        Raises:
            TypeError: If ``store`` is not a :class:`ConversationStore`.
        """
        missing = [name for name in ("load", "append") if not hasattr(store, name)]
        if missing:
            raise TypeError(
                f"this conversation store is missing {missing}; one answers "
                f"load(conversation_id) and append(conversation_id, role, text)")
        self._store = store
        self._conversation_id: str | None = None
        self._chat = ChatWrapper(config, on_message=self._persist, **kwargs)

    def begin(self, conversation_id: str, system_prompt: str,
              tools: ToolRegistry | list[ToolSpec] | None = None) -> None:
        """Select the conversation and prefill its recent history.

        A failed begin selects nothing new, so later turns cannot persist into
        a conversation whose history was never loaded.

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
        self._ensure_conversation_selected()
        return self._chat.inject(text, role)

    # ----- asking: without tools -----------------------------------------------
    def reply_without_tools_stream(self, text: str, **kwargs) -> ReplyStream[str]:
        self._ensure_conversation_selected()
        return self._chat.reply_without_tools_stream(text, **kwargs)

    def reply_without_tools(self, text: str, **kwargs) -> Reply:
        self._ensure_conversation_selected()
        return self._chat.reply_without_tools(text, **kwargs)

    # ----- asking: with tools --------------------------------------------------
    def reply_with_tools_stream(self, text: str,
                                registry: ToolRegistry | None = None,
                                **kwargs) -> ReplyStream[str]:
        self._ensure_conversation_selected()
        return self._chat.reply_with_tools_stream(text, registry, **kwargs)

    def reply_with_tools(self, text: str, registry: ToolRegistry | None = None,
                          **kwargs) -> Reply:
        self._ensure_conversation_selected()
        return self._chat.reply_with_tools(text, registry, **kwargs)

    # ----- tools: driving the loop yourself ------------------------------------
    def reply_until_tool(self, text: str,
                        **kwargs) -> ReplyStream[str | ToolCall]:
        self._ensure_conversation_selected()
        return self._chat.reply_until_tool(text, **kwargs)

    def resume_until_tool(self, **kwargs) -> ReplyStream[str | ToolCall]:
        return self._chat.resume_until_tool(**kwargs)

    def inject_tool_result(self, call: ToolCall | str, result) -> int:
        return self._chat.inject_tool_result(call, result)

    def declare_unknown_tools(self, tools: ToolRegistry | list[ToolSpec]) -> int:
        return self._chat.declare_unknown_tools(tools)

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self._chat.total_tokens

    @property
    def context_size(self) -> int:
        return self._chat.context_size

    @property
    def evict_above_tokens(self) -> int:
        return self._chat.evict_above_tokens

    @property
    def max_history_messages(self) -> int | None:
        return self._chat.max_history_messages

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

    @property
    def reply_open(self) -> bool:
        return self._chat.reply_open

    # ----- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._chat.close()

    def __enter__(self) -> PersistentChat:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- internals -----------------------------------------------------------
    def _ensure_conversation_selected(self) -> None:
        """Refuse before ``begin``, which is what selects where turns go.

        Checked at every entry point that can write a message, and again where
        one is persisted: a message noticed only on its way to the store has
        already been prefilled, and the cache would keep what the store never
        got.

        Raises:
            RuntimeError: If no conversation is selected.
        """
        if self._conversation_id is None:
            raise RuntimeError("no conversation selected; call begin() first")

    def _persist(self, role: str, text: str) -> None:
        self._ensure_conversation_selected()
        self._store.append(self._conversation_id, role, text)
