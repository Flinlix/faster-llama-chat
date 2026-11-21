"""A caller of the public interface as a type checker reads it, checked with mypy's strict settings by ``tests/test_static.py``: a conversation saved and begun again, a history of plain dictionaries, and a reply's pieces."""

from typing import Any, assert_type

from llama_cpp_conversation import ChatMessage, ChatWrapper, Conversation, ToolCall


def resumed(chat: ChatWrapper, history: list[dict[str, Any]]) -> str:
    saved = chat.conversation()
    assert_type(saved, Conversation)
    assert_type(saved["messages"], list[ChatMessage])
    chat.begin(**saved)
    chat.begin(history)
    chat.begin([{"role": "user", "content": "Hi"}])
    for piece in chat.reply("Hi"):
        assert_type(piece, str | ToolCall)
    reply = chat.reply("Again").finish()
    assert reply is not None            # None only for a stream closed before it started
    return reply.text
