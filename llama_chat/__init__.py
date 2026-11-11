"""llama_chat: a KV-cache managing chat wrapper around llama.cpp.

The core actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin`: reset and prefill the system prompt and history,
  with or without reasoning.
* :meth:`ChatWrapper.inject`: prefill one message as context, no generation.
* :meth:`ChatWrapper.reply`: prefill the request text, then stream the reply,
  refusing, handing over or running the tool calls the model makes.
* :meth:`ChatWrapper.truncate_last_reply`: shorten the newest reply to what was
  read or heard of it.
* :meth:`ChatWrapper.conversation`: the system prompt and the history the
  cache holds, tool turns included, as :meth:`ChatWrapper.begin` takes them.

Crossing either eviction limit removes the oldest exchanges and shifts the
survivors down, reusing their KV without re-prefilling. A model whose own
template describes tool calls can use them, the tags and the argument spelling
being recovered from that template.
"""

from .cache_context import CacheContext, ModelReader
from .config import Config, InvalidSettingError, Sampling
from .tools import ToolCall, ToolCallStatus, ToolRegistry, ToolResult, ToolSpec, ToolTurn
from .reply import (STOP_REASONS, TOOL_CALL_POLICIES, ContextOverflowError, Reply, ReplyStream,
                    StopReason, ToolCallPolicy)
from .wrapper import ChatWrapper

__all__ = [
    "Config",
    "InvalidSettingError",
    "Sampling",
    "ChatWrapper",
    "ContextOverflowError",
    "Reply",
    "ReplyStream",
    "StopReason",
    "STOP_REASONS",
    "ToolCallPolicy",
    "TOOL_CALL_POLICIES",
    "ToolCall",
    "ToolCallStatus",
    "ToolResult",
    "ToolTurn",
    "ToolSpec",
    "ToolRegistry",
    # For a cache context of one's own, ``ChatWrapper(context=...)``.
    "CacheContext",
    "ModelReader",
]
