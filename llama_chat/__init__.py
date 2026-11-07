"""llama_chat: a KV-cache managing chat wrapper around llama.cpp.

The core actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin`: reset and prefill the system prompt and history,
  with or without reasoning.
* :meth:`ChatWrapper.inject`: prefill one message as context, no generation.
* :meth:`ChatWrapper.reply`: prefill the request text, then stream the reply,
  refusing, handing over or running the tool calls the model makes.
* :meth:`ChatWrapper.truncate_last_reply`: shorten the newest reply to what was
  read or heard of it.
* :meth:`ChatWrapper.history`: the conversation the cache holds, tool turns
  included, as :meth:`ChatWrapper.begin` takes it.

Crossing either eviction limit removes the oldest exchanges and shifts the
survivors down, reusing their KV without re-prefilling. A model whose own
template describes tool calls can use them, the tags and the argument spelling
being recovered from that template.
"""

from .config import Config, Sampling
from .messages import Eviction
from .template import CallBody, TemplateFragments, ToolCallFragments
from .tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn
from .reply import ContextOverflowError, Reply, ReplyStream
from .wrapper import ChatWrapper

__all__ = [
    "Config",
    "Sampling",
    "ChatWrapper",
    "ContextOverflowError",
    "Reply",
    "ReplyStream",
    "ToolCall",
    "ToolResult",
    "ToolTurn",
    "ToolSpec",
    "ToolRegistry",
    # For a cache context of one's own, ``ChatWrapper(context=...)``, and a
    # model whose template describes too little, ``fragments=`` and ``tool_calls=``.
    "Eviction",
    "TemplateFragments",
    "ToolCallFragments",
    "CallBody",
]
