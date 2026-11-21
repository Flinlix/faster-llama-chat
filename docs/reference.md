# API reference

Everything the package exports, as `from llama_cpp_conversation import ...`
takes it.

```{eval-rst}
.. automodule:: llama_cpp_conversation
   :members:
   :imported-members:
   :exclude-members: ChatMessage, ChatToolCall, ChatFunction, Conversation
```

## Messages and conversations

The typed dictionaries `begin` takes and `conversation()` gives, with their
keys.

```{eval-rst}
.. autoclass:: llama_cpp_conversation.ChatMessage
   :members:
   :undoc-members:

.. autoclass:: llama_cpp_conversation.ChatToolCall
   :members:
   :undoc-members:

.. autoclass:: llama_cpp_conversation.ChatFunction
   :members:
   :undoc-members:

.. autoclass:: llama_cpp_conversation.Conversation
   :members:
   :undoc-members:
```

## Constants and types

The values and type aliases the package exports, under the module that
defines them.

```{eval-rst}
.. autodata:: llama_cpp_conversation.messages.QUESTION_CUT_MARK
.. autodata:: llama_cpp_conversation.reply.StopReason
.. autodata:: llama_cpp_conversation.reply.STOP_REASONS
.. autodata:: llama_cpp_conversation.reply.ToolCallPolicy
.. autodata:: llama_cpp_conversation.reply.TOOL_CALL_POLICIES
.. autodata:: llama_cpp_conversation.reply.EvictionCause
.. autodata:: llama_cpp_conversation.reply.EVICTION_CAUSES
.. autodata:: llama_cpp_conversation.tools.ToolCallStatus
.. autodata:: llama_cpp_conversation.cache_context.TOOL_USE_TEMPLATE
.. autodata:: llama_cpp_conversation.conversation_types.GeneratedTurns
```

## For tests

`llama_cpp_conversation.testing` holds a stand-in model for tests of code that
drives a `ChatWrapper` without loading one.

```{eval-rst}
.. autoclass:: llama_cpp_conversation.testing.fake_context.FakeContext
   :members:

.. autoclass:: llama_cpp_conversation.testing.fake_context.ScriptedCall

.. automodule:: llama_cpp_conversation.testing.invented_templates
```
