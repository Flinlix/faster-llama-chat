"""Every wrapper a test builds on the model-free cache ends the test with that cache equal to its message table, and, where nothing is kept otherwise, to the template's render."""

import pytest

from llama_chat.tools import ToolTurn
from llama_chat.wrapper import ChatWrapper

from .fake_context import FakeContext


@pytest.fixture(autouse=True)
def _the_cache_holds_the_table_and_reads_as_the_template_renders_it(monkeypatch):
    built = []
    build = ChatWrapper.__init__

    def recording(self, *args, **kwargs):
        build(self, *args, **kwargs)
        if isinstance(self._context, FakeContext):
            built.append(self)

    monkeypatch.setattr(ChatWrapper, "__init__", recording)
    yield
    for wrapper in built:
        tokens = [t for m in wrapper._table.messages for t in m.token_ids]
        assert wrapper._context.cache == tokens, "the cache no longer holds the message table"
        if _rendered_as_kept(wrapper):
            rendered = _render(wrapper)
            if rendered is not None:
                assert wrapper._context.text_of(wrapper._context.cache) == rendered, \
                    "the cache does not read as the template renders the conversation"


def _rendered_as_kept(wrapper) -> bool:
    """Whether the cache must read as the template renders the conversation's history.

    Not where a turn keeps what it was generated as, a tool turn or a thought
    the template keeps, nor where a message was cut to fit, nor where the
    fragments were handed in rather than read from the template.
    """
    reasoning = wrapper._template.reasoning
    return (wrapper._begun and not wrapper.reply_open and not wrapper._closed
            and not any(wrapper._template_given.values())
            and not (wrapper._reasoning and reasoning and reasoning.past_keeps_thought)
            and wrapper.config.oversize_policy != "truncate"
            and not any(m.holds_call for m in wrapper._table.messages))


def _render(wrapper) -> str | None:
    """The fake template's render of the system prompt and ``history()``, as ``begin`` would read them in."""
    messages = [{"role": "system", "content": wrapper.system_prompt}] if wrapper.system_prompt else []
    for item in wrapper.history():
        if isinstance(item, ToolTurn):
            return None             # covered above: such a turn keeps its tokens
        messages.append({"role": item[0], "content": item[1]})
    specs = [spec.as_declaration() for spec in wrapper._system_specs + wrapper._moving_specs]
    return wrapper._context.render_with_model_template(messages, specs or None,
                                                       reasoning=wrapper._reasoning)

