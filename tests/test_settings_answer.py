"""Whatever the settings and the system prompt, a conversation that begin accepts answers a question: no state is reachable in which the model is ready but cannot answer."""

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from llama_cpp_conversation import ChatWrapper, Config, ContextOverflowError
from tests.chat_messages import opened
from tests.chat_templates import STEMS, fixture
from llama_cpp_conversation.testing.fake_context import FakeContext
from llama_cpp_conversation.testing.invented_templates import PLAIN, REASONING, TOOLS

_TEMPLATES = {"plain": PLAIN, "tools": TOOLS, "reasoning": REASONING,
              **{stem: fixture(stem) for stem in STEMS}}


@settings(max_examples=300, derandomize=True, database=None, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(name=st.sampled_from(sorted(_TEMPLATES)), context_size=st.integers(64, 4096),
       evict=st.floats(0.05, 1.0), min_reply=st.integers(0, 600),
       system=st.integers(0, 3000), question=st.integers(0, 20000),
       reasoning=st.booleans())
def test_a_conversation_begin_accepts_answers_a_question_of_any_length(
        name, context_size, evict, min_reply, system, question, reasoning):
    try:
        config = Config(context_size=context_size, evict_above_fraction=evict,
                        min_reply_tokens=min_reply)
    except ValueError:
        return                          # settings refused where they are made
    wrapper = ChatWrapper(config, context=FakeContext(_TEMPLATES[name]))
    can = wrapper.capabilities(tools=False)
    try:
        wrapper.begin(opened("s" * system if can.can_take_system_prompt else ""),
                      reasoning=bool(reasoning and can.can_reason))
    except ContextOverflowError:
        return                          # refused at begin, with its cause: never ready
    reply = wrapper.reply("x" * question + " Frage?").finish()
    assert reply.text.strip() or reply.stop_reason == "token_limit"
    assert not wrapper.turn_open


@pytest.mark.parametrize("name", ["plain", "reasoning"])
def test_a_reply_room_too_small_for_a_word_is_still_a_reply(name):
    wrapper = ChatWrapper(Config(context_size=256, min_reply_tokens=1),
                          context=FakeContext(_TEMPLATES[name]))
    wrapper.begin(opened(""))
    assert wrapper.reply("Frage?", max_reply_tokens=1).finish().stop_reason == "token_limit"
