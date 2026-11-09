"""What a template's fragments are read for: a message's text as the template writes it, and the tags around a reply."""

import pytest

from llama_chat.resolved_template import ResolvedTemplate
from tests.fake_context import FakeContext
from tests.wrappers import fragments_with


def test_the_generation_prompt_is_the_reply_tag_where_the_template_adds_nothing_to_it():
    context = FakeContext()
    template = ResolvedTemplate(context)
    assert template.fragments.generation_prompt == ""
    assert template.generation_prompt() == "<[turn>model\n"
    assert template.answer_opening() == ""
    assert template.n_closing_tokens == len(context.tokenize("<turn]>\n", add_special=False))
    with pytest.raises(ValueError, match="reasoning"):
        template.answer_opening(True)           # a model that cannot reason refuses it


def test_a_tag_in_a_content_stays_in_it():
    assert fragments_with().content("hi <turn]> there") == "hi <turn]> there"
