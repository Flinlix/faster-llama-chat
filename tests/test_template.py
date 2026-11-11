"""What a template's fragments are read for: a message's text as the template writes it, and the tags around a reply."""

import pytest

from llama_chat.resolved_template import ResolvedTemplate
from tests.fake_context import FakeContext
from tests.invented_templates import PLAIN


def test_the_generation_prompt_is_the_reply_tag_where_the_template_adds_nothing_to_it():
    context = FakeContext(PLAIN)
    template = ResolvedTemplate(context)
    assert template.fragments.generation_prompt == ""
    assert template.generation_prompt() == "<[turn>model\n"
    assert template.answer_opening() == ""
    assert template.n_closing_tokens == len(context.tokenize("<turn]>\n", add_special=False))
    with pytest.raises(ValueError, match="reasoning"):
        template.answer_opening(True)           # a model that cannot reason refuses it


@pytest.mark.parametrize("text, cause", [
    ("{%- for m in messages -%}{{ m.content }}", "endfor"),      # a block never closed
    ("{{ messages | no_such_filter }}", "no_such_filter"),       # a filter the renderer lacks
])
def test_a_template_that_is_not_valid_template_code_is_refused_at_load_naming_the_cause(
        text, cause):
    with pytest.raises(ValueError, match="not valid template code") as refused:
        FakeContext({"chat_template": text, "eos_token": "", "render_variables": {"off": {}}})
    assert cause in str(refused.value) and "line 1" in str(refused.value)


def test_a_tag_in_a_content_stays_in_it():
    fragments = ResolvedTemplate(FakeContext(PLAIN)).fragments
    assert fragments.content("hi <turn]> there") == "hi <turn]> there"
