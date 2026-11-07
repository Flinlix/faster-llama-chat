"""Writing messages with a template's fragments."""

import pytest

from llama_chat.template import TemplateFormatter
from tests.wrappers import fragments_with


# ----- template formatting -------------------------------------------------
def test_template_formatter_wraps_each_role_in_its_tags():
    fmt = TemplateFormatter(fragments_with())
    assert fmt.fragment("user", "hi") == "<[turn>user\nhi<turn]>\n"
    assert fmt.generation_prompt() == "<[turn>model\n"
    assert fmt.assistant_close() == "<turn]>\n"


def test_template_formatter_combines_fragments():
    fmt = TemplateFormatter(
        fragments_with(user_prefix="<u>", user_suffix="</u>",
               assistant_prefix="<a>", assistant_suffix="</a>")
    )
    assert fmt.fragment("user", "hi") == "<u>hi</u>"
    assert fmt.fragment("assistant", "yo") == "<a>yo</a>"
    assert fmt.generation_prompt() == "<a>"
    assert fmt.assistant_close() == "</a>"
    with pytest.raises(ValueError):
        fmt.fragment("unknown-role", "x")


def test_trim_content_strips_by_default():
    fmt = TemplateFormatter(fragments_with())
    assert fmt.fragment("user", "  hi\n") == "<[turn>user\nhi<turn]>\n"


def test_trim_content_disabled_keeps_whitespace():
    fmt = TemplateFormatter(fragments_with(trim_content=False))
    assert fmt.fragment("user", "  hi\n") == "<[turn>user\n  hi\n<turn]>\n"
