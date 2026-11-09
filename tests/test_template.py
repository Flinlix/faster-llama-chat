"""What a template's fragments are read for: a message's text as the template writes it, and the tags around a reply."""

from llama_chat.template import TemplateFormatter
from tests.wrappers import fragments_with


def test_the_formatter_reads_the_prompt_and_the_closing_around_a_reply():
    fmt = TemplateFormatter(fragments_with(assistant_prefix="<a>", assistant_suffix="</a>"))
    assert fmt.generation_prompt() == "<a>"
    assert fmt.assistant_close() == "</a>"


def test_a_tag_in_a_content_stays_in_it():
    assert fragments_with().content("hi <turn]> there") == "hi <turn]> there"
