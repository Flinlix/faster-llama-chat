"""What a template's fragments are read for: a message's text as the template writes it, and the tags around a reply."""

import json
import pytest

from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN
from llama_chat.errors import TemplateRefusedError
from llama_chat.split_render import split_render


def test_the_generation_prompt_is_the_reply_tag_where_the_template_adds_nothing_to_it():
    context = FakeContext(PLAIN)
    template = ResolvedTemplate(context)
    assert template.properties.fragments.generation_prompt == ""
    assert template.generation_prompt() == "<[turn>model\n"
    assert template.answer_opening() == ""
    assert template.properties.n_closing_tokens == len(context.tokenize("<turn]>\n", add_special=False))
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
    fragments = ResolvedTemplate(FakeContext(PLAIN)).properties.fragments
    assert fragments.content("hi <turn]> there") == "hi <turn]> there"


def test_a_split_render_takes_time_in_proportion_to_the_messages():
    import time

    from llama_chat.split_render import split_render

    def render(messages, declarations):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def timed(n):
        """The shortest of three splits of ``n`` messages, so that a pause of the machine counts once at most."""
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"text {i}"}
                    for i in range(n)]
        times = []
        for _ in range(3):
            started = time.thread_time()        # this thread's own processor time
            split = split_render(render, messages, lambda text: text, reads_clock=False)
            times.append(time.thread_time() - started)
        assert len(split.own) == n
        return min(times)

    timed(2000)                                 # warmed up
    # Twice the messages take about twice as long, not four times.
    assert timed(40000) < 3 * timed(20000)


def _closed_by_the_next_question(after_reply: str = "", footer: str = "", head: str = "") -> dict:
    """A template that writes nothing after a reply but ``after_reply``, whose question opens with a tag that ends a generation."""
    return {"chat_template": head + (
        "{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}<|ask|>{{ m.content }}" + footer +
        "{%- elif m.role == 'system' -%}<|rules|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|answer|>{{ m.content }}" + after_reply +
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|answer|>{%- endif -%}"),
        "eos_token": "<|stop|>", "ends_generation": ["<|ask|>"],
        "render_variables": {"off": {}, "on": {}}}


@pytest.mark.parametrize("after_reply, footer", [
    ("{{ '\\n' }}", ""),
    # Text only where the reasoning switch is left unset or turned on, which a
    # template that cannot reason is never written with.
    ("{%- if enable_thinking is not defined or enable_thinking -%}~{%- endif -%}", ""),
    ("", "{%- if loop.index0 and messages[loop.index0 - 1].role == 'assistant' -%}"
         "<|quote|>{{ messages[loop.index0 - 1].content }}{%- endif -%}"),
], ids=["space-before-the-question", "text-in-a-mode-never-written", "reply-quoted-later"])
def test_a_reply_closed_by_nothing_loads_where_the_question_after_it_opens_with_an_ending_tag(
        after_reply, footer):
    template = ResolvedTemplate(FakeContext(_closed_by_the_next_question(after_reply, footer)))
    assert not template.properties.fragments.assistant_suffix.strip()
    assert not template.properties.can_reason


def test_a_reply_closed_by_nothing_on_a_template_that_refuses_a_second_question_is_refused_at_load():
    one_question = _closed_by_the_next_question(head=(
        "{%- if messages | selectattr('role', 'eq', 'user') | list | length > 1 -%}"
        "{{ raise_exception('one question only') }}{%- endif -%}"))
    with pytest.raises(ValueError, match="writes nothing after a reply"):
        ResolvedTemplate(FakeContext(one_question))


@pytest.mark.parametrize("arguments, options", [
    ("", {}),
    ("(ensure_ascii=true)", {"ensure_ascii": True}),
    ("(indent=4)", {"indent": 4}),
    ("(separators=(',', ':'))", {"separators": (",", ":")}),
    ("(sort_keys=true)", {"sort_keys": True}),
])
def test_tojson_writes_json_as_json_dumps_does_keeping_letters_beyond_ascii_unless_asked(
        arguments, options):
    from llama_chat.model_template import compiled
    value = {"zeta": "café 漢", "alpha": [1, {"b": 2.5, "a": None, "c": True}]}
    rendered = compiled("{{ value | tojson" + arguments + " }}").render(value=value)
    assert rendered == json.dumps(value, **{"ensure_ascii": False, **options})


def _render_marking_repeated_questions(drops_repeated: bool):
    """A render that tags a question equal to an earlier one, and writes it, or drops its text where ``drops_repeated``."""
    def render(messages, declarations):
        text = "".join(f"<decl>{d['function']['description']}</decl>" for d in declarations or ())
        asked = []
        for m in messages:
            repeated = m["role"] == "user" and m["content"] in asked
            asked += [m["content"]] if m["role"] == "user" else []
            body = "" if repeated and drops_repeated else m["content"]
            text += ("<again>" if repeated else "") + f"<{m['role']}>{body}</{m['role']}>"
        return text
    return render


_ASKED_TWICE = [{"role": "user", "content": "Same?"}, {"role": "assistant", "content": "Fine."},
                {"role": "user", "content": "Same?"}]


_ANSWERED_TWICE_ALIKE = _ASKED_TWICE + [{"role": "assistant", "content": "Fine."}]


_DECLARED_ALIKE = [{"type": "function", "function": {"name": name, "description": "Looks it up."}}
                   for name in ("weather", "clock")]


@pytest.mark.parametrize("messages", [_ASKED_TWICE, _ANSWERED_TWICE_ALIKE],
                         ids=["a-question-asked-twice", "a-question-and-its-reply-twice"])
@pytest.mark.parametrize("declarations", [None, _DECLARED_ALIKE], ids=["undeclared", "declared"])
def test_a_split_where_the_template_compares_texts_places_each_text_in_its_own_place(
        messages, declarations):
    render = _render_marking_repeated_questions(drops_repeated=False)
    split = split_render(render, messages, lambda text: text, declarations, reads_clock=False)
    assert split.text == render(messages, declarations)
    places = [split.contents[i] for i in range(len(messages))]
    assert all(place is not None for place in places)
    assert [split.text[start:end] for start, end in places] == [m["content"] for m in messages]
    assert places == sorted(places)
    own = [split.text[start:end] for start, end in split.own]
    descriptions = [d["function"]["description"] for d in declarations or ()]
    assert own == descriptions + [m["content"] for m in messages]


def test_a_split_where_the_template_drops_a_repeated_text_is_refused():
    render = _render_marking_repeated_questions(drops_repeated=True)
    with pytest.raises(TemplateRefusedError, match="text in its place"):
        split_render(render, _ASKED_TWICE, lambda text: text, reads_clock=False)
