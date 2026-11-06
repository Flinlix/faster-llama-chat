"""Real chat templates of many families, each encoded as it renders itself.

``tests/templates/`` holds one template per family, read from a published
model file that each file names. The renderer is the one a model is loaded
with; only the tokenizer is a stand-in, one token per character, so what is
compared is the text.
"""

import json
import re
from pathlib import Path

import pytest

from llama_chat._backend import Backend
from llama_chat.reasoning import ThoughtFilter, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_CONVERSATION = [("user", " ping "), ("assistant", " pong "), ("user", " again ")]


class _TemplateContext:
    """A cache context reduced to rendering through one real template."""

    def __init__(self, fixture: dict) -> None:
        backend = Backend()
        self._renderers = {prompted: backend.template_renderer(
            fixture["chat_template"], fixture["eos_token"], prompted)
            for prompted in (False, True)}

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None):
        switch = {} if reasoning is None else {"enable_thinking": reasoning}
        try:
            return self._renderers[generation_prompt](
                messages=messages, tools=tools, **switch).prompt
        except Exception:
            return None

    def tokenize(self, text, add_special=False, parse_special=True):
        return [ord(c) for c in text]

    def tokenize_fragment(self, prefix, content, suffix, add_special=False):
        return self.tokenize(prefix + content + suffix)

    def extract_fragments(self):
        from llama_chat.template_extract import extract_fragments
        return extract_fragments(self.render_with_model_template)

    def first_special_token(self, text):
        return None             # no vocabulary, so no call tag resolves

    def token_text(self, token):
        return ""


def _load(path):
    fixture = json.loads(path.read_text())
    context = _TemplateContext(fixture)
    # Construction runs the check every model meets at load: a system
    # prompt, a past reply and a newest turn, in each reasoning mode.
    return fixture, context, ResolvedTemplate(context)


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_the_template_loads_and_supports_what_it_is_known_to(path):
    fixture, _context, template = _load(path)
    assert (template.reasoning is not None) == fixture["marks_thought"]
    assert template.can_reason == fixture["can_reason"]
    assert (template._tool_calls is not None) == fixture["calls_tools"]


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda p: p.stem)
def test_a_conversation_without_a_system_prompt_opens_as_the_template_opens_it(path):
    _fixture, context, template = _load(path)
    for reasoning in (False, True) if template.can_reason else (False,):
        formatter = template.formatter(reasoning)
        messages = [{"role": r, "content": t} for r, t in _CONVERSATION]
        rendered = context.render_with_model_template(
            messages, generation_prompt=True, reasoning=reasoning)
        ours = (template.preamble(reasoning)
                + "".join(formatter.fragment(r, t) for r, t in _CONVERSATION)
                + formatter.generation_prompt())
        assert ours == rendered


@pytest.mark.parametrize("path", [p for p in _TEMPLATES
                                  if json.loads(p.read_text())["can_reason"]],
                         ids=lambda p: p.stem)
def test_a_reply_that_reasons_streams_only_its_answer(path):
    _fixture, _context, template = _load(path)
    tags = template.reasoning
    opening = template.formatter(True).generation_prompt()
    inside = begins_in_thought(tags, opening)
    generated = ("" if inside else tags.thought_open) + "Denken" \
        + tags.thought_close + "Antwort."
    thought = ThoughtFilter(tags, inside)
    shown = "".join(thought.feed(piece) for piece in generated) + thought.flush()
    assert shown == "Antwort."
    assert thought.thought_pieces > 0


# ----- tool turns, with a vocabulary for the call tags ----------------------
_TAG = re.compile(r"<\|[^<>\s]*?\|?>|<[^<>\s|]+\|>|<[/a-z_]+(?::[a-z_]+)?>|\[[A-Z_]+\]|<｜[^｜]+｜>")


class _TaggedContext(_TemplateContext):
    """The same, with tags single tokens.

    A stand-in for a vocabulary that holds every tag-shaped text, such as one
    closing each argument of a call, and the tags that close those, spelled as
    tags are paired (``<x>`` and ``</x>``, ``<|x>`` and ``<x|>``, ``start`` and
    ``end``). The earliest of them in a text is its first tag.
    """

    def __init__(self, fixture: dict) -> None:
        super().__init__(fixture)
        self._tags: list[str] = []

    def first_special_token(self, text):
        # The earliest tag in the text, as a vocabulary finds it.
        known = [(text.index(tag), tag) for tag in self._tags if tag in text]
        found = _TAG.search(text)
        if found is not None and found.group() not in self._tags:
            known.append((found.start(), found.group()))
        if not known:
            return None
        tag = min(known)[1]
        if tag not in self._tags:
            self._tags += [tag] + [closer for closer in _closers(tag)
                                   if closer not in self._tags]
        return self._id(tag)

    def _id(self, tag):
        return -1 - self._tags.index(tag)

    def token_text(self, token):
        return self._tags[-1 - token] if token < 0 else chr(token)


def _closers(tag: str) -> list[str]:
    """The tags that would close ``tag``."""
    inner = tag.strip("<>[]")
    return [f"</{inner}>", f"<{inner.lstrip('|')}|>", tag.replace("start", "end"),
            tag.replace("calls", "calls").replace("<｜", "</｜", 1)]


# Families whose calls cannot be read, so tools are refused, and why.
_TOOLS_REFUSED = {
    "llama-3.2": "plain text",          # a call is bare JSON, without a tag
    "deepseek-v4.1": "plain text",      # its call tag is not one this stand-in pairs
    "minimax-m3.1": "call format",      # each key's name is in its own closing tag
    "hunyuan-3": "call format",         # a call's body is not recovered
    "hunyuan-memory": "call format",    # a call's body is not recovered
    "kimi-k3": "call format",           # a call's body is not recovered
    "minicpm-5": "plain text",          # a call is bare JSON, without a tag
    "hunyuan-4": "plain text",          # no call tag resolves in this stand-in
}

_TOOL = ToolSpec(name="get_weather", description="Weather for a city",
                 parameters={"type": "object", "properties": {"city": {"type": "string"}}})


def _tool_templates():
    for path in _TEMPLATES:
        fixture = json.loads(path.read_text())
        if fixture["calls_tools"]:
            yield path


@pytest.mark.parametrize("path", list(_tool_templates()), ids=lambda p: p.stem)
def test_a_conversation_with_a_past_tool_turn_is_encoded_as_it_renders(path):
    fixture = json.loads(path.read_text())
    context = _TaggedContext(fixture)
    template = ResolvedTemplate(context)
    if path.stem in _TOOLS_REFUSED:
        with pytest.raises(ValueError, match=_TOOLS_REFUSED[path.stem]):
            template.ensure_tools_usable()
        return
    template.ensure_tools_usable()
    turn = ToolTurn((ToolCall("get_weather", {"city": "Berlin"}),
                     ToolResult("get_weather", "sunny"), "It is sunny."))
    head, tail = template.system_turn_declaring([_TOOL])
    formatter = template.formatter()
    ours = (head + "Be brief." + tail + formatter.fragment("user", "Weather?")
            + "".join(text for text, _ in template.past_tool_turn(turn))
            + formatter.fragment("user", "Thanks.") + formatter.generation_prompt())
    messages = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
                {"role": "tool", "name": "get_weather", "tool_call_id": "call_0",
                 "content": "sunny"},
                {"role": "assistant", "content": "It is sunny."},
                {"role": "user", "content": "Thanks."}]
    rendered = context.render_with_model_template(
        messages, tools=[_TOOL.as_declaration()], generation_prompt=True,
        reasoning=False)
    assert ours == rendered


@pytest.mark.parametrize("path", [p for p in _tool_templates()
                                  if p.stem not in _TOOLS_REFUSED],
                         ids=lambda p: p.stem)
def test_tools_without_a_system_prompt_open_as_the_template_opens_them(path):
    fixture = json.loads(path.read_text())
    context = _TaggedContext(fixture)
    template = ResolvedTemplate(context)
    formatter = template.formatter()
    ours = (template.preamble_declaring([_TOOL]) + formatter.fragment("user", "Weather?")
            + formatter.generation_prompt())
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "Weather?"}], tools=[_TOOL.as_declaration()],
        generation_prompt=True, reasoning=False)
    assert ours == rendered


@pytest.mark.parametrize("path", [p for p in _TEMPLATES if p.stem in ("gpt-oss", "muse-glimmer")],
                         ids=lambda p: p.stem)
def test_a_reply_routed_through_a_channel_streams_only_its_answer(path):
    from llama_chat.reasoning import AnswerOpening
    _fixture, _context, template = _load(path)
    formatter = template.formatter()
    assert formatter.answer_opening()
    generated = formatter.answer_opening() + "Antwort."
    thought, opening = ThoughtFilter(template.reasoning), AnswerOpening(formatter.answer_opening())
    shown = "".join(opening.feed(thought.feed(c)) for c in generated)
    shown += opening.feed(thought.flush()) + opening.flush()
    assert shown == "Antwort."
