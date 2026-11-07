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
from llama_chat.call_parsing import parse_calls
from llama_chat.reasoning import ThoughtFilter, begins_in_thought
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.template_extract import reasoning_switch
from llama_chat.tools import ToolCall, ToolResult, ToolSpec, ToolTurn

_TEMPLATES = sorted((Path(__file__).parent / "templates").glob("*.json"))
_CONVERSATION = [("user", " ping "), ("assistant", " pong "), ("user", " again ")]


class _TemplateContext:
    """A cache context reduced to rendering through one real template."""

    def __init__(self, fixture: dict) -> None:
        backend = Backend()
        self._template = fixture["chat_template"]
        self._renderers = {prompted: backend.template_renderer(
            fixture["chat_template"], fixture["eos_token"], prompted)
            for prompted in (False, True)}
        self._switch = reasoning_switch(lambda messages, generation_prompt=False, **variables:
                                        self._renderers[generation_prompt](
                                            messages=messages, **variables).prompt)

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None):
        try:
            return self._renderers[generation_prompt](
                messages=messages, tools=tools, **self._switch(reasoning)).prompt
        except Exception:
            return None

    def tokenize(self, text, add_special=False, parse_special=True):
        return [ord(c) for c in text]

    def tags(self):
        # As a vocabulary holds them: every tag-shaped text, and what each closer closes.
        found = set(_TAG.findall(self._template))
        return tuple(sorted(found | {"<" + tag[2:] for tag in found if tag.startswith("</")}))

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
def test_two_user_turns_in_a_row_are_written_as_the_template_writes_them_or_refused(path):
    _fixture, context, template = _load(path)
    messages = [{"role": "user", "content": " ping "}, {"role": "user", "content": " again "}]
    rendered = context.render_with_model_template(messages, generation_prompt=True,
                                                  reasoning=False)
    formatter = template.formatter()
    ours = (template.preamble() + formatter.fragment("user", " ping ")
            + (template.user_turn_join or "") + formatter.fragment("user", " again ")
            + formatter.generation_prompt())
    if template.user_turn_join is None:
        assert rendered != ours
    else:
        assert rendered == ours


@pytest.mark.parametrize("stem, join", [("deepseek-r1-qwen3", "<｜Assistant｜>"), ("qwen-3.6", ""),
                                        ("mistral-3", None), ("deepseek-v4.1", None)])
def test_the_join_of_two_user_turns_is_read_from_the_template(stem, join):
    _fixture, _context, template = _load(Path(__file__).parent / "templates" / f"{stem}.json")
    assert template.user_turn_join == join


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


class _TagTokenizingContext(_TaggedContext):
    """The same, with each tag tokenized as the one token it resolves to."""

    def tokenize(self, text, add_special=False, parse_special=True):
        tokens, pos = [], 0
        for found in _TAG.finditer(text):
            tokens += [ord(c) for c in text[pos:found.start()]]
            tokens.append(self.first_special_token(found.group()))
            pos = found.end()
        return tokens + [ord(c) for c in text[pos:]]


@pytest.mark.parametrize("stem, first_tag", [("cohere-aya", "<|END_RESPONSE|>"),
                                             ("qwen-3.6", "<|im_end|>")])
def test_a_reply_ends_at_the_first_tag_of_its_closing(stem, first_tag):
    fixture = json.loads((Path(__file__).parent / "templates" / f"{stem}.json").read_text())
    context = _TagTokenizingContext(fixture)
    template = ResolvedTemplate(context)
    assert context.token_text(template.turn_end_token) == first_tag


@pytest.mark.parametrize("stem", ["gpt-oss", "muse-glimmer", "qwen-3.6"])
def test_a_thought_closing_with_the_turns_end_does_not_end_the_reply(stem):
    """gpt-oss closes a thought with the tag that also closes its turn."""
    from llama_chat.reasoning import AnswerOpening, turn_end_test
    fixture = json.loads((Path(__file__).parent / "templates" / f"{stem}.json").read_text())
    context = _TagTokenizingContext(fixture)
    template = ResolvedTemplate(context)
    tags, turn_end = template.reasoning, template.turn_end_token
    formatter = template.formatter(True)
    inside = begins_in_thought(tags, formatter.generation_prompt())
    thought = ThoughtFilter(tags, inside)
    opening = AnswerOpening(formatter.answer_opening())
    ends = turn_end_test(turn_end, context.token_text(turn_end), tags, thought)
    reply = (("" if inside else tags.thought_open) + "Denken" + tags.thought_close
             + "Antwort." + template._fragments.assistant_suffix)
    shown = ""
    for token in context.tokenize(reply):
        if ends(token):
            break
        shown += opening.feed(thought.feed(context.token_text(token)))
    assert shown + opening.feed(thought.flush()) + opening.flush() == "Antwort."


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
    "ministral-2410": "declares no tools",  # declared before the newest user turn only
    "mistral-small-3.2": "declares no tools",  # declared before the newest user turn only
    "kimi-k3": "call format",           # a call's body is not recovered
    "gpt-oss": "plain text",            # the call is named in the turn's header, before any tag
    "muse-glimmer": "plain text",       # the call is named in the turn's header, before any tag
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
                    {"id": "call00000", "type": "function", "function": {
                        "name": "get_weather", "arguments": {"city": "Berlin"}}}]},
                {"role": "tool", "name": "get_weather", "tool_call_id": "call00000",
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
def test_a_call_written_as_the_template_writes_it_reads_back_as_made(path):
    fixture = json.loads(path.read_text())
    context = _TaggedContext(fixture)
    template = ResolvedTemplate(context)
    template.ensure_tools_usable()
    rendered = context.render_with_model_template(
        [{"role": "user", "content": "Weather?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call00000", "type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "Berlin"}}}]}],
        tools=[_TOOL.as_declaration()], reasoning=False)
    opener, closer = (context.token_text(t) for t in template.tool_tokens)
    # The last call: a declaration may show an example call first.
    body = rendered.rsplit(opener, 1)[1].split(closer, 1)[0]
    assert parse_calls(body, template.call_body, template.call_lead_in) == [
        ToolCall("get_weather", {"city": "Berlin"})]


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
