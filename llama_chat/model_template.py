"""A model's own chat template, rendered by this library from its text, with its reasoning switch applied."""

from __future__ import annotations

import json
from datetime import datetime
from collections.abc import Mapping

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .template_extract import reasoning_switch


# The template variables the renderer sets for every template itself, so that no
# render's ``variables`` holds one of them.
RENDERER_VARIABLES = frozenset({"messages", "tools", "add_generation_prompt", "bos_token",
                                "eos_token", "raise_exception", "strftime_now"})


class _GenerationBlock(jinja2.ext.Extension):
    """Writes the body of a ``{% generation %}`` block as it is, the block itself marking nothing."""

    tags = {"generation"}

    def parse(self, parser):
        parser.stream.skip(1)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


class TemplateRaised(ValueError):
    """The template refused what it was given, through its own ``raise_exception``."""


def _raise_exception(message: str):
    raise TemplateRaised(message)


def _tojson(value, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def _strftime_now(form: str) -> str:
    return datetime.now().strftime(form)


def _environment() -> jinja2.Environment:
    """The environment chat templates are written for: sandboxed, blocks trimmed, ``tojson`` writing JSON as chat templates expect."""
    environment = ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True,
        extensions=[_GenerationBlock, jinja2.ext.loopcontrols])
    environment.filters["tojson"] = _tojson
    return environment


def compiled(text: str) -> jinja2.Template:
    """The template ``text``, compiled.

    Raises:
        ValueError: If it is not valid template code, naming where and why.
    """
    try:
        return _environment().from_string(text)
    except jinja2.TemplateSyntaxError as error:
        raise ValueError(f"the model's chat template is not valid template code, so no "
                         f"conversation could be written with it: line {error.lineno}: "
                         f"{error.message}") from error


def names_read(text: str) -> frozenset[str]:
    """Every name the template ``text`` reads, a variable it gives a default when undefined included."""
    compiled(text)              # refuses text that is no template code, naming where
    parsed = _environment().parse(text)
    return frozenset(node.name for node in parsed.find_all(jinja2.nodes.Name)
                     if node.ctx == "load")


class ModelTemplate:
    """The chat template a model ships, and renders of it.

    Built from the template's text and the text of the model's end-of-sequence
    token, which some templates write.

    Raises:
        ValueError: If ``text`` is not valid template code, naming where and why.
    """

    def __init__(self, text: str, eos_text: str) -> None:
        self.text = text
        self._template = compiled(text)
        self._eos_text = eos_text
        self._switch = reasoning_switch(
            lambda messages, generation_prompt, variables:
            self._rendered(messages, generation_prompt, variables))

    def render(self, messages: list[dict], tools: list[dict] | None = None, *,
               generation_prompt: bool = False, reasoning: bool | None = None,
               variables: Mapping[str, object] | None = None) -> str | None:
        """``messages`` as the template writes them, or ``None`` where it refuses that shape.

        Args:
            generation_prompt: Append the prompt a reply is generated after.
            reasoning: Turn the template's reasoning switch on or off, as
                :func:`~llama_chat.template_extract.reasoning_switch` reads it.
                ``None`` leaves it unset.
            variables: Further template variables, by name, each in place of
                the switch's own value where it names the switch.
        """
        try:
            return self._rendered(messages, generation_prompt,
                                  {"tools": tools, **self._switch(reasoning), **(variables or {})})
        except Exception:
            return None

    def refusal_reason(self, messages: list[dict], tools: list[dict] | None = None, *,
                       generation_prompt: bool = False, reasoning: bool | None = None,
                       variables: Mapping[str, object] | None = None) -> str | None:
        """What the template itself says where it refuses to write ``messages``, through its ``raise_exception``; ``None`` where it says nothing or writes them."""
        try:
            self._rendered(messages, generation_prompt,
                           {"tools": tools, **self._switch(reasoning), **(variables or {})})
        except TemplateRaised as refused:
            return str(refused)
        except Exception:
            return None
        return None

    def _rendered(self, messages: list[dict], generation_prompt: bool, variables: dict) -> str:
        """The render, raising where the template does."""
        return self._template.render(
            messages=messages, eos_token=self._eos_text, bos_token="",
            raise_exception=_raise_exception, add_generation_prompt=generation_prompt,
            strftime_now=_strftime_now, functions=None, function_call=None, tool_choice=None,
            **{"tools": None, **variables})
