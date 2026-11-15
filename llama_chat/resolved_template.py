"""One model's chat template as a conversation uses it: its renders, the prompt a reply follows, and its calls read, with what a load read of it in one record.

See :mod:`llama_chat.template_reading` for what a load reads and checks.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping

from .cache_context import MODEL_READER, TOOL_USE_TEMPLATE, ensure_answers
from .call_parsing import parse_calls
from .errors import UnsupportedError
from .messages import with_thought_as
from .model_template import ModelTemplate
from .reply_text import reply_header
from .template import CallBoundary, shared_tags_length
from .template_reading import TemplateProperties, TemplateReader
from .tools import ToolCall

_log = logging.getLogger(__name__)


class ResolvedTemplate:
    """This model's chat template, recovered and proven usable.

    Args:
        context: The cache context to read the model through.
        addon_call: How the model's add-on file says a call is written, its
            ``opening``, ``name_close`` and ``closing``; empty or ``None``
            where it says nothing.
        template_name: Which of the model's templates: ``None`` for its
            default one, else the name of a further one it ships.
        reasoning_on: The template variables the model's add-on file says
            turn reasoning on, each over the switch the template shows; empty
            or ``None`` where it says nothing.

    Raises:
        TypeError: If ``context`` does not answer :class:`~llama_chat.cache_context.ModelReader`.
        ValueError: If the template cannot be recovered, if it fails
            validation against the model, or if ``reasoning_on`` writes the
            prompt as reasoning off does.
    """

    def __init__(self, context, addon_call: Mapping[str, str] | None = None, *,
                 template_name: str | None = None,
                 reasoning_on: Mapping[str, object] | None = None) -> None:
        ensure_answers(context, MODEL_READER)
        text = context.chat_template_text(template_name)
        if text is None:
            raise ValueError("the model ships no chat template, which this library writes every "
                             "conversation with")
        model = ModelTemplate(text, context.eos_token_text())
        # A template for tools is written for a list of them, so it is given one, empty where none is declared.
        listed = template_name == TOOL_USE_TEMPLATE
        self._model_render = with_reasoning_on(with_tool_list(model.render, listed),
                                               dict(reasoning_on or {}))
        self._model_refusal = with_reasoning_on(with_tool_list(model.refusal_reason, listed),
                                                dict(reasoning_on or {}))
        self.properties: TemplateProperties = TemplateReader(
            context, model, self._model_render, addon_call, reasoning_on).properties
        self._reply_headers: dict[bool, tuple[str, str] | None] = {}

    def render(self, messages: list[dict], tools: list[dict] | None = None, *,
               generation_prompt: bool = False, reasoning: bool | None = None) -> str | None:
        """``messages`` as the model's template writes them, ``None`` where it refuses that shape.

        A reply's thought, held under ``reasoning_content``, is handed to the
        template in the form it reads a past reply's thought in.
        """
        return self._model_render(self._as_template_reads(messages), tools,
                                  generation_prompt=generation_prompt, reasoning=reasoning,
                                  variables=self.properties.declaration_variables)

    def refusal_reason(self, messages: list[dict], tools: list[dict] | None = None, *,
                       generation_prompt: bool = False, reasoning: bool | None = None) -> str | None:
        """What the template itself says where :meth:`render` refuses ``messages``; ``None`` where it says nothing."""
        return self._model_refusal(self._as_template_reads(messages), tools,
                                   generation_prompt=generation_prompt, reasoning=reasoning,
                                   variables=self.properties.declaration_variables)

    def _as_template_reads(self, messages: list[dict]) -> list[dict]:
        """``messages``, each reply's thought in the form the template reads a past reply's thought in."""
        reasoning = self.properties.reasoning
        if reasoning is not None and reasoning.past_keeps_thought:
            return [with_thought_as(message, reasoning.past_thought_form) for message in messages]
        return messages

    def ensure_can_reason(self) -> None:
        """Refuse to have this model reason where its template marks no thought.

        Raises:
            UnsupportedError: If this model cannot reason, naming ``reasoning``.
        """
        if not self.properties.can_reason:
            raise UnsupportedError("reasoning",
                                   "this model's template marks no thought that could be told "
                                   "from the answer, or its prompt with reasoning on closes the "
                                   "thought at once, so it cannot be asked for reasoning; an "
                                   "add-on file's reasoning_on can name what turns it on")

    def generation_prompt(self, reasoning: bool = False) -> str:
        """The generation prompt a reply is sampled after, in the reasoning mode given.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        if not reasoning:
            return self.properties.fragments.generation_prompt_without_reasoning
        self.ensure_can_reason()
        return self.properties.reasoning.generation_prompt

    def reply_header(self, reasoning: bool = False) -> tuple[str, str] | None:
        """The start and end of a header naming a reply's reader, as :func:`~llama_chat.reply_text.reply_header` reads them, worked out once for each reasoning mode."""
        if reasoning not in self._reply_headers:
            properties = self.properties
            self._reply_headers[reasoning] = reply_header(
                self.answer_opening(reasoning), properties.reasoning, properties.tag_texts)
        return self._reply_headers[reasoning]

    def answer_opening(self, reasoning: bool = False) -> str:
        """What a past reply's tag holds that the generation prompt leaves to the model.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        prompt, tag = self.generation_prompt(reasoning), self.properties.fragments.assistant_prefix
        if tag.startswith(prompt):
            return tag[len(prompt):]
        shared = shared_tags_length(tag, prompt, self.properties.tag_texts)
        return tag[shared:] if self._closes_empty_thought(prompt[shared:], tag[shared:]) else ""

    def _closes_empty_thought(self, written: str, opening: str) -> bool:
        """Whether ``written`` is a thought with nothing in it, closed, and ``opening`` the rest of its closing, whitespace aside."""
        reasoning = self.properties.reasoning
        if reasoning is None:
            return False
        def unspaced(text: str) -> str:
            return "".join(text.split())

        thought_open = unspaced(reasoning.thought_open)
        written = unspaced(written)
        return (written.startswith(thought_open) and len(written) > len(thought_open)
                and written + unspaced(opening)
                == thought_open + unspaced(reasoning.thought_close))

    def call_boundary(self, recognizes_calls: bool = True, withhold: bool = False) -> CallBoundary:
        """Where a generated call opens and closes and where a reply ends, as generation reads it.

        Without ``recognizes_calls`` no call is recognized, so the reply is certain
        to close. With ``withhold`` a call's opening token is also never
        drawn; a call that opens with plain text cannot be kept from the model
        that way, so the reply ends where such a call opens.
        """
        boundary = self.properties.boundary
        if recognizes_calls:
            return boundary
        if withhold and boundary.opens_as_text and boundary.call_texts:
            return dataclasses.replace(boundary, tool_tokens=None, ends_at_call=True)
        tokens = boundary.tool_tokens
        return CallBoundary(opens_as_text=boundary.opens_as_text,
                            turn_end_text=boundary.turn_end_text,
                            withheld_tokens=(tokens[0],) if tokens and withhold else ())

    def read_calls(self, generated_call_text: str,
                   parameters: dict[str, dict] | None = None) -> list[ToolCall]:
        """The calls in what a model generated between a call's opening and its closing.

        ``parameters`` is the parameter schema of each declared tool, by its
        name, as :func:`~llama_chat.call_parsing.parse_calls` reads it.

        Raises:
            ValueError: If the text is not a call this model's format reads.
        """
        properties, text = self.properties, generated_call_text
        if properties.call_texts is not None:
            text = properties.call_lead_in + text    # the opening text holds the lead-in's start
        return parse_calls(text, None if properties.json_call_keys else properties.call_body,
                           properties.call_lead_in, parameters, properties.json_call_keys,
                           properties.json_call_id_key)

    def ensure_tools_usable(self, field: str) -> None:
        """Refuse to declare tools this model could never use.

        Args:
            field: The parameter the tools were given in, which a refusal names.

        Raises:
            UnsupportedError: If the call tags, the declarations, the argument
                spelling or the result block cannot be had from this model.
        """
        if not self.properties.calls_marked and self.properties.tool_fragments is not None:
            raise UnsupportedError(
                field,
                "this model writes a tool call as plain text, without a tag "
                "token of its own, so a call could not be told from a reply; "
                "its add-on file can say how it writes one")
        if not self.properties.calls_marked:
            raise UnsupportedError(
                field,
                "this model's template describes no tool call, so a tool "
                "could never be reached")
        if not self.properties.can_declare_tools:
            raise UnsupportedError(
                field,
                "this model's template declares no tools, so the model would "
                "never be told the tool exists")
        if self.properties.call_body is None:
            raise UnsupportedError(
                field,
                "this model's call format could not be recovered from its "
                "template, so a call would be detected but never read")
        if not self.properties.can_write_results:
            raise UnsupportedError(
                field,
                "this model's template renders no tool result, so a call could "
                "be made but never answered")


def with_tool_list(render, listed: bool):
    """``render``, given an empty list of tools in place of none where ``listed``."""
    if not listed:
        return render

    def rendered(messages, tools=None, **options):
        return render(messages, [] if tools is None else tools, **options)
    return rendered


def with_reasoning_on(render, reasoning_on: Mapping[str, object]):
    """``render``, with ``reasoning_on``'s variables over the template's switch wherever it renders with reasoning on."""
    if not reasoning_on:
        return render

    def rendered(messages, tools=None, *, reasoning=None, variables=None, **options):
        if reasoning:
            variables = {**reasoning_on, **(variables or {})}
        return render(messages, tools, reasoning=reasoning, variables=variables, **options)
    return rendered


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What :meth:`~llama_chat.wrapper.ChatWrapper.begin` accepts for one kind of conversation, settled when the model loads.

    Attributes:
        can_call_tools: Tools can be used every step of the way: the template
            describes a call, renders the declarations that tell the model a
            tool exists, has a readable argument spelling, and renders a block
            to answer a call in. Where only the spelling is missing, a call is
            still detected, so a reply can still end with
            ``"unreadable_tool_call"``, but nothing runnable comes out of it.
        can_reason: ``reasoning=True`` is accepted: the template marks a
            thought, and its prompt with reasoning on leaves the thought open.
        can_turn_off_reasoning: A conversation begun without reasoning is
            answered without it, as the template's own off writes it; true
            where the template marks no thought. Where a template that marks
            one has no off, the model reasons in every reply, and the thought
            is kept out of the answer either way.
        can_take_system_prompt: A system prompt is accepted: the template
            writes it where no later question moves it.
    """

    can_call_tools: bool
    can_reason: bool
    can_turn_off_reasoning: bool
    can_take_system_prompt: bool


class ResolvedTemplates:
    """This model's default chat template and, where it ships one, its template for a conversation that declares tools, each recovered and proven usable.

    Args:
        context: The cache context to read the model through.
        addon_call: How the model's add-on file says a call is written, for
            both templates; see :class:`ResolvedTemplate`.
        reasoning_on: What the add-on file says turns reasoning on, for both
            templates; see :class:`ResolvedTemplate`.

    A template for tools that cannot be recovered leaves the model with its
    default template, every conversation with tools refused with the cause.

    Raises:
        TypeError: If ``context`` does not answer :class:`~llama_chat.cache_context.ModelReader`.
        ValueError: If the default template cannot be recovered, or fails
            validation against the model.
    """

    def __init__(self, context, addon_call: Mapping[str, str] | None = None,
                 reasoning_on: Mapping[str, object] | None = None) -> None:
        self._default = ResolvedTemplate(context, addon_call, reasoning_on=reasoning_on)
        self._for_tools = self._default
        self._tools_refused: str | None = None     # why the template for tools is not used
        if context.chat_template_text(TOOL_USE_TEMPLATE) is not None:
            try:
                self._for_tools = ResolvedTemplate(context, addon_call,
                                                   template_name=TOOL_USE_TEMPLATE,
                                                   reasoning_on=reasoning_on)
            except ValueError as refused:
                self._tools_refused = str(refused)
                _log.warning("the model's template for tools cannot be used, so no "
                             "conversation can call tools: %s", refused)

    def for_conversation(self, uses_tools: bool) -> ResolvedTemplate:
        """The template a conversation is written with: the one for tools where it is begun with tools."""
        return self._for_tools if uses_tools else self._default

    def ensure_accepts(self, uses_tools: bool, *, tools_field: str | None = None,
                       reasoning: bool = False, system_prompt_field: str | None = None) -> None:
        """Refuse what a conversation written with :meth:`for_conversation` of ``uses_tools`` cannot take, exactly where :meth:`capabilities` says it does not.

        Args:
            tools_field: The parameter that names tools or holds a call, where one does.
            reasoning: Whether reasoning is asked for.
            system_prompt_field: The parameter a system prompt comes with, where one does.

        Raises:
            UnsupportedError: Naming that parameter, with the cause.
        """
        capabilities, template = self.capabilities(uses_tools), self.for_conversation(uses_tools)
        if tools_field is not None and not capabilities.can_call_tools:
            if self._tools_refused is not None:
                raise UnsupportedError(tools_field, "this model's template for tools could not "
                                       f"be read, so it calls no tools: {self._tools_refused}")
            template.ensure_tools_usable(tools_field)
        if reasoning and not capabilities.can_reason:
            template.ensure_can_reason()
        if system_prompt_field is not None and not capabilities.can_take_system_prompt:
            raise UnsupportedError(
                system_prompt_field,
                "this model's template does not keep a system prompt in place: it writes one "
                "where the questions after it change it, so a system prompt kept for the whole "
                "conversation cannot be; begin without one")

    def capabilities(self, uses_tools: bool) -> Capabilities:
        """What a conversation written with :meth:`for_conversation` of ``uses_tools`` accepts."""
        properties = self.for_conversation(uses_tools).properties
        return Capabilities(properties.can_call_tools and self._tools_refused is None,
                            properties.can_reason, properties.can_turn_off_reasoning,
                            properties.can_take_system_prompt)
