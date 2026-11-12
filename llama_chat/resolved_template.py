"""What one model's own chat template supports, read from its renders once and checked.

Everything here is settled while a wrapper is being built: the tags around a
turn and a thought, where a reply ends, and the four steps tool calling needs.
An instance is therefore a template known to work. Nothing here touches a
conversation.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping

from .cache_context import MODEL_READER, RENDERER_VARIABLES, TOOL_USE_TEMPLATE, ensure_answers
from .call_format import (extract_call_body, extract_call_close_text, extract_call_separator,
                          extract_named_twice_opening, extract_tool_fragments)
from .call_parsing import parse_calls
from .messages import call_id, calling_message, result_message, with_thought_as
from .model_template import names_read
from .template import (CallBoundary, CallBody, Reasoning, TemplateFragments,
                       ToolCallFragments, common_prefix_length, shared_tags_length)
from .template_extract import (ESCAPABLE, extract_content_rewrites, extract_fragments,
                               extract_reasoning, extract_removed_thought, refusing)
from .tokenizing import first_special_token, split_at_tags, tag_texts
from .tools import ToolCall, ToolSpec

# Where the prompt and a tool result go in a render. Neither contains the
# other, so a split on either lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
_NEXT = "NEXTMSG"
_SYSTEM = "SYSTEMMSG"
# A reply and a user turn after the first, so a template that declares tools
# before the newest user turn shows it after the first turn, not ahead of it.
_EXCHANGE = ({"role": "assistant", "content": "REPLYMSG"},
             {"role": "user", "content": "LATERMSG"})
# A question, a past reply to it, and the next question.
_PAST_REPLY = ({"role": "user", "content": _PROMPT}, {"role": "assistant", "content": _NEXT},
               *_EXCHANGE[1:])

# The stand-in a probe declares and answers a call for. What a template renders
# around a declaration or around a result does not depend on which function it
# is, so one probe settles both for every tool. It never reaches a model.
_PROBE_FUNCTION = "probe"
_PROBE_TOOL = ToolSpec(name=_PROBE_FUNCTION, description="probe",
                       parameters={"type": "object", "properties": {}})

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
        self._context = context
        self._addon_call = dict(addon_call or {})
        self._template_name = template_name
        self._model_render = with_reasoning_on(
            functools.partial(context.render_with_model_template, template_name=template_name),
            dict(reasoning_on or {}))
        self._probe_render = refusing(self._model_render)
        self._variables: dict = {}      # template variables every render passes
        self._tag_texts = tag_texts(context)
        self._fragments = self._resolve_fragments()
        self._has_system_turn = self._resolve_has_system_turn()
        self._validate_template()
        turn_end_token = self._resolve_turn_end_token()
        # None where this model cannot call tools at all.
        self._tool_calls = extract_tool_fragments(self._render, self._fragments)
        (self._boundary, self._call_lead_in, self._call_from_file,
         self._turn_end_token) = self._resolve_calls(turn_end_token)
        self._call_body = self._resolve_call_body()
        self._declares_in_newest_turn = self._resolve_declares_in_newest_turn()
        if not self._declares_in_newest_turn:
            self._variables = self._resolve_declaration_variables()
        self._can_declare_tools = self._resolve_can_declare_tools()
        self._can_write_results = self._resolve_can_write_results()
        self._reasoning = self._resolve_reasoning()
        if self._reasoning is not None:
            tags = (self._reasoning.thought_open.strip(), self._reasoning.thought_close.strip())
            self._boundary = dataclasses.replace(self._boundary,
                                                 thought_tags=tuple(t for t in tags if t))
        if reasoning_on:
            self._ensure_reasoning_on_changes_the_prompt()
        self._check_what_generation_reads()
        self._n_closing_tokens = len(context.tokenize(self._fragments.assistant_suffix,
                                                      add_special=False))

    def render(self, messages: list[dict], tools: list[dict] | None = None, *,
               generation_prompt: bool = False, reasoning: bool | None = None) -> str | None:
        """``messages`` as the model's template writes them, ``None`` where it refuses that shape.

        A reply's thought, held under ``reasoning_content``, is handed to the
        template in the form it reads a past reply's thought in.
        """
        if self._reasoning is not None and self._reasoning.past_keeps_thought:
            messages = [with_thought_as(message, self._reasoning.past_thought_form)
                        for message in messages]
        return self._model_render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=reasoning, variables=self._variables)

    def _render(self, messages, tools=None, *, variables=None, **options):
        """A probe's render of ``messages``, ``None`` where the template refuses them or raises."""
        return self._probe_render(messages, tools, **options,
                                  variables={**self._variables, **(variables or {})})

    def _ensure_reasoning_on_changes_the_prompt(self) -> None:
        """Refuse an add-on file's reasoning on that writes the prompt as reasoning off does.

        Raises:
            ValueError: If it does.
        """
        question = [{"role": "user", "content": _PROMPT}]
        on, off = (self._render(question, generation_prompt=True, reasoning=reasoning)
                   for reasoning in (True, False))
        if on is None or on == off:
            raise ValueError("the add-on file's reasoning_on writes the prompt as reasoning "
                             "off does; name the template variables that turn this "
                             "model's reasoning on")

    # ----- what the wrapper reads ----------------------------------------------
    @property
    def fragments(self) -> TemplateFragments:
        """The tags a turn is wrapped in, and how a message's text is written."""
        return self._fragments

    @property
    def turn_end_token(self) -> int | None:
        """The tag a reply's closing begins with, which ends a reply where sampled.

        ``None`` where the closing does not begin with one of the model's tags.
        """
        return self._turn_end_token

    @property
    def reasoning(self) -> Reasoning | None:
        """How this model marks a thought and reasons, or ``None`` where it marks none."""
        return self._reasoning

    @property
    def can_reason(self) -> bool:
        """Whether a conversation can be asked to reason, as the template does it."""
        return self._reasoning is not None

    @property
    def can_turn_off_reasoning(self) -> bool:
        """Whether a reply asked for without reasoning is generated without it, as the template's own off writes it.

        True where the template marks no thought.
        """
        return self._reasoning is None or self._reasoning.can_turn_off

    @property
    def has_system_turn(self) -> bool:
        """Whether a system prompt can be kept: no later question moves where the template writes it."""
        return self._has_system_turn

    def _resolve_has_system_turn(self) -> bool:
        """Whether a system prompt lies in the opening of two renders that differ only in the oldest question."""
        def render(oldest):
            return self._render([{"role": "system", "content": _SYSTEM},
                                 {"role": "user", "content": oldest}, *_EXCHANGE[:1],
                                 {"role": "user", "content": _NEXT}],
                                generation_prompt=True, reasoning=False)
        first, second = render(_PROMPT), render(_RESULT)
        if first is None or second is None:
            return False
        return _SYSTEM in first[:common_prefix_length(first, second)]

    def ensure_can_reason(self) -> None:
        """Refuse to have this model reason where its template marks no thought.

        Raises:
            ValueError: If this model cannot reason.
        """
        if not self.can_reason:
            raise ValueError("this model's template marks no thought that could be "
                             "told from the answer, so it cannot be asked for reasoning")

    def generation_prompt(self, reasoning: bool = False) -> str:
        """The generation prompt a reply is sampled after, in the reasoning mode given.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        if not reasoning:
            return self._fragments.generation_prompt_without_reasoning
        self.ensure_can_reason()
        return self._reasoning.generation_prompt

    def answer_opening(self, reasoning: bool = False) -> str:
        """What a past reply's tag holds that the generation prompt leaves to the model.

        A template that routes replies through channels ends its generation
        prompt before the channel's own header, which the model writes and
        the answer does not include. One whose prompt ends in an empty thought
        leaves the tag's own opening of the answer to the model after it,
        which is then what the tag holds beyond what it shares with the
        prompt.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        prompt, tag = self.generation_prompt(reasoning), self._fragments.assistant_prefix
        if tag.startswith(prompt):
            return tag[len(prompt):]
        shared = shared_tags_length(tag, prompt)
        return tag[shared:] if self._closes_empty_thought(prompt[shared:], tag[shared:]) else ""

    def _closes_empty_thought(self, written: str, opening: str) -> bool:
        """Whether ``written`` is a thought with nothing in it, closed, and ``opening`` the rest of its closing, whitespace aside."""
        if self._reasoning is None:
            return False
        def unspaced(text: str) -> str:
            return "".join(text.split())

        thought_open = unspaced(self._reasoning.thought_open)
        written = unspaced(written)
        return (written.startswith(thought_open) and len(written) > len(thought_open)
                and written + unspaced(opening)
                == thought_open + unspaced(self._reasoning.thought_close))

    @property
    def n_closing_tokens(self) -> int:
        """How many tokens the closing after a reply's text takes."""
        return self._n_closing_tokens

    @property
    def can_call_tools(self) -> bool:
        """Whether all four steps tool calling needs came out of this template.

        The call tags, the declarations that tell the model a tool exists, the
        argument spelling, and the block a result goes in.
        :meth:`ensure_tools_usable` says which one is missing.
        """
        return (self._calls_marked and self._can_declare_tools
                and self._call_body is not None and self._can_write_results)

    @property
    def _calls_marked(self) -> bool:
        """Whether a call can be told from a reply, by its tags or by its texts."""
        return self._boundary.tool_tokens is not None or self._boundary.call_texts is not None

    @property
    def call_close_text(self) -> str:
        """The text a call ends with, empty where this model marks no call."""
        if self._boundary.call_texts is not None:
            return self._boundary.call_texts[1]
        if self._boundary.tool_tokens is not None:
            return self._context.token_text(self._boundary.tool_tokens[1])
        return ""

    @property
    def declares_before_newest_turn(self) -> bool:
        """Whether the template writes tool declarations right before the newest user turn, so that they move with each question."""
        return self._declares_in_newest_turn

    def ensure_tools_usable(self) -> None:
        """Refuse to declare tools this model could never use.

        Declaring tools, and handing over a registry to run them, are the same
        promise, so every step between the promise and a result the model can
        read is checked before the promise is made rather than at the first
        call, where the reply would already be running. All four were settled
        at construction, which is what :attr:`can_call_tools` reports; this only
        says which one is missing.

        Raises:
            ValueError: If the call tags, the declarations, the argument
                spelling or the result block cannot be had from this model.
        """
        if not self._calls_marked and self._tool_calls is not None:
            raise ValueError(
                "this model writes a tool call as plain text, without a tag "
                "token of its own, so a call could not be told from a reply; "
                "its add-on file can say how it writes one")
        if not self._calls_marked:
            raise ValueError(
                "this model's template describes no tool call, so a tool "
                "could never be reached")
        if not self._can_declare_tools:
            raise ValueError(
                "this model's template declares no tools, so the model would "
                "never be told the tool exists")
        if self._call_body is None:
            raise ValueError(
                "this model's call format could not be recovered from its "
                "template, so a call would be detected but never read")
        if not self._can_write_results:
            raise ValueError(
                "this model's template renders no tool result, so a call could "
                "be made but never answered")

    def _resolve_fragments(self) -> TemplateFragments:
        """Recover the template fragments from the model's own chat template, which a context that reports no template text ships none of."""
        has_text = self._context.chat_template_text(self._template_name) is not None
        fragments = extract_fragments(self._model_render) if has_text else None
        if fragments is None:
            raise ValueError(
                "the model ships no chat template, which this library writes every "
                "conversation with"
            )
        rewrites = extract_content_rewrites(self._model_render, (*self._tag_texts, *ESCAPABLE))
        return dataclasses.replace(fragments, rewrites=rewrites) if rewrites else fragments

    def _validate_template(self) -> None:
        """Ensure the turn-terminator opens with one of this model's own tags.

        The model ends a reply by generating that tag; a terminator that opens
        with plain text never lets it stop, so it would run to the token cap
        every generation. Where the terminator is empty, the reply ends where
        the model draws the tag that opens the next turn, which must therefore
        end a generation.
        """
        terminator = self._fragments.assistant_suffix.strip()
        if not terminator:
            self._ensure_next_turn_ends_a_reply()
            return
        pieces = split_at_tags(self._context, terminator)
        if not (pieces and pieces[0][1]):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not open with one of the model's own tags, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed, so the model cannot be used."
            )

    def _ensure_next_turn_ends_a_reply(self) -> None:
        """Ensure the turn after a reply opens with a tag that ends a generation.

        Raises:
            ValueError: If it opens with plain text, or with a tag the model
                generates without ending.
        """
        past = self._render(list(_PAST_REPLY), reasoning=False)
        after = past.split(_NEXT, 1)[-1].lstrip() if past is not None else ""
        first = self._context.tokenize(after, add_special=False)[:1] if after else []
        if first and self._context.is_special(first[0]):
            if self._context.ends_generation(first[0]):
                return
            opens_with = (f"the tag {self._context.token_text(first[0])!r}, which does not "
                          "end a generation for this model")
        else:
            opens_with = "no tag of the model's own"
        raise ValueError(
            "this model's template writes nothing after a reply, and the turn after "
            f"one opens with {opens_with}, so a reply would never stop; the model "
            "cannot be used")

    def _resolve_turn_end_token(self) -> int | None:
        """The first token of the turn-terminator, where it is one of the model's tags."""
        terminator = self._fragments.assistant_suffix.strip()
        if not terminator:
            return None
        first = first_special_token(self._context, terminator)
        tokens = self._context.tokenize(terminator, add_special=False)
        return first if tokens and tokens[0] == first else None

    def _check_what_generation_reads(self) -> None:
        """Ensure the tags read around a generation are the ones the template writes.

        In each reasoning mode: the generation prompt ends the render of a
        question, and the closing after a reply follows a past reply's text.
        What else a conversation holds is the render itself.

        Raises:
            ValueError: If the template writes either otherwise.
        """
        question = [{"role": "user", "content": _PROMPT}]
        closing = self._fragments.assistant_suffix.strip()
        for reasoning in (False, True) if self.can_reason else (False,):
            prompted = self._render(question, generation_prompt=True, reasoning=reasoning)
            past = self._render(list(_PAST_REPLY), reasoning=reasoning)
            if prompted is None or past is None or not prompted.endswith(
                    self.generation_prompt(reasoning)) or not past.split(_NEXT, 1)[-1].lstrip() \
                    .startswith(closing):
                raise ValueError(
                    "this model's template writes its generation prompt or the closing "
                    "after a reply otherwise than as its own probes read them, so a "
                    "reply's start and end could not be told; the model cannot be used")

    def _resolve_reasoning(self) -> Reasoning | None:
        """What turns this model's reasoning on, recovered from its template."""
        render = self._render
        found = extract_reasoning(render, self._fragments, self._tool_calls, self._tag_texts)
        if found is None:
            found = extract_removed_thought(render, self._fragments, self._tag_texts)
        return found

    def _resolve_tool_tokens(self) -> tuple[tuple[int, int] | None, str]:
        """This model's call tags as tokens, and the text it writes before a name."""
        fragments = self._tool_calls
        if fragments is None:
            return None, ""
        opener = first_special_token(self._context, fragments.call_open)
        closer = self._call_closer(fragments)
        if opener is None or closer is None:
            return None, ""
        tag = self._context.token_text(opener)
        reply = self._fragments
        if any(tag in text for text in (reply.assistant_prefix, reply.assistant_suffix,
                                        reply.generation_prompt)):
            return None, ""         # it marks every reply, not a call
        # Text the template writes before the tag belongs to the reply, not to the call.
        lead_in = fragments.call_open.partition(tag)[2]
        return (opener, closer), lead_in

    def _resolve_calls(self, turn_end_token: int | None):
        """How a generated call is marked, in one step, as nothing later changes it.

        Returns:
            The call boundary, the text a call writes before its function's
            name, whether the add-on file gave the call's texts, and the token
            that ends a reply, which a reply ended by text has none of.
        """
        tool_tokens, lead_in = self._resolve_tool_tokens()
        call_texts, from_file = (None, False) if tool_tokens is not None else self._resolve_call_texts()
        opens_as_text = (call_texts is not None
                         and first_special_token(self._context, call_texts[0]) is None)
        if call_texts is not None:
            lead_in = self._text_after_last_tag(call_texts[0])
        turn_end_text = self._resolve_turn_end_text(call_texts)
        boundary = CallBoundary(tool_tokens, call_texts, opens_as_text, turn_end_text)
        return boundary, lead_in, from_file, None if turn_end_text else turn_end_token

    def _resolve_call_texts(self) -> tuple[tuple[str, str] | None, bool]:
        """The texts around a call, from the template, else from the model's add-on file.

        Returns:
            The texts, and whether they came from the add-on file.
        """
        found = self._template_call_texts()
        if found is not None:
            return found, False
        call = self._addon_call
        if not call or self._tool_calls is None:
            return None, False
        rendered = self._render([{"role": "user", "content": _PROMPT},
                                 calling_message("", [ToolCall(_PROBE_FUNCTION, {})], 0)])
        if rendered is None or call["closing"] not in rendered:
            raise ValueError("the model's add-on file closes a call with text its "
                             "template never writes after one")
        return (call["opening"], call["closing"]), True

    def _template_call_texts(self) -> tuple[str, str] | None:
        """The texts around a call, for a template whose call tags also mark every reply."""
        if self._tool_calls is None:
            return None
        call_open = self._tool_calls.call_open
        if first_special_token(self._context, call_open) is None:
            # Named in the header first: the call opens where the name is written again.
            opening = extract_named_twice_opening(self._render, call_open)
            if opening:
                close = extract_call_close_text(self._render, opening)
                return (opening, close) if close else None
            if self._opens_in_reply_header(call_open):
                return None
        close = extract_call_close_text(self._render, call_open)
        return (call_open, close) if close else None

    def _text_after_last_tag(self, text: str) -> str:
        """Where a call's own text begins in its opening: after the opening's last tag."""
        tags = self._tags_in(text)
        last = self._context.token_text(tags[-1]) if tags else ""
        return text.rpartition(last)[2] if last else text

    def call_boundary(self, recognizes_calls: bool = True, withhold: bool = False) -> CallBoundary:
        """Where a generated call opens and closes and where a reply ends, as generation reads it.

        Without ``recognizes_calls`` no call is recognized, so the reply is certain
        to close. With ``withhold`` a call's opening token is also never
        drawn; a call that opens with plain text cannot be kept from the model
        that way, and stays text.
        """
        if recognizes_calls:
            return self._boundary
        tokens = self._boundary.tool_tokens
        return CallBoundary(opens_as_text=self._boundary.opens_as_text,
                            turn_end_text=self._boundary.turn_end_text,
                            withheld_tokens=(tokens[0],) if tokens and withhold else ())

    def read_calls(self, generated_call_text: str) -> list[ToolCall]:
        """The calls in what a model generated between a call's opening and its closing.

        Raises:
            ValueError: If the text is not a call this model's format reads.
        """
        text = generated_call_text
        if self._boundary.call_texts is not None:
            text = self._call_lead_in + text    # the opening text holds the lead-in's start
        return parse_calls(text, self._call_body, self._call_lead_in)

    def _opens_in_reply_header(self, call_open: str) -> bool:
        """Whether a plain-text opening begins with text of a reply's own opening.

        A model may write such a header in another order than its template
        does, so the template cannot tell how a generated call begins.
        """
        header, text = self._fragments.assistant_prefix, ""
        for tag in self._tags_in(header):
            text_before, _, header = header.partition(self._context.token_text(tag))
            text += " " + text_before
        words = (text + " " + header).split()
        return any(call_open.lstrip().startswith(word) for word in words)

    def _resolve_turn_end_text(self, call_texts: tuple[str, str] | None) -> str | None:
        """Where a reply ends, where a call opens with the start of a reply's closing."""
        if call_texts is None:
            return None
        closing, call_open = self._fragments.assistant_suffix, call_texts[0]
        shared = shared_tags_length(closing, call_open)
        if not shared:
            return None
        tag = first_special_token(self._context, closing[shared:])
        text = self._context.token_text(tag) if tag is not None else ""
        if not text or not closing[shared:].startswith(text):
            return None
        return closing[:shared] + text

    def _call_closer(self, fragments: ToolCallFragments) -> int | None:
        """The tag that ends a call.

        The last tag between two calls of one turn, where the template writes
        one there that ``call_close`` also holds: a call's ending may close its
        last argument with a tag of its own first. Otherwise the first tag of
        ``call_close``.
        """
        separator = extract_call_separator(self._render,
                                           fragments.call_open)
        tags = self._tags_in(separator or "")
        if tags and self._context.token_text(tags[-1]) in fragments.call_close:
            return tags[-1]
        return first_special_token(self._context, fragments.call_close)

    def _tags_in(self, text: str) -> list[int]:
        """The model's own tags in ``text``, in order."""
        found = (first_special_token(self._context, piece)
                 for piece, is_tag in split_at_tags(self._context, text) or () if is_tag)
        return [tag for tag in found if tag is not None]

    def _resolve_call_body(self) -> CallBody | None:
        """This model's argument spelling, or ``None`` if it cannot be read.

        A call stays detectable without it, and only its contents cannot be
        read.
        """
        if not self._calls_marked or self._tool_calls is None:
            return None
        if self._call_from_file:
            # The name ends where the file says; the arguments are read as written.
            return CallBody(name_close=self._addon_call["name_close"], key_open="",
                            key_close="", pair_sep="", body_close="", empty_close="")
        call_open = (self._boundary.call_texts[0] if self._boundary.call_texts
                     else self._tool_calls.call_open)
        return extract_call_body(self._render, call_open, self.call_close_text)

    def _resolve_can_declare_tools(self) -> bool:
        """Whether this model's template renders tool declarations at all.

        Probed with a stand-in tool, for the same reason the result block is:
        a template renders declarations or it does not, and does not branch on
        which tools it was handed.
        """
        if not self._calls_marked:
            return False
        return self._declares_in_newest_turn or self._declares_in_system_turn()

    def _resolve_declaration_variables(self) -> dict:
        """A template variable that puts tool declarations into the system turn, where needed.

        One the template reads qualifies where setting it changes nothing in a
        conversation without tools; empty where the system turn declares them
        already, or no variable makes it.
        """
        text = self._context.chat_template_text(self._template_name)
        if not self._calls_marked or text is None or self._declares_in_system_turn():
            return {}
        conversation = [{"role": "system", "content": _PROMPT}, {"role": "user", "content": _NEXT}]
        plain = self._render(conversation)
        names = names_read(text)
        for name in sorted(names - RENDERER_VARIABLES):
            for value in (False, True):
                self._variables = {}
                if self._render(conversation, variables={name: value}) != plain:
                    continue
                self._variables = {name: value}
                if self._declares_in_system_turn():
                    return {name: value}
        self._variables = {}
        return {}

    def _resolve_declares_in_newest_turn(self) -> bool:
        """Whether the template declares tools right before the newest user turn alone.

        Probed with a stand-in tool's declaration, which must read the same in a
        first turn as in a later one.
        """
        if not self._calls_marked or self._declares_in_system_turn():
            return False
        tools = [_PROBE_TOOL.as_declaration()]
        found = set()
        for messages in ([{"role": "user", "content": _NEXT}],
                         [{"role": "user", "content": _PROMPT}, *_EXCHANGE[:1],
                          {"role": "user", "content": _NEXT}]):
            rendered = self._render(messages, tools=tools)
            plain = self._render(messages)
            newest = self._fragments.user_prefix + _NEXT
            if rendered is None or plain is None or newest not in rendered \
                    or newest not in plain:
                return False
            declared_head, _, declared_tail = rendered.rpartition(newest)
            plain_head, _, plain_tail = plain.rpartition(newest)
            if declared_tail != plain_tail or not declared_head.startswith(plain_head) \
                    or declared_head == plain_head:
                return False
            found.add(declared_head[len(plain_head):])
        return len(found) == 1

    def _resolve_can_write_results(self) -> bool:
        """Whether this model's template renders a block to answer a call in.

        Probed once, with a stand-in name: a template renders a result block or
        it does not, and does not branch on which function is being answered.
        """
        if not self._calls_marked:
            return False
        rendered = self._render([
            {"role": "user", "content": _PROMPT},
            calling_message("", [ToolCall(_PROBE_FUNCTION, {})], 0),
            result_message(_PROBE_FUNCTION, call_id(0), _RESULT)],
            generation_prompt=True, reasoning=False)
        return rendered is not None and _RESULT in rendered

    def _system_turn(self, tools: list[dict] | None) -> str | None:
        """The system turn rendered ahead of a user turn, which some templates require.

        Cut where the user turn begins, so whatever the template writes between
        the two belongs to the system turn.
        """
        rendered = self._render(
            [{"role": "system", "content": _PROMPT},
             {"role": "user", "content": _NEXT}, *_EXCHANGE], tools=tools)
        opened = self._fragments.user_prefix + _NEXT
        if rendered is None or opened not in rendered:
            return None
        return rendered.split(opened, 1)[0]

    def _declares_in_system_turn(self) -> bool:
        """Whether the system turn the template renders holds a stand-in tool's declaration."""
        rendered = self._system_turn([_PROBE_TOOL.as_declaration()])
        plain = self._system_turn(None)
        return (rendered is not None and plain is not None and _PROMPT in rendered
                and _PROMPT in plain and rendered != plain)


def with_reasoning_on(render, reasoning_on: Mapping[str, object]):
    """``render``, with ``reasoning_on``'s variables over the template's switch wherever it renders with reasoning on."""
    if not reasoning_on:
        return render

    def rendered(messages, tools=None, *, reasoning=None, variables=None, **options):
        if reasoning:
            variables = {**reasoning_on, **(variables or {})}
        return render(messages, tools, reasoning=reasoning, variables=variables, **options)
    return rendered


class ResolvedTemplates:
    """This model's default chat template and, where it ships one, its template for a conversation that declares tools, each recovered and proven usable.

    Args:
        context: The cache context to read the model through.
        addon_call: How the model's add-on file says a call is written, for
            both templates; see :class:`ResolvedTemplate`.
        reasoning_on: What the add-on file says turns reasoning on, for both
            templates; see :class:`ResolvedTemplate`.

    Raises:
        TypeError: If ``context`` does not answer :class:`~llama_chat.cache_context.ModelReader`.
        ValueError: If either template cannot be recovered, or fails validation
            against the model.
    """

    def __init__(self, context, addon_call: Mapping[str, str] | None = None,
                 reasoning_on: Mapping[str, object] | None = None) -> None:
        self._default = ResolvedTemplate(context, addon_call, reasoning_on=reasoning_on)
        self._for_tools = (ResolvedTemplate(context, addon_call, template_name=TOOL_USE_TEMPLATE,
                                            reasoning_on=reasoning_on)
                           if context.chat_template_text(TOOL_USE_TEMPLATE) is not None
                           else self._default)

    def for_conversation(self, declares_tools: bool) -> ResolvedTemplate:
        """The template a conversation is written with: the one for tools where it declares tools."""
        return self._for_tools if declares_tools else self._default
