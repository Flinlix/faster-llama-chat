"""What one model's own chat template supports, read from its renders once and checked.

Everything here is settled while a wrapper is being built: the tags around a
turn and a thought, where a reply ends, and the four steps tool calling needs.
An instance is therefore a template known to work. Nothing here touches a
conversation.
"""

from __future__ import annotations

import dataclasses

from .cache_context import MODEL_READER, ensure_answers
from .messages import call_id, calling_message, result_message
from .template import (CallBoundary, CallBody, Reasoning, TemplateFormatter,
                       TemplateFragments, ToolCallFragments, shared_tags_length)
from .template_extract import (ESCAPABLE, extract_call_body, extract_call_close_text,
                               extract_content_rewrites,
                               extract_call_separator, extract_fragments,
                               extract_named_twice_opening, extract_reasoning, extract_removed_thought,
                               extract_tool_fragments, names_read, refusing)
from .tools import ToolCall, ToolSpec

# Where the prompt and a tool result go in a render. Neither contains the
# other, so a split on either lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
_NEXT = "NEXTMSG"
# A reply and a user turn after the first, so a template that declares tools
# before the newest user turn shows it after the first turn, not ahead of it.
_EXCHANGE = ({"role": "assistant", "content": "REPLYMSG"},
             {"role": "user", "content": "LATERMSG"})

# The stand-in a probe declares and answers a call for. What a template renders
# around a declaration or around a result does not depend on which function it
# is, so one probe settles both for every tool. It never reaches a model.
_PROBE_FUNCTION = "probe"
_PROBE_TOOL = ToolSpec(name=_PROBE_FUNCTION, description="probe",
                       parameters={"type": "object", "properties": {}})

class ResolvedTemplate:
    """This model's chat template, recovered or supplied, and proven usable.

    Args:
        context: The cache context to read the model through.
        tool_calls: Explicit call tags, and optionally the argument spelling
            inside them, for a model whose template describes neither. Whatever
            it leaves out is recovered from the model.

    Raises:
        TypeError: If ``context`` does not answer :class:`~llama_chat.cache_context.ModelReader`.
        ValueError: If the template cannot be recovered, or if it fails
            validation against the model.
    """

    def __init__(self, context, *, tool_calls: ToolCallFragments | None = None) -> None:
        ensure_answers(context, MODEL_READER)
        self._context = context
        self._probe_render = refusing(context.render_with_model_template)
        self._variables: dict = {}      # template variables every render passes
        self._fragments = self._resolve_fragments()
        self._validate_template()
        turn_end_token = self._resolve_turn_end_token()
        # None where this model cannot call tools at all.
        self._tool_calls = (tool_calls if tool_calls is not None
                            else extract_tool_fragments(self._render, self._fragments))
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
        self._check_what_generation_reads()

    def render(self, messages: list[dict], tools: list[dict] | None = None, *,
               generation_prompt: bool = False, reasoning: bool | None = None) -> str | None:
        """``messages`` as the model's template writes them, ``None`` where it refuses that shape."""
        return self._context.render_with_model_template(
            messages, tools, generation_prompt=generation_prompt, reasoning=reasoning,
            **self._variables)

    def _render(self, messages, tools=None, **options):
        """A probe's render of ``messages``, ``None`` where the template refuses them or raises."""
        return self._probe_render(messages, tools, **self._variables, **options)

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
    def has_system_turn(self) -> bool:
        """Whether a system prompt has a turn of its own, rather than the user tags'."""
        frags = self._fragments
        return (frags.system_prefix, frags.system_suffix) != \
            (frags.user_prefix, frags.user_suffix)

    def formatter(self, reasoning: bool = False) -> TemplateFormatter:
        """What renders one message, and what is decoded around sampling.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        if reasoning and not self.can_reason:
            raise ValueError("this model's template marks no thought that could be "
                             "told from the answer, so it cannot be asked for reasoning")
        return TemplateFormatter(self._fragments,
                                 self._reasoning if reasoning else None)

    @property
    def tool_tokens(self) -> tuple[int, int] | None:
        """The ``(open, close)`` tokens bracketing a call, or ``None``.

        ``None`` also where :attr:`call_texts` marks a call instead.
        """
        return self._boundary.tool_tokens

    @property
    def call_texts(self) -> tuple[str, str] | None:
        """The texts that open and end a call, where a call has no tag of its own.

        ``None`` where :attr:`tool_tokens` marks a call, or nothing does.
        """
        return self._boundary.call_texts

    @property
    def turn_end_text(self) -> str | None:
        """Where a reply ends, where a call goes on from the start of its closing.

        Such a reply ends once the text has passed the last point a call could
        still begin from; ``None`` where :attr:`turn_end_token` says it.
        """
        return self._boundary.turn_end_text

    @property
    def call_lead_in(self) -> str:
        """Fixed text this model writes between its call tag and the name."""
        return self._call_lead_in

    @property
    def call_body(self) -> CallBody | None:
        """How a call spells its arguments, or ``None`` if it cannot be read."""
        return self._call_body

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
                "token of its own, so a call could not be told from a reply")
        if not self._calls_marked:
            raise ValueError(
                "this model's template describes no tool call, so a tool "
                "could never be reached; supply the tags with "
                "ChatWrapper(tool_calls=...)")
        if not self._can_declare_tools:
            raise ValueError(
                "this model's template declares no tools, so the model would "
                "never be told the tool exists")
        if self._call_body is None:
            raise ValueError(
                "this model's call format could not be recovered from its "
                "template, so a call would be detected but never read; supply "
                "it as the body of ChatWrapper(tool_calls=...)")
        if not self._can_write_results:
            raise ValueError(
                "this model's template renders no tool result, so a call could "
                "be made but never answered")

    def _resolve_fragments(self) -> TemplateFragments:
        """Recover the template fragments from the model's own chat template."""
        fragments = extract_fragments(self._context.render_with_model_template)
        if fragments is None:
            raise ValueError(
                "the model ships no chat template, which this library writes every "
                "conversation with"
            )
        rewrites = extract_content_rewrites(self._context.render_with_model_template,
                                            (*self._context.tags(), *ESCAPABLE))
        return dataclasses.replace(fragments, rewrites=rewrites) if rewrites else fragments

    def _validate_template(self) -> None:
        """Ensure the turn-terminator opens with one of this model's own tags.

        The model ends a reply by generating that tag; a terminator that opens
        with plain text never lets it stop, so it would run to the token cap
        every generation.
        """
        terminator = self._fragments.assistant_suffix.strip()
        pieces = self._context.split_at_tags(terminator) if terminator else None
        if terminator and not (pieces and pieces[0][1]):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not open with one of the model's own tags, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed, so the model cannot be used."
            )

    def _resolve_turn_end_token(self) -> int | None:
        """The first token of the turn-terminator, where it is one of the model's tags."""
        terminator = self._fragments.assistant_suffix.strip()
        if not terminator:
            return None
        first = self._context.first_special_token(terminator)
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
        exchange = question + [{"role": "assistant", "content": _NEXT}, *_EXCHANGE[1:]]
        for reasoning in (False, True) if self.can_reason else (False,):
            formatter = self.formatter(reasoning)
            prompted = self._render(question, generation_prompt=True, reasoning=reasoning)
            past = self._render(exchange, reasoning=reasoning)
            closing = formatter.assistant_close().strip()
            if prompted is None or past is None or not prompted.endswith(
                    formatter.generation_prompt()) or not past.split(_NEXT, 1)[-1].lstrip() \
                    .startswith(closing):
                raise ValueError(
                    "this model's template writes its generation prompt or the closing "
                    "after a reply otherwise than as its own probes read them, so a "
                    "reply's start and end could not be told; the model cannot be used")

    def _resolve_reasoning(self) -> Reasoning | None:
        """What turns this model's reasoning on, recovered from its template."""
        render = self._render
        found = extract_reasoning(render, self._fragments, self._tool_calls)
        if found is None:
            found = extract_removed_thought(render, self._fragments, self._context.tags())
        return found

    def _resolve_tool_tokens(self) -> tuple[tuple[int, int] | None, str]:
        """This model's call tags as tokens, and the text it writes before a name."""
        fragments = self._tool_calls
        if fragments is None:
            return None, ""
        opener = self._context.first_special_token(fragments.call_open)
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
                         and self._context.first_special_token(call_texts[0]) is None)
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
        call = self._context.addon_call()
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
        if self._context.first_special_token(call_open) is None:
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

    @property
    def call_opens_as_text(self) -> bool:
        """Whether a call opens with plain text, which counts only after a tag or at the start."""
        return self._boundary.opens_as_text

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
        tag = self._context.first_special_token(closing[shared:])
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
        return self._context.first_special_token(fragments.call_close)

    def _tags_in(self, text: str) -> list[int]:
        """The model's own tags in ``text``, in order."""
        found = (self._context.first_special_token(piece)
                 for piece, is_tag in self._context.split_at_tags(text) or () if is_tag)
        return [tag for tag in found if tag is not None]

    def _resolve_call_body(self) -> CallBody | None:
        """This model's argument spelling, or ``None`` if it cannot be read.

        A supplied one wins; a call stays detectable without either, and only
        its contents cannot be read.
        """
        if not self._calls_marked or self._tool_calls is None:
            return None
        if self._tool_calls.body is not None:
            return self._tool_calls.body
        if self._call_from_file:
            # The name ends where the file says; the arguments are read as written.
            return CallBody(name_close=self._context.addon_call()["name_close"], key_open="",
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
        if not self._calls_marked or self._context.chat_template_text() is None \
                or self._declares_in_system_turn():
            return {}
        conversation = [{"role": "system", "content": _PROMPT}, {"role": "user", "content": _NEXT}]
        plain = self._render(conversation)
        names = names_read(self._context.chat_template_text() or "")
        for name in sorted(names - {"messages", "tools", "add_generation_prompt",
                                    "generation_prompt", "reasoning"}):
            for value in (False, True):
                self._variables = {}
                if self._render(conversation, **{name: value}) != plain:
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
