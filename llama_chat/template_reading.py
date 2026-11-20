"""What one model's chat template supports, read from its renders once, while a wrapper is built, and checked.

The tags around a turn and a thought, where a reply ends, and the four steps
tool calling needs. :class:`TemplateReader` makes :class:`TemplateProperties`;
nothing here touches a conversation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from .call_format import (extract_call_body, extract_call_close_text, extract_call_separator,
                          extract_json_call_id_key, extract_json_call_keys, holds_the_probe_id, extract_named_twice_opening,
                          extract_tool_fragments)
from .messages import call_id, calling_message, result_message
from .model_template import RENDERER_VARIABLES, ModelTemplate, names_read
from .render_checks import writes_two_user_turns
from .sequences import common_prefix_length
from .template import (CallBoundary, CallBody, Reasoning, TemplateFragments,
                       ToolCallFragments, shared_tags_length)
from .template_extract import (ESCAPABLE, as_probes_read, extract_content_rewrites,
                               extract_fragments, extract_reasoning, extract_removed_thought,
                               refusing)
from .tokenizing import first_special_token, split_at_tags, tag_texts
from .tools import ToolCall, ToolSpec

# The function a template calls to read the clock, as the renderer provides it.
_CLOCK = "strftime_now"
# Where the prompt and a tool result go in a render. Neither contains the
# other, so a split on either lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
_NEXT = "NEXTMSG"
_TEXT = "TEXTMSG"
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


class TemplateReader:
    """Reads one model's chat template from its renders and checks it against the model; :attr:`properties` holds what it read.

    Args:
        context: The cache context to read the model through.
        model: The template.
        model_render: Renders ``model`` as every render of it is made, the
            add-on file's reasoning switch and a list of tools applied.
        addon_call: How the model's add-on file says a call is written.
        reasoning_on: What the add-on file says turns reasoning on.

    Raises:
        ValueError: If the template cannot be recovered, if it fails
            validation against the model, or if ``reasoning_on`` writes the
            prompt as reasoning off does.
    """

    def __init__(self, context, model: ModelTemplate, model_render,
                 addon_call: Mapping[str, str] | None = None,
                 reasoning_on: Mapping[str, object] | None = None) -> None:
        self._context = context
        self._model = model
        self._model_render = model_render
        self._addon_call = dict(addon_call or {})
        self._probe_render = refusing(model_render)
        self._variables: dict = {}      # template variables every render passes
        self._read_as_probes_for: tuple | None = None   # the variables it was made for
        self._declares_in_system_turn_for: dict[tuple, bool] = {}
        self._tag_texts = tag_texts(context)
        self._ensure_tags_listed()
        self._reads_clock = _CLOCK in model.text
        self._fragments = self._resolve_fragments()
        self._can_take_system_prompt = self._resolve_can_take_system_prompt()
        self._validate_template()
        turn_end_token = self._resolve_turn_end_token()
        # None where this model cannot call tools at all.
        self._tool_calls = extract_tool_fragments(self._read_as_probes, self._fragments,
                                                  self._tag_texts)
        (self._boundary, self._call_lead_in, self._call_from_file,
         self._turn_end_token) = self._resolve_calls(turn_end_token)
        self._call_body = self._resolve_call_body()
        self._json_keys = (None if self._call_from_file or not self._calls_marked
                           else extract_json_call_keys(self._read_as_probes))
        self._json_id_key = (None if self._json_keys is None
                             else extract_json_call_id_key(self._read_as_probes, self._json_keys))
        if self._call_body is not None and self._json_keys is None \
                and holds_the_probe_id(self._call_body):
            self._call_body = None          # an id it writes where it cannot be read
        self._declares_before_newest_turn = self._resolve_declares_before_newest_turn()
        if not self._declares_before_newest_turn:
            self._variables = self._resolve_declaration_variables()
        self._can_declare_tools = self._resolve_can_declare_tools()
        self._can_write_results = self._resolve_can_write_results()
        self._writes_past_calls = self._resolve_writes_past_calls()
        self._reasoning = self._resolve_reasoning()
        # A prompt "with reasoning" that closes the thought already reasons not at all.
        self._can_reason = self._reasoning is not None and not (
            self._reasoning.thought_close.strip()
            and self._reasoning.generation_prompt.rstrip().endswith(
                self._reasoning.thought_close.strip()))
        if self._reasoning is not None:
            tags = (self._reasoning.thought_open.strip(), self._reasoning.thought_close.strip())
            self._boundary = dataclasses.replace(self._boundary,
                                                 thought_tags=tuple(t for t in tags if t))
        if reasoning_on:
            self._ensure_reasoning_on_changes_the_prompt()
        if self._can_reason and not self._fragments.assistant_suffix.strip():
            self._ensure_next_turn_ends_a_reply(reasoning=True)
        self._check_what_generation_reads()
        boundary = self._boundary
        # A turn may end inside a call, which is then closed as the template closes one.
        call_closing = (len(context.tokenize(boundary.call_texts[1], add_special=False,
                                             parse_special=True)) if boundary.call_texts
                        else 1 if boundary.tool_tokens else 0)
        self._n_closing_tokens = max(call_closing, len(context.tokenize(
            self._fragments.assistant_suffix, add_special=False)))
        self.properties = self._read_properties()

    def _read_properties(self) -> TemplateProperties:
        """Every property this load read from the template, in one record."""
        return TemplateProperties(
            reads_clock=self._reads_clock, fragments=self._fragments,
            can_take_system_prompt=self._can_take_system_prompt, turn_end_token=self._turn_end_token,
            content_rewrites=self._fragments.rewrites, tool_fragments=self._tool_calls,
            boundary=self._boundary, call_close_text=self._call_close_text,
            call_lead_in=self._call_lead_in,
            call_body=self._call_body,
            json_call_keys=self._json_keys, json_call_id_key=self._json_id_key,
            declares_before_newest_turn=self._declares_before_newest_turn,
            declaration_variables=dict(self._variables),
            can_declare_tools=self._can_declare_tools, writes_past_calls=self._writes_past_calls,
            can_write_results=self._can_write_results, reasoning=self._reasoning,
            can_reason=self._can_reason, n_closing_tokens=self._n_closing_tokens,
            tag_texts=self._tag_texts,
            writes_reply_as_prompted=self._writes_reply_as_prompted(),
            writes_two_user_turns=writes_two_user_turns(
                lambda messages, prompt: self._model_render(
                    messages, generation_prompt=prompt, variables=self._variables)))

    def _writes_reply_as_prompted(self) -> bool:
        """Whether, with reasoning off, a past reply stands where its question's generation prompt left it, as the text it was generated as: the cache then never goes back behind it."""
        prompted = self._render([{"role": "user", "content": _PROMPT}], generation_prompt=True,
                                reasoning=False)
        past = self._render(list(_PAST_REPLY), reasoning=False)
        return prompted is not None and past is not None and past.startswith(prompted + _NEXT)

    def _generation_prompt(self, reasoning: bool) -> str:
        """The generation prompt a reply is sampled after, in the reasoning mode given."""
        if not reasoning:
            return self._fragments.generation_prompt_without_reasoning
        assert self._reasoning is not None, "a reply reasons only where the template does"
        return self._reasoning.generation_prompt

    @property
    def _read_as_probes(self):
        """:meth:`_render` as the load checks read it, made once for each set of template variables."""
        key = tuple(sorted(self._variables.items()))
        if self._read_as_probes_for != key:
            self._probes_render, self._read_as_probes_for = (
                as_probes_read(self._render, self._tag_texts), key)
        return self._probes_render

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

    def _resolve_can_take_system_prompt(self) -> bool:
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

    @property
    def _calls_marked(self) -> bool:
        """Whether a call can be told from a reply, by its tags or by its texts."""
        return self._boundary.tool_tokens is not None or self._boundary.call_texts is not None

    @property
    def _call_close_text(self) -> str:
        """The text a call ends with, empty where this model marks no call."""
        if self._boundary.call_texts is not None:
            return self._boundary.call_texts[1]
        if self._boundary.tool_tokens is not None:
            return self._context.token_text(self._boundary.tool_tokens[1])
        return ""

    def _ensure_tags_listed(self) -> None:
        """Refuse a model whose tokenizer reads a tag in its template's render that its list of tags leaves out, so that no tag's end is taken for text.

        Raises:
            ValueError: If it does.
        """
        probe = [{"role": "system", "content": _PROMPT}, {"role": "user", "content": _NEXT},
                 {"role": "assistant", "content": _PROMPT}, {"role": "user", "content": _NEXT}]
        rendered = self._probe_render(probe, generation_prompt=True, reasoning=False)
        if rendered is None:
            return
        read = {self._context.token_text(token)
                for token in self._context.tokenize(rendered, add_special=False, parse_special=True)
                if self._context.is_special(token)}
        missing = sorted(text for text in read - set(self._tag_texts) if text)
        if missing:
            raise ValueError(
                f"the model's tokenizer reads {', '.join(map(repr, missing[:3]))} as tags, which "
                "its list of tags leaves out, so where its tags end could not be told; the "
                "cache context must list every tag")

    def _resolve_fragments(self) -> TemplateFragments:
        """Recover the template fragments from the model's own chat template."""
        fragments = extract_fragments(self._model_render, self._tag_texts)
        if fragments is None:
            raise ValueError(
                "this model's chat template writes no turn whose reply can be told from the "
                "question, so a conversation cannot be written with it"
            )
        rewrites = extract_content_rewrites(self._model_render, (*self._tag_texts, *ESCAPABLE),
                                            self._tag_texts)
        return dataclasses.replace(fragments, rewrites=rewrites) if rewrites else fragments

    def _validate_template(self) -> None:
        """Ensure the closing after a reply opens with one of this model's own tags.

        Where the closing is empty, the tag that opens the next turn must end a
        generation instead.
        """
        closing = self._fragments.assistant_suffix.strip()
        if not closing:
            self._ensure_next_turn_ends_a_reply(reasoning=False)
            return
        pieces = split_at_tags(self._context, closing)
        if not (pieces and pieces[0][1]):
            raise ValueError(
                "chat template is not valid for this model: its closing after a reply "
                f"{closing!r} does not open with one of the model's own tags, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed, so the model cannot be used."
            )

    def _ensure_next_turn_ends_a_reply(self, reasoning: bool) -> None:
        """Ensure the turn after a reply opens with a tag that ends a generation, in the reasoning mode given.

        Raises:
            ValueError: If it opens with plain text, or with a tag the model
                generates without ending, or if where a reply ends cannot be
                read.
        """
        mode = "with reasoning on, " if reasoning else ""
        past = self._render(list(_PAST_REPLY), reasoning=reasoning)
        if past is None:
            raise ValueError(
                f"{mode}this model's template writes nothing after a reply and refuses a "
                "question after one, so where a reply ends could not be read; the model "
                "cannot be used")
        after = self._after_a_reply(past, reasoning)
        if after is None:
            raise ValueError(
                f"{mode}this model's template writes nothing after a reply and no past "
                "reply where it stands, so where a reply ends could not be read; the model "
                "cannot be used")
        after = after.lstrip()
        first = self._context.tokenize(after, add_special=False)[:1] if after else []
        if first and self._context.is_special(first[0]):
            if self._context.ends_generation(first[0]):
                return
            opens_with = (f"the tag {self._context.token_text(first[0])!r}, which does not "
                          "end a generation for this model")
        else:
            opens_with = "no tag of the model's own"
        raise ValueError(
            f"{mode}this model's template writes nothing after a reply, and the turn after "
            f"one opens with {opens_with}, so a reply would never stop; the model "
            "cannot be used")

    def _after_a_reply(self, past: str, reasoning: bool) -> str | None:
        """``past``, the render of a question, a reply and a question, from where the reply ends: after its text, or after its turn's opening, as a reply is prompted in either mode, where the template leaves the text out; ``None`` where neither stands."""
        if _NEXT in past:
            return past.split(_NEXT, 1)[1]
        asked = past.find(_PROMPT)
        if asked < 0:
            return None
        for opening in dict.fromkeys((self._generation_prompt(reasoning).strip(),
                                      self._generation_prompt(False).strip())):
            at = past.find(opening, asked + len(_PROMPT)) if opening else -1
            if at >= 0:
                return past[at + len(opening):]
        return None

    def _resolve_turn_end_token(self) -> int | None:
        """The first token of the closing after a reply, where it is one of the model's tags."""
        closing = self._fragments.assistant_suffix.strip()
        if not closing:
            return None
        first = first_special_token(self._context, closing)
        tokens = self._context.tokenize(closing, add_special=False)
        return first if tokens and tokens[0] == first else None

    def _check_what_generation_reads(self) -> None:
        """Ensure the tags read around a generation are the ones the template writes.

        In each reasoning mode: the generation prompt ends the render of a
        question, and the closing after a reply follows a past reply's text.

        Raises:
            ValueError: If the template writes either otherwise.
        """
        question = [{"role": "user", "content": _PROMPT}]
        closing = self._fragments.assistant_suffix.strip()
        for reasoning in (False, True) if self._can_reason else (False,):
            prompted = self._render(question, generation_prompt=True, reasoning=reasoning)
            past = self._render(list(_PAST_REPLY), reasoning=reasoning)
            if prompted is None or past is None or not prompted.endswith(
                    self._generation_prompt(reasoning)) or not past.split(_NEXT, 1)[-1].lstrip() \
                    .startswith(closing):
                raise ValueError(
                    "this model's template writes its generation prompt or the closing "
                    "after a reply otherwise than as its own probes read them, so a "
                    "reply's start and end could not be told; the model cannot be used")

    def _resolve_reasoning(self) -> Reasoning | None:
        """What turns this model's reasoning on, recovered from its template."""
        render = self._read_as_probes
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
        # An add-on file's call wins over what the template shows of one.
        tool_tokens, lead_in = (None, "") if self._addon_call else self._resolve_tool_tokens()
        call_texts, from_file = (None, False) if tool_tokens is not None else self._resolve_call_texts()
        opens_as_text = (call_texts is not None
                         and first_special_token(self._context, call_texts[0]) is None)
        if call_texts is not None:
            lead_in = self._text_after_last_tag(call_texts[0])
        turn_end_text = self._resolve_turn_end_text(call_texts)
        boundary = CallBoundary(tool_tokens, call_texts, opens_as_text, turn_end_text,
                                follows_text=opens_as_text and call_texts is not None
                                and self._writes_call_after_text(call_texts[0]))
        return boundary, lead_in, from_file, None if turn_end_text else turn_end_token

    def _resolve_call_texts(self) -> tuple[tuple[str, str] | None, bool]:
        """The texts around a call, from the model's add-on file, else from the template.

        Returns:
            The texts, and whether they came from the add-on file.
        """
        call = self._addon_call
        if not call:
            return self._template_call_texts(), False
        if self._tool_calls is None:
            return None, False
        rendered = self._render([{"role": "user", "content": _PROMPT},
                                 calling_message("", [ToolCall(_PROBE_FUNCTION, {})], 0)])
        if rendered is None or call["closing"] not in rendered:
            raise ValueError("the model's add-on file closes a call with text its "
                             "template never writes after one")
        return (call["opening"], call["closing"]), True

    def _writes_call_after_text(self, opening: str) -> bool:
        """Whether the template writes a call's ``opening`` right after a reply's text, whitespace apart."""
        rendered = self._render([{"role": "user", "content": _PROMPT},
                                 calling_message(_TEXT, [ToolCall(_PROBE_FUNCTION, {})], 0)])
        if rendered is None or _TEXT not in rendered:
            return False
        return rendered.partition(_TEXT)[2].lstrip().startswith(opening.strip())

    def _template_call_texts(self) -> tuple[str, str] | None:
        """The texts around a call, for a template whose call tags also mark every reply."""
        if self._tool_calls is None:
            return None
        call_open = self._tool_calls.call_open
        if first_special_token(self._context, call_open) is None:
            # Named in the header first: the call opens where the name is written again.
            opening = extract_named_twice_opening(self._read_as_probes, call_open)
            if opening:
                close = extract_call_close_text(self._read_as_probes, opening)
                return (opening, close) if close else None
            if self._opens_in_reply_header(call_open):
                return None
        close = extract_call_close_text(self._read_as_probes, call_open)
        return (call_open, close) if close else None

    def _text_after_last_tag(self, text: str) -> str:
        """Where a call's own text begins in its opening: after the opening's last tag."""
        tags = self._tags_in(text)
        last = self._context.token_text(tags[-1]) if tags else ""
        return text.rpartition(last)[2] if last else text

    def _opens_in_reply_header(self, call_open: str) -> bool:
        """Whether a plain-text opening begins with text of a reply's own opening."""
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
        shared = shared_tags_length(closing, call_open, self._tag_texts)
        if not shared:
            return None
        tag = first_special_token(self._context, closing[shared:])
        text = self._context.token_text(tag) if tag is not None else ""
        if not text or not closing[shared:].startswith(text):
            return None
        return closing[:shared] + text

    def _call_closer(self, fragments: ToolCallFragments) -> int | None:
        """The tag that ends a call.

        The last tag between two calls of one turn, where the template writes one
        there that ``call_close`` also holds; otherwise the first tag of
        ``call_close``.
        """
        separator = extract_call_separator(self._read_as_probes,
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
        """This model's argument spelling, or ``None`` if it cannot be read."""
        if not self._calls_marked or self._tool_calls is None:
            return None
        if self._call_from_file:
            # The name ends where the file says; the arguments are read as written.
            return CallBody(name_close=self._addon_call["name_close"], key_open="",
                            key_close="", pair_sep="", body_close="", empty_close="",
                            arguments_json=True)
        call_open = (self._boundary.call_texts[0] if self._boundary.call_texts
                     else self._tool_calls.call_open)
        return extract_call_body(self._read_as_probes, call_open, self._call_close_text)

    def _resolve_can_declare_tools(self) -> bool:
        """Whether this model's template renders tool declarations at all."""
        if not self._calls_marked:
            return False
        return self._declares_before_newest_turn or self._declares_in_system_turn()

    def _resolve_declaration_variables(self) -> dict:
        """A template variable that puts tool declarations into the system turn, where needed.

        One the template reads qualifies where setting it changes nothing in a
        conversation without tools; empty where the system turn declares them
        already, or no variable makes it.
        """
        if not self._calls_marked or self._declares_in_system_turn():
            return {}
        conversation = [{"role": "system", "content": _PROMPT}, {"role": "user", "content": _NEXT}]
        plain = self._render(conversation)
        names = names_read(self._model.text)
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

    def _resolve_declares_before_newest_turn(self) -> bool:
        """Whether the template declares tools right before the newest user turn alone."""
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
        """Whether this model's template renders a block to answer a call in."""
        if not self._calls_marked:
            return False
        rendered = self._render([
            {"role": "user", "content": _PROMPT},
            calling_message("", [ToolCall(_PROBE_FUNCTION, {})], 0),
            result_message(_PROBE_FUNCTION, call_id(0), _RESULT)],
            generation_prompt=True, reasoning=False)
        return rendered is not None and _RESULT in rendered

    def _resolve_writes_past_calls(self) -> bool:
        """Whether a call the template writes while its turn is the newest stays written once a later question follows."""
        if not self._calls_marked:
            return False
        answered = [{"role": "user", "content": _PROMPT},
                    calling_message("", [ToolCall(_PROBE_FUNCTION, {})], 0),
                    result_message(_PROBE_FUNCTION, call_id(0), _RESULT)]
        newest = self._render(answered, generation_prompt=True, reasoning=False)
        past = self._render([*answered, {"role": "assistant", "content": _TEXT},
                             {"role": "user", "content": _NEXT}],
                            generation_prompt=True, reasoning=False)
        if newest is None or past is None or _RESULT not in newest or _RESULT not in past:
            return True
        call_open = (self._boundary.call_texts[0] if self._boundary.call_texts
                     else self._tool_calls.call_open if self._tool_calls is not None else "").strip()
        written = (render.split(_RESULT, 1)[0] for render in (newest, past))
        return not call_open or next(written).count(call_open) <= next(written).count(call_open)

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
        """Whether the system turn the template renders holds a stand-in tool's declaration, worked out once for each set of template variables."""
        key = tuple(sorted(self._variables.items()))
        if key not in self._declares_in_system_turn_for:
            rendered = self._system_turn([_PROBE_TOOL.as_declaration()])
            plain = self._system_turn(None)
            self._declares_in_system_turn_for[key] = (
                rendered is not None and plain is not None and _PROMPT in rendered
                and _PROMPT in plain and rendered != plain)
        return self._declares_in_system_turn_for[key]


@dataclasses.dataclass(frozen=True)
class TemplateProperties:
    """Everything a load reads from a model's chat template, in one place; see each field's reader on :class:`ResolvedTemplate`.

    Attributes:
        reads_clock: The template reads the date, so a render may change with it.
        fragments: The texts around each kind of turn.
        can_take_system_prompt: A system prompt stays where it is written: the
            template does not move it with later questions.
        turn_end_token: The tag that ends a reply, where one does.
        content_rewrites: Texts the template writes otherwise inside a message.
        tool_fragments: The texts around a past call and its result; ``None``
            where the template describes no call.
        boundary: Where a generated call opens and closes and where a reply
            ends, as generation reads it.
        call_close_text: The text a call ends with, empty where the model
            marks no call.
        call_lead_in: Fixed text the model writes before a call's name.
        call_body: How a call's arguments are spelled, where not as JSON.
        json_call_keys: The keys a JSON call names its function and arguments
            by, where a call is JSON.
        json_call_id_key: Of those keys, the one a call's id is written
            under, where the template writes one.
        declares_before_newest_turn: The declarations move with each question.
        declaration_variables: The template variables the declarations are
            passed under.
        can_declare_tools: The template writes the declarations.
        can_write_results: The template writes a call's result.
        writes_past_calls: The template still writes a call once a later
            question follows its turn.
        reasoning: How a thought is marked and turned on, where it is.
        can_reason: A conversation can be asked to reason.
        n_closing_tokens: Tokens the closing after a reply takes at most.
        tag_texts: The texts of the model's own tags.
        writes_reply_as_prompted: With reasoning off, a past reply stands as
            its question's generation prompt left it, as the text it was
            generated as.
        writes_two_user_turns: The template writes two questions in a row.
    """

    reads_clock: bool
    fragments: TemplateFragments
    can_take_system_prompt: bool
    turn_end_token: int | None
    content_rewrites: tuple
    tool_fragments: ToolCallFragments | None
    boundary: CallBoundary
    call_close_text: str
    call_lead_in: str
    call_body: CallBody | None
    json_call_keys: tuple[str, ...] | None
    json_call_id_key: str | None
    declares_before_newest_turn: bool
    declaration_variables: dict
    can_declare_tools: bool
    writes_past_calls: bool
    can_write_results: bool
    reasoning: Reasoning | None
    can_reason: bool
    n_closing_tokens: int
    tag_texts: tuple[str, ...]
    writes_reply_as_prompted: bool
    writes_two_user_turns: bool

    @property
    def call_tags(self) -> tuple[int, int] | None:
        """The tags that open and close a generated call, where it has tags of its own."""
        return self.boundary.tool_tokens

    @property
    def call_texts(self) -> tuple[str, str] | None:
        """The texts that open and close a generated call, where it has no tags."""
        return self.boundary.call_texts

    @property
    def call_follows_text(self) -> bool:
        """Whether a call may follow a reply's text."""
        return self.boundary.follows_text

    @property
    def calls_marked(self) -> bool:
        """Whether a call can be told from a reply, by its tags or by its texts."""
        return self.call_tags is not None or self.call_texts is not None

    @property
    def can_call_tools(self) -> bool:
        """Whether the call tags, the declarations, the argument spelling and the result block all came out of the template, and a past call stays written."""
        return (self.calls_marked and self.can_declare_tools and self.call_body is not None
                and self.can_write_results and self.writes_past_calls)

    @property
    def can_turn_off_reasoning(self) -> bool:
        """Whether a reply asked for without reasoning is generated without it, as the template's own off writes it; true where the template marks no thought."""
        return self.reasoning is None or self.reasoning.can_turn_off
