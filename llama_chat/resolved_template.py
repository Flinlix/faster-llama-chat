"""What one model's own chat template supports, resolved once and validated.

Everything here is settled while a wrapper is being built: the per-role
fragments, what turns reasoning on, and the four steps tool calling needs. An
instance is therefore a template known to work, and the
questions a caller would otherwise ask per turn are already answered.

Nothing here touches a conversation. It reads a cache context and renders
through the model's own template; what is done with the result belongs to
:class:`~llama_chat.wrapper.ChatWrapper`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .template import (CallBody, Reasoning, TemplateFormatter,
                       TemplateFragments, ToolCallFragments)
from .template_extract import (call_id, extract_call_body, extract_call_separator,
                               extract_reasoning, extract_tool_fragments)
from .tools import ToolCall, ToolResult, ToolSpec, ToolTurn, result_text

# Where the prompt and a tool result go in a render. Neither contains the
# other, so a split on either lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
# The results of a batch, numbered; none contains another.
_RESULT_N = "RESULT{}NTH"

# Where a past turn's own text goes in a render, numbered; the user turns around
# it are found by the two markers above and below.
_FIELD = "FIELD{}MSG"
_FIELD_PATTERN = re.compile(r"FIELD(\d+)MSG")
_NEXT = "NEXTMSG"
# A reply and a user turn after the first, so a template that declares tools
# before the newest user turn shows it after the first turn, not ahead of it.
_EXCHANGE = ({"role": "assistant", "content": "REPLYMSG"},
             {"role": "user", "content": "LATERMSG"})

# The conversation the fragments are checked on: a system prompt and two user
# turns, so a past reply and a newest turn are both in it. Padded, so a wrong
# `trim_content` shows too.
_CHECK_CONVERSATION = ((("system", " sys "),), (("user", " ping "), ("assistant", " pong "),
                                     ("user", " again ")))

# The stand-in a probe declares and answers a call for. What a template renders
# around a declaration or around a result does not depend on which function it
# is, so one probe settles both for every tool. It never reaches a model.
_PROBE_FUNCTION = "probe"
_PROBE_TOOL = ToolSpec(name=_PROBE_FUNCTION, description="probe",
                       parameters={"type": "object", "properties": {}})

# What resolving a model with tool tags calls on its cache context.
_TOOL_SURFACE = ("first_special_token", "token_text", "render_with_model_template")


def ensure_context_answers(context, surface: tuple[str, ...]) -> None:
    """Refuse a cache context missing something its caller will call on it.

    Checked where a context is adopted, and where resolving reaches for the
    tool half, rather than at the first call that finds what is missing, which
    for an eviction is a whole conversation later.

    Raises:
        TypeError: If any name in ``surface`` is missing.
    """
    missing = [name for name in surface
               if not hasattr(context, name)]
    if missing:
        raise TypeError(
            f"this cache context is missing {missing}; see the README, "
            f"\"Layout\", for what a cache context has to answer")


@dataclass(frozen=True)
class _SystemTurnSplit:
    """The system turn this model renders, with tools declared and without.

    Split where the prompt goes, so the declarations are what the two renders
    differ by. Which side of the prompt they land on is the template's business.

    Attributes:
        declared_head, declared_tail: What the render with tools puts before
            and after the prompt.
        plain_head, plain_tail: The same for the render without them.
    """

    declared_head: str
    declared_tail: str
    plain_head: str
    plain_tail: str


class ResolvedTemplate:
    """This model's chat template, recovered or supplied, and proven usable.

    Args:
        context: The cache context to read the model through.
        fragments: Explicit per-role tags, for a model that ships no embedded
            template. Recovered from the model otherwise.
        tool_calls: Explicit call tags, and optionally the argument spelling
            inside them, for a model whose template describes neither. Whatever
            it leaves out is recovered from the model.

    Raises:
        TypeError: If ``context`` does not answer what resolving needs of it.
        ValueError: If the template cannot be recovered, or if it fails
            validation against the model.
    """

    def __init__(self, context, *, fragments: TemplateFragments | None = None,
                 tool_calls: ToolCallFragments | None = None) -> None:
        self._context = context
        self._fragments = (fragments if fragments is not None
                           else self._resolve_fragments())
        self._validate_template()
        self._turn_end_token = self._resolve_turn_end_token()
        # None where this model cannot call tools at all.
        self._tool_calls = (tool_calls if tool_calls is not None
                            else self._resolve_tool_calls())
        self._tool_tokens, self._call_lead_in = self._resolve_tool_tokens()
        self._call_body = self._resolve_call_body()
        self._can_declare_tools = self._resolve_can_declare_tools()
        self._can_write_results = self._resolve_can_write_results()
        self._reasoning = self._resolve_reasoning()
        self._check_fragments_match_model()
        self._preambles = {reasoning: self._resolve_preamble(reasoning)
                           for reasoning in (False, True)}
        self._later_system = self._resolve_later_system()

    # ----- what the wrapper reads ----------------------------------------------
    @property
    def fragments(self) -> TemplateFragments:
        """The per-role tags a turn is wrapped in."""
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
        return self._reasoning is not None and self._reasoning.system_prefix is not None

    @property
    def has_system_turn(self) -> bool:
        """Whether a system prompt has a turn of its own, rather than the user tags'."""
        frags = self._fragments
        return (frags.system_prefix, frags.system_suffix) != \
            (frags.user_prefix, frags.user_suffix)

    @property
    def later_system_tags(self) -> tuple[str, str] | None:
        """The tags of a system turn after the first exchange, or ``None``."""
        return self._later_system

    def preamble(self, reasoning: bool = False) -> str | None:
        """What the template writes ahead of the first turn when there is no system prompt.

        ``None`` where the model ships no template, the fragments' system turn
        then standing in.
        """
        return self._preambles[reasoning]

    def preamble_declaring(self, specs: list[ToolSpec], reasoning: bool = False) -> str:
        """What the template writes ahead of the first turn for ``specs`` and no system prompt.

        Raises:
            ValueError: If the template does not write it where it can be located.
        """
        messages = [{"role": "user", "content": _PROMPT}, *_EXCHANGE]
        rendered = self._context.render_with_model_template(
            messages, tools=[spec.as_declaration() for spec in specs], reasoning=reasoning)
        plain = self._context.render_with_model_template(messages, reasoning=reasoning)
        opened = self._fragments.user_prefix + _PROMPT
        if rendered is None or plain is None or opened not in rendered or opened not in plain \
                or (rendered != plain
                    and rendered.split(opened, 1)[0] == plain.split(opened, 1)[0]):
            raise ValueError("this model's template declares tools without a system "
                             "prompt in a way that cannot be located")
        return rendered.split(opened, 1)[0]

    def formatter(self, reasoning: bool = False) -> TemplateFormatter:
        """What renders one message, and what is decoded around sampling.

        Raises:
            ValueError: If ``reasoning`` is asked of a model that cannot reason.
        """
        if reasoning and not self.can_reason:
            raise ValueError("this model's template has no reasoning switch that "
                             "can be turned on as it does, or marks no thought "
                             "that could be told from the answer")
        return TemplateFormatter(self._fragments,
                                 self._reasoning if reasoning else None)

    @property
    def tool_tokens(self) -> tuple[int, int] | None:
        """The ``(open, close)`` tokens bracketing a call, or ``None``."""
        return self._tool_tokens

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
        return (self._tool_tokens is not None and self._can_declare_tools
                and self._call_body is not None and self._can_write_results)

    def system_turn_declaring(self, specs: list[ToolSpec],
                              reasoning: bool = False) -> tuple[str, str]:
        """What this model wraps a system prompt in when ``specs`` are declared.

        Args:
            reasoning: Whether the conversation reasons, which the template may
                switch in this turn.

        Returns:
            The text before and after the prompt. Which side the declarations
            land on is the template's business.

        Raises:
            ValueError: If the template renders no declarations.
        """
        split = self._declaration_turn(specs, reasoning)
        return split.declared_head, split.declared_tail

    def result_wrapping(self, name: str, reasoning: bool = False, *,
                        batch: list[str] | None = None,
                        index: int = 0) -> tuple[str, str]:
        """What this model's template wraps ``name``'s result in.

        Recovered per result, because the wrapping names the function, and per
        place in a batch: the template writes the results of calls made
        together one after another, with what ends the block after the last.
        That ending is where the reply goes on, so it includes whatever the
        generation prompt adds after a result, which may depend on
        ``reasoning``. Only meaningful where :attr:`tool_tokens` is set, a
        call's closing tag being what the first result is found against.

        Args:
            batch: The names of the calls made together, in order; ``name``
                alone where omitted.
            index: Which of them this result answers.

        Returns:
            What goes before the result, and what goes after it: nothing but
            for the last of a batch.

        Raises:
            ValueError: If the template renders no result block.
        """
        names = batch or [name]
        markers = [_RESULT_N.format(i) for i in range(len(names))]
        rendered = self._context.render_with_model_template([
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id(i), "type": "function",
                 "function": {"name": called, "arguments": {}}}
                for i, called in enumerate(names)]},
            *({"role": "tool", "name": called, "tool_call_id": call_id(i),
               "content": markers[i]} for i, called in enumerate(names)),
        ], generation_prompt=True, reasoning=reasoning)
        closer = self._context.token_text(self._tool_tokens[1])
        if rendered is None or any(m not in rendered for m in markers) \
                or closer not in rendered:
            raise ValueError("this model's template renders no tool result")
        before = rendered.split(markers[index], 1)[0]
        prefix = (before.rsplit(closer, 1)[1] if index == 0
                  else before.split(markers[index - 1], 1)[1])
        suffix = rendered.split(markers[-1], 1)[1] if index == len(names) - 1 else ""
        return prefix, suffix

    def past_tool_turn(self, turn: ToolTurn, open_turn: bool = False,
                       reasoning: bool = False) -> list[tuple[str, bool]]:
        """A turn of the history that called tools, as this model's template writes it.

        Rendered by the template, with a marker for every piece of the turn's
        own text: what the model wrote, each string argument, each result. So
        the order and the spelling are the template's, and nothing of the
        turn's text is read as a tag.

        Args:
            turn: The turn.
            open_turn: Leave the turn open after its last result, for the reply
                to go on with: rendered as the generation prompt ends it, with
                nothing after it, and without the closing tag a template may
                add there.
            reasoning: Whether the conversation reasons, which the generation
                prompt of an open turn may depend on.

        Returns:
            The turn from its opening tag to its closing one, or to its last
            result where it is left open, in pieces: the text, and whether it is
            the turn's own text, to be tokenized with special-token parsing
            off, rather than the template's.

        Raises:
            ValueError: If this model cannot call tools, or the template does not
                render the turn where a turn goes.
        """
        self.ensure_tools_usable()
        fields: list[str] = []

        def field(text: str) -> str:
            fields.append(text)
            return _FIELD.format(len(fields) - 1)

        frags = self._fragments
        after = [] if open_turn else [{"role": "user", "content": _NEXT}]
        rendered = self._context.render_with_model_template(
            [{"role": "user", "content": _PROMPT}]
            + _turn_messages(turn, field, frags) + after,
            generation_prompt=open_turn, reasoning=reasoning)
        head = frags.user_prefix + _PROMPT + frags.user_suffix
        if rendered is None or head not in rendered \
                or not (open_turn or _NEXT in rendered):
            raise ValueError("this model's template renders no past tool turn")
        body = rendered.partition(head)[2]
        if open_turn:
            body = body.removesuffix(frags.assistant_suffix)
        else:
            body = body.rpartition(frags.user_prefix + _NEXT)[0]
        pieces, at = [], 0
        for found in _FIELD_PATTERN.finditer(body):
            text = fields[int(found.group(1))]
            quoted = (body[found.start() - 1:found.start()] == '"'
                      and body[found.end():found.end() + 1] == '"')
            pieces.append((body[at:found.start()], False))
            pieces.append((json.dumps(text, ensure_ascii=False)[1:-1] if quoted else text,
                           True))
            at = found.end()
        pieces.append((body[at:], False))
        return [(text, own) for text, own in pieces if text]

    def declaration_block(self, specs: list[ToolSpec]) -> str:
        """Only the declarations, for adding to a conversation already begun.

        What the two renders differ by, from whichever side of the prompt the
        template put them on.

        Raises:
            ValueError: If the template renders no declarations, or renders
                them where they cannot be told apart from the system turn.
        """
        split = self._declaration_turn(specs)
        block = ""
        if split.declared_head.startswith(split.plain_head):
            block += split.declared_head[len(split.plain_head):]
        if split.declared_tail.endswith(split.plain_tail):
            block += split.declared_tail[:len(split.declared_tail)
                                         - len(split.plain_tail)]
        if not block:
            raise ValueError(
                "this model's template does not render its declarations as a "
                "block that can be lifted out of the system turn, so they "
                "cannot be added as a message of their own; declare them with "
                "begin(tools=...) instead")
        return block

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
        if self._tool_tokens is None and self._tool_calls is not None:
            raise ValueError(
                "this model writes a tool call as plain text, without a tag "
                "token of its own, so a call could not be told from a reply")
        if self._tool_tokens is None:
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
        if not hasattr(self._context, "extract_fragments"):
            raise ValueError("context cannot supply template fragments")
        fragments = self._context.extract_fragments()
        if fragments is None:
            raise ValueError(
                "the model ships no chat template, so the fragments cannot be "
                "derived; pass them explicitly with ChatWrapper(fragments=...)"
            )
        return fragments

    def _validate_template(self) -> None:
        """Ensure the turn-terminator is made of this model's own tags.

        A terminator that splits into plain text never lets the model stop, so
        it would run to the token cap every generation.
        """
        if not hasattr(self._context, "tokenizes_to_tags"):
            return
        terminator = self._fragments.assistant_suffix.strip()
        if terminator and not self._context.tokenizes_to_tags(terminator):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not tokenize into the model's own tags, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _resolve_turn_end_token(self) -> int | None:
        """The first token of the turn-terminator, where it is one of the model's tags."""
        terminator = self._fragments.assistant_suffix.strip()
        if not terminator or not hasattr(self._context, "first_special_token"):
            return None
        first = self._context.first_special_token(terminator)
        tokens = self._context.tokenize(terminator, add_special=False)
        return first if tokens and tokens[0] == first else None

    def _check_fragments_match_model(self) -> None:
        """Assert the fragments encode a conversation as the model's own template does.

        A system prompt, a past reply and a newest turn, followed by the
        generation prompt, in each reasoning mode the model has. Tags that do
        not reproduce what the model saw in training make incremental prefill
        write tokens it never learned. A system turn is left out where the
        template has none of its own and the user tags stand in for it.
        """
        if not hasattr(self._context, "render_with_model_template"):
            return
        system, exchange = _CHECK_CONVERSATION
        probe = list(system if self.has_system_turn else ()) + list(exchange)
        for reasoning in (False, True) if self.can_reason else (False,):
            messages = [{"role": role, "content": text} for role, text in probe]
            rendered = self._context.render_with_model_template(
                messages, reasoning=reasoning)
            prompted = self._context.render_with_model_template(
                messages, generation_prompt=True, reasoning=reasoning)
            if rendered is None:
                if self._context.render_with_model_template(
                        [{"role": "user", "content": _PROMPT}]) is None:
                    return          # the model ships no template to check against
                raise ValueError(
                    "this model's template refuses a plain conversation of a "
                    "system prompt and two user turns, so it cannot be checked, "
                    "and a conversation could not be encoded as it renders one")
            formatter = self.formatter(reasoning)
            # Built exactly as real prefill builds it, tags special-on and
            # content special-off, so any mismatch is a tag or trim divergence.
            fragment_tokens: list[int] = []
            fragment_text = ""
            for role, text in probe:
                prefix, content, suffix = formatter.parts(role, text)
                fragment_tokens += self._context.tokenize_fragment(prefix, content, suffix)
                fragment_text += prefix + content + suffix
            # The generation prompt is compared where the renderer adds one.
            if prompted is not None and prompted != rendered:
                rendered = prompted
                opening = formatter.generation_prompt()
                fragment_tokens += self._context.tokenize(opening, add_special=False)
                fragment_text += opening
            if self._context.tokenize(rendered, add_special=False) != fragment_tokens:
                raise ValueError(
                    "chat template fragments do not match the model's trained "
                    "template:\n"
                    f"  model renders:    {rendered!r}\n"
                    f"  fragments render: {fragment_text!r}\n"
                    "The recovered fragments do not round-trip through the "
                    "model's template; supply correct tags with "
                    "ChatWrapper(fragments=...).")

    def _resolve_tool_calls(self) -> ToolCallFragments | None:
        """What this model's template wraps a generated call in, if anything."""
        if not hasattr(self._context, "render_with_model_template"):
            return None
        return extract_tool_fragments(self._context.render_with_model_template,
                                      self._fragments)

    def _resolve_preamble(self, reasoning: bool) -> str | None:
        """What a conversation without a system message opens with, before its first turn."""
        if not hasattr(self._context, "render_with_model_template"):
            return None
        rendered = self._context.render_with_model_template(
            [{"role": "user", "content": _PROMPT}], reasoning=reasoning)
        opened = self._fragments.user_prefix + _PROMPT
        if rendered is None:
            return None             # no template, so the fragments' system turn
        if opened not in rendered:
            raise ValueError("this model's template opens a conversation without a "
                             "system prompt in a way that cannot be located")
        return rendered.split(opened, 1)[0]

    def _resolve_later_system(self) -> tuple[str, str] | None:
        """The tags of a system turn that follows an exchange, read from a render."""
        if not hasattr(self._context, "render_with_model_template"):
            return None
        frags = self._fragments
        rendered = self._context.render_with_model_template([
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "content": _RESULT},
            {"role": "system", "content": _NEXT},
            {"role": "user", "content": _PROMPT},
        ])
        after = _RESULT + frags.assistant_suffix
        before = frags.user_prefix + _PROMPT
        if rendered is None or after not in rendered or _NEXT not in rendered:
            return None
        head, _, tail = rendered.split(after, 1)[1].partition(_NEXT)
        if before not in tail or _PROMPT in head:
            return None
        return head, tail.split(before, 1)[0]

    def _resolve_reasoning(self) -> Reasoning | None:
        """What turns this model's reasoning on, recovered from its template."""
        if not hasattr(self._context, "render_with_model_template"):
            return None
        return extract_reasoning(self._context.render_with_model_template,
                                 self._fragments, self._tool_calls)

    def _resolve_tool_tokens(self) -> tuple[tuple[int, int] | None, str]:
        """This model's call tags as tokens, and the text it writes before a name."""
        fragments = self._tool_calls
        if fragments is None:
            return None, ""
        ensure_context_answers(self._context, _TOOL_SURFACE)
        opener = self._context.first_special_token(fragments.call_open)
        closer = self._call_closer(fragments)
        if opener is None or closer is None:
            return None, ""
        tag = self._context.token_text(opener)
        # Text the template writes before the tag belongs to the reply, not to the call.
        lead_in = fragments.call_open.partition(tag)[2]
        return (opener, closer), lead_in

    def _call_closer(self, fragments: ToolCallFragments) -> int | None:
        """The tag that ends a call.

        The last tag between two calls of one turn, where the template writes
        one there that ``call_close`` also holds: a call's ending may close its
        last argument with a tag of its own first. Otherwise the first tag of
        ``call_close``.
        """
        separator = extract_call_separator(self._context.render_with_model_template,
                                           fragments.call_open)
        tags = self._tags_in(separator or "")
        if tags and self._context.token_text(tags[-1]) in fragments.call_close:
            return tags[-1]
        return self._context.first_special_token(fragments.call_close)

    def _tags_in(self, text: str) -> list[int]:
        """The model's own tags in ``text``, in order."""
        tags = []
        while (tag := self._context.first_special_token(text)) is not None:
            tags.append(tag)
            text = text.partition(self._context.token_text(tag))[2]
        return tags

    def _resolve_call_body(self) -> CallBody | None:
        """This model's argument spelling, or ``None`` if it cannot be read.

        A supplied one wins; a call stays detectable without either, and only
        its contents cannot be read.
        """
        if self._tool_tokens is None or self._tool_calls is None:
            return None
        if self._tool_calls.body is not None:
            return self._tool_calls.body
        return extract_call_body(self._context.render_with_model_template,
                                 self._tool_calls.call_open,
                                 self._context.token_text(self._tool_tokens[1]))

    def _resolve_can_declare_tools(self) -> bool:
        """Whether this model's template renders tool declarations at all.

        Probed with a stand-in tool, for the same reason the result block is:
        a template renders declarations or it does not, and does not branch on
        which tools it was handed. Any failure answers ``False``, because this
        decides an optional capability and must not stop a conversation that
        wants no tools.
        """
        if self._tool_tokens is None:
            return False
        try:
            self._declaration_turn([_PROBE_TOOL])
            return True
        except Exception:
            return False

    def _resolve_can_write_results(self) -> bool:
        """Whether this model's template renders a block to answer a call in.

        Probed once, with a stand-in name: a template renders a result block or
        it does not, and does not branch on which function is being answered.
        Any failure answers ``False``, because this decides an optional
        capability and must not stop a conversation that wants no tools.
        """
        if self._tool_tokens is None:
            return False
        try:
            self.result_wrapping(_PROBE_FUNCTION)
            return True
        except Exception:
            return False

    def _plain_system_turn(self, reasoning: bool = False) -> str | None:
        """The system turn this model renders with no tools, prompt sentinelled."""
        return self._system_turn(None, reasoning)

    def _system_turn(self, tools: list[dict] | None, reasoning: bool) -> str | None:
        """The system turn rendered ahead of a user turn, which some templates require.

        Cut where the user turn begins, so whatever the template writes between
        the two belongs to the system turn.
        """
        rendered = self._context.render_with_model_template(
            [{"role": "system", "content": _PROMPT},
             {"role": "user", "content": _NEXT}, *_EXCHANGE], tools=tools, reasoning=reasoning)
        opened = self._fragments.user_prefix + _NEXT
        if rendered is None or opened not in rendered:
            return None
        return rendered.split(opened, 1)[0]

    def _declaration_turn(self, specs: list[ToolSpec],
                          reasoning: bool = False) -> _SystemTurnSplit:
        """The system turn this model renders for ``tools``, and without them.

        Rendered by the template, so no declaration shape is written down here.

        Raises:
            ValueError: If the template renders no declarations.
        """
        rendered = self._system_turn([spec.as_declaration() for spec in specs],
                                     reasoning)
        plain = self._plain_system_turn(reasoning)
        if rendered is None or plain is None or _PROMPT not in rendered \
                or _PROMPT not in plain or rendered == plain:
            raise ValueError("this model's template declares no tools")
        declared_head, _, declared_tail = rendered.partition(_PROMPT)
        plain_head, _, plain_tail = plain.partition(_PROMPT)
        return _SystemTurnSplit(declared_head, declared_tail,
                                plain_head, plain_tail)


def _turn_messages(turn: ToolTurn, field, fragments: TemplateFragments) -> list[dict]:
    """``turn`` as the chat messages a template renders, its own text as markers.

    The model's text and the calls it makes belong to one assistant message
    until a result follows; each result is a tool message, answering the calls
    in the order they were made.
    """
    messages: list[dict] = []
    calls = results = 0
    assistant = None
    for step in turn.steps:
        if isinstance(step, ToolResult):
            messages.append({"role": "tool", "name": step.name,
                             "tool_call_id": call_id(results),
                             "content": field(result_text(step.result))})
            results += 1
            assistant = None
            continue
        if assistant is None:
            assistant = {"role": "assistant", "content": "", "tool_calls": []}
            messages.append(assistant)
        if isinstance(step, ToolCall):
            assistant["tool_calls"].append({
                "id": call_id(calls), "type": "function",
                "function": {"name": step.name,
                             "arguments": _marked(step.arguments, field)}})
            calls += 1
        elif text := fragments.content("assistant", step):
            assistant["content"] = field(text)
    for m in messages:
        if m["role"] == "assistant" and not m["tool_calls"]:
            del m["tool_calls"]
    return messages


def _marked(value, field):
    """``value`` with every string in it replaced by a marker."""
    if isinstance(value, str):
        return field(value)
    if isinstance(value, dict):
        return {k: _marked(v, field) for k, v in value.items()}
    if isinstance(value, list):
        return [_marked(v, field) for v in value]
    return value
