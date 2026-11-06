"""What one model's own chat template supports, resolved once and validated.

Everything here is settled while a wrapper is being built: the per-role
fragments, the formatter that opens and closes a turn, and the four steps tool
calling needs. An instance is therefore a template known to work, and the
questions a caller would otherwise ask per turn are already answered.

Nothing here touches a conversation. It reads a cache context and renders
through the model's own template; what is done with the result belongs to
:class:`~llama_chat.wrapper.ChatWrapper`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .template import (CallBody, TemplateFormatter, TemplateFragments,
                       ToolCallFragments)
from .template_extract import extract_call_body, extract_tool_fragments
from .tools import ToolCall, ToolResult, ToolSpec, ToolTurn, result_text

# What builds the formatter once the fragments are known.
_FormatterFactory = Callable[[TemplateFragments], TemplateFormatter]

# Where the prompt, a tool result and the reply after it go in a render. No one
# of them contains another, so a split on any lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
_TAIL = "TAILMSG"

# Where a past turn's own text goes in a render, numbered; the user turns around
# it are found by the two markers above and below.
_FIELD = "FIELD{}MSG"
_FIELD_PATTERN = re.compile(r"FIELD(\d+)MSG")
_NEXT = "NEXTMSG"

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
        formatter: Factory building the :class:`~llama_chat.template.TemplateFormatter`
            from the resolved fragments.

    Raises:
        TypeError: If ``context`` does not answer what resolving needs of it.
        ValueError: If the template cannot be recovered, if it fails validation
            against the model, or if ``formatter`` builds one that does not keep
            the model's own turn tags.
    """

    def __init__(self, context, *, fragments: TemplateFragments | None = None,
                 tool_calls: ToolCallFragments | None = None,
                 formatter: _FormatterFactory | None = None) -> None:
        self._context = context
        self._fragments = (fragments if fragments is not None
                           else self._resolve_fragments())
        self._formatter = (formatter or TemplateFormatter)(self._fragments)
        self._validate_template()
        self._validate_formatter()
        self._check_fragments_match_model()
        # None where this model cannot call tools at all.
        self._tool_calls = (tool_calls if tool_calls is not None
                            else self._resolve_tool_calls())
        self._tool_tokens, self._call_lead_in = self._resolve_tool_tokens()
        self._call_body = self._resolve_call_body()
        self._can_declare_tools = self._resolve_can_declare_tools()
        self._can_write_results = self._resolve_can_write_results()

    # ----- what the wrapper reads ----------------------------------------------
    @property
    def fragments(self) -> TemplateFragments:
        """The per-role tags a turn is wrapped in."""
        return self._fragments

    @property
    def formatter(self) -> TemplateFormatter:
        """What renders one message, and what is decoded around sampling."""
        return self._formatter

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
                              opening: str = "") -> tuple[str, str]:
        """What this model wraps a system prompt in when ``specs`` are declared.

        Args:
            opening: The formatter's ``system_open()``, placed where the turn's
                own head without declarations ends.

        Returns:
            The text before and after the prompt. Which side the declarations
            land on is the template's business.

        Raises:
            ValueError: If the template renders no declarations, or renders
                them before its own head, where no opening can follow it.
        """
        split = self._declaration_turn(specs)
        head = split.declared_head
        if opening:
            if not head.startswith(split.plain_head):
                raise ValueError("this template declares tools ahead of its system "
                                 "turn's head, so system_open() has no place")
            head = split.plain_head + opening + head[len(split.plain_head):]
        return head, split.declared_tail

    def result_wrapping(self, name: str) -> tuple[str, str]:
        """What this model's template wraps ``name``'s result in.

        Recovered per result, because the wrapping names the function. Only
        meaningful where :attr:`tool_tokens` is set, a call's closing tag being
        what the result block is found against.

        Raises:
            ValueError: If the template renders no result block.
        """
        rendered = self._context.render_with_model_template([
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": name, "arguments": {}}}]},
            {"role": "tool", "name": name, "content": _RESULT},
            {"role": "assistant", "content": _TAIL},
        ])
        closer = self._context.token_text(self._tool_tokens[1])
        if rendered is None or _RESULT not in rendered \
                or _TAIL not in rendered or closer not in rendered:
            raise ValueError("this model's template renders no tool result")
        head, _, tail = rendered.partition(_RESULT)
        return head.rsplit(closer, 1)[1], tail.partition(_TAIL)[0]

    def past_tool_turn(self, turn: ToolTurn,
                       open_turn: bool = False) -> list[tuple[str, bool]]:
        """A turn of the history that called tools, as this model's template writes it.

        Rendered by the template, with a marker for every piece of the turn's
        own text: what the model wrote, each string argument, each result. So
        the order and the spelling are the template's, and nothing of the
        turn's text is read as a tag.

        Args:
            turn: The turn.
            open_turn: Leave the turn open after its last result, for the reply
                to go on with: rendered with nothing after it, and without the
                closing tag a template may add there.

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
            + _turn_messages(turn, field, frags.trim_content) + after)
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
        """Ensure the turn-terminator is a real special token of this model.

        A terminator that splits into plain text never lets the model stop, so
        it would run to the token cap every generation.
        """
        if not hasattr(self._context, "tokenizes_to_special"):
            return
        terminator = self._fragments.assistant_suffix.strip()
        if terminator and not self._context.tokenizes_to_special(terminator):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not tokenize to a special token, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _validate_formatter(self) -> None:
        """Ensure the formatter still opens and closes the model's own turn.

        A formatter may add to what is decoded around sampling, an empty
        reasoning channel for instance, but the turn tags stay the model's own
        or the conversation is encoded as the model never saw it. Read here
        rather than trusted at the first generation, and at the first resumed
        turn, which is later still.

        Raises:
            ValueError: If the generation prompt does not open the assistant
                turn, if the terminator does not close it, or if a resumption
                would open the turn a second time.
        """
        prefix = self._fragments.assistant_prefix
        suffix = self._fragments.assistant_suffix
        if not self._formatter.assistant_open().startswith(prefix):
            raise ValueError(
                f"the formatter's assistant_open() must begin with the "
                f"assistant tag {prefix!r}: it opens the turn the reply is "
                f"generated into")
        if not self._formatter.assistant_close().endswith(suffix):
            raise ValueError(
                f"the formatter's assistant_close() must end with the turn "
                f"terminator {suffix!r}: it is what closes the generated turn")
        if prefix and prefix in self._formatter.resume_open():
            raise ValueError(
                f"the formatter's resume_open() must not repeat the assistant "
                f"tag {prefix!r}: the turn it would open is already open")
        system_prefix = self._fragments.system_prefix
        if system_prefix and system_prefix in self._formatter.system_open():
            raise ValueError(
                f"the formatter's system_open() must not repeat the system tag "
                f"{system_prefix!r}: it is decoded inside that turn")

    def _check_fragments_match_model(self) -> None:
        """Assert the user and assistant fragments match the model's own template.

        Tags that do not reproduce what the model saw in training make
        incremental prefill write tokens it never learned. The system fragment
        is not probed, there being no portable ground truth for it.
        """
        if not hasattr(self._context, "render_with_model_template"):
            return
        # The surrounding whitespace catches a wrong `trim_content` too.
        probe = [
            {"role": "user", "content": " ping "},
            {"role": "assistant", "content": " pong "},
        ]
        rendered = self._context.render_with_model_template(probe)
        if rendered is None:
            return
        model_tokens = self._context.tokenize(rendered, add_special=False)
        # Built exactly as real prefill builds it, tags special-on and content
        # special-off, so any mismatch is a tag or trim divergence.
        fragment_tokens: list[int] = []
        for m in probe:
            prefix, content, suffix = self._formatter.parts(m["role"], m["content"])
            fragment_tokens.extend(
                self._context.tokenize_fragment(prefix, content, suffix))
        if model_tokens != fragment_tokens:
            fragment_text = "".join(self._formatter.fragment(m["role"], m["content"]) for m in probe)
            raise ValueError(
                "chat template fragments do not match the model's trained "
                "template:\n"
                f"  model renders:    {rendered!r}\n"
                f"  fragments render: {fragment_text!r}\n"
                "The recovered fragments do not round-trip through the model's "
                "template; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _resolve_tool_calls(self) -> ToolCallFragments | None:
        """What this model's template wraps a generated call in, if anything."""
        if not hasattr(self._context, "render_with_model_template"):
            return None
        return extract_tool_fragments(self._context.render_with_model_template,
                                      self._fragments)

    def _resolve_tool_tokens(self) -> tuple[tuple[int, int] | None, str]:
        """This model's call tags as tokens, and the text it writes before a name."""
        fragments = self._tool_calls
        if fragments is None:
            return None, ""
        ensure_context_answers(self._context, _TOOL_SURFACE)
        opener = self._context.first_special_token(fragments.call_open)
        closer = self._context.first_special_token(fragments.call_close)
        if opener is None or closer is None:
            return None, ""
        tag = self._context.token_text(opener)
        lead_in = fragments.call_open[len(tag):] \
            if fragments.call_open.startswith(tag) else ""
        return (opener, closer), lead_in

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

    def _plain_system_turn(self) -> str | None:
        """The system turn this model renders with no tools, prompt sentinelled."""
        return self._context.render_with_model_template(
            [{"role": "system", "content": _PROMPT}])

    def _declaration_turn(self, specs: list[ToolSpec]) -> _SystemTurnSplit:
        """The system turn this model renders for ``tools``, and without them.

        Rendered by the template, so no declaration shape is written down here.

        Raises:
            ValueError: If the template renders no declarations.
        """
        rendered = self._context.render_with_model_template(
            [{"role": "system", "content": _PROMPT}],
            tools=[spec.as_declaration() for spec in specs])
        plain = self._plain_system_turn()
        if rendered is None or plain is None or _PROMPT not in rendered \
                or _PROMPT not in plain or rendered == plain:
            raise ValueError("this model's template declares no tools")
        declared_head, _, declared_tail = rendered.partition(_PROMPT)
        plain_head, _, plain_tail = plain.partition(_PROMPT)
        return _SystemTurnSplit(declared_head, declared_tail,
                                plain_head, plain_tail)


def _turn_messages(turn: ToolTurn, field, trim: bool) -> list[dict]:
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
                             "tool_call_id": f"call_{results}",
                             "content": field(result_text(step.result))})
            results += 1
            assistant = None
            continue
        if assistant is None:
            assistant = {"role": "assistant", "content": "", "tool_calls": []}
            messages.append(assistant)
        if isinstance(step, ToolCall):
            assistant["tool_calls"].append({
                "id": f"call_{calls}", "type": "function",
                "function": {"name": step.name,
                             "arguments": _marked(step.arguments, field)}})
            calls += 1
        elif step.strip() if trim else step:
            assistant["content"] = field(step.strip() if trim else step)
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
