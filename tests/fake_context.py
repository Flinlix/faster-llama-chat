"""A model-free stand-in for :class:`~llama_chat.context.KVContext`, worked over a list of ints.

It renders with a test template's renderer, the one a model is loaded with, or
without one with an invented format of its own. Only the tokenizer is a
stand-in, one token per character and one per tag, so what is compared is the
text. The cache is a plain list of token ids, and the model is a script.
"""

from __future__ import annotations

import functools
import re
from dataclasses import fields, replace

from llama_chat.addon_files import addon_file
from llama_chat.model_template import ModelTemplate
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.template import TemplateFragments


@functools.cache
def _compiled(text: str, eos_text: str) -> ModelTemplate:
    """A template's renderer, compiled once for every context over the same text."""
    return ModelTemplate(text, eos_text)


# Every tag-shaped text of a test template, and the invented format's turn tags.
TAG = re.compile(r"<\|[^<>\s]*?\|?>|<[^<>\s|]+\|>|<[/a-z_]+(?::[a-z0-9_]+)?>|\[[A-Z_]+\]"
                 r"|<｜[^｜]+｜>|\]~!?[a-z][\[\]]|\[[a-z]~\[|<\[[a-z]+>|<[a-z]+\]>")

# The sequence opener, which spells no text, and the token that ends a
# generation, which no text tokenizes into. Tags count down below both.
BOS = -1
END_OF_GENERATION = -2
_FIRST_TAG = -3


# ----- the invented format ---------------------------------------------------------
# The tags are invented, not any real model's, so nothing can pass by matching a
# format hard-coded somewhere.
SYSTEM_PREFIX, SYSTEM_SUFFIX = "<[turn>system\n", "<turn]>\n"
DEFAULT_FRAGMENTS = TemplateFragments(
    user_prefix="<[turn>user\n", user_suffix="<turn]>\n",
    assistant_prefix="<[turn>model\n", assistant_suffix="<turn]>\n",
)
# Its reasoning, where it has any: a switch at the top of the system turn, an
# empty thought opening a reply that does not reason, and the tags around a thought.
THINK_ON = "<think_on>\n"
THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"
EMPTY_THOUGHT = THOUGHT_OPEN + THOUGHT_CLOSE
# The tags around a call, and what it writes between its opening and the function's name.
CALL_TAGS = ("<call>", "</call>")
CALL_LEAD_IN = "fn:"


def _invented_value(value):
    """The invented format quotes a string and writes anything else bare."""
    return f"'{value}'" if isinstance(value, str) else str(value)


def _call_blocks(calls) -> list[str]:
    """Calls as the invented format writes them, one block each."""
    blocks = []
    for call in calls:
        fn = call["function"]
        body = ",".join(f"{k}:{_invented_value(v)}" for k, v in fn["arguments"].items())
        blocks.append(f"{CALL_TAGS[0]}{CALL_LEAD_IN}{fn['name']}{{{body}}}{CALL_TAGS[1]}")
    return blocks


class _InventedFormat:
    """A template of the context's own, rendered by hand from its fragments and flags.

    A call leaves its turn open, and the result and the prose after it stay
    inside that same turn. A newest reply's thought is written only in a turn
    that calls a tool, as one real template does.

    Args:
        calls_tools: Whether a call is written; without it, a call is dropped.
        closes_turn_before_next: Whether a turn left open after a result is
            closed when another message follows it.
        can_reason: Whether the template has a reasoning switch.
        user_join: What is written between two user turns in a row; ``None``
            refuses them.
        keeps_thought: Whether a past reply keeps its thought, where one that
            called no tool otherwise loses it.
        declares_in_newest_turn: Whether tools are declared right before the
            newest user turn rather than in the system turn.
    """

    def __init__(self, fragments: TemplateFragments = DEFAULT_FRAGMENTS,
                 calls_tools: bool = False, closes_turn_before_next: bool = False,
                 can_reason: bool = False, user_join: str | None = "",
                 keeps_thought: bool = False, declares_in_newest_turn: bool = False) -> None:
        self.fragments = fragments
        self.calls_tools = calls_tools
        self.closes_turn_before_next = closes_turn_before_next
        self.can_reason = can_reason
        self.user_join = user_join
        self.keeps_thought = keeps_thought
        self.declares_in_newest_turn = declares_in_newest_turn
        if can_reason:
            self.fragments = replace(
                fragments, generation_prompt=fragments.assistant_prefix + EMPTY_THOUGHT)

    def tag_source(self) -> str:
        """The text its listed tags are read from: its fragments."""
        return "".join(getattr(self.fragments, field.name) for field in fields(self.fragments)
                       if isinstance(getattr(self.fragments, field.name), str))

    def render(self, messages, tools=None, *, generation_prompt=False, reasoning=None):
        if any(not isinstance(m.get("content", ""), str) for m in messages):
            return None             # content blocks: a shape this template refuses
        renders_reasoning = self.can_reason and reasoning is not False
        frags = self.fragments
        opening = {"system": SYSTEM_PREFIX, "user": frags.user_prefix,
                   "model": frags.assistant_prefix}
        closing = {"system": SYSTEM_SUFFIX, "user": frags.user_suffix,
                   "model": frags.assistant_suffix}
        out, turn_open, previous = [], False, None
        newest_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
        for index, m in enumerate(messages):
            role = "model" if m["role"] == "assistant" else m["role"]
            if role == previous == "user":
                if self.user_join is None:
                    return None
                out.append(self.user_join)
            previous = role
            if turn_open and role != "tool" and role != "model" \
                    and self.closes_turn_before_next:
                out.append("<turn]>\n")
                turn_open = False
            if role == "tool":
                out.append(f"<result>{m['name']}({m['content']})</result>")
                continue
            calls = m.get("tool_calls", ()) if self.calls_tools else ()
            if role == "model" and turn_open:
                # Text after a result, and another round of calls, go on in the same turn.
                out.append(m["content"].strip())
                out.extend(_call_blocks(calls))
                if not calls:
                    out.append(closing["model"])
                    turn_open = False
                continue
            if self.declares_in_newest_turn and index == newest_user:
                for declared in tools or ():
                    out.append(f"<decl>{declared['function']['name']}</decl>")
            out.append(opening[role])
            if role == "system" and renders_reasoning:
                out.append(THINK_ON)
            if (calls or self.keeps_thought) and self.can_reason and m.get("reasoning_content"):
                out.append(THOUGHT_OPEN + m["reasoning_content"] + THOUGHT_CLOSE)
            out.extend(_call_blocks(calls))
            out.append(m["content"].strip())
            if role == "system" and not self.declares_in_newest_turn:
                for declared in tools or ():
                    out.append(f"<decl>{declared['function']['name']}</decl>")
            if calls:
                turn_open = True               # the result belongs in this turn
            else:
                out.append(closing[role])
        if generation_prompt and not turn_open:
            out.append(opening["model"])
            if self.can_reason and not renders_reasoning:
                out.append(EMPTY_THOUGHT)
        return "".join(out)


# ----- the context -----------------------------------------------------------------
class ScriptedCall:
    """A call the context's model writes in place of a reply, ``body`` between its tags, ``before`` it as reply text."""

    def __init__(self, body: str, before: str = "") -> None:
        self.body = body
        self.before = before


def call_marks(template, context) -> tuple[str, str]:
    """The texts a call opens and closes with in a render: its own texts, else its tags' text."""
    boundary = template.call_boundary()
    if boundary.call_texts is not None:
        return boundary.call_texts
    opener, closer = boundary.tool_tokens
    return context.token_text(opener), context.token_text(closer)


class FakeContext:
    """A cache context without a model: a template's renderer, a tokenizer stand-in, a list as the cache, and a scripted model.

    The tokenizer gives one token per character and, where special-token
    parsing is on, one per tag-shaped text, each a token of its own from the
    first time it is met: a stand-in for a vocabulary that holds every tag,
    such as one closing each argument of a call. A tag typed into a message,
    which is tokenized with parsing off, stays its characters and cannot
    become the model's own. The tags it lists are those the template spells,
    and the opening of each closing tag among them. The invented format's
    tokenizer puts :data:`BOS` first where asked to; a test template's puts
    nothing there.

    A generation is the library's own loop drawing, in this order: the next of
    ``call_script`` where a call's opening may be drawn, else the next of
    ``replies``, else ``gen_text`` cut or padded to ``gen_tokens``; then the
    end of generation. It first hands ``before_reply`` the cache's text, so a
    test can compare it with the template's render at that moment.

    Every decode lands at the cache's end, and each cache edit asserts the
    ability it needs, so a wrapper that cuts a cache which cannot cut, or
    shifts one which cannot shift, fails here.

    Args:
        fixture: The test template to render with, as its file holds it.
            Without one, the invented format renders, configured by the
            keywords :class:`_InventedFormat` takes.
        replies: What the model generates, one per generation in order: a
            text, or a :class:`ScriptedCall`.
        tokens_spell_text: Whether a generated token is its character's own,
            as the tokenizer would give it; otherwise an id no tokenization
            gives, as a model that generates a word in other pieces than its
            text tokenizes into.
    """

    def __init__(self, fixture: dict | None = None, replies=(), *, gen_tokens: int = 5,
                 gen_text: str = "reply", can_shift: bool = True, can_cut_tail: bool = False,
                 tokens_spell_text: bool = True, **invented) -> None:
        assert fixture is None or not invented, "a test template takes none of the invented flags"
        self._fixture = fixture
        if fixture is None:
            self._format = _InventedFormat(**invented)
            self._template, tag_source = None, self._format.tag_source()
        else:
            self._format = None
            self._model = _compiled(fixture["chat_template"], fixture["eos_token"])
            self._template = tag_source = fixture["chat_template"]
            self._stated = fixture["render_variables"]
        self._adds_bos = fixture is None
        self._tag_ids: dict[str, int] = {}
        self._tag_texts: dict[int, str] = {}
        found = set(TAG.findall(tag_source))
        self._listed = tuple(self._tag_token(tag) for tag in sorted(
            found | {"<" + tag[2:] for tag in found if tag.startswith("</")}))
        self._call_mark_tokens = None
        self.cache: list[int] = []
        self.can_shift = can_shift
        self.can_cut_tail = can_cut_tail
        self.context_size = 1 << 20      # no limit of its own: the config's is the one tested
        # What the model generates.
        self.call_script: list[str] = []
        self.replies = list(replies)
        self.gen_tokens = gen_tokens
        self.gen_text = gen_text
        self.before_reply = None
        self._tokens_spell_text = tokens_spell_text
        self._next_generated = 1000
        self._generated_chars: dict[int, str] = {}   # what each generated token spells
        # What was asked of it, in order.
        self.tokenize_calls: list[tuple[str, bool]] = []      # (text, parse_special)
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.rendered: list[int] = []            # how many messages each render was given
        # ("cut", position), ("shift", start, tokens) and ("restore", position).
        self.edits: list[tuple] = []
        self.samplings = []                      # the sampling each generation was handed
        self.resets = 0
        self.warmups = 0
        self.saved_states = 0

    # ----- the model as it is read ---------------------------------------------
    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        self.tokenize_calls.append((text, parse_special))
        opener = [BOS] if add_special and self._adds_bos else []
        return opener + self._tokens(text, parse_special)

    def _tokens(self, text: str, parse_special: bool) -> list[int]:
        """``text`` tokenized, unrecorded."""
        if not parse_special:
            return [ord(c) for c in text]
        tokens, at = [], 0
        for found in TAG.finditer(text):
            tokens += [ord(c) for c in text[at:found.start()]]
            tokens.append(self._tag_token(found.group()))
            at = found.end()
        return tokens + [ord(c) for c in text[at:]]

    def _tag_token(self, tag: str) -> int:
        """The one token ``tag`` is, given an id of its own the first time it is met."""
        if tag not in self._tag_ids:
            token = _FIRST_TAG - len(self._tag_ids)
            self._tag_ids[tag], self._tag_texts[token] = token, tag
        return self._tag_ids[tag]

    def is_special(self, token: int) -> bool:
        return token in self._tag_texts

    def special_tokens(self) -> tuple[int, ...]:
        return self._listed

    def token_text(self, token: int) -> str:
        if token in self._tag_texts:
            return self._tag_texts[token]
        if token in self._generated_chars:
            return self._generated_chars[token]
        return chr(token) if 0 < token < 0x110000 else ""

    def text_of(self, tokens: list[int]) -> str:
        """What ``tokens`` spell: characters, the generated ones as they were generated, and the tags."""
        return "".join(self.token_text(token) for token in tokens)

    def token_bytes(self, token: int) -> bytes:
        return self.token_text(token).encode()

    def render_with_model_template(self, messages: list[dict],
                                   tools: list[dict] | None = None, *,
                                   generation_prompt: bool = False,
                                   reasoning: bool | None = None, **variables) -> str | None:
        self.rendered.append(len(messages))
        if self._format is not None:
            return self._format.render(messages, tools, generation_prompt=generation_prompt,
                                       reasoning=reasoning)
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=reasoning, **variables)

    def stated_render(self, messages, tools=None, *, generation_prompt=False, reasoning=False):
        """The render with what the template states for ``reasoning``, nothing of it read off the template.

        A test template's is rendered with the variables its file states, and
        is not recorded; the invented format reads its reasoning switch itself.
        """
        if self._format is not None:
            return self.render_with_model_template(messages, tools,
                                                   generation_prompt=generation_prompt,
                                                   reasoning=reasoning)
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=None, **self._stated["on" if reasoning else "off"])

    def chat_template_text(self) -> str | None:
        return self._template

    def addon_call(self) -> dict:
        # The shipped add-on file that names this template, as a model's load finds it.
        return {} if self._template is None else addon_file(None, self._template)["call"]

    # ----- the cache ---------------------------------------------------------------
    def reset(self) -> None:
        self.cache = []
        self.resets += 1

    def warmup(self) -> None:
        # A real warmup nets to an empty cache; begin calls it after a reset.
        self.warmups += 1

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        assert start_pos == len(self.cache), (
            f"non-append prefill at {start_pos} (cache has {len(self.cache)}): "
            "the wrapper re-prefilled existing context"
        )
        self.cache.extend(token_ids)
        self.prefill_calls.append((start_pos, len(token_ids), want_logits))

    def save_state(self):
        self.saved_states += 1
        return list(self.cache)

    def restore_state(self, state, pos: int) -> None:
        assert len(state) == pos
        self.cache = list(state)
        self.edits.append(("restore", pos))

    def remove_from(self, pos: int) -> None:
        assert self.can_shift or self.can_cut_tail, "a cache that cannot cut was cut"
        assert 0 <= pos <= len(self.cache)
        self.edits.append(("cut", pos))
        del self.cache[pos:]

    def apply_eviction(self, ev) -> None:
        assert self.can_shift, "a cache that cannot shift was shifted"
        assert len(self.cache) == ev.old_total
        self.edits.append(("shift", ev.remove_start, ev.shift_delta))
        del self.cache[ev.remove_start:ev.remove_end]  # remove, then close the gap

    def decode_generated(self, token_ids: list[int], start_pos: int) -> None:
        assert start_pos == len(self.cache), (start_pos, len(self.cache))
        self.cache.extend(token_ids)

    def close(self) -> None:
        pass

    # ----- the model -------------------------------------------------------------
    def sampled_tokens(self, sampling, withheld_tokens):
        """The next scripted call where its opening may be drawn, else the next scripted reply, else the default one; then the end of generation."""
        self.samplings.append(sampling)
        if self.before_reply is not None:
            self.before_reply(self.text_of(self.cache))
        if self.call_script and self._call_marks()[0][0] not in withheld_tokens:
            body = self.call_script.pop(0)
            yield from self._call_tokens(ScriptedCall(
                body if body.startswith(CALL_LEAD_IN) else CALL_LEAD_IN + body))
        elif self.replies:
            scripted = self.replies.pop(0)
            if isinstance(scripted, ScriptedCall):
                yield from self._call_tokens(scripted)
            else:
                yield from (self._generated(ch) for ch in scripted)
        else:
            for i in range(self.gen_tokens):
                yield self._generated(self.gen_text[i] if i < len(self.gen_text) else "x")
        while True:
            yield END_OF_GENERATION

    def ends_generation(self, token: int) -> bool:
        return token == END_OF_GENERATION

    def _call_tokens(self, call: ScriptedCall) -> list[int]:
        """The scripted call as the model writes it: its ``before``, then its opening, its body, its closing."""
        opening, closing = self._call_marks()
        return (self._tokens(call.before, parse_special=True) + opening
                + [self._generated(ch) for ch in call.body] + closing)

    def _call_marks(self) -> tuple[list[int], list[int]]:
        """The template's own call tags as tokens, else its call texts as characters."""
        if self._call_mark_tokens is None:
            if self._format is not None:
                marks, by_tags = CALL_TAGS, True
            else:
                reader = FakeContext(self._fixture)
                template = ResolvedTemplate(reader)
                marks = call_marks(template, reader)
                by_tags = template.call_boundary().tool_tokens is not None
            self._call_mark_tokens = tuple(self._tokens(text, parse_special=by_tags)
                                           for text in marks)
        return self._call_mark_tokens

    def _generated(self, ch: str) -> int:
        """The token this model generates ``ch`` as."""
        if self._tokens_spell_text:
            token = ord(ch)
        else:
            token, self._next_generated = self._next_generated, self._next_generated + 1
        self._generated_chars[token] = ch
        return token

    # ----- what was asked of it ------------------------------------------------------
    def total_prefilled(self) -> int:
        return sum(n for _, n, _ in self.prefill_calls)

    def edit_count(self, kind: str) -> int:
        """How many cache edits of ``kind`` were made: ``"cut"``, ``"shift"`` or ``"restore"``."""
        return sum(edit[0] == kind for edit in self.edits)
