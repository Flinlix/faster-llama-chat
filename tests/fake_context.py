"""A model-free stand-in for :class:`~llama_chat.context.KVContext`, worked over a list of ints.

It renders with a test template's renderer, the one a model is loaded with.
Only the tokenizer is a stand-in, one token per character and one per tag, so
what is compared is the text. The cache is a plain list of token ids, and the
model is a script.
"""

from __future__ import annotations

import functools
import json
import re
import sys

from llama_chat.addon_files import addon_file
from llama_chat.model_template import ModelTemplate
from llama_chat.resolved_template import ResolvedTemplate


class _Renders:
    """A template's renderer, and its renders remembered, up to a bound, where only their inputs decide them."""

    _BOUND = 200                    # renders kept before the memory starts over

    def __init__(self, text: str, eos_text: str) -> None:
        self._model = ModelTemplate(text, eos_text)
        # A template that reads the clock may render the same inputs otherwise later.
        self._remembers = "strftime_now" not in text
        self._known: dict = {}

    def render(self, messages, tools=None, *, generation_prompt=False, reasoning=None,
               variables=None) -> str | None:
        key = self._key(messages, tools, generation_prompt, reasoning, variables or {})
        if key is not None and key in self._known:
            return self._known[key]
        rendered = self._model.render(messages, tools, generation_prompt=generation_prompt,
                                      reasoning=reasoning, variables=variables)
        if key is not None:
            if len(self._known) > self._BOUND:
                self._known.clear()
            self._known[key] = rendered
        return rendered

    def _key(self, messages, tools, generation_prompt, reasoning, variables) -> str | None:
        """The inputs spelled exactly, ``None`` where they are not plain data or the template reads the clock."""
        if not self._remembers:
            return None
        plain = (messages, tools, generation_prompt, reasoning, sorted(variables.items()))
        try:
            json.dumps(plain)       # plain data only, whose spelling holds no address
        except (TypeError, ValueError):
            return None
        return repr(plain)


@functools.cache
def _compiled(text: str, eos_text: str) -> _Renders:
    """A template's renderer, compiled once for every context over the same text."""
    return _Renders(text, eos_text)


# Every tag-shaped text of a test template, the invented ones' turn tags included.
TAG = re.compile(r"<\|[^<>\s]*?\|?>|<[^<>\s|]+\|>|<[/a-z_]+(?::[a-z0-9_]+)?>|\[[A-Z_]+\]"
                 r"|<｜[^｜]+｜>|\]~!?[a-z][\[\]]|\[[a-z]~\[|<\[[a-z]+>|<[a-z]+\]>")

# The sequence opener, which spells no text, and the token that ends a
# generation, which no text tokenizes into. Tags count down below both.
BOS = -1
END_OF_GENERATION = -2
_FIRST_TAG = -3
# Where the ids of generated tokens that spell no character's own start: past
# every character's code point.
_FIRST_GENERATED = sys.maxunicode + 1


class ScriptedCall:
    """A call the context's model writes in place of a reply, ``body`` between its tags, ``before`` it as reply text."""

    def __init__(self, body: str, before: str = "") -> None:
        self.body = body
        self.before = before


def shipped_addon_call(template_text: str) -> dict:
    """How the shipped add-on file that names ``template_text`` says a call is written, as a wrapper finds it."""
    return addon_file(None, template_text)["call"]


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
    and the opening of each closing tag among them. Where the fixture says
    ``adds_bos``, :data:`BOS` comes first where the opener is asked for.

    A generation is the library's own loop drawing, in this order: the next of
    ``call_script`` where a call's opening may be drawn, else the next of
    ``replies``, else ``gen_text`` cut or padded to ``gen_tokens``; then the
    end of generation. It first hands ``before_reply`` the cache's text, so a
    test can compare it with the template's render at that moment.

    Every decode lands at the cache's end, and each cache edit asserts the
    ability it needs, so a wrapper that cuts a cache which cannot cut, or
    shifts one which cannot shift, fails here.

    Args:
        fixture: The template to render with, as a test template's fixture
            holds it.
        replies: What the model generates, one per generation in order: a
            text, or a :class:`ScriptedCall`.
        tokens_spell_text: Whether a generated token is its character's own,
            as the tokenizer would give it; otherwise an id no tokenization
            gives, as a model that generates a word in other pieces than its
            text tokenizes into.
    """

    def __init__(self, fixture: dict, replies=(), *, gen_tokens: int = 5,
                 gen_text: str = "reply", can_shift: bool = True, can_cut_tail: bool = False,
                 tokens_spell_text: bool = True) -> None:
        self._fixture = fixture
        self._model = _compiled(fixture["chat_template"], fixture["eos_token"])
        self._template = fixture["chat_template"]
        self._stated = fixture["render_variables"]
        self._adds_bos = fixture.get("adds_bos", False)
        self._tag_ids: dict[str, int] = {}
        self._tag_texts: dict[int, str] = {}
        found = set(TAG.findall(self._template))
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
        self._next_generated = _FIRST_GENERATED
        self._generated_chars: dict[int, str] = {}   # what each generated token spells
        # What was asked of it, in order.
        self.tokenize_calls: list[tuple[str, bool]] = []      # (text, parse_special)
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.rendered: list[int] = []            # how many messages each render was given
        # ("cut", position), ("shift", start, tokens) and ("restore", position).
        self.edits: list[tuple] = []
        self.samplings = []                      # the sampling each generation was handed
        self.recommended = {}                    # the sampling the model states for itself
        self.resets = 0
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
        return chr(token) if 0 < token <= sys.maxunicode else ""

    def text_of(self, tokens: list[int]) -> str:
        """What ``tokens`` spell: characters, the generated ones as they were generated, and the tags."""
        return "".join(self.token_text(token) for token in tokens)

    def token_bytes(self, token: int) -> bytes:
        return self.token_text(token).encode()

    def render_with_model_template(self, messages: list[dict],
                                   tools: list[dict] | None = None, *,
                                   generation_prompt: bool = False,
                                   reasoning: bool | None = None,
                                   variables: dict | None = None) -> str | None:
        self.rendered.append(len(messages))
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=reasoning, variables=variables)

    def stated_render(self, messages, tools=None, *, generation_prompt=False, reasoning=False):
        """The render with the variables the fixture states for ``reasoning``, none of them read off the template, and not recorded."""
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=None, variables=self._stated["on" if reasoning else "off"])

    def chat_template_text(self) -> str | None:
        return self._template

    def recommended_sampling(self) -> dict:
        return dict(self.recommended)

    # ----- the cache ---------------------------------------------------------------
    def reset(self) -> None:
        self.cache = []
        self.resets += 1

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

    def apply_eviction(self, start: int, end: int) -> None:
        assert self.can_shift, "a cache that cannot shift was shifted"
        assert 0 <= start < end <= len(self.cache), (start, end, len(self.cache))
        self.edits.append(("shift", start, end - start))
        del self.cache[start:end]  # remove, then close the gap

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
            yield from self._call_tokens(ScriptedCall(self.call_script.pop(0)))
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
            reader = FakeContext(self._fixture)
            template = ResolvedTemplate(reader, shipped_addon_call(self._template))
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
