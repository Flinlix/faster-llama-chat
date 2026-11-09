"""A model-free stand-in for :class:`~llama_chat.context.KVContext`.

It implements the surface :class:`~llama_chat.wrapper.ChatWrapper` and its
resolved template call on a cache context: a template of its own, rendered
from its fragments, and a *physical cache* (a plain list of token ids) so tests
can assert the cache holds what the wrapper holds it to be.

``prefill`` and ``decode_generated`` assert ``start_pos == len(cache)``:
every decode lands at the cache's end, after whatever cut, restore or rebuild
put it there.

A generation is the library's own loop drawing this fake model's scripted
tokens: its next call where its call tag may be drawn, else its reply, then
the end of generation.

Each cache edit asserts the ability it needs, so a wrapper that cuts a cache
which cannot cut, or shifts one which cannot shift, fails here.

Tokenizer: one token per character and, where special-token parsing is on, one
per tag, with a BOS sentinel (``-1``) prepended when ``add_special`` is set. A
tag is any text spelled inside angle brackets, each a token of its own from
the first time it is met; the tags this fake model lists are those its
fragments hold. Deterministic and easy to reason about.
"""

from __future__ import annotations

import re
from dataclasses import fields, replace

from llama_chat.cache_context import Eviction
from llama_chat.template import TemplateFragments

BOS = -1

# Every tag this fake model writes is spelled inside angle brackets.
_TAG = re.compile(r"<[^<>\s]+>")

# Tag tokens, outside the characters' range so that ordinary text never
# produces them: the call tags, and every other tag below _FIRST_TAG.
TOOL_OPEN, TOOL_CLOSE = -10, -11
_FIRST_TAG = -100
# The token that ends a generation, which no text produces either.
END_OF_GENERATION = -2
# What this fake model writes between its call tag and the function's name.
CALL_LEAD_IN = "fn:"

# The default fragments for tests. The tags are invented, not any real
# model's, so nothing can pass by matching a format hard-coded somewhere.
SYSTEM_PREFIX, SYSTEM_SUFFIX = "<[turn>system\n", "<turn]>\n"
DEFAULT_FRAGMENTS = TemplateFragments(
    user_prefix="<[turn>user\n", user_suffix="<turn]>\n",
    assistant_prefix="<[turn>model\n", assistant_suffix="<turn]>\n",
)


# This fake model's reasoning, when it has any: a switch at the top of the
# system turn, an empty thought opening a reply that does not reason, and the
# tags around a thought. Invented, as the tags above are.
THINK_ON = "<think-on>\n"
THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"
EMPTY_THOUGHT = THOUGHT_OPEN + THOUGHT_CLOSE


def _fake_value(value):
    """This fake model quotes a string and emits anything else bare."""
    return f"'{value}'" if isinstance(value, str) else str(value)


def _call_blocks(calls) -> list[str]:
    """This fake model's calls, one block each."""
    blocks = []
    for call in calls:
        fn = call["function"]
        body = ",".join(f"{k}:{_fake_value(v)}" for k, v in fn["arguments"].items())
        blocks.append(f"<call>{CALL_LEAD_IN}{fn['name']}{{{body}}}</call>")
    return blocks


class FakeContext:
    def __init__(self, gen_tokens: int = 5, gen_text: str = "reply",
                 can_shift: bool = True,
                 fragments: TemplateFragments = DEFAULT_FRAGMENTS,
                 calls_tools: bool = False,
                 tokens_spell_text: bool = True,
                 closes_turn_before_next: bool = False,
                 can_reason: bool = False,
                 user_join: str | None = "",
                 keeps_thought: bool = False,
                 declares_in_newest_turn: bool = False,
                 can_cut_tail: bool = False) -> None:
        self.cache: list[int] = []
        self.can_cut_tail = can_cut_tail
        # Call bodies this fake model will produce, one per generation that may
        # draw its call tag.
        self.call_script: list[str] = []
        self._fragments = fragments
        # A second model, one that writes tool calls. Its tags are recovered
        # from the renders below, the way a real model's are.
        self._calls_tools = calls_tools
        # As llama.cpp's own renderer does: a turn left open after a result is
        # closed when another message follows it.
        self._closes_turn_before_next = closes_turn_before_next
        # A third model, one with a reasoning switch.
        self._can_reason = can_reason
        # What this template writes between two user turns in a row; ``None``
        # for one that refuses them.
        self._user_join = user_join
        # Whether a past reply keeps its thought, where one that called no tool
        # otherwise loses it.
        self._keeps_thought = keeps_thought
        # Whether tools are declared right before the newest user turn rather
        # than in the system turn.
        self._declares_in_newest_turn = declares_in_newest_turn
        if can_reason:
            self._fragments = replace(
                fragments, generation_prompt=fragments.assistant_prefix + EMPTY_THOUGHT)
        self._tag_ids: dict[str, int] = {"<call>": TOOL_OPEN, "</call>": TOOL_CLOSE}
        self._tag_texts: dict[int, str] = {TOOL_OPEN: "<call>", TOOL_CLOSE: "</call>"}
        self._listed = tuple(self._tag_token(tag) for tag in sorted(set(_TAG.findall("".join(
            getattr(self._fragments, field.name) for field in fields(self._fragments)
            if isinstance(getattr(self._fragments, field.name), str))))))
        self.tokenize_calls: list[tuple[str, bool]] = []  # (text, parse_special)
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.evictions: list[Eviction] = []
        self.can_shift = can_shift
        self.context_size = 1 << 20      # no limit of its own: the config's is the one tested
        self.rebuilds = 0  # how many times the cache was rebuilt (no-shift path)
        self.restores = 0  # how many times a saved state was gone back to
        self.saved_states = 0
        self.warmups = 0  # how many times warmup() was called
        self._gen_tokens = gen_tokens
        self._gen_text = gen_text
        # Generated tokens are the characters' own, as the tokenizer would
        # give them; otherwise ids no tokenization reproduces, as a model that
        # generates a word in other pieces than its text tokenizes into.
        self._tokens_spell_text = tokens_spell_text
        self._next_gen = 1000
        self._generated_chars: dict[int, str] = {}   # what each generated token spells
        self.samplings = []  # the sampling each generation was handed

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        # Recorded so tests can assert content is tokenized with parse_special off.
        self.tokenize_calls.append((text, parse_special))
        toks, at = [], 0
        for found in _TAG.finditer(text) if parse_special else ():
            toks += [ord(c) for c in text[at:found.start()]]
            toks.append(self._tag_token(found.group()))
            at = found.end()
        toks += [ord(c) for c in text[at:]]
        return [BOS] + toks if add_special else toks

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

    def chat_template_text(self) -> str | None:
        return None                 # the renders below stand in for one

    def addon_call(self) -> dict:
        return {}

    # ----- cache edits ---------------------------------------------------
    def reset(self) -> None:
        self.cache = []
        self.rebuilds += 1  # begin() and every no-shift rebuild call this

    def warmup(self) -> None:
        # Real warmup nets to an empty cache; begin calls it post-reset, so just
        # record the call.
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
        self.restores += 1

    def remove_from(self, pos: int) -> None:
        assert self.can_shift or self.can_cut_tail, "a cache that cannot cut was cut"
        assert 0 <= pos <= len(self.cache)
        del self.cache[pos:]

    def apply_eviction(self, ev: Eviction) -> None:
        assert self.can_shift, "a cache that cannot shift was shifted"
        assert len(self.cache) == ev.old_total
        del self.cache[ev.remove_start:ev.remove_end]  # remove + implicit shift
        self.evictions.append(ev)

    def token_text(self, token: int) -> str:
        return self._tag_texts.get(token, "")

    def ends_generation(self, token: int) -> bool:
        return token == END_OF_GENERATION

    def token_bytes(self, token: int) -> bytes:
        return self.text_of([token]).encode()

    def text_of(self, tokens: list[int]) -> str:
        """What ``tokens`` spell: characters, the generated ones as they were generated, and the tags."""
        return "".join(self._generated_chars.get(t) or self.token_text(t)
                       or (chr(t) if 0 < t < 0x110000 else "") for t in tokens)

    def render_with_model_template(self, messages: list[dict],
                                   tools: list[dict] | None = None, *,
                                   generation_prompt: bool = False,
                                   reasoning: bool | None = None, **variables) -> str:
        """Render just enough for the wrapper to slice declarations and results.

        A call leaves its turn open, and the result and the prose after it stay
        inside that same turn. That is the property the whole exchange rests on,
        so this fake has to have it too. A newest reply's thought is written
        only in a turn that calls a tool, as one real template does.
        """
        if any(not isinstance(m.get("content", ""), str) for m in messages):
            return None             # content blocks: a shape this template refuses
        renders_reasoning = self._can_reason and reasoning is not False
        frags = self._fragments
        opening = {"system": SYSTEM_PREFIX, "user": frags.user_prefix,
                   "model": frags.assistant_prefix}
        closing = {"system": SYSTEM_SUFFIX, "user": frags.user_suffix,
                   "model": frags.assistant_suffix}
        out, turn_open, previous = [], False, None
        newest_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
        for index, m in enumerate(messages):
            role = "model" if m["role"] == "assistant" else m["role"]
            if role == previous == "user":
                if self._user_join is None:
                    return None
                out.append(self._user_join)
            previous = role
            if turn_open and role != "tool" and role != "model" \
                    and self._closes_turn_before_next:
                out.append("<turn]>\n")
                turn_open = False
            if role == "tool":
                out.append(f"<result>{m['name']}({m['content']})</result>")
                continue
            calls = m.get("tool_calls", ()) if self._calls_tools else ()
            if role == "model" and turn_open:
                # Text after a result, and another round of calls, go on in the same turn.
                out.append(m["content"].strip())
                out.extend(_call_blocks(calls))
                if not calls:
                    out.append(closing["model"])
                    turn_open = False
                continue
            if self._declares_in_newest_turn and index == newest_user:
                for declared in tools or ():
                    out.append(f"<decl>{declared['function']['name']}</decl>")
            out.append(opening[role])
            if role == "system" and renders_reasoning:
                out.append(THINK_ON)
            if (calls or self._keeps_thought) and self._can_reason and m.get("reasoning_content"):
                out.append(THOUGHT_OPEN + m["reasoning_content"] + THOUGHT_CLOSE)
            out.extend(_call_blocks(calls))
            out.append(m["content"].strip())
            if role == "system" and not self._declares_in_newest_turn:
                for declared in tools or ():
                    out.append(f"<decl>{declared['function']['name']}</decl>")
            if calls:
                turn_open = True               # the result belongs in this turn
            else:
                out.append(closing[role])
        if generation_prompt and not turn_open:
            out.append(opening["model"])
            if self._can_reason and not renders_reasoning:
                out.append(EMPTY_THOUGHT)
        return "".join(out)

    def stated_render(self, messages, tools=None, *, generation_prompt=False, reasoning=False):
        """Its render: this template reads its reasoning switch itself, and nothing is read off it."""
        return self.render_with_model_template(messages, tools, generation_prompt=generation_prompt,
                                               reasoning=reasoning)

    def sampled_tokens(self, sampling, withheld_tokens):
        """The next scripted call where its call tag may be drawn, else the scripted reply; then the end of generation."""
        self.samplings.append(sampling)
        if self.call_script and TOOL_OPEN not in withheld_tokens:
            yield from self._call_tokens(self.call_script.pop(0))
        else:
            for i in range(self._gen_tokens):
                yield self._generated(self._gen_text[i] if i < len(self._gen_text) else "x")
        while True:
            yield END_OF_GENERATION

    def _call_tokens(self, body: str):
        """A scripted call: its tags around the call's text, which opens with the lead-in."""
        yield TOOL_OPEN
        for ch in body if body.startswith(CALL_LEAD_IN) else CALL_LEAD_IN + body:
            yield self._generated(ch)
        yield TOOL_CLOSE

    def _generated(self, ch: str) -> int:
        """The token this fake model generates ``ch`` as."""
        if self._tokens_spell_text:
            token = ord(ch)
        else:
            token, self._next_gen = self._next_gen, self._next_gen + 1
        self._generated_chars[token] = ch
        return token

    def decode_generated(self, token_ids: list[int], start_pos: int) -> None:
        assert start_pos == len(self.cache), (start_pos, len(self.cache))
        self.cache.extend(token_ids)

    def close(self) -> None:
        pass

    # ----- test helpers --------------------------------------------------
    def total_prefilled(self) -> int:
        return sum(n for _, n, _ in self.prefill_calls)
