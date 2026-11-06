"""A model-free stand-in for :class:`~llama_chat.context.KVContext`.

It implements the surface :class:`~llama_chat.wrapper.ChatWrapper` and its
resolved template call on a cache context, and maintains a *physical cache* (a
plain list of token ids) so tests can assert the cache stays in lockstep with
the message table.

Two correctness guarantees fall out of this fake:

* ``prefill`` asserts ``start_pos == len(cache)``, so the wrapper only ever
  *appends*. It never re-prefills survivors. That is the design guarantee,
  checked mechanically.
* After eviction the physical cache equals the concatenation of the surviving
  messages' tokens, proving remove-and-shift closes the gap correctly.

Tokenizer: one token per character, with a BOS sentinel (``-1``) prepended when
``add_special`` is set. Deterministic and easy to reason about.
"""

from __future__ import annotations

from llama_chat.context import GenerationAccumulator
from llama_chat.messages import Eviction
from llama_chat.template import TemplateFragments

BOS = -1

# Synthetic control tokens, outside the 1:1 char tokenizer's range so they can
# never be produced by ordinary text.
TOOL_OPEN, TOOL_CLOSE = -10, -11
# What this fake model writes between its call tag and the function's name.
CALL_LEAD_IN = "fn:"

# The default fragments for tests. The tags are invented, not any real
# model's, so nothing can pass by matching a format hard-coded somewhere.
DEFAULT_FRAGMENTS = TemplateFragments(
    system_prefix="<|turn>user\n", system_suffix="<turn|>\n",
    user_prefix="<|turn>user\n", user_suffix="<turn|>\n",
    assistant_prefix="<|turn>model\n", assistant_suffix="<turn|>\n",
    trim_content=True,
)


def _fake_value(value):
    """This fake model quotes a string and emits anything else bare."""
    return f"'{value}'" if isinstance(value, str) else str(value)


class FakeContext:
    def __init__(self, gen_tokens: int = 5, gen_text: str = "reply",
                 can_shift: bool = True,
                 fragments: TemplateFragments = DEFAULT_FRAGMENTS,
                 calls_tools: bool = False,
                 tokens_spell_text: bool = False,
                 closes_turn_before_next: bool = False) -> None:
        self.cache: list[int] = []
        # Call bodies this fake model will produce, one per generation.
        self.call_script: list[str] = []
        self._fragments = fragments
        # A second model, one that writes tool calls. Its tags are recovered
        # from the renders below, the way a real model's are.
        self._calls_tools = calls_tools
        # As llama.cpp's own renderer does: a turn left open after a result is
        # closed when another message follows it.
        self._closes_turn_before_next = closes_turn_before_next
        self.tokenize_calls: list[tuple[str, bool]] = []  # (text, parse_special)
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.evictions: list[Eviction] = []
        self.can_shift = can_shift
        self.rebuilds = 0  # how many times the cache was rebuilt (no-shift path)
        self.warmups = 0  # how many times warmup() was called
        self._gen_tokens = gen_tokens
        self._gen_text = gen_text
        # Generated tokens are the characters' own, as the tokenizer would
        # give them, rather than ids no tokenization reproduces.
        self._tokens_spell_text = tokens_spell_text
        self._next_gen = 1000
        self.samplings = []  # the sampling each generation was handed

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        # 1:1 char tokenizer: tokens never merge, so parse_special is a no-op
        # here (a literal tag is the same char tokens either way). Recorded so
        # tests can assert content is tokenized with parse_special off.
        self.tokenize_calls.append((text, parse_special))
        toks = [ord(c) for c in text]
        return [BOS] + toks if add_special else toks

    def tokenize_fragment(self, prefix: str, content: str, suffix: str,
                          add_special: bool = False) -> list[int]:
        return (self.tokenize(prefix, add_special=add_special, parse_special=True)
                + self.tokenize(content, parse_special=False)
                + self.tokenize(suffix, parse_special=True))

    def extract_fragments(self) -> TemplateFragments:
        return self._fragments

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

    def remove_from(self, pos: int) -> None:
        assert 0 <= pos <= len(self.cache)
        del self.cache[pos:]

    def apply_eviction(self, ev: Eviction) -> None:
        assert len(self.cache) == ev.old_total
        del self.cache[ev.remove_start:ev.remove_end]  # remove + implicit shift
        self.evictions.append(ev)

    def first_special_token(self, text: str) -> int | None:
        """Resolve this fake model's two tags; everything else is plain text.

        Answering this here is why a wrong criterion in the real backend went
        unnoticed; `tests/test_backend.py` pins that one directly.
        """
        for tag, token in (("<call>", TOOL_OPEN), ("</call>", TOOL_CLOSE)):
            if tag in text:
                return token
        return None

    def token_text(self, token: int) -> str:
        return {TOOL_OPEN: "<call>", TOOL_CLOSE: "</call>"}.get(token, "")

    def render_with_model_template(self, messages: list[dict],
                                   tools: list[dict] | None = None) -> str:
        """Render just enough for the wrapper to slice declarations and results.

        A call leaves its turn open, and the result and the prose after it stay
        inside that same turn. That is the property the whole exchange rests on,
        so this fake has to have it too.
        """
        out, turn_open = [], False
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            if turn_open and role != "tool" and role != "model" \
                    and self._closes_turn_before_next:
                out.append("<turn|>\n")
                turn_open = False
            if role == "tool":
                out.append(f"<result>{m['name']}({m['content']})</result>")
                continue
            if role == "model" and turn_open:
                out.append(m["content"].strip())
                out.append("<turn|>\n")
                turn_open = False
                continue
            out.append(f"<|turn>{role}\n")
            calls = m.get("tool_calls", ()) if self._calls_tools else ()
            for call in calls:
                fn = call["function"]
                body = ",".join(
                    f"{k}:{_fake_value(v)}" for k, v in fn["arguments"].items())
                out.append(f"<call>{CALL_LEAD_IN}{fn['name']}{{{body}}}</call>")
            out.append(m["content"].strip())   # these fragments trim
            if role == "system":
                for declared in tools or ():
                    out.append(f"<decl>{declared['function']['name']}</decl>")
            if calls:
                turn_open = True               # the result belongs in this turn
            else:
                out.append("<turn|>\n")
        return "".join(out)

    def generate(self, start_pos: int, max_generated_tokens: int, stop: list[str],
                 out: GenerationAccumulator | None = None, cancel=None,
                 tool_tokens: tuple[int, int] | None = None, sampling=None):
        """Stream generated tokens, mirroring KVContext.generate's contract.

        Each token is appended to the cache *before* its text delta is yielded,
        and ``out`` is filled incrementally, so an early ``close()`` leaves the
        cache and ``out.token_ids`` in agreement. Both abort routes record
        ``stop_reason == "interrupted"``: the ``cancel`` event (checked once per
        token) and a ``close()`` of the generator.
        """
        assert start_pos == len(self.cache)
        self.samplings.append(sampling)
        generated = out if out is not None else GenerationAccumulator()
        if tool_tokens is not None and self.call_script:
            yield from self._generate_call(generated, tool_tokens)
            return generated
        n = max(0, min(self._gen_tokens, max_generated_tokens))
        for i in range(n):
            if cancel is not None and cancel.is_set():
                generated.stop_reason = "interrupted"
                break
            ch = self._gen_text[i] if i < len(self._gen_text) else "x"
            tok = ord(ch) if self._tokens_spell_text else self._next_gen + i
            self.cache.append(tok)  # decoded into the cache before surfacing text
            generated.token_ids.append(tok)
            generated.text += ch
            try:
                yield ch
            except GeneratorExit:
                generated.stop_reason = "interrupted"
                raise
        else:
            generated.stop_reason = ("end_of_turn" if n == self._gen_tokens
                                    else "token_limit")
        self._next_gen += n
        return generated

    def _generate_call(self, generated: GenerationAccumulator,
                       tool_tokens: tuple[int, int]):
        """Emit the next scripted call: open tag, body, close tag, no deltas.

        Mirrors the real contract: every token reaches the cache, none of the
        call's text is yielded, and the body lands on ``tool_call`` rather than
        on ``text``.
        """
        body = self.call_script.pop(0)
        opener, closer = tool_tokens
        spelled = ([ord(c) for c in CALL_LEAD_IN + body] if self._tokens_spell_text
                   else range(len(body)))
        for token in (opener, *spelled, closer):
            self.cache.append(token)
            generated.token_ids.append(token)
        generated.tool_call = body
        generated.stop_reason = "tool_call"
        return
        yield        # a generator either way, so the caller can drive it

    def close(self) -> None:
        pass

    # ----- test helpers --------------------------------------------------
    def total_prefilled(self) -> int:
        return sum(n for _, n, _ in self.prefill_calls)
