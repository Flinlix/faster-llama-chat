"""A model-free stand-in for :class:`~llama_chat.context.KVContext`.

It implements the exact surface :class:`~llama_chat.wrapper.ChatWrapper` uses and
maintains a *physical cache* (a plain list of token ids) so tests can assert the
cache stays in lockstep with the message table.

Two correctness guarantees fall out of this fake:

* ``prefill`` asserts ``start_pos == len(cache)`` - i.e. the wrapper only ever
  *appends*. It never re-prefills survivors. That is the whole point of the
  design, enforced mechanically.
* After eviction the physical cache equals the concatenation of the surviving
  messages' tokens - proving remove-and-shift closes the gap correctly.

Tokenizer: one token per character, with a BOS sentinel (``-1``) prepended when
``add_special`` is set. Deterministic and easy to reason about.
"""

from __future__ import annotations

from llama_chat.context import GenerationAccumulator
from llama_chat.messages import Eviction
from llama_chat.template import Fragments

BOS = -1

# Gemma-style tags; the default fragments for tests.
GEMMA_FRAGMENTS = Fragments(
    system_prefix="<|turn>user\n", system_suffix="<turn|>\n",
    user_prefix="<|turn>user\n", user_suffix="<turn|>\n",
    assistant_prefix="<|turn>model\n", assistant_suffix="<turn|>\n",
    trim_content=True,
)


class FakeContext:
    def __init__(self, gen_len: int = 5, gen_text: str = "reply",
                 can_shift: bool = True, fragments: Fragments = GEMMA_FRAGMENTS) -> None:
        self.cache: list[int] = []
        self._fragments = fragments
        self.tokenize_calls: list[tuple[str, bool]] = []  # (text, parse_special)
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.evictions: list[Eviction] = []
        self.can_shift = can_shift
        self.rebuilds = 0  # how many times the cache was rebuilt (no-shift path)
        self.warmups = 0  # how many times warmup() was called
        self._gen_len = gen_len
        self._gen_text = gen_text
        self._next_gen = 1000

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

    def extract_fragments(self) -> Fragments:
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

    def apply_eviction(self, ev: Eviction) -> None:
        assert len(self.cache) == ev.old_total
        del self.cache[ev.remove_start:ev.remove_end]  # remove + implicit shift
        self.evictions.append(ev)

    def generate(self, start_pos: int, n_predict_max: int, stop: list[str],
                 out: GenerationAccumulator | None = None):
        """Stream generated tokens, mirroring KVContext.generate's contract.

        Each token is appended to the cache *before* its text delta is yielded,
        and ``out`` is filled incrementally, so an early ``close()`` leaves the
        cache and ``out.token_ids`` in agreement (the barge-in guarantee).
        """
        assert start_pos == len(self.cache)
        gen = out if out is not None else GenerationAccumulator()
        n = max(0, min(self._gen_len, n_predict_max))
        for i in range(n):
            tok = self._next_gen + i
            self.cache.append(tok)  # decoded into the cache before surfacing text
            gen.token_ids.append(tok)
            ch = self._gen_text[i] if i < len(self._gen_text) else "x"
            gen.text += ch
            yield ch
        self._next_gen += n
        gen.stop_reason = "eog" if n == self._gen_len else "length"
        return gen

    def close(self) -> None:
        pass

    # ----- test helpers --------------------------------------------------
    def total_prefilled(self) -> int:
        return sum(n for _, n, _ in self.prefill_calls)
