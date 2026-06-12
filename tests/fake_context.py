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

BOS = -1


class FakeContext:
    def __init__(self, gen_len: int = 5, gen_text: str = "reply",
                 can_shift: bool = True) -> None:
        self.cache: list[int] = []
        self.prefill_calls: list[tuple[int, int, bool]] = []  # (start_pos, n, want_logits)
        self.evictions: list[Eviction] = []
        self.can_shift = can_shift
        self.rebuilds = 0  # how many times the cache was rebuilt (no-shift path)
        self._gen_len = gen_len
        self._gen_text = gen_text
        self._next_gen = 1000

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        toks = [ord(c) for c in text]
        return [BOS] + toks if add_special else toks

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t >= 0)

    # ----- cache edits ---------------------------------------------------
    def reset(self) -> None:
        self.cache = []
        self.rebuilds += 1  # begin() and every no-shift rebuild call this

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
