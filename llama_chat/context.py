"""KV-cache operations: tokenize, prefill, generate, evict-and-shift.

This is the only module that drives llama.cpp directly. It owns the model,
context and sampler, and translates the wrapper's intent into the low-level
calls resolved by :class:`~llama_chat._backend.Backend`. It deliberately knows
nothing about *which* messages to keep - that policy lives in the wrapper and
:mod:`~llama_chat.messages`.
"""

from __future__ import annotations

import codecs
import warnings
from dataclasses import dataclass, field

from ._backend import Backend, set_log_level
from .config import KV_CACHE_GGML_TYPES, Config
from .messages import Eviction
from .template import Fragments

SEQ = 0  # single active conversation -> single sequence id


class _TextBuffer:
    """Accumulated decoded text with a cursor tracking what has been emitted."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def append(self, text: str) -> None:
        self._buf += text

    def emit(self, upto: int) -> str | None:
        if upto > self._emitted:
            delta = self._buf[self._emitted:upto]
            self._emitted = upto
            return delta
        return None

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def text(self) -> str:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class GenerationAccumulator:
    """Live accumulator for one generation, updated as tokens stream in.

    It is filled incrementally by :meth:`KVContext.generate` so that a caller
    which abandons the stream early (barge-in) can still read exactly what
    reached the cache: every token in ``token_ids`` was decoded before its text
    was yielded, so ``token_ids`` and the KV cache never disagree.

    Attributes:
        token_ids: Every token decoded into the cache (the cache truth).
        text: Visible reply text so far, trimmed at the stop string if one hit.
        stop_reason: ``"eog"``, ``"stop"`` or ``"length"``.
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "length"


class KVContext:
    """Owns the llama.cpp model/context and performs all cache edits."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        set_log_level(config.log_level)  # filter llama.cpp's load-time logs too
        self._b = Backend()
        self._model = self._b.load_model(config.model_path, config.gpu_layers)
        self._ctx = None
        self._sampler = None
        try:
            kv_type = KV_CACHE_GGML_TYPES.get(config.kv_cache_type)
            self._ctx = self._b.new_context(
                self._model, config.context_size, config.threads, config.batch_size,
                flash_attn=config.flash_attention, type_k=kv_type, type_v=kv_type)
            self._vocab = self._b.vocab(self._model)
            self._batch_size = config.batch_size
            self._can_shift = self._b.can_shift(self._ctx)
            self._sampler = self._build_sampler(config)
            self._model_formatter = self._build_model_formatter()
        except BaseException:
            # Construction can legitimately fail (bad config, OOM); free the
            # native resources created so far instead of leaking them.
            if self._sampler is not None:
                self._b.lc.llama_sampler_free(self._sampler)
            if self._ctx is not None:
                self._b.free_context(self._ctx)
            self._b.free_model(self._model)
            raise
        self._closed = False

    @property
    def can_shift(self) -> bool:
        """Whether eviction can shift survivors in place vs. rebuild the cache."""
        return self._can_shift

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        return self._b.tokenize(self._vocab, text, add_special, parse_special)

    def tokenize_fragment(self, prefix: str, content: str, suffix: str,
                          add_special: bool = False) -> list[int]:
        """Tokenize one templated message: tags special-on, content special-off.

        The structural ``prefix``/``suffix`` tags are parsed with special-token
        parsing on so they become their control tokens, while ``content`` is
        parsed with it off so a literal tag in untrusted text becomes ordinary
        text tokens and cannot forge a turn boundary. ``add_special`` (BOS)
        attaches only to the leading ``prefix``.
        """
        toks = self.tokenize(prefix, add_special=add_special, parse_special=True)
        toks += self.tokenize(content, add_special=False, parse_special=False)
        toks += self.tokenize(suffix, add_special=False, parse_special=True)
        return toks

    def tokenizes_to_special(self, text: str) -> bool:
        """True if ``text`` tokenizes to exactly one control/special token.

        Used to validate that a configured turn-terminator tag is a real special
        token of *this* model, rather than being silently split into plain-text
        pieces (which would stop generation from ever terminating).
        """
        toks = self.tokenize(text, add_special=False)
        return len(toks) == 1 and self._b.is_special(self._vocab, toks[0])

    def render_with_model_template(self, messages: list[dict]) -> str | None:
        """Render ``messages`` with the model's own GGUF chat template.

        Returns ``None`` when the model ships no ``tokenizer.chat_template`` or
        the template rejects this message shape (e.g. Gemma refusing a system
        role) - callers treat that as "nothing to validate against". BOS is
        suppressed and no generation prompt is added so the render aligns with a
        fragment prefill (``add_special=False``).
        """
        if self._model_formatter is None:
            return None
        try:
            return self._model_formatter(messages=messages).prompt
        except Exception:
            return None

    def extract_fragments(self) -> Fragments | None:
        """Recover the per-role template fragments from the model's own template.

        Returns a :class:`~llama_chat.template.Fragments` derived from the GGUF
        ``tokenizer.chat_template``, or ``None`` when the model ships no template
        (callers must then supply ``ChatWrapper(fragments=...)``).
        """
        from .template_extract import extract_fragments
        return extract_fragments(self.render_with_model_template)

    def _build_model_formatter(self):
        template = self._b.metadata_value(self._model, "tokenizer.chat_template")
        if not template:
            return None
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        return Jinja2ChatFormatter(
            template=template,
            eos_token=self._b.eos_text(self._vocab),
            bos_token="",
            add_generation_prompt=False,
        )

    # ----- cache edits ---------------------------------------------------
    def reset(self) -> None:
        self._b.kv_clear(self._ctx)
        # Forget sampler state (e.g. the repeat-penalty window) along with the
        # cache, so a fresh conversation is not penalized for the previous one.
        self._b.sampler_reset(self._sampler)

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        self._b.decode(self._ctx, token_ids, start_pos, SEQ, want_logits, self._batch_size)

    def warmup(self) -> None:
        """Warm the single-token generation graph (batch-1 decode + logits).

        ``begin``'s prefill only warms the batched-prefill graph; the batch-1
        decode used during generation is first built on the first request, paying
        a one-time graph-capture/allocation cost. Decode one throwaway token with
        logits on the empty cache, then clear it so the real prefill that follows
        starts clean. Mirrors llama.cpp's own model warmup.
        """
        tok = self._b.token_eos(self._vocab)
        if tok < 0:  # some models expose no EOS; any valid id warms the graph
            tok = 0
        self._b.decode(self._ctx, [tok], 0, SEQ, True, self._batch_size)
        self.reset()  # kv_clear -> cache empty again for the real prefill

    def apply_eviction(self, ev: Eviction) -> None:
        """Remove an evicted message's tokens, then shift survivors to close the gap."""
        self._b.kv_seq_rm(self._ctx, SEQ, ev.remove_start, ev.remove_end)
        if ev.remove_end < ev.old_total:
            self._b.kv_seq_add(
                self._ctx, SEQ, ev.remove_end, ev.old_total, -ev.shift_delta
            )

    # ----- generation ----------------------------------------------------
    def generate(
        self, start_pos: int, n_predict_max: int, stop: list[str],
        out: GenerationAccumulator | None = None,
    ):
        """Stream a reply, yielding text deltas as tokens are sampled.

        Generation begins from the logits of the token most recently prefilled
        (the caller must have prefilled the generation prompt with
        ``want_logits=True``). Each sampled token is decoded into the cache at the
        next free position **before** its text is yielded, so the cache and
        ``out.token_ids`` never diverge even if the consumer abandons the stream
        early (barge-in).

        Bytes are run through an incremental UTF-8 decoder, so a codepoint split
        across two tokens is never emitted as a replacement character. When
        ``stop`` strings are configured, the trailing few characters are held back
        until they are known not to begin a stop string.

        Args:
            start_pos: Cache position of the first generated token.
            n_predict_max: Hard cap on tokens to generate.
            stop: Stop strings; generation ends just before the earliest match.
            out: Accumulator to fill (token_ids/text/stop_reason). Pass one in to
                observe progress after barge-in; a fresh one is used otherwise.

        Yields:
            Visible reply text deltas; their concatenation equals ``out.text``.
        """
        acc = out if out is not None else GenerationAccumulator()
        pos = start_pos
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        max_stop = max((len(s) for s in stop if s), default=0)
        tbuf = _TextBuffer()

        for _ in range(max(0, n_predict_max)):
            tok = self._b.lc.llama_sampler_sample(self._sampler, self._ctx, -1)
            if self._b.is_eog(self._vocab, tok):
                acc.stop_reason = "eog"
                break
            # Commit to the cache first, then record, then surface text: this
            # ordering is what keeps token_ids == cache across an early close.
            self._b.decode(self._ctx, [tok], pos, SEQ, True, self._batch_size)
            pos += 1
            acc.token_ids.append(tok)
            tbuf.append(decoder.decode(self._b.token_to_piece_bytes(self._vocab, tok)))

            if max_stop:
                # Emitted text is already known to start no stop string (that is
                # what the hold-back guarantees), so only the unemitted tail
                # needs scanning - the total scan stays linear in the reply.
                hit = _stop_hit(tbuf.text, stop, search_from=tbuf.emitted)
                if hit is not None:
                    acc.stop_reason = "stop"
                upto = hit if hit is not None else len(tbuf) - (max_stop - 1)
            else:
                upto = len(tbuf)
            delta = tbuf.emit(upto)
            if delta:
                acc.text += delta
                yield delta
            if acc.stop_reason == "stop":
                break
        else:
            acc.stop_reason = "length"

        # Flush the held-back tail (and any dangling bytes) unless a stop string
        # truncated the reply.
        if acc.stop_reason != "stop":
            tbuf.append(decoder.decode(b"", final=True))
            delta = tbuf.emit(len(tbuf))
            if delta:
                acc.text += delta
                yield delta

    # ----- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Free the sampler, context and model. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._b.lc.llama_sampler_free(self._sampler)
        finally:
            self._b.free_context(self._ctx)
            self._b.free_model(self._model)

    # ----- internals -----------------------------------------------------
    def _build_sampler(self, cfg: Config):
        lc = self._b.lc
        params = lc.llama_sampler_chain_default_params()
        chain = lc.llama_sampler_chain_init(params)
        add = lc.llama_sampler_chain_add
        if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
            try:
                add(chain, lc.llama_sampler_init_penalties(
                    cfg.repetition_window, cfg.repetition_penalty, 0.0, 0.0))
            except TypeError:
                # Older builds use a different signature; degrade without the
                # penalty rather than crash - but say so.
                warnings.warn(
                    "this llama_cpp build has an incompatible "
                    "llama_sampler_init_penalties signature; "
                    f"repetition_penalty={cfg.repetition_penalty} is ignored",
                    RuntimeWarning,
                )
        if cfg.temperature <= 0.0:
            add(chain, lc.llama_sampler_init_greedy())
            return chain
        if cfg.top_k > 0:
            add(chain, lc.llama_sampler_init_top_k(cfg.top_k))
        add(chain, lc.llama_sampler_init_top_p(cfg.top_p, 1))
        add(chain, lc.llama_sampler_init_temp(cfg.temperature))
        # LLAMA_DEFAULT_SEED asks llama.cpp for a fresh random seed per run.
        seed = getattr(lc, "LLAMA_DEFAULT_SEED", 0xFFFFFFFF)
        add(chain, lc.llama_sampler_init_dist(seed))
        return chain


def _stop_hit(text: str, stop: list[str], search_from: int = 0) -> int | None:
    """Return the index where the earliest stop string begins, or ``None``.

    ``search_from`` skips text already known to contain no stop start (e.g.
    previously emitted output), keeping repeated scans linear overall.
    """
    earliest: int | None = None
    for s in stop:
        if not s:
            continue
        i = text.find(s, search_from)
        if i != -1 and (earliest is None or i < earliest):
            earliest = i
    return earliest
