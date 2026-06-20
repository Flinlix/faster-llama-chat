"""Thin adapter over llama-cpp-python's low-level (``llama_cpp.*``) API.

llama.cpp's C API has churned across releases - most importantly the KV-cache
functions migrated ``llama_kv_cache_seq_*`` -> ``llama_kv_self_seq_*`` -> the
newer ``llama_memory_*`` handle API. This module resolves whichever symbols the
installed build exposes, once, so the rest of the package can speak a single
stable vocabulary on both the CPU and CUDA builds.

Only the handful of primitives the wrapper actually needs are exposed here.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Callable

try:  # pragma: no cover - import guard exercised only without the dep installed
    import llama_cpp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llama-cpp-python is required to run the model. Install it with:\n"
        '  CPU:       pip install llama-cpp-python\n'
        '  GPU:       CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python'
    ) from exc

# Config log levels -> Python logging levels. llama-cpp-python installs a ggml
# log callback at import that prints a message only when its "llama-cpp-python"
# logger level is at or below the message's mapped level, so setting that
# logger's level is all it takes to filter llama.cpp's output. "none" sits above
# CRITICAL to also drop llama.cpp's unconditional (level-0) banners.
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "none": logging.CRITICAL + 1,
}


def set_log_level(level: str) -> None:
    """Filter llama.cpp/ggml log output to ``level`` and above, process-wide.

    Args:
        level: One of :data:`~llama_chat.config.LOG_LEVELS`.
    """
    logging.getLogger("llama-cpp-python").setLevel(_LOG_LEVELS[level])


def _maybe(*names: str) -> Callable | None:
    """Return the first attribute of ``llama_cpp`` that exists among ``names``.

    Handles API renames across llama_cpp versions; returns ``None`` when no
    name is present (the caller treats the capability as unavailable).
    """
    for name in names:
        fn = getattr(llama_cpp, name, None)
        if fn is not None:
            return fn
    return None


def _first(*names: str) -> Callable:
    """Like ``_maybe``, but raises when the symbol is mandatory and absent."""
    fn = _maybe(*names)
    if fn is None:
        raise AttributeError(
            f"none of {names} found in llama_cpp (version {getattr(llama_cpp, '__version__', '?')})"
        )
    return fn


class Backend:
    """Resolved, version-stable handles for the low-level calls we use."""

    def __init__(self) -> None:
        self.lc = llama_cpp
        self._load_model = _first("llama_model_load_from_file", "llama_load_model_from_file")
        self._free_model = _first("llama_model_free", "llama_free_model")
        self._new_ctx = _first("llama_init_from_model", "llama_new_context_with_model")
        self._get_vocab = _maybe("llama_model_get_vocab")

        # Memory / KV-cache handle ops, newest-API-first.
        self._get_memory = _maybe("llama_get_memory")
        self._mem_clear = _maybe("llama_memory_clear")
        self._mem_rm = _maybe("llama_memory_seq_rm")
        self._mem_add = _maybe("llama_memory_seq_add")
        self._mem_can_shift = _maybe("llama_memory_can_shift")
        self._kv_clear = _maybe("llama_kv_self_clear", "llama_kv_cache_clear")
        self._kv_rm = _maybe("llama_kv_self_seq_rm", "llama_kv_cache_seq_rm")
        self._kv_add = _maybe("llama_kv_self_seq_add", "llama_kv_cache_seq_add")

        self._is_eog = _first("llama_vocab_is_eog", "llama_token_is_eog")
        self._token_eos = _first("llama_vocab_eos", "llama_token_eos")
        self._token_to_piece = _first("llama_token_to_piece")
        self._is_control = _maybe("llama_vocab_is_control")
        self._vocab_n_tokens = _first("llama_vocab_n_tokens", "llama_n_vocab")
        self._meta_val_str = _maybe("llama_model_meta_val_str")
        self._sampler_reset = _maybe("llama_sampler_reset")

    # ----- model / context lifecycle ------------------------------------
    def load_model(self, path: str, n_gpu_layers: int):
        mparams = self.lc.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        model = self._load_model(path.encode("utf-8"), mparams)
        if not model:
            raise RuntimeError(f"failed to load model: {path}")
        return model

    def new_context(self, model, n_ctx: int, n_threads: int | None, n_batch: int,
                    flash_attn: bool = False, type_k: int | None = None,
                    type_v: int | None = None):
        cparams = self.lc.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = n_batch
        if n_threads:
            cparams.n_threads = n_threads
            cparams.n_threads_batch = n_threads
        self._set_flash_attn(cparams, flash_attn)
        if type_k is not None:
            cparams.type_k = type_k
        if type_v is not None:
            cparams.type_v = type_v
        ctx = self._new_ctx(model, cparams)
        if not ctx:
            raise RuntimeError("failed to create llama context")
        return ctx

    def _set_flash_attn(self, cparams, enable: bool) -> None:
        """Set flash attention across the param-struct churn.

        Older builds expose a bool ``flash_attn``; newer ones use a
        ``flash_attn_type`` enum (DISABLED=0, ENABLED=1, AUTO=-1).
        Always set explicitly so the build's default (AUTO on newer builds)
        never silently overrides the caller's intent.
        """
        if hasattr(cparams, "flash_attn"):
            cparams.flash_attn = enable
        elif hasattr(cparams, "flash_attn_type"):
            name = ("LLAMA_FLASH_ATTN_TYPE_ENABLED" if enable
                    else "LLAMA_FLASH_ATTN_TYPE_DISABLED")
            cparams.flash_attn_type = getattr(self.lc, name, 1 if enable else 0)
        elif enable:
            raise RuntimeError(
                "this llama_cpp build exposes no flash-attention context param")

    def vocab(self, model):
        return self._get_vocab(model) if self._get_vocab else model

    def metadata_value(self, model, key: str) -> str | None:
        """Return the GGUF metadata string for ``key``, or ``None`` if absent."""
        if self._meta_val_str is None:
            return None
        size = 8192
        buf = ctypes.create_string_buffer(size)
        n = self._meta_val_str(model, key.encode("utf-8"), buf, size)
        if n < 0:
            return None
        if n >= size:  # truncated; n is the length the value needs
            buf = ctypes.create_string_buffer(n + 1)
            n = self._meta_val_str(model, key.encode("utf-8"), buf, n + 1)
            if n < 0:
                return None
        return buf.value.decode("utf-8", errors="replace")

    def free_model(self, model) -> None:
        self._free_model(model)

    def free_context(self, ctx) -> None:
        self.lc.llama_free(ctx)

    # ----- tokenize / detokenize ----------------------------------------
    def tokenize(self, vocab, text: str, add_special: bool) -> list[int]:
        """Convert ``text`` to a list of token IDs, optionally prepending a BOS token.

        Special-token parsing is always enabled so template tags tokenize to
        their control tokens. Message content is kept safe upstream:
        :class:`~llama_chat.template.TemplateFormatter` strips the model's
        special-token pieces from content before it reaches this method, so a
        literal tag in user text cannot forge a turn boundary here.
        """
        raw = text.encode("utf-8")
        n_max = len(raw) + 16
        buf = (self.lc.llama_token * n_max)()
        n = self.lc.llama_tokenize(vocab, raw, len(raw), buf, n_max, add_special, True)
        if n < 0:  # buffer too small; n is the negative required size
            n_max = -n
            buf = (self.lc.llama_token * n_max)()
            n = self.lc.llama_tokenize(vocab, raw, len(raw), buf, n_max, add_special, True)
            if n < 0:
                raise RuntimeError(f"llama_tokenize failed: required {-n} tokens")
        return list(buf[:n])

    def token_to_piece_bytes(self, vocab, token: int) -> bytes:
        """Return the raw UTF-8 bytes for ``token`` (may be a partial codepoint).

        SentencePiece can split one codepoint across tokens, so callers stream
        these bytes through an incremental decoder rather than decoding each
        token in isolation.
        """
        buf = (ctypes.c_char * 64)()
        n = self._token_to_piece(vocab, token, buf, len(buf), 0, True)
        if n < 0:
            buf = (ctypes.c_char * (-n))()
            n = self._token_to_piece(vocab, token, buf, len(buf), 0, True)
            if n < 0:
                raise RuntimeError(f"llama_token_to_piece failed for token {token}")
        return bytes(buf[:n])

    def token_to_piece(self, vocab, token: int) -> str:
        """Decode a single token to text (standalone; not boundary-safe)."""
        return self.token_to_piece_bytes(vocab, token).decode("utf-8", errors="replace")

    def is_eog(self, vocab_or_model, token: int) -> bool:
        return bool(self._is_eog(vocab_or_model, token))

    def eos_text(self, vocab) -> str:
        """Rendered text of the end-of-sequence token (e.g. ``<eos>``)."""
        return self.token_to_piece(vocab, self._token_eos(vocab))

    def token_eos(self, vocab) -> int:
        """The end-of-sequence token id (may be negative if the model has none)."""
        return self._token_eos(vocab)

    def is_special(self, vocab, token: int) -> bool:
        """True if ``token`` is a control/special token (rendered, not literal text)."""
        if self._is_control is None:
            return False
        return bool(self._is_control(vocab, token))

    def supports_special(self) -> bool:
        """Whether this build can classify control/special tokens."""
        return self._is_control is not None

    def vocab_size(self, vocab_or_model) -> int:
        """Number of tokens in the vocabulary.

        Takes the same vocab-or-model object that :meth:`vocab` returns: the
        newer ``llama_vocab_n_tokens`` takes a vocab, the older ``llama_n_vocab``
        a model, and the object generation tracks the resolved symbol - so this
        is correct on both builds (same coupling as ``is_eog``/``token_eos``).
        """
        return int(self._vocab_n_tokens(vocab_or_model))

    # ----- sampler -------------------------------------------------------
    def sampler_reset(self, sampler) -> None:
        """Reset sampler-chain state (e.g. the repeat-penalty window), if supported."""
        if self._sampler_reset is not None:
            self._sampler_reset(sampler)

    # ----- KV / memory edits --------------------------------------------
    def _mem(self, ctx):
        return self._get_memory(ctx) if self._get_memory else None

    def kv_clear(self, ctx) -> None:
        mem = self._mem(ctx)
        if mem is not None and self._mem_clear is not None:
            self._mem_clear(mem, True)
        elif self._kv_clear is not None:
            self._kv_clear(ctx)
        else:
            raise RuntimeError("no KV-cache clear function available in this llama_cpp build")

    def kv_seq_rm(self, ctx, seq: int, p0: int, p1: int) -> None:
        mem = self._mem(ctx)
        if mem is not None and self._mem_rm is not None:
            self._mem_rm(mem, seq, p0, p1)
        elif self._kv_rm is not None:
            self._kv_rm(ctx, seq, p0, p1)
        else:
            raise RuntimeError("no KV-cache seq_rm function available in this llama_cpp build")

    def kv_seq_add(self, ctx, seq: int, p0: int, p1: int, delta: int) -> None:
        mem = self._mem(ctx)
        if mem is not None and self._mem_add is not None:
            self._mem_add(mem, seq, p0, p1, delta)
        elif self._kv_add is not None:
            self._kv_add(ctx, seq, p0, p1, delta)
        else:
            raise RuntimeError("no KV-cache seq_add function available in this llama_cpp build")

    def can_shift(self, ctx) -> bool:
        """True if the cache supports in-place position shifting after removal.

        False for caches that lose position information when tokens are dropped
        (e.g. compact sliding-window or recurrent state). When unknown, assume
        True - the modern handle API only exposes this for caches that need it.
        """
        mem = self._mem(ctx)
        if mem is None or self._mem_can_shift is None:
            return True
        return bool(self._mem_can_shift(mem))

    # ----- decode / batch ------------------------------------------------
    def decode(self, ctx, token_ids: list[int], start_pos: int, seq: int,
               want_logits: bool, n_batch: int) -> None:
        """Prefill ``token_ids`` at positions ``[start_pos, start_pos + n)``.

        Long inputs are split into ``n_batch``-sized decode calls so a single
        prefill never exceeds the context's batch limit. Logits are requested
        only on the very last token, and only when generation will follow -
        injection and history prefill skip them.
        """
        n = len(token_ids)
        if n == 0:
            return
        # One batch allocation sized for the largest chunk, reused across chunks.
        batch = self.lc.llama_batch_init(min(n, n_batch), 0, 1)
        try:
            for off in range(0, n, n_batch):
                chunk = token_ids[off:off + n_batch]
                m = len(chunk)
                is_last_chunk = off + m >= n
                for i, tok in enumerate(chunk):
                    batch.token[i] = tok
                    batch.pos[i] = start_pos + off + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = seq
                    batch.logits[i] = 0
                batch.logits[m - 1] = 1 if (want_logits and is_last_chunk) else 0
                batch.n_tokens = m
                rc = self.lc.llama_decode(ctx, batch)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed (rc={rc})")
        finally:
            self.lc.llama_batch_free(batch)
