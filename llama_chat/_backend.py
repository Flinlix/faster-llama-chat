"""Adapter over llama-cpp-python's low-level API, resolved once per build.

Each C symbol is resolved once, under the names it has had since the oldest
release the package supports, and the rest of the package uses a single name
for it. Only the primitives the wrapper needs are exposed.
"""

from __future__ import annotations

import ctypes
import inspect
import logging
import warnings
from typing import Callable

try:  # pragma: no cover (reachable only without the dependency installed)
    import llama_cpp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llama-cpp-python 0.3.23 or later is required to run the model, built from "
        "source for this machine. Build it with the repository's install script:\n"
        "  ./install.sh            (CUDA where nvidia-smi is present, else the CPU)\n"
        "  ./install.sh --cpu      or --cuda to choose the backend\n"
        "or by hand:\n"
        '  CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python '
        '"llama-cpp-python>=0.3.23"\n'
        '  CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python '
        '"llama-cpp-python>=0.3.23"'
    ) from exc

# "none" sits above CRITICAL to drop llama.cpp's unconditional banners too.
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "none": logging.CRITICAL + 1,
}


# The C API's floor on how many candidates a truncating stage may leave.
_MIN_CANDIDATES = 1


def set_log_level(level: str) -> None:
    """Filter llama.cpp log output to ``level`` and above, process-wide."""
    logging.getLogger("llama-cpp-python").setLevel(_LOG_LEVELS[level])


def template_renderer(template: str, eos_token: str, generation_prompt: bool = False):
    """A renderer for one GGUF chat template, called with messages and tools.

    BOS is left out: the sequence opener is the tokenizer's to add.
    Template variables pass through as keywords.
    """
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    return Jinja2ChatFormatter(template=template, eos_token=eos_token,
                               bos_token="",
                               add_generation_prompt=generation_prompt)


def _parameter_count(fn) -> int:
    """How many arguments a binding takes, from its ctypes types or its signature."""
    argtypes = getattr(fn, "argtypes", None)
    if argtypes is not None:
        return len(argtypes)
    return len(inspect.signature(fn).parameters)


def _required_symbol(*names: str):
    """Return the first of ``names`` this build defines.

    Raises:
        AttributeError: If it defines none of them.
    """
    fn = next((getattr(llama_cpp, name) for name in names if hasattr(llama_cpp, name)), None)
    if fn is None:
        raise AttributeError(
            f"none of {names} found in llama_cpp (version {getattr(llama_cpp, '__version__', '?')})"
        )
    return fn


class Backend:
    """Version-stable handles for the low-level calls this package makes."""

    # Control tokens and the ones a fine-tune added. Both count as tags; see
    # design_history.md, "How a call's format is recovered".
    _TAG_ATTRS = llama_cpp.LLAMA_TOKEN_ATTR_CONTROL | llama_cpp.LLAMA_TOKEN_ATTR_USER_DEFINED

    def __init__(self) -> None:
        self._llama_cpp = llama_cpp
        self._load_model = _required_symbol("llama_model_load_from_file")
        self._free_model = _required_symbol("llama_model_free")
        self._new_ctx = _required_symbol("llama_init_from_model")
        self._get_vocab = _required_symbol("llama_model_get_vocab")

        self._get_memory = _required_symbol("llama_get_memory")
        self._mem_clear = _required_symbol("llama_memory_clear")
        self._mem_rm = _required_symbol("llama_memory_seq_rm")
        self._mem_add = _required_symbol("llama_memory_seq_add")
        self._mem_can_shift = _required_symbol("llama_memory_can_shift")
        self._state_size = _required_symbol("llama_state_seq_get_size_ext")
        self._state_get = _required_symbol("llama_state_seq_get_data_ext")
        self._state_set = _required_symbol("llama_state_seq_set_data_ext")
        # Only the part of a sequence's state that cannot be cut back; the
        # rest is cut with a removal after the restore.
        self._state_flags = llama_cpp.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY

        self._is_eog = _required_symbol("llama_vocab_is_eog")
        self._token_eos = _required_symbol("llama_vocab_eos")
        self._token_to_piece = _required_symbol("llama_token_to_piece")
        self._get_attr = _required_symbol("llama_vocab_get_attr")
        self._meta_val_str = _required_symbol("llama_model_meta_val_str")

        # Sampler chain: one name each so far, resolved here anyway so a build
        # missing one is refused at construction like every other symbol.
        self._chain_params = _required_symbol("llama_sampler_chain_default_params")
        self._chain_init = _required_symbol("llama_sampler_chain_init")
        self._chain_add = _required_symbol("llama_sampler_chain_add")
        self._init_penalties = _required_symbol("llama_sampler_init_penalties")
        # Releases differ in whether the vocabulary size comes first; one that
        # asks for it must also say it.
        self._penalties_take_vocab_size = _parameter_count(self._init_penalties) == 5
        self._vocab_n_tokens = _required_symbol("llama_vocab_n_tokens")
        self._init_greedy = _required_symbol("llama_sampler_init_greedy")
        self._init_top_k = _required_symbol("llama_sampler_init_top_k")
        self._init_top_p = _required_symbol("llama_sampler_init_top_p")
        self._init_min_p = _required_symbol("llama_sampler_init_min_p")
        self._init_temp = _required_symbol("llama_sampler_init_temp")
        self._init_dist = _required_symbol("llama_sampler_init_dist")
        self._init_logit_bias = _required_symbol("llama_sampler_init_logit_bias")
        self._sampler_sample = _required_symbol("llama_sampler_sample")
        self._sampler_free = _required_symbol("llama_sampler_free")

    # ----- model / context lifecycle ------------------------------------
    def load_model(self, path: str, n_gpu_layers: int,
                   progress: Callable[[float], None] | None = None):
        mparams = self._llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        if progress is not None:
            @self._llama_cpp.llama_progress_callback
            def _report_progress(frac, _user_data):
                progress(float(frac))
                return True  # False aborts the load, and so does a raise
            # Held as a raw C pointer, so a Python reference must outlive the load.
            self._progress_callback = _report_progress
            mparams.progress_callback = _report_progress
        model = self._load_model(path.encode("utf-8"), mparams)
        if not model:
            raise RuntimeError(f"failed to load model: {path}")
        return model

    def new_context(self, model, n_ctx: int, n_threads: int | None, n_batch: int,
                    flash_attn: bool = False, type_k: int | None = None,
                    type_v: int | None = None, kv_cache_on_gpu: bool = True,
                    n_threads_batch: int | None = None):
        cparams = self._llama_cpp.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = n_batch
        if n_threads:
            cparams.n_threads = n_threads
            cparams.n_threads_batch = n_threads
        if n_threads_batch:
            cparams.n_threads_batch = n_threads_batch
        self._set_flash_attn(cparams, flash_attn)
        self._set_kv_cache_on_gpu(cparams, kv_cache_on_gpu)
        if type_k is not None:
            cparams.type_k = type_k
        if type_v is not None:
            cparams.type_v = type_v
        ctx = self._new_ctx(model, cparams)
        if not ctx:
            raise RuntimeError(
                "failed to create llama context, most often for want of memory on "
                "the device: lower context_size or max_batch_tokens, or use a "
                "quantized kv_cache_type")
        return ctx

    def _set_kv_cache_on_gpu(self, cparams, enable: bool) -> None:
        """Place the KV cache in VRAM (``True``) or in host RAM (``False``)."""
        cparams.offload_kqv = enable

    def _set_flash_attn(self, cparams, enable: bool) -> None:
        """Set flash attention explicitly on or off."""
        cparams.flash_attn_type = (self._llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED if enable
                                   else self._llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED)

    def vocab(self, model):
        return self._get_vocab(model)

    def kv_cache_type_id(self, name: str) -> int:
        """The type id this build gives the KV-cache quantization ``name``.

        Raises:
            AttributeError: If this build defines no quantization of that name.
        """
        return int(_required_symbol(f"GGML_TYPE_{name.upper()}"))

    def metadata_value(self, model, key: str) -> str | None:
        """Return the GGUF metadata string for ``key``, or ``None`` if absent."""
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
        self._llama_cpp.llama_free(ctx)

    # ----- tokenize / detokenize ----------------------------------------
    def tokenize(self, vocab, text: str, add_special: bool,
                 parse_special: bool = True) -> list[int]:
        """Convert ``text`` to token ids, prepending BOS when ``add_special``.

        ``parse_special`` decides whether tag text resolves to its control
        token. See the README, "Special tokens in message content".
        """
        raw = text.encode("utf-8")
        n_max = len(raw) + 16
        buf = (self._llama_cpp.llama_token * n_max)()
        n = self._llama_cpp.llama_tokenize(vocab, raw, len(raw), buf, n_max, add_special, parse_special)
        if n < 0:  # buffer too small; n is the negative required size
            n_max = -n
            buf = (self._llama_cpp.llama_token * n_max)()
            n = self._llama_cpp.llama_tokenize(vocab, raw, len(raw), buf, n_max, add_special, parse_special)
            if n < 0:
                raise RuntimeError(f"llama_tokenize failed: required {-n} tokens")
        return list(buf[:n])

    def token_to_piece_bytes(self, vocab, token: int) -> bytes:
        """Return the raw UTF-8 bytes for ``token``, possibly a partial codepoint.

        A codepoint can be split across tokens, so callers stream these bytes
        through an incremental decoder.
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
        """Decode one token to text. Not safe across a codepoint boundary."""
        return self.token_to_piece_bytes(vocab, token).decode("utf-8", errors="replace")

    def is_eog(self, vocab_or_model, token: int) -> bool:
        return bool(self._is_eog(vocab_or_model, token))

    def eos_text(self, vocab) -> str:
        """Rendered text of the end-of-sequence token."""
        return self.token_to_piece(vocab, self._token_eos(vocab))

    def token_eos(self, vocab) -> int:
        """The end-of-sequence token id, negative where the model has none."""
        return self._token_eos(vocab)

    def n_tokens(self, vocab) -> int:
        """How many tokens the vocabulary holds."""
        return int(self._vocab_n_tokens(vocab))

    def is_special(self, vocab, token: int) -> bool:
        """Whether the tokenizer resolves ``token`` as a tag rather than as text."""
        return bool(self._get_attr(vocab, token) & self._TAG_ATTRS)

    # ----- sampler -------------------------------------------------------
    # Each of these adds one stage to a chain, in the order the caller adds
    # them. Which stages a chain gets is the wrapper's policy, not this file's.
    def sampler_chain(self):
        """An empty sampler chain, to be freed with :meth:`free_sampler`."""
        return self._chain_init(self._chain_params())

    def add_penalties(self, chain, vocab, window_tokens: int, repetition: float,
                      frequency: float, presence: float) -> None:
        """Add the three repetition penalties over the last ``window_tokens``.

        Passes ``vocab``'s size first where this build's function takes it.
        Warns and adds no stage where its signature is another still, leaving
        generation unpenalized rather than refusing to start.
        """
        leading = ((self._vocab_n_tokens(vocab),)
                   if self._penalties_take_vocab_size else ())
        try:
            self._chain_add(chain, self._init_penalties(
                *leading, window_tokens, repetition, frequency, presence))
        except TypeError:
            warnings.warn(
                "this llama_cpp build has an incompatible "
                "llama_sampler_init_penalties signature; "
                "repetition/presence/frequency penalties are ignored",
                RuntimeWarning,
            )

    def add_withheld(self, chain, vocab, tokens: tuple[int, ...]) -> None:
        """Add the stage that keeps ``tokens`` from ever being drawn."""
        biases = (self._llama_cpp.llama_logit_bias * len(tokens))(
            *(self._llama_cpp.llama_logit_bias(token, float("-inf")) for token in tokens))
        self._chain_add(chain, self._init_logit_bias(
            self._vocab_n_tokens(vocab), len(tokens), biases))

    def add_greedy(self, chain) -> None:
        """Add the stage that takes the most likely token."""
        self._chain_add(chain, self._init_greedy())

    def add_top_k(self, chain, top_k: int) -> None:
        self._chain_add(chain, self._init_top_k(top_k))

    def add_top_p(self, chain, top_p: float) -> None:
        self._chain_add(chain, self._init_top_p(top_p, _MIN_CANDIDATES))

    def add_min_p(self, chain, min_p: float) -> None:
        self._chain_add(chain, self._init_min_p(min_p, _MIN_CANDIDATES))

    def add_temperature(self, chain, temperature: float) -> None:
        self._chain_add(chain, self._init_temp(temperature))

    def add_distribution(self, chain, seed: int) -> None:
        """Add the stage that draws from what the earlier stages left."""
        self._chain_add(chain, self._init_dist(seed))

    @property
    def fresh_seed_per_run(self) -> int:
        """The seed value that asks for a new random seed on every run."""
        return getattr(self._llama_cpp, "LLAMA_DEFAULT_SEED", 0xFFFFFFFF)

    def sample(self, sampler, ctx) -> int:
        """Draw the next token from the logits of the most recent decode."""
        return self._sampler_sample(sampler, ctx, -1)

    def free_sampler(self, sampler) -> None:
        self._sampler_free(sampler)

    # ----- KV / memory edits --------------------------------------------
    def kv_clear(self, ctx) -> None:
        self._mem_clear(self._get_memory(ctx), True)

    def kv_seq_rm(self, ctx, seq: int, p0: int, p1: int) -> bool:
        """Remove positions ``[p0, p1)``; ``False`` where the cache refused."""
        return self._mem_rm(self._get_memory(ctx), seq, p0, p1) is not False

    def kv_seq_add(self, ctx, seq: int, p0: int, p1: int, delta: int) -> None:
        self._mem_add(self._get_memory(ctx), seq, p0, p1, delta)

    def save_state(self, ctx, seq: int):
        """The part of ``seq``'s state a cut cannot reach, as a host buffer.

        Raises:
            RuntimeError: If the build saves nothing.
        """
        size = self._state_size(ctx, seq, self._state_flags)
        buffer = (ctypes.c_uint8 * size)()
        if self._state_get(ctx, buffer, size, seq, self._state_flags) != size:
            raise RuntimeError("the cache did not save its state")
        return buffer

    def restore_state(self, ctx, seq: int, buffer) -> None:
        """Put ``seq`` back into the state ``buffer`` was saved from.

        Raises:
            RuntimeError: If the cache refuses it.
        """
        if self._state_set(ctx, buffer, len(buffer), seq, self._state_flags) == 0:
            raise RuntimeError("the cache refused the saved state")

    def can_shift(self, ctx) -> bool:
        """Whether the cache supports in-place position shifting after removal."""
        return bool(self._mem_can_shift(self._get_memory(ctx)))

    # ----- decode / batch ------------------------------------------------
    def decode(self, ctx, token_ids: list[int], start_pos: int, seq: int,
               want_logits: bool, n_batch: int) -> None:
        """Prefill ``token_ids`` at positions ``[start_pos, start_pos + n)``.

        Chunked into ``n_batch``-sized decode calls. Logits are requested on the
        last token only, and only when ``want_logits``.
        """
        n = len(token_ids)
        if n == 0:
            return
        # One allocation sized for the largest chunk, reused across chunks.
        batch = self._llama_cpp.llama_batch_init(min(n, n_batch), 0, 1)
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
                rc = self._llama_cpp.llama_decode(ctx, batch)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed (rc={rc})")
        finally:
            self._llama_cpp.llama_batch_free(batch)
