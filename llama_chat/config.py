"""Configuration for the chat wrapper.

The ``Config`` object is the only thing that differs between deployment targets
(a Raspberry Pi versus an i9 + RTX 3090): the model, the context size, and the
eviction threshold. The wrapper logic itself is hardware agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemplateConfig:
    """Chat-template fragments used to wrap each message.

    Defaults follow the ChatML convention (``<|im_start|>role ... <|im_end|>``).
    The fragments must match the model's *trained* template exactly, because the
    turn-terminator tag is what lets the model stop generating; a mismatched
    template tokenizes as plain text, so the model never emits its end-of-turn
    token and rambles to the generation cap. Use :meth:`preset` or the
    ``ChatWrapper`` auto-detection rather than hand-editing these for a known
    model family.

    Attributes:
        system: Format string for the system prompt; must contain ``{text}``.
        user: Format string for a user message; must contain ``{text}``.
        assistant: Format string for a complete assistant message; must contain
            ``{text}``.
        assistant_open: The generation prompt decoded before sampling begins
            (everything up to where the model starts writing its reply).
        assistant_close: Tokens decoded after generation to terminate the
            assistant turn cleanly, so the next turn's boundary is correct.
    """

    system: str = "<|im_start|>system\n{text}<|im_end|>\n"
    user: str = "<|im_start|>user\n{text}<|im_end|>\n"
    assistant: str = "<|im_start|>assistant\n{text}<|im_end|>\n"
    assistant_open: str = "<|im_start|>assistant\n"
    assistant_close: str = "<|im_end|>\n"

    @staticmethod
    def chatml() -> "TemplateConfig":
        """The ChatML template (Qwen, many fine-tunes)."""
        return TemplateConfig()

    @staticmethod
    def gemma() -> "TemplateConfig":
        """Gemma 2 / 3 template (``<start_of_turn>`` markers, no system role).

        Gemma has no system role, so the system prompt is folded into the first
        user turn by the caller; here we expose user/model turns only.
        """
        return TemplateConfig(
            system="<start_of_turn>user\n{text}<end_of_turn>\n",
            user="<start_of_turn>user\n{text}<end_of_turn>\n",
            assistant="<start_of_turn>model\n{text}<end_of_turn>\n",
            assistant_open="<start_of_turn>model\n",
            assistant_close="<end_of_turn>\n",
        )

    @staticmethod
    def gemma4() -> "TemplateConfig":
        """Gemma 4 template (``<|turn>{role} ... <turn|>`` markers).

        Distinct from Gemma 2/3: Gemma 4 uses ``<|turn>`` / ``<turn|>`` tokens,
        and ``<turn|>`` doubles as the end-of-generation token.
        """
        return TemplateConfig(
            system="<|turn>system\n{text}<turn|>\n",
            user="<|turn>user\n{text}<turn|>\n",
            assistant="<|turn>model\n{text}<turn|>\n",
            assistant_open="<|turn>model\n",
            assistant_close="<turn|>\n",
        )

    @classmethod
    def preset(cls, name: str) -> "TemplateConfig":
        """Return a named preset.

        Args:
            name: One of ``"chatml"``, ``"gemma"``, ``"gemma4"``.

        Raises:
            ValueError: If ``name`` is not a known preset.
        """
        presets = {"chatml": cls.chatml, "gemma": cls.gemma, "gemma4": cls.gemma4}
        if name not in presets:
            raise ValueError(
                f"unknown template preset {name!r}; choose from {sorted(presets)}"
            )
        return presets[name]()


@dataclass
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

    Attributes:
        model_path: Path to the GGUF model file.
        n_ctx: General context size in tokens (the hard wall — generation must
            never cross it).
        threshold_pct: Fraction of ``n_ctx`` above which prefilled content
            triggers eviction of the oldest messages. Must be in ``(0, 1]``.
        n_gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only Pi build;
            ``-1`` to offload everything on the RTX 3090.
        n_threads: CPU threads for decode (``None`` lets llama.cpp decide).
        n_batch: Maximum tokens submitted to a single ``llama_decode`` call;
            longer prefills are chunked to this size.
        seed: RNG seed for sampling.
        max_tokens: Upper bound on generated tokens per ``request`` (further
            capped so total never exceeds ``n_ctx``).
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        top_k: Top-k sampling cutoff (``<= 0`` disables).
        top_p: Nucleus sampling cutoff.
        repeat_penalty: Repetition penalty applied over ``repeat_last_n`` tokens.
        repeat_last_n: Window for the repetition penalty.
        stop: Additional stop strings that end generation.
        oversize_policy: What to do when a single message cannot fit under the
            threshold even after evicting everything but the system prompt:
            ``"reject"`` raises, ``"truncate"`` clips the message to fit.
        template: Chat-template fragments (see :class:`TemplateConfig`). Leave as
            ``None`` to auto-detect the right preset from the model's built-in
            chat template at load time; set explicitly to override.
        verbose: Pass through to llama.cpp's logging.
    """

    model_path: str
    n_ctx: int = 4096
    threshold_pct: float = 0.8

    # Backend / hardware
    n_gpu_layers: int = 0
    n_threads: int | None = None
    n_batch: int = 512
    seed: int = 0

    # Generation defaults (overridable per request)
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    stop: list[str] = field(default_factory=list)

    # Policy
    oversize_policy: str = "reject"

    template: TemplateConfig | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_pct <= 1.0:
            raise ValueError("threshold_pct must be in (0, 1]")
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.oversize_policy not in ("reject", "truncate"):
            raise ValueError("oversize_policy must be 'reject' or 'truncate'")

    @property
    def threshold_tokens(self) -> int:
        """Eviction trigger expressed as an absolute token count."""
        return int(self.threshold_pct * self.n_ctx)
