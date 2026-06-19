"""End-to-end demo of the KV-cache manager against a real GGUF model.

Config defaults to Gemma 4; change it to match your model and hardware.

Install the package first (see install.sh or README), then run:

    python examples/demo.py /path/to/gemma_4_model.gguf                  # CPU
    python examples/demo.py /path/to/gemma_4_model.gguf --gpu-layers -1  # GPU

The demo proves the core claim by logging how many tokens are *prefilled* per
action: after ``begin`` primes the conversation, each ``request`` prefills only
its own new prompt - never the surviving history. The eviction stress run uses
a deliberately tiny context to show the oldest messages being dropped while the
system prompt survives and the total never crosses ``n_ctx``.
"""

from __future__ import annotations

import argparse

from llama_chat import ChatWrapper, Config, KVContext


class CountingContext(KVContext):
    """KVContext that records prefill volume so the demo can report reuse."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.prefilled = 0

    def prefill(self, token_ids, start_pos, want_logits):
        self.prefilled += len(token_ids)
        super().prefill(token_ids, start_pos, want_logits)


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(w: ChatWrapper) -> None:
    rows = w.snapshot()
    print(f"  cache: {w.total_tokens} tokens across {len(rows)} messages")
    for r in rows:
        print(f"    [{r['pos_start']:>5},{r['pos_end']:>5})  {r['role']:<10} ({r['n_tokens']} tok)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a .gguf model")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    # ---- Normal conversation: prove only-new-tokens prefill ----
    banner("CONVERSATION (reuse): only new tokens are prefilled per turn")
    cfg = Config(
        model_path=args.model,
        n_ctx=args.n_ctx,
        threshold_pct=args.threshold,
        n_gpu_layers=args.gpu_layers,
        max_tokens=64,
    )
    ctx = CountingContext(cfg)
    w = ChatWrapper(cfg, context=ctx)

    w.begin(
        "You are a concise, helpful assistant.",
        [("user", "Remember my name is Felix."), ("assistant", "Got it, Felix.")],
    )
    print(f"begin: prefilled {ctx.prefilled} tokens (system + history)")
    show(w)

    w.inject("Context: the user is building a llama.cpp wrapper in Python.")
    print(f"\nafter inject: total prefilled so far {ctx.prefilled}")
    show(w)

    for q in ["What's my name?", "What am I building?", "Suggest one optimization."]:
        before = ctx.prefilled
        turn = w.request(q)
        print(f"\n> {q}")
        print(f"  reply: {turn.text.strip()[:200]}")
        print(
            f"  prefilled {ctx.prefilled - before} new tokens "
            f"(prompt={turn.n_prefilled}, generated={turn.n_generated}, "
            f"evicted={turn.n_evicted}, stop={turn.stop_reason})"
        )
    show(w)
    w.close()

    # ---- Eviction stress: tiny context, watch the eviction ----
    banner("EVICTION STRESS: tiny context - oldest messages get cut, system kept")
    small = Config(
        model_path=args.model,
        n_ctx=512,
        threshold_pct=0.5,
        n_gpu_layers=args.gpu_layers,
        max_tokens=32,
    )
    sw = ChatWrapper(small, context=KVContext(small))
    sw.begin("You are a terse assistant. Keep replies under 10 words.")
    for i in range(10):
        turn = sw.request(f"Give me fact number {i} about the planet Mars, briefly.")
        roles = [r["role"] for r in sw.snapshot()]
        assert roles[0] == "system", "system prompt was evicted!"
        assert sw.total_tokens <= small.n_ctx, "context overflow!"
        print(
            f"turn {i:>2}: total={sw.total_tokens:>4}/{small.n_ctx} "
            f"messages={len(roles):>2} evicted={turn.n_evicted}"
        )
    show(sw)
    sw.close()
    print("\nOK: system prompt survived every eviction; context never overflowed.")


if __name__ == "__main__":
    main()
