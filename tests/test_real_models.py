"""Long conversations on real model files: the cache is checked against a fresh render after every action, a compact window cache holds what the next token attends over, and memory stops growing; a conversation switched away from comes back whole.

The model files are named by the environment, never here:
``LLAMA_CPP_CONVERSATION_TEST_MODELS`` holds their paths, separated as the system separates
paths. Without it every test here is skipped. GPU memory is measured by
``LLAMA_CPP_CONVERSATION_TEST_GPU_MEMORY_COMMAND``, a shell command that prints the MiB the
process ``{pid}`` holds on the GPU; without it only RAM is bounded.
``LLAMA_CPP_CONVERSATION_TEST_GPU_LAYERS`` sets the layers to offload, all by default, and
``LLAMA_CPP_CONVERSATION_TEST_COMPACT_WINDOW_CACHE=1`` runs every file with the compact
window cache. Run with ``-m gpu``.
"""

import os
import subprocess

import pytest

from llama_cpp_conversation import ChatWrapper, Config, ToolRegistry
from tests import internals

_MODELS = [path for path in os.environ.get("LLAMA_CPP_CONVERSATION_TEST_MODELS", "").split(os.pathsep) if path]
# The turns of each conversation, the first of them to fill the bounded memories before
# growth is measured.
_TURNS, _WARM_UP_TURNS = 120, 30
# How much the process may grow over the turns after the warm-up: in RAM, the bounded memories
# reaching their bound; on the GPU, nothing.
_MAX_RAM_GROWTH_MIB, _MAX_GPU_GROWTH_MIB = 64, 16
_CONTEXT_TOKENS, _REPLY_TOKENS = 4096, 48
# Room for the conversation switched away from, which the switch back takes out again.
_KEPT_BYTES = 1 << 30
_GPU_LAYERS = int(os.environ.get("LLAMA_CPP_CONVERSATION_TEST_GPU_LAYERS", "-1"))
_COMPACT_WINDOW_CACHE = os.environ.get("LLAMA_CPP_CONVERSATION_TEST_COMPACT_WINDOW_CACHE") == "1"

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _MODELS, reason="LLAMA_CPP_CONVERSATION_TEST_MODELS names no model file")]


def _ram_mib() -> float:
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) / 1024 for line in status if line.startswith("VmRSS:"))


_GPU_MEMORY_COMMAND = os.environ.get("LLAMA_CPP_CONVERSATION_TEST_GPU_MEMORY_COMMAND")


def _gpu_mib() -> int | None:
    """The GPU memory this process holds, by the command the environment names; ``None`` without one."""
    if not _GPU_MEMORY_COMMAND:
        return None
    # Replaced as text: the command's own braces, such as awk's, are no placeholders.
    printed = subprocess.run(_GPU_MEMORY_COMMAND.replace("{pid}", str(os.getpid())), shell=True,
                             capture_output=True, text=True, check=True).stdout
    return int(float(printed.split()[0])) if printed.split() else 0


def _clock_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The current time.", name="clock",
                  parameters={"type": "object", "properties": {}})
    return registry


def _assert_the_window_layers_hold_the_window(wrapper, turn) -> None:
    # The full-size cache leaves a gap after an eviction; see
    # docs/decisions/0060-the-full-size-cache-is-not-checked.md.
    reach = internals.window_layers_reach(wrapper) if _COMPACT_WINDOW_CACHE else None
    if reach is not None:
        window, oldest = reach
        held = len(internals.held_tokens(wrapper))
        assert oldest <= max(0, held - window), (
            f"turn {turn}: the window layers hold positions from {oldest} only, the next "
            f"token attends back to {held - window + 1}")


@pytest.mark.time_limit_s(3600)
@pytest.mark.parametrize("model_path", _MODELS, ids=[os.path.basename(p) for p in _MODELS])
def test_a_long_conversation_keeps_the_cache_the_render_and_memory_bounded(model_path):
    wrapper = ChatWrapper(Config(model_path=model_path, context_size=_CONTEXT_TOKENS, gpu_layers=_GPU_LAYERS,
                                 max_reply_tokens=_REPLY_TOKENS, seed=7, check_cache=True,
                                 compact_window_cache=_COMPACT_WINDOW_CACHE))
    try:
        registry = _clock_registry()
        can = wrapper.capabilities(tools=True)
        system = [{"role": "system", "content": "Be brief. Use the clock tool when asked the time."}]
        wrapper.begin(system if can.can_take_system_prompt else [],
                      tools=registry.specs() if can.can_call_tools else None)
        policy = registry if can.can_call_tools else None
        measured = []                   # RAM and GPU memory after each turn past the warm-up
        for turn in range(_TURNS):
            question = ("What time is it?" if turn % 5 == 1 else
                        f"Tell me one fact about the number {turn}.")
            reply = wrapper.reply(question, tool_call_policy=policy).finish()
            if turn % 7 == 3 and reply.text:
                wrapper.truncate_last_reply(reply.text[:len(reply.text) // 2])
            if turn % 11 == 5:
                wrapper.begin(**wrapper.conversation())
            wrapper.wait_idle(timeout_s=30)
            _assert_the_window_layers_hold_the_window(wrapper, turn)
            if turn >= _WARM_UP_TURNS:
                measured.append((_ram_mib(), _gpu_mib()))
        assert wrapper._conversation._found_fault is None, "check_cache found a fault"
        # Peaks, not single readings: a bounded memory swings, and only growth raises its peak.
        early, late = measured[:_WARM_UP_TURNS], measured[-_WARM_UP_TURNS:]
        grown_ram = max(ram for ram, _ in late) - max(ram for ram, _ in early)
        assert grown_ram <= _MAX_RAM_GROWTH_MIB, f"RAM grew by {grown_ram:.0f} MiB"
        if late[0][1] is not None:
            grown_gpu = max(gpu for _, gpu in late) - max(gpu for _, gpu in early)
            assert grown_gpu <= _MAX_GPU_GROWTH_MIB, f"GPU memory grew by {grown_gpu} MiB"
    finally:
        wrapper.close()


@pytest.mark.time_limit_s(1800)
@pytest.mark.parametrize("model_path", _MODELS, ids=[os.path.basename(p) for p in _MODELS])
def test_a_conversation_switched_away_from_and_back_to_comes_back_whole(model_path):
    wrapper = ChatWrapper(Config(model_path=model_path, context_size=_CONTEXT_TOKENS, gpu_layers=_GPU_LAYERS,
                                 max_reply_tokens=_REPLY_TOKENS, seed=7, check_cache=True,
                                 compact_window_cache=_COMPACT_WINDOW_CACHE,
                                 kept_conversation_bytes=_KEPT_BYTES))
    try:
        can = wrapper.capabilities(tools=False)
        wrapper.begin([{"role": "system", "content": "Be brief."}] if can.can_take_system_prompt else [])
        filler = " ".join(f"word{i}" for i in range(120))
        for turn in range(24):
            wrapper.reply(f"Answer in one word. {filler}").finish()
            wrapper.wait_idle(timeout_s=30)
            if turn % 6 == 5:
                # Away and back, the newest exchange gone: the kept cache is restored and cut.
                kept = wrapper.conversation()
                wrapper.begin([{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}])
                kept["messages"] = kept["messages"][:-2]
                wrapper.begin(**kept)
                wrapper.wait_idle(timeout_s=30)
                _assert_the_window_layers_hold_the_window(wrapper, turn)
        assert wrapper._conversation._found_fault is None, "check_cache found a fault"
    finally:
        wrapper.close()
