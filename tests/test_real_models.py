"""Long conversations on real model files: the cache is checked against a fresh render after every action, and memory stops growing.

The model files are named by the environment, never here:
``LLAMA_CHAT_TEST_MODELS`` holds their paths, separated as the system separates
paths. Without it every test here is skipped. GPU memory is measured by
``LLAMA_CHAT_TEST_GPU_MEMORY_COMMAND``, a shell command that prints the MiB the
process ``{pid}`` holds on the GPU; without it only RAM is bounded.
``LLAMA_CHAT_TEST_GPU_LAYERS`` sets the layers to offload, all by default. Run
with ``-m gpu``.
"""

import os
import subprocess

import pytest

from llama_chat import ChatWrapper, Config, ToolRegistry

_MODELS = [path for path in os.environ.get("LLAMA_CHAT_TEST_MODELS", "").split(os.pathsep) if path]
# The turns of each conversation, the first of them to fill the bounded memories before
# growth is measured.
_TURNS, _WARM_UP_TURNS = 120, 30
# How much the process may grow over the turns after the warm-up: in RAM, the bounded memories
# reaching their bound; on the GPU, nothing.
_MAX_RAM_GROWTH_MIB, _MAX_GPU_GROWTH_MIB = 64, 16
_CONTEXT_TOKENS, _REPLY_TOKENS = 4096, 48
_GPU_LAYERS = int(os.environ.get("LLAMA_CHAT_TEST_GPU_LAYERS", "-1"))

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _MODELS, reason="LLAMA_CHAT_TEST_MODELS names no model file")]


def _ram_mib() -> float:
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) / 1024 for line in status if line.startswith("VmRSS:"))


_GPU_MEMORY_COMMAND = os.environ.get("LLAMA_CHAT_TEST_GPU_MEMORY_COMMAND")


def _gpu_mib() -> int | None:
    """The GPU memory this process holds, by the command the environment names; ``None`` without one."""
    if not _GPU_MEMORY_COMMAND:
        return None
    printed = subprocess.run(_GPU_MEMORY_COMMAND.format(pid=os.getpid()), shell=True,
                             capture_output=True, text=True, check=True).stdout
    return int(float(printed.split()[0])) if printed.split() else 0


def _clock_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The current time.", name="clock",
                  parameters={"type": "object", "properties": {}})
    return registry


@pytest.mark.time_limit_s(3600)
@pytest.mark.parametrize("model_path", _MODELS, ids=[os.path.basename(p) for p in _MODELS])
def test_a_long_conversation_keeps_the_cache_the_render_and_memory_bounded(model_path):
    wrapper = ChatWrapper(Config(model_path=model_path, context_size=_CONTEXT_TOKENS, gpu_layers=_GPU_LAYERS,
                                 max_reply_tokens=_REPLY_TOKENS, seed=7, check_cache=True))
    try:
        registry = _clock_registry()
        can = wrapper.capabilities(tools=True)
        system = [{"role": "system", "content": "Be brief. Use the clock tool when asked the time."}]
        wrapper.begin(system if can.can_take_system_prompt else [],
                      tools=registry.specs() if can.can_call_tools else None)
        policy = registry if can.can_call_tools else None
        measured = {}
        for turn in range(_TURNS):
            question = ("What time is it?" if turn % 5 == 1 else
                        f"Tell me one fact about the number {turn}.")
            reply = wrapper.reply(question, tool_call_policy=policy).finish()
            if turn % 7 == 3 and reply.text:
                wrapper.truncate_last_reply(reply.text[:len(reply.text) // 2])
            if turn % 11 == 5:
                wrapper.begin(**wrapper.conversation())
            wrapper.wait_idle(timeout_s=30)
            if turn == _WARM_UP_TURNS:
                measured["start"] = (_ram_mib(), _gpu_mib())
        assert wrapper._conversation._found_fault is None, "check_cache found a fault"
        ram, gpu = _ram_mib(), _gpu_mib()
        start_ram, start_gpu = measured["start"]
        assert ram - start_ram <= _MAX_RAM_GROWTH_MIB, f"RAM grew by {ram - start_ram:.0f} MiB"
        if gpu is not None:
            assert gpu - start_gpu <= _MAX_GPU_GROWTH_MIB, f"GPU memory grew by {gpu - start_gpu} MiB"
    finally:
        wrapper.close()
