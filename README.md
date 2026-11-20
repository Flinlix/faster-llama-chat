# faster-llama-chat: a KV-cache manager for llama.cpp

<!-- --8<-- [start:lead] -->
A Python chat library over llama.cpp whose goal is to **prefill as little as
possible**. It keeps the KV cache alive across turns, decodes only what is new,
and when the context fills it *removes the oldest exchanges and shifts the
survivors down to close the gap*, reusing their KV instead of decoding them
again.

Every model whose GGUF ships a chat template is encoded exactly as that
template writes the conversation, without code particular to the model, or
refused: the template renders the conversation, and the cache is compared
with that render and decoded only where the two differ. The work for the next
question is done right after a reply, while the reply is read, so a question
waits only for its own tokens. Tool calls work on any model whose template
describes them: the tags and the argument spelling are recovered from the
template. What a template cannot tell, such as a model's recommended sampling,
comes from a small JSON add-on file.
<!-- --8<-- [end:lead] -->

The [documentation](#documentation) has a guide per task and the API
reference. The reasons behind each design choice are in the
[decision records](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/decisions/README.md).

## Why

<!-- --8<-- [start:why] -->
1. **Minimum prefill:** only genuinely new tokens are processed; the cache from
   previous turns survives and is reused.
2. **Rolling eviction:** when the context fills, the oldest exchanges are
   dropped, so a chat runs indefinitely.

Trimming the prompt defeats the first goal: removing old messages from the
start shifts every later token's position and invalidates the whole cached
prefix, so every turn decodes in full. The library edits the cache in place
instead: it removes the oldest exchange's tokens and shifts the survivors'
positions down, so kept messages are never processed again.
<!-- --8<-- [end:why] -->

## Install

<!-- --8<-- [start:install] -->
```bash
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python "git+https://github.com/Flinlix/faster-llama-chat@v3.0.0"   # CPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python "git+https://github.com/Flinlix/faster-llama-chat@v3.0.0"     # Nvidia GPU
```

The package is installed from GitHub; it is not published elsewhere. `@v3.0.0`
names a release; without it, the newest commit is installed. Each
[release](https://github.com/Flinlix/faster-llama-chat/releases) also carries
the built package.

llama-cpp-python is built **from source per machine**, so it uses that CPU's
own kernels or the CUDA backend. From a checkout of this repository,
`install.sh` does the same:

```bash
./install.sh            # CUDA where nvidia-smi is present, else a native-CPU build
./install.sh --cpu      # force the native-CPU build
./install.sh --cuda     # force the Nvidia GPU build
./install.sh --dev      # add the test, development and documentation tools (append to any of the above)
```

The same by hand:

```bash
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.23"   # CPU
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.23"     # Nvidia GPU
pip install -e .                    # or -e ".[test,dev,docs]" with those tools
git config core.hooksPath hooks     # with them: the checks before every push
```

It installs as `faster-llama-chat` and is imported as `llama_chat`.
<!-- --8<-- [end:install] -->

## Quick start

<!-- --8<-- [start:quick-start] -->
```python
from llama_chat import ChatWrapper, Config

chat = ChatWrapper(Config(model_path="models/your-model.gguf", context_size=8192,
                          gpu_layers=-1))

# begin: prefill the system prompt and as much recent history as fits, as Chat
# Completions messages. A template that writes a system prompt where later
# questions move it refuses one (capabilities(tools=False).can_take_system_prompt
# is False); leave the system message out there.
chat.begin([{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}])

# inject: prefill one message as context, without generating
chat.inject("Reference: the deadline is Friday. What's the deadline?")

# reply: prefill only what is new, then stream the reply, which answers the
# injected message
for delta in chat.reply():
    print(delta, end="", flush=True)

reply = chat.reply("Summarize the plan.").finish()    # or read it to its end at once
print(reply.text, reply.stop_reason)
```
<!-- --8<-- [end:quick-start] -->

How a reply streams, how to stop one, and how the wrapper behaves across
threads: [Streams, cancelling and threads](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/streams-and-threads.md).

## Supported models

Any model whose GGUF file ships a chat template runs, and the library is not
limited to the models below. They are the models whose templates the tests
run every conversation on; a template that differs from all of them is read
the same way, and refused with its cause where it cannot be.

| Model | Template taken from | Reasoning | Tool calls |
|---|---|---|---|
| DeepSeek-R1-0528-Qwen3-8B | [unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) | yes | yes |
| DeepSeek-R1-Distill-Qwen-7B | [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) | yes |  |
| DeepSeek-V4.1-Flash-1017 | [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | yes | yes |
| Devstral-Small-2-24B-Instruct-2512 | [unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |  | yes |
| Dolphin3-Cyber-8B | [RavichandranJ/Dolphin3-Cyber-8B-GGUF](https://huggingface.co/RavichandranJ/Dolphin3-Cyber-8B-GGUF) |  |  |
| Flint-abliterated | [orcarouter/Flint-abliterated-GGUF](https://huggingface.co/orcarouter/Flint-abliterated-GGUF) | yes | yes |
| gemma-3-1b-it | [ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) |  |  |
| gemma-4-26B-A4B-it | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | yes | yes |
| GLM-4.7-Flash | [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) | yes | yes |
| GLM-5.3-Flash-Maya | [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) | yes | yes |
| gpt-oss-20b | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | yes | yes |
| granite-4.2-8b | [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) | yes | yes |
| Hermes-2-Pro-Llama-3-8B | [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) |  | yes |
| Hy-Memory-30B-A3B | [rapatsky/Hy-Memory-30B-A3B-GGUF](https://huggingface.co/rapatsky/Hy-Memory-30B-A3B-GGUF) | yes | yes |
| Hy-MT2-1.8B | [tencent/Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |  |  |
| Hy3 | [AngelSlim/Hy3-GGUF](https://huggingface.co/AngelSlim/Hy3-GGUF) | yes | yes |
| Hy4-preview | [AngelSlim/Hy4-preview-GGUF](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) | yes | yes |
| HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M | [rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF](https://huggingface.co/rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF) |  |  |
| Inkling-Small | [unsloth/Inkling-Small-GGUF](https://huggingface.co/unsloth/Inkling-Small-GGUF) | yes |  |
| Kimi-K3 | [unsloth/Kimi-K3-GGUF](https://huggingface.co/unsloth/Kimi-K3-GGUF) | yes | yes |
| Laguna-XS-2.1-APEX | [mudler/Laguna-XS-2.1-APEX-GGUF](https://huggingface.co/mudler/Laguna-XS-2.1-APEX-GGUF) | yes | yes |
| LFM2.5-24B-A2B-Uncensored | [AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF) | yes | yes |
| LFM3-Omni-2B | [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | yes | yes |
| Ling-3.0-tiny | [bartowski/Ling-3.0-tiny-GGUF](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF) | yes | yes |
| Mage-VL-Cowork | [Hansda/Mage-VL-Cowork-GGUF](https://huggingface.co/Hansda/Mage-VL-Cowork-GGUF) | yes | yes |
| microsoft_Phi-4-mini-instruct | [bartowski/microsoft_Phi-4-mini-instruct-GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF) |  |  |
| MiniCPM5-2B | [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) | yes | yes |
| MiniMax-M3.1-Flash-abliterated | [orcarouter/MiniMax-M3.1-Flash-abliterated-GGUF](https://huggingface.co/orcarouter/MiniMax-M3.1-Flash-abliterated-GGUF) | yes | yes |
| Ministral-3-8B-Instruct-2512 | [mistralai/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF) |  | yes |
| Ministral-3-8B-Instruct-2512 | [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) |  | yes |
| Ministral-3-8B-Reasoning-2512 | [unsloth/Ministral-3-8B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Reasoning-2512-GGUF) | yes | yes |
| Ministral-8B-Instruct-2410 | [bartowski/Ministral-8B-Instruct-2410-GGUF](https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF) |  | yes |
| Mistral-Large-4-1T-A52B | [unsloth/Mistral-Large-4-1T-A52B-GGUF](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF) | yes | yes |
| Mistral-Small-3.2-24B-Instruct-2506 | [unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF) |  | yes |
| Muse-Glimmer-30B | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | yes | yes |
| Nanbeige_Nanbeige4.3-33B | [bartowski/Nanbeige_Nanbeige4.3-33B-GGUF](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF) | yes | yes |
| North-Mini-Code-1.0 | [unsloth/North-Mini-Code-1.0-GGUF](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) | yes | yes |
| NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | [AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF) | yes | yes |
| NVIDIA-Nemotron-Labs-Dory1.0-4B | [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF) | yes | yes |
| Orpheus-3b-German-FT-Q8_0 | [lex-au/Orpheus-3b-German-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-German-FT-Q8_0.gguf) |  | yes |
| PaddleOCR-VL-1.6 | [PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF) |  |  |
| Qwen2.5-Coder-7B-Instruct | [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) |  | yes |
| Qwen3-8B | [unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | yes | yes |
| Qwen3-Coder-30B-A3B-Instruct | [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) |  | yes |
| Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | yes | yes |
| Qwen3.6-27B | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | yes | yes |
| Qwen3.8-27B-OBLITERATED | [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) | yes |  |
| Qwopus3.8-27B-Coder-Flash-TQ | [YTan2000/Qwopus3.8-27B-Coder-Flash-TQ](https://huggingface.co/YTan2000/Qwopus3.8-27B-Coder-Flash-TQ) | yes | yes |
| Sharp-Spark-X2.5-4B | [peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF](https://huggingface.co/peculiar-ragdoll/Sharp-Spark-X2.5-4B-GGUF) | yes | yes |
| Spark-X2.6-4B | [XHToken/Spark-X2.6-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF) | yes | yes |
| tiny-aya-global | [CohereLabs/tiny-aya-global-GGUF](https://huggingface.co/CohereLabs/tiny-aya-global-GGUF) |  |  |
| TwIL-8B | [webAI-Official/TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B) | yes | yes |
| TwIL-LM3 | [webAI-Official/TwIL-LM3](https://huggingface.co/webAI-Official/TwIL-LM3) | yes |  |

## Documentation

- [What to call, by task](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/actions.md)
- [Streams, cancelling and threads](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/streams-and-threads.md)
- [Tool calling](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/tool-calling.md)
- [Reasoning](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/reasoning.md)
- [Keeping a conversation](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/keeping-a-conversation.md)
- [Configuration](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/configuration.md)
- [Add-on files](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/add-on-files.md)
- [A model without a usable template](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/without-a-template.md)
- [Your own cache context](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/cache-context.md)
- [Verify](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/guide/verify.md)
- [API reference](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/reference.md)
- [Decision records](https://github.com/Flinlix/faster-llama-chat/blob/main/docs/decisions/README.md)
