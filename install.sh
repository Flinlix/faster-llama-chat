#!/usr/bin/env bash
# Install llama-cpp-python built from source for this machine's backend.
#
#   ./install.sh           auto-detect: CUDA if nvidia-smi is present, else CPU
#   ./install.sh --cpu     force the native-CPU build
#   ./install.sh --cuda    force the Nvidia GPU build
#   add --dev to any of the above to also install test dependencies
#
# The current llama_chat package imports the `llama_cpp` module (the pip package,
# which bundles its own llama.cpp) -- it does NOT use a separate llama.cpp clone.
#
# Why build from source instead of the prebuilt wheel: the prebuilt wheel targets
# a baseline ISA and ships no CUDA. GGML_NATIVE=ON compiles with -march=native,
# enabling the CPU's own kernels (dot-product on Armv8.2, AVX2 on x86) -- the main
# prefill speedup; GGML_CUDA=on compiles the CUDA backend so layers can offload.
set -euo pipefail

cd "$(dirname "$0")"

# Pick the build backend: explicit flag wins, otherwise auto-detect.
BACKEND=""
case "${1:-}" in
    --cpu) BACKEND="cpu" ;;
    --cuda) BACKEND="cuda" ;;
    "") BACKEND="$(command -v nvidia-smi >/dev/null && echo cuda || echo cpu)" ;;
    *) echo "usage: $0 [--cpu|--cuda]" >&2; exit 2 ;;
esac

# Build prerequisites (cmake, a C++ compiler, Python headers).
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
    echo "Installing build tools (needs sudo)..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake python3-dev
fi

# Use the project venv if present, otherwise the active python.
PIP="${PIP:-.venv/bin/pip}"
[ -x "$PIP" ] || PIP="pip"
PYTHON="$(dirname "$PIP")/python"
[ -x "$PYTHON" ] || PYTHON="python"

if [ "$BACKEND" = "cuda" ]; then
    CMAKE_ARGS="-DGGML_CUDA=on"
else
    CMAKE_ARGS="-DGGML_NATIVE=ON"
fi

echo "Building llama-cpp-python from source ($BACKEND backend: $CMAKE_ARGS)..."
CMAKE_ARGS="$CMAKE_ARGS" \
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
    "$PIP" install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.0"

# Install llama_chat itself (editable) so `import llama_chat` and the demo work
# without setting PYTHONPATH. Pass --dev to also install test dependencies.
echo "Installing llama_chat (editable)..."
if [ "${1:-}" = "--dev" ] || [ "${2:-}" = "--dev" ]; then
    "$PIP" install -e ".[test]"
else
    "$PIP" install -e "."
fi

echo "Verifying the build and the llama.cpp symbols the package relies on..."
"$PYTHON" - <<'PY'
import sys, llama_cpp

# The backend's own symbol resolution is the source of truth: constructing it runs
# every required-symbol lookup (with the version fallbacks it supports) and raises
# if a mandatory call is genuinely absent.
from llama_chat._backend import Backend
b = Backend()

# The evict-and-shift core needs one working path per memory op -- either the new
# llama_memory_* API or the older llama_kv_self_* / llama_kv_cache_* fallback, the
# same OR the backend itself uses.
kv_ok = bool(
    (b._mem_clear or b._kv_clear)
    and (b._mem_rm or b._kv_rm)
    and (b._mem_add or b._kv_add)
)

print("llama-cpp-python", llama_cpp.__version__)
print("required symbols:", "OK")
print("KV-cache eviction ops:", "OK" if kv_ok else "MISSING")
print("system info:", llama_cpp.llama_print_system_info().decode().strip())
if not kv_ok:
    sys.exit(1)
PY

echo "Done!"
