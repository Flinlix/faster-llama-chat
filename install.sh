#!/usr/bin/env bash
# Install llama-cpp-python built from source for this machine's backend.
#
#   ./install.sh           auto-detect: CUDA if nvidia-smi is present, else CPU
#   ./install.sh --cpu     force the native-CPU build
#   ./install.sh --cuda    force the Nvidia GPU build
#   add --dev to any of the above to also install the test, development and documentation tools
#
# Why build from source: the prebuilt wheel targets a baseline instruction set and
# ships no CUDA. GGML_NATIVE=ON compiles with -march=native so the CPU's own
# kernels are used; GGML_CUDA=on compiles the CUDA backend so layers can offload.
set -euo pipefail

cd "$(dirname "$0")"

# Parse flags in any order: backend (--cpu/--cuda, else auto-detect) and --dev.
BACKEND=""
DEV=0
for arg in "$@"; do
    case "$arg" in
        --cpu) BACKEND="cpu" ;;
        --cuda) BACKEND="cuda" ;;
        --dev) DEV=1 ;;
        *) echo "usage: $0 [--cpu|--cuda] [--dev]" >&2; exit 2 ;;
    esac
done
[ -n "$BACKEND" ] || BACKEND="$(command -v nvidia-smi >/dev/null && echo cuda || echo cpu)"

# Build prerequisites (cmake, a C++ compiler, Python headers), installed only when asked.
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
    echo "The build needs cmake, a C++ compiler and the Python headers."
    if command -v apt-get >/dev/null && [ -t 0 ]; then
        read -r -p "Install them with: sudo apt-get install -y build-essential cmake python3-dev? [y/N] " answer
        [ "$answer" = "y" ] || [ "$answer" = "Y" ] || { echo "Install them, then run $0 again." >&2; exit 1; }
        sudo apt-get update
        sudo apt-get install -y build-essential cmake python3-dev
    else
        echo "Install them with your system's package manager, then run $0 again." >&2
        exit 1
    fi
fi

# The project venv's Python if present, otherwise the active one; pip always runs as its module.
PYTHON="${PYTHON:-.venv/bin/python}"
command -v "$PYTHON" >/dev/null || PYTHON="$(command -v python || command -v python3 || echo python)"

# The CUDA compiler where the build looks for it: CUDACXX, PATH, or the toolkit's default place.
if [ "$BACKEND" = "cuda" ] && [ -z "${CUDACXX:-}" ] && ! command -v nvcc >/dev/null \
        && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "The CUDA build needs the CUDA compiler (nvcc): not in CUDACXX, on PATH or in /usr/local/cuda." >&2
    echo "Install the CUDA toolkit, or run $0 --cpu." >&2
    exit 1
fi

if [ "$BACKEND" = "cuda" ]; then
    CMAKE_ARGS="-DGGML_CUDA=on"
else
    CMAKE_ARGS="-DGGML_NATIVE=ON"
fi

echo "Building llama-cpp-python from source ($BACKEND backend: $CMAKE_ARGS)..."
CMAKE_ARGS="$CMAKE_ARGS" \
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
    "$PYTHON" -m pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.23"

# Install faster-llama-chat itself (editable) so `import llama_chat` and the demo work
# without setting PYTHONPATH. Pass --dev to also install the test, development and documentation tools.
echo "Installing faster-llama-chat (editable)..."
if [ "$DEV" = 1 ]; then
    "$PYTHON" -m pip install -e ".[test,dev,docs]"
    # The checks before every push: hooks/pre-push.
    git config core.hooksPath hooks
else
    "$PYTHON" -m pip install -e "."
fi

echo "Verifying the build and the llama.cpp symbols the package relies on..."
"$PYTHON" - <<'PY'
import llama_cpp

# The backend's own symbol resolution is the source of truth: constructing it
# looks up every symbol the package calls, the cache edits included, and raises
# if this build lacks one.
from llama_chat._backend import Backend
Backend()

print("llama-cpp-python", llama_cpp.__version__)
print("required symbols:", "OK")
print("system info:", llama_cpp.llama_print_system_info().decode().strip())
PY

echo "Done."
