#!/usr/bin/env bash
# Install llama-cpp-python built from source, optimized for the Raspberry Pi 5.
#
# The current llama_chat package imports the `llama_cpp` module (the pip package,
# which bundles its own llama.cpp) -- it does NOT use a separate llama.cpp clone.
#
# Why build from source instead of the prebuilt wheel: the generic aarch64 wheel
# targets a baseline ISA. GGML_NATIVE=ON compiles with -march=native, which on the
# Pi 5's Cortex-A76 (Armv8.2) auto-enables the asimddp dot-product kernels -- the
# main prefill speedup -- while avoiding instructions the chip lacks (e.g. i8mm).
set -euo pipefail

cd "$(dirname "$0")"

# Build prerequisites (cmake, a C++ compiler, Python headers).
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
    echo "Installing build tools (needs sudo)..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake python3-dev
fi

# Use the project venv if present, otherwise the active python.
PIP="${PIP:-.venv/bin/pip}"
[ -x "$PIP" ] || PIP="pip"

echo "Building llama-cpp-python from source with native Pi 5 flags..."
CMAKE_ARGS="-DGGML_NATIVE=ON" \
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
    "$PIP" install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python>=0.3.0"

# Install llama_chat itself (editable) plus the test extra, so `import
# llama_chat`, the demo, and `pytest` all work without setting PYTHONPATH.
echo "Installing llama_chat (editable) + test dependencies..."
"$PIP" install -e ".[test]"

echo "Verifying the build and the llama.cpp symbols the package relies on..."
"$(dirname "$PIP")/python" - <<'PY'
import sys, llama_cpp

# 1) KV-memory API (the evict-and-shift scalpel).
kv = ["llama_get_memory", "llama_memory_seq_rm", "llama_memory_seq_add"]
# 2) Template auto-detection + turn-terminator validation.
tmpl = ["llama_model_chat_template", "llama_vocab_is_control"]
missing = [n for n in kv + tmpl if not hasattr(llama_cpp, n)]

# 3) Confirm the native build took effect: DOTPROD is the Cortex-A76 dot-product
#    kernel and the whole reason for building from source. A generic wheel would
#    report DOTPROD = 0 and silently lose most of the prefill speedup.
info = llama_cpp.llama_print_system_info().decode()
dotprod = "DOTPROD = 1" in info

print("llama-cpp-python", llama_cpp.__version__)
print("required symbols:", "OK" if not missing else f"MISSING {missing}")
print("native DOTPROD kernels:", "OK" if dotprod else "NOT ENABLED (generic build?)")
if missing or not dotprod:
    sys.exit(1)
PY

echo "Done. Run the tests with: .venv/bin/python -m pytest"
