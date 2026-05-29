# On the Raspberry Pi (or any ARM64 Linux)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j4