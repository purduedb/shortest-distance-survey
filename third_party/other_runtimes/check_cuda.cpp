// Download `libtorch` from: libtorch-shared-with-deps-2.4.0%2Bcu121.zip
// Unzip it and use its "full-path" in the subsequent commands
//
// Usage:
//
// Compile command (CPU):
// g++ check_cuda.cpp -o check_cuda \
//     -I$PWD/libtorch/include \
//     -I$PWD/libtorch/include/torch/csrc/api/include \
//     -L$PWD/libtorch/lib \
//     -ltorch -ltorch_cpu -lc10 \
//     -Wl,-rpath,$PWD/libtorch/lib \
//     -D_GLIBCXX_USE_CXX11_ABI=0
//
// Compile command (CUDA):
// g++ check_cuda.cpp -o check_cuda \
//     -I$HOME/scratch/shortest-distance-survey/tmp/cpp_runtime/libtorch/include \
//     -I$HOME/scratch/shortest-distance-survey/tmp/cpp_runtime/libtorch/include/torch/csrc/api/include \
//     -L$HOME/scratch/shortest-distance-survey/tmp/cpp_runtime/libtorch/lib \
//     -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda\
//     -Wl,-rpath,$HOME/scratch/shortest-distance-survey/tmp/cpp_runtime/libtorch/lib \
//     -D_GLIBCXX_USE_CXX11_ABI=0
//
// Run command:
// ./check_cuda

#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Torch version: " << TORCH_VERSION << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is NOT available." << std::endl;
    }

    return 0;
}
