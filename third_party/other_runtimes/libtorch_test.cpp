#include <torch/torch.h>
#include <iostream>
#include <chrono>


// Define the model class
class SimpleEmbeddingModel : public torch::nn::Module {
public:
    SimpleEmbeddingModel(int64_t num_nodes, int64_t embedding_dim) {
        embedding = register_parameter("embedding",
            torch::randn({num_nodes, embedding_dim}));
        embedding.requires_grad_(false);  // Freeze parameters
    }

    torch::Tensor forward(const torch::Tensor& src, const torch::Tensor& dst) {
        auto src_emb = embedding.index_select(0, src);
        auto dst_emb = embedding.index_select(0, dst);
        return torch::l1_loss(src_emb, dst_emb, torch::Reduction::None).sum(1);
    }

private:
    torch::Tensor embedding;
};

// Custom Dataset class
class DistanceDataset : public torch::data::Dataset<DistanceDataset> {
public:
    DistanceDataset(size_t num_samples, int64_t num_nodes) {
        src = torch::randint(0, num_nodes, {num_samples}).cuda();
        dst = torch::randint(0, num_nodes, {num_samples}).cuda();
        dist = torch::rand({num_samples}).cuda();  // Random distances between 0 and 1
    }

    torch::data::Example<> get(size_t index) override {
        return {torch::stack({src[index], dst[index]}), dist[index]};
    }

    torch::optional<size_t> size() const override {
        return src.size(0);
    }

private:
    torch::Tensor src, dst, dist;
};

int main() {
    std::cout << "Torch version: " << TORCH_VERSION << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is NOT available." << std::endl;
    }

    std::cout << "Testing Libtorch functionality..." << std::endl;


    // Parameters
    const int64_t NUM_NODES = 10000;
    const int64_t EMBEDDING_DIM = 2;
    const size_t NUM_SAMPLES = 1000000;
    const int64_t BATCH_SIZE = 1024*1024;
    const int64_t NUM_WORKERS = 4;
    const int64_t EVAL_RUNS = 10;

    // Create model
    auto model = std::make_shared<SimpleEmbeddingModel>(NUM_NODES, EMBEDDING_DIM);
    model->to(torch::kCUDA);
    model->eval();  // Set to evaluation mode

    // Create dataset and dataloader
    auto dataset = DistanceDataset(NUM_SAMPLES, NUM_NODES)
        .map(torch::data::transforms::Stack<>());

    auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(NUM_WORKERS)
    );

    // Inference timing
    torch::NoGradGuard no_grad;  // Disable gradient computation

    std::cout << "Starting inference..." << std::endl;

    double total_time;
    size_t total_samples;

    std::cout << "Running evaluation "  << EVAL_RUNS << " times..." << std::endl;
    for (int i = 0; i < EVAL_RUNS; ++i) {
        total_time = 0.0;
        total_samples = 0;
        auto start_eval = std::chrono::high_resolution_clock::now();
        for (const auto& batch : *dataloader) {

            // Extract src and dst from the input
            auto indices = batch.data;
            auto src = indices.slice(1, 0, 1).squeeze();
            auto dst = indices.slice(1, 1, 2).squeeze();

            // Forward pass
            auto start = std::chrono::high_resolution_clock::now();
            auto output = model->forward(src, dst);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            total_time += duration.count();
            total_samples += batch.data.size(0);
        }
        auto end_eval = std::chrono::high_resolution_clock::now();
        auto duration_eval = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval);
        double adjusted_time_per_sample = static_cast<double>(duration_eval.count()) / total_samples;

        // Calculate and print results
        double avg_time_per_sample = total_time / total_samples;
        // std::cout << "Average inference time per sample: " << avg_time_per_sample
        //         << " microseconds" << std::endl;
        std::cout << "[Eval Run][Libtorch] Query time per sample: " << avg_time_per_sample << " microseconds" << std::endl;
        std::cout << "[Eval Run][Libtorch] Adjusted query time per sample: " << adjusted_time_per_sample << " microseconds" << std::endl;
        // std::cout << "Total samples processed: " << total_samples << std::endl;
        // std::cout << "Total time: " << total_time / 1000000 << " seconds" << std::endl;
    }


    return 0;
}

// WITH CPU (not needed, but we may need to remove `.cuda()` and `kCUDA` calls):
// Compile command (CPU):
// g++ libtorch_test.cpp -o libtorch_test \
//     -I$PWD/libtorch/include \
//     -I$PWD/libtorch/include/torch/csrc/api/include \
//     -L$PWD/libtorch/lib \
//     -ltorch -ltorch_cpu -lc10 \
//     -Wl,-rpath,$PWD/libtorch/lib \
//     -D_GLIBCXX_USE_CXX11_ABI=0
//
// #################################
// Results with GPU on libtorch (C++)
// #################################
//
// WITH CUDA:
// Compile command (CUDA):
// g++ libtorch_test.cpp -o libtorch_test \
//     -I$PWD/libtorch/include \
//     -I$PWD/libtorch/include/torch/csrc/api/include \
//     -L$PWD/libtorch/lib \
//     -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda \
//     -Wl,-rpath,$PWD/libtorch/lib \
//     -D_GLIBCXX_USE_CXX11_ABI=0
// Run command: `$ ./libtorch_test`
//
// Running evaluation 10 times...
// [Eval Run][Libtorch] Query time per sample: 2.19747 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 26.334 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000219 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 23.329 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000215 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 24.4807 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000413 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 23.6782 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000198 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 21.6423 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.0002 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 21.5274 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000222 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 20.8389 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000246 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 21.5921 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000201 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 20.6067 microseconds
// [Eval Run][Libtorch] Query time per sample: 0.000188 microseconds
// [Eval Run][Libtorch] Adjusted query time per sample: 20.465 microseconds
