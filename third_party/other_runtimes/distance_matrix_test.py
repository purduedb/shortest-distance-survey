import torch
import numpy as np
import time

# Check PyTorch and GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters (same as TensorFlow version)
NUM_NODES = 10_000
EMBEDDING_DIM = 2
NUM_SAMPLES = 1_000_000
BATCH_SIZE = 1024 * 1024
NUM_WORKERS = 4
EVAL_RUNS = 10
DEVICE = "cpu"
device = DEVICE
print("Using device: ", device)

# Model definition
class SimpleEmbeddingModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        # # Initialize random embeddings and freeze them
        # initial_embedding = torch.randn(num_nodes, embedding_dim)
        # self.embedding = torch.nn.Parameter(initial_embedding, requires_grad=False)
        # NOTE: nn.Parameters shows up in model.parameters(), so we use a buffer instead
        self.distance_matrix = torch.nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=False)

    def forward(self, src, dst):
        # # Get embeddings for source and destination nodes
        # src_emb = self.embedding[src]
        # dst_emb = self.embedding[dst]
        # # Compute L1 distance
        # return torch.norm(src_emb - dst_emb, p=1, dim=1)
        return self.distance_matrix[src, dst]

# Create data
src = torch.randint(0, NUM_NODES, (NUM_SAMPLES,), dtype=torch.long)
dst = torch.randint(0, NUM_NODES, (NUM_SAMPLES,), dtype=torch.long)
dist = torch.rand(NUM_SAMPLES)  # Random distances between 0 and 1

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(src, dst, dist)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Create model and move to device
model = SimpleEmbeddingModel(NUM_NODES, EMBEDDING_DIM).to(device)
print("\nModel operations running on:", device)

print("Starting inference...")
print(f"Running evaluation {EVAL_RUNS} times to get average query time...")

total_avg_time = 0
for run in range(EVAL_RUNS):
    total_time = 0
    total_samples = 0
    data_move_time = 0
    inference_time = 0

    start_time_eval = time.perf_counter()
    with torch.no_grad():
        for batch_src, batch_dst, _ in dataloader:

            # Forward pass with timing
            start_time = time.perf_counter()

            # Move inputs to device
            start_time_data_move = time.perf_counter()
            batch_src = batch_src.to(device)
            batch_dst = batch_dst.to(device)
            data_move_time = (time.perf_counter() - start_time_data_move) * 1e6

            # Forward pass
            start_time_inference = time.perf_counter()
            output = model(batch_src, batch_dst)
            inference_time = (time.perf_counter() - start_time_inference) * 1e6
            # torch.cuda.synchronize()  # Ensure GPU operations are completed

            # Move outputs to device
            start_time_output_move = time.perf_counter()
            output = output.to(device)
            data_move_time += (time.perf_counter() - start_time_output_move) * 1e6

            # Measure elapsed time
            end_time = time.perf_counter()

            duration = (end_time - start_time) * 1e6  # Convert to microseconds
            total_time += duration
            total_samples += batch_src.size(0)

    end_time_eval = time.perf_counter()
    adjusted_time_per_sample = (end_time_eval - start_time_eval) * 1e6 / total_samples

    # Calculate and print results
    avg_time_per_sample = total_time / total_samples
    print(f"[Eval Run][PyTorch] Query time per sample: {avg_time_per_sample:.6f} microseconds")
    print(f"[Eval Run][PyTorch] Data movement time per sample: {data_move_time/total_samples:.6f} microseconds")
    print(f"[Eval Run][PyTorch] Inference time per sample: {inference_time/total_samples:.6f} microseconds")
    print(f"[Eval Run][PyTorch] Adjusted query time per sample: {adjusted_time_per_sample:.6f} microseconds")

    total_avg_time += avg_time_per_sample

print(f"[Eval Run] Average query time per sample: {total_avg_time/EVAL_RUNS:.6f} microseconds")


#################################
# Results with GPU on PyTorch
#################################
# Run command: `$ python pytorch_test.py`

# Without data movement calls

# Running evaluation 10 times to get average query time...
# [Eval Run][PyTorch] Query time per sample: 0.058760 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.372234 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000438 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.313291 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000525 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.258798 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000432 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 21.107192 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000499 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 20.914008 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000435 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 20.865387 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000441 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 21.101920 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000451 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.043809 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000449 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.161786 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.000455 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 26.187884 microseconds
# [Eval Run] Average query time per sample: 0.006289 microseconds

# With data movement calls

# Running evaluation 10 times to get average query time...
# [Eval Run][PyTorch] Query time per sample: 0.095431 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 20.251347 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002809 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.250726 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002159 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.738785 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002790 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.426810 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002150 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.920271 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002165 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.282696 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002833 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.745437 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002049 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.145924 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002036 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.000076 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002049 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.888962 microseconds
# [Eval Run] Average query time per sample: 0.011647 microseconds

# Verbose

# Running evaluation 10 times to get average query time...
# [Eval Run][PyTorch] Query time per sample: 0.056291 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.002059 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.054220 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.155739 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002107 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001528 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.000571 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 17.887180 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002806 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001734 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.001057 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 17.761624 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.001918 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001500 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.000412 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 17.636826 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002077 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001541 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.000530 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 17.748746 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002776 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001728 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.001033 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.084280 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002827 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001747 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.001066 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.172329 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002768 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001719 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.001035 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 19.303808 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.002010 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001508 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.000495 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.144984 microseconds
# [Eval Run][PyTorch] Query time per sample: 0.001968 microseconds
# [Eval Run][PyTorch] Data movement time per sample: 0.001518 microseconds
# [Eval Run][PyTorch] Inference time per sample: 0.000444 microseconds
# [Eval Run][PyTorch] Adjusted query time per sample: 18.305622 microseconds
# [Eval Run] Average query time per sample: 0.007755 microseconds
