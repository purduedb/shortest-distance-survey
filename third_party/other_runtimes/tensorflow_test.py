import tensorflow as tf
import numpy as np
import time

# Check TF and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Parameters (same as C++ version)
NUM_NODES = 10000
EMBEDDING_DIM = 2
NUM_SAMPLES = 1000000
BATCH_SIZE = 1024 * 1024
NUM_WORKERS = 4
EVAL_RUNS = 10

# Model definition
class SimpleEmbeddingModel(tf.keras.Model):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        # Initialize random embeddings and freeze them
        initial_embedding = tf.random.normal([num_nodes, embedding_dim])
        self.embedding = tf.Variable(initial_embedding, trainable=False)

    def call(self, inputs):
        src, dst = inputs
        # Get embeddings for source and destination nodes
        src_emb = tf.gather(self.embedding, src)
        dst_emb = tf.gather(self.embedding, dst)
        # Compute L1 distance
        return tf.norm(src_emb - dst_emb, ord=1, axis=1)

# Create dataset
src = tf.random.uniform([NUM_SAMPLES], 0, NUM_NODES, dtype=tf.int32)
dst = tf.random.uniform([NUM_SAMPLES], 0, NUM_NODES, dtype=tf.int32)
dist = tf.random.uniform([NUM_SAMPLES])  # Random distances between 0 and 1

# Create TF dataset
dataset = tf.data.Dataset.from_tensor_slices(((src, dst), dist))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Create model
model = SimpleEmbeddingModel(NUM_NODES, EMBEDDING_DIM)

print("\nModel operations running on:", model.embedding.device)

print("Starting inference...")

print("Running evaluation", EVAL_RUNS, "times to get average query time...")
for _ in range(EVAL_RUNS):
    # Main evaluation loop
    total_time = 0
    total_samples = 0

    start_time_eval = time.perf_counter()
    for batch in dataset:
        # Forward pass with timing
        start_time = time.perf_counter()
        output = model(batch[0])
        end_time = time.perf_counter()

        duration = (end_time - start_time) * 1e6  # Convert to microseconds
        total_time += duration
        total_samples += batch[0][0].shape[0]
    end_time_eval = time.perf_counter()
    adjusted_time_per_sample = (end_time_eval - start_time_eval) * 1e6 / total_samples

    # Calculate and print results
    avg_time_per_sample = total_time / total_samples
    # print(f"Average inference time per sample: {avg_time_per_sample:.3f} microseconds")
    print(f"[Eval Run][Tensorflow] Query time per sample: {avg_time_per_sample:.3f} microseconds")
    print(f"[Eval Run][Tensorflow] Adjusted query time per sample: {adjusted_time_per_sample:.3f} microseconds")
    # print(f"Total samples processed: {total_samples}")
    # print(f"Total time: {total_time/1e6:.3f} seconds")


#################################
# Results with GPU on Tensorflow
#################################
# Run command: `$ python tensorflow_test.py``

# Running evaluation 10 times to get average query time...
# [Eval Run][Tensorflow] Query time per sample: 0.033 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 2.742 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.005 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.146 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.006 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 2.966 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.004 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.076 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.006 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.012 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.005 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.513 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.004 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 2.921 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.004 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.284 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.005 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 4.038 microseconds
# [Eval Run][Tensorflow] Query time per sample: 0.004 microseconds
# [Eval Run][Tensorflow] Adjusted query time per sample: 3.112 microseconds

#################################
# Results with GPU on PyTorch
#################################
# Run command: `$ python train.py --model_class lpnorm --data_dir Harbin --eval_runs 10`

# Running evaluation 10 times to get average query time...
# [Eval Run] Query time per sample: 0.008 microseconds
# [Eval Run] Adjusted query time per sample: 7.494 microseconds
# [Eval Run] Query time per sample: 0.007 microseconds
# [Eval Run] Adjusted query time per sample: 7.461 microseconds
# [Eval Run] Query time per sample: 0.003 microseconds
# [Eval Run] Adjusted query time per sample: 7.435 microseconds
# [Eval Run] Query time per sample: 0.003 microseconds
# [Eval Run] Adjusted query time per sample: 7.428 microseconds
# [Eval Run] Query time per sample: 0.007 microseconds
# [Eval Run] Adjusted query time per sample: 7.454 microseconds
# [Eval Run] Query time per sample: 0.003 microseconds
# [Eval Run] Adjusted query time per sample: 7.442 microseconds
# [Eval Run] Query time per sample: 0.007 microseconds
# [Eval Run] Adjusted query time per sample: 7.441 microseconds
# [Eval Run] Query time per sample: 0.007 microseconds
# [Eval Run] Adjusted query time per sample: 7.455 microseconds
# [Eval Run] Query time per sample: 0.003 microseconds
# [Eval Run] Adjusted query time per sample: 7.434 microseconds
# [Eval Run] Query time per sample: 0.007 microseconds
# [Eval Run] Adjusted query time per sample: 7.435 microseconds
# [Eval Run] Average query time per sample: 0.005 microseconds
