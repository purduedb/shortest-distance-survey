# Usage:
## All pairs - Surat
#   python generate_query_data.py --data_name Surat --query_strategy all --save_dir all_pairs

## Real workload query_perturbation (perturb_k / k_hop are dataset-specific)
#   python generate_query_data.py --data_name W_Jinan    --query_dir real_workload --query_strategy query_perturbation --perturb_k 60  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_Shenzhen --query_dir real_workload --query_strategy query_perturbation --perturb_k 60  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_Chengdu  --query_dir real_workload --query_strategy query_perturbation --perturb_k 30  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_Beijing  --query_dir real_workload --query_strategy query_perturbation --perturb_k 50  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_Shanghai --query_dir real_workload --query_strategy query_perturbation --perturb_k 10  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_NewYork  --query_dir real_workload --query_strategy query_perturbation --perturb_k 10  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_data.py --data_name W_Chicago  --query_dir real_workload --query_strategy query_perturbation --perturb_k 120 --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k

## landmark_hl 30M - DIMACS
# python generate_query_data.py --data_name FLA --query_strategy landmark_hl --num_landmarks 30 --save_dir landmark_30M
# python generate_query_data.py --data_name E   --query_strategy landmark_hl --num_landmarks 10 --save_dir landmark_30M
# python generate_query_data.py --data_name W   --query_strategy landmark_hl --num_landmarks 5  --save_dir landmark_30M
# python generate_query_data.py --data_name CTR --query_strategy landmark_hl --num_landmarks 2  --save_dir landmark_30M
# python generate_query_data.py --data_name USA --query_strategy landmark_hl --num_landmarks 1  --save_dir landmark_30M

## random_hl 500k
#   python generate_query_data.py --data_name W_Jinan --query_strategy random_hl --save_dir random_500k --num_random_pairs 500000

## random_hl 1M
#   python generate_query_data.py --data_name FLA --query_strategy random_hl --save_dir random_1M


# Imports
import os
import sys
import argparse
import subprocess
import tempfile

import numpy as np

sys.path.insert(0, "..")
from utils.data_utils import (
    load_graph,
    print_summary_stats,
    seed_everything,
    augment_nodes,
    augment_queries,
)

from utils.torch_utils import (
    save_dataset,
    load_dataset,
    read_query_file,
    AllPairsDataset,
    LandmarkPairsDataset,
    RandomPairsDataset,
    WorkloadDataset,
)

# NOTE: Import torch after igraph (here via data_utils) to avoid `ImportError: ... CXXABI_1.3.15 not found`
from torch.utils.data import Dataset, random_split


def _hl_distances(index_path, pairs, inference_bin):
    """Compute distances for (src, dst) pairs via the C++ inference binary.
    Pairs are 0-indexed; conversion to/from the 1-indexed C++ index is handled here."""
    # C++ binary: non_ml_index/Hierarchical-Cut-Labelling/src/inference.cpp
    if not os.path.isfile(inference_bin):
        raise FileNotFoundError(
            f"C++ inference binary not found at {inference_bin}. "
            "Rebuild with `make compile` in non_ml_index/Hierarchical-Cut-Labelling/."
        )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pairs', delete=False) as tf:
        pairs_file = tf.name
        for s, d in pairs:
            tf.write(f"{s + 1} {d + 1}\n")  # 0→1-indexed for C++ index
    try:
        proc = subprocess.run(
            [inference_bin, index_path, pairs_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
    finally:
        os.unlink(pairs_file)
    for line in proc.stderr.decode().splitlines():
        print(f"  [C++] {line}")
    result = []
    for line in proc.stdout.decode().splitlines():
        s, d, dist = map(int, line.split())
        result.append((s - 1, d - 1, dist))  # 1→0-indexed
    return result


# Setup argument parser
parser = argparse.ArgumentParser(description="Generate query workloads for shortest-distance datasets.")
parser.add_argument('--data_name', type=str, default='Surat', help='Name of the dataset to use')
parser.add_argument('--data_dir', type=str, default='../../data', help='Directory containing datasets')
parser.add_argument('--query_strategy', type=str, default='query_perturbation', help='Strategy for generating query workload')
parser.add_argument('--query_dir', type=str, default='real_workload', help='Directory for reading queries within data directory')
parser.add_argument('--save_dir', type=str, default='default', help='Directory for saving queries within data directory')
parser.add_argument('--num_landmarks', type=int, default=20, help='Number of landmarks (if using landmark strategy)')
parser.add_argument('--num_random_pairs', type=int, default=1_000_000, help='Number of random pairs (if using random strategy)')
parser.add_argument('--perturb_k', type=int, default=4, help='Number of neighbor perturbations to use when augmenting queries')
parser.add_argument('--k_hop', type=int, default=1, help='Number of hops to consider for k-hop neighbors')
parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to generate (if applicable)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--index_dir', type=str,
                    default='../../non_ml_index/Hierarchical-Cut-Labelling/saved_indexes',
                    help='Directory containing built .hl index files (run from repo root)')

args = parser.parse_args()

# Get arguments
DATA_NAME = args.data_name
DATA_DIR = args.data_dir
QUERY_WORKLOAD_STRATEGY = args.query_strategy
QUERY_DIR = args.query_dir
SAVE_DIR = args.save_dir
NUM_LANDMARKS = args.num_landmarks
NUM_RANDOM_PAIRS = args.num_random_pairs
PERTURB_K = args.perturb_k
K_HOP = args.k_hop
MAX_QUERIES = args.max_queries
SEED = args.seed
INDEX_DIR = args.index_dir

print("Arguments:")
for arg, value in vars(args).items():
    print(f"  - {arg:<20}: {value}")

# Seed for reproducibility
seed_everything(SEED)

# Set up directories
dir_name = os.path.join(DATA_DIR, DATA_NAME)
if not os.path.exists(dir_name):
    raise FileNotFoundError(f"Dataset directory {dir_name} does not exist.")
query_dir = os.path.join(DATA_DIR, DATA_NAME, QUERY_DIR)
save_dir = os.path.join(DATA_DIR, DATA_NAME, SAVE_DIR)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load graph
print(f"Processing dataset: {DATA_NAME} ({dir_name})")
G = load_graph(dir_name)
print_summary_stats(G)

# Create dataset using graph
if QUERY_WORKLOAD_STRATEGY == "all":
    print("Using all pairs strategy...")
    train_dataset = AllPairsDataset(G)
    val_dataset   = train_dataset
    test_dataset  = train_dataset  # Since we are using all pairs, train/val/test datasets are the same
elif QUERY_WORKLOAD_STRATEGY == "landmark":
    print("Using landmark strategy...")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234)  ## Using different seeds compared to model seeds
    val_dataset   = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+2345)
    test_dataset  = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+3456)
elif QUERY_WORKLOAD_STRATEGY == "landmark_split":
    print("Using landmark split strategy (80/10/10 split)...")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234)  ## Using different seeds compared to model seeds
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [0.8, 0.1, 0.1])  # 80-10-10 split
elif QUERY_WORKLOAD_STRATEGY == "random":
    print("Using random pairs strategy...")
    train_dataset = RandomPairsDataset(G, k=NUM_RANDOM_PAIRS, seed=SEED+3456)
    val_dataset   = RandomPairsDataset(G, k=NUM_RANDOM_PAIRS, seed=SEED+4567)
    test_dataset  = RandomPairsDataset(G, k=NUM_RANDOM_PAIRS, seed=SEED+5678)
elif QUERY_WORKLOAD_STRATEGY == "query_aware_landmark":
    print(f"Using query-aware landmark strategy with {NUM_LANDMARKS} landmarks...")
    query_file_name = os.path.join(query_dir, f"{DATA_NAME}.queries.npz")
    full_data = read_query_file(query_file_name, drop_duplicates=True)
    unique_nodes = np.unique(np.array(full_data)[:, :2].astype(np.int64)).tolist()
    # if DATA_NAME == "W_Jinan":
    #     k = 2
    # elif DATA_NAME == "W_Shenzhen":
    #     k = 7
    # elif DATA_NAME == "W_Chengdu":
    #     k = 0
    # elif DATA_NAME == "W_Beijing":
    #     k = 3
    # elif DATA_NAME == "W_NewYork":
    #     k = 15
    # elif DATA_NAME == "W_Chicago":
    #     k = 600
    k = 5  # Default value
    print(f"Using k={k} neighbors for augmenting unique nodes.")
    print(f"Initial no. of unique nodes from queries: {len(unique_nodes)} ({len(unique_nodes)/G.vcount()*100:.2f}%)")
    unique_nodes = augment_nodes(G, unique_nodes, k=k)
    print(f"No. of unique nodes after adding neighbors: {len(unique_nodes)} ({len(unique_nodes)/G.vcount()*100:.2f}%)")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234, subset_nodes=unique_nodes)
    print(f"Total {len(train_dataset)} queries have {len(unique_nodes)} unique nodes.")
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [0.8, 0.1, 0.1])  # 80-10-10 split
elif QUERY_WORKLOAD_STRATEGY == "query_perturbation":
    print(f"Using query perturbation strategy...")
    query_file_name = os.path.join(query_dir, f"{DATA_NAME}.queries.npz")
    data = read_query_file(query_file_name, drop_duplicates=True)

    # Augment queries by making slight perturbations to the source and target nodes
    augmented_queries = augment_queries(G, data, perturb_k=PERTURB_K, k_hop=K_HOP)

    # Limit number of queries, using random sampling
    if MAX_QUERIES is not None and len(augmented_queries) > MAX_QUERIES:
        print("Total augmented queries before limiting: ", len(augmented_queries))
        np.random.seed(SEED+5678)
        sampled_indices = np.random.choice(len(augmented_queries), size=MAX_QUERIES, replace=False)
        augmented_queries = [augmented_queries[i] for i in sampled_indices]
        print("Total augmented queries after limiting: ", len(augmented_queries))
        print("No. of unique queries in augmented queries:", len(set([(src, dst) for src, dst, _ in augmented_queries])))

    # Compute true distances for augmented queries
    index_path = os.path.join(INDEX_DIR, f"{DATA_NAME}.hl")
    inference_bin = os.path.join(os.path.dirname(INDEX_DIR), "build", "inference")
    pairs = [(int(src), int(dst)) for src, dst, _ in augmented_queries]
    print(f"Computing distances for {len(pairs):,} augmented queries...")
    augmented_queries_with_distances = _hl_distances(index_path, pairs, inference_bin)

    # Create Workload Dataset and split into train/val/test
    train_dataset = WorkloadDataset(augmented_queries_with_distances)
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [0.8, 0.1, 0.1])  # 80-10-10 split
elif QUERY_WORKLOAD_STRATEGY == "landmark_hl":
    print(f"Using landmark pairs + HL index strategy ({NUM_LANDMARKS} landmarks, 80/10/10 split)...")
    num_nodes = G.vcount()

    # Select landmarks randomly
    rng = np.random.default_rng(SEED + 1234)
    landmarks = rng.choice(num_nodes, size=NUM_LANDMARKS, replace=False)
    print(f"Selected {NUM_LANDMARKS} landmarks: {landmarks.tolist()}")

    # Generate all (landmark, node) pairs, excluding self-pairs
    srcs = np.repeat(landmarks, num_nodes)
    dsts = np.tile(np.arange(num_nodes), NUM_LANDMARKS)
    mask = srcs != dsts
    srcs, dsts = srcs[mask], dsts[mask]
    pairs = list(zip(srcs.tolist(), dsts.tolist()))
    print(f"Total pairs to compute: {len(pairs):,}")

    # Compute distances for landmark pairs using HL index
    index_path = os.path.join(INDEX_DIR, f"{DATA_NAME}.hl")
    inference_bin = os.path.join(os.path.dirname(INDEX_DIR), "build", "inference")
    print(f"Computing {len(pairs):,} landmark distances via HL index...")
    queries_with_distances = _hl_distances(index_path, pairs, inference_bin)

    # Create Workload Dataset and split into train/val/test
    train_dataset = WorkloadDataset(queries_with_distances)
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [0.8, 0.1, 0.1])
elif QUERY_WORKLOAD_STRATEGY == "random_hl":
    print(f"Using random pairs + HL index strategy (k={NUM_RANDOM_PAIRS:,}, 80/10/10 split)...")
    num_nodes = G.vcount()

    # Sample k unique, non-self (src != dst) pairs
    rng = np.random.default_rng(SEED + 6789)
    srcs = rng.integers(0, num_nodes, size=NUM_RANDOM_PAIRS)
    dsts = rng.integers(0, num_nodes, size=NUM_RANDOM_PAIRS)
    while True:
        _, first_occ = np.unique(np.stack([srcs, dsts], axis=1), axis=0, return_index=True)
        is_first = np.zeros(NUM_RANDOM_PAIRS, dtype=bool)
        is_first[first_occ] = True
        mask = (srcs == dsts) | ~is_first
        if not mask.any():
            break
        srcs[mask] = rng.integers(0, num_nodes, size=mask.sum())
        dsts[mask] = rng.integers(0, num_nodes, size=mask.sum())

    # Compute distances for random pairs using HL index
    index_path = os.path.join(INDEX_DIR, f"{DATA_NAME}.hl")
    inference_bin = os.path.join(os.path.dirname(INDEX_DIR), "build", "inference")
    pairs = list(zip(srcs.tolist(), dsts.tolist()))
    print(f"Computing {NUM_RANDOM_PAIRS:,} distances...")
    queries_with_distances = _hl_distances(index_path, pairs, inference_bin)

    # Create Workload Dataset and split into train/val/test
    train_dataset = WorkloadDataset(queries_with_distances)
    train_dataset, val_dataset, test_dataset = random_split(train_dataset, [0.8, 0.1, 0.1])
else:
    raise ValueError(f"Invalid QUERY_WORKLOAD_STRATEGY: {QUERY_WORKLOAD_STRATEGY}")

# TODO: Shall we limit the dataset size to 1M pairs?
# e.g., train_dataset = Subset(train_dataset, 1_000_000)

# Save dataset
print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))
print("Sample from train dataset:", train_dataset[0])
print("Sample from val dataset:", val_dataset[0])
print("Sample from test dataset:", test_dataset[0])
save_dataset(train_dataset, val_dataset, test_dataset, save_dir, DATA_NAME)

train_dataset1, val_dataset1, test_dataset1 = load_dataset(save_dir)
print("Loaded train dataset size:", len(train_dataset1))
print("Loaded val dataset size:", len(val_dataset1))
print("Loaded test dataset size:", len(test_dataset1))
print("Sample from loaded train dataset:", train_dataset1[0])
print("Sample from loaded val dataset:", val_dataset1[0])
print("Sample from loaded test dataset:", test_dataset1[0])
