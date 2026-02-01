# Usage:
## Example, Real workload perturbation k=4
#   python generate_query_workload.py --data_name W_Jinan    --query_dir real_workload --query_strategy query_perturbation --perturb_k 5   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_Shenzhen --query_dir real_workload --query_strategy query_perturbation --perturb_k 7   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_Chengdu  --query_dir real_workload --query_strategy query_perturbation --perturb_k 36  --k_hop 2 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_Beijing  --query_dir real_workload --query_strategy query_perturbation --perturb_k 64  --k_hop 3 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_Shanghai --query_dir real_workload --query_strategy query_perturbation --perturb_k 10  --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_NewYork  --query_dir real_workload --query_strategy query_perturbation --perturb_k 5   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
#   python generate_query_workload.py --data_name W_Chicago  --query_dir real_workload --query_strategy query_perturbation --perturb_k 128 --k_hop 3 --max_queries 500000 --save_dir real_workload_perturb_500k

## Example, All pairs
#   python generate_query_workload.py --data_name Surat_subgraph  --query_strategy all --save_dir all_pairs
#   python generate_query_workload.py --data_name Surat           --query_strategy all --save_dir all_pairs
#   python generate_query_workload.py --data_name Dongguan        --query_strategy all --save_dir all_pairs


# Imports
import os
import sys
import argparse

import numpy as np

from torch.utils.data import Dataset, random_split

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
    AllPairsDataset,
    LandmarkPairsDataset,
    RandomPairsDataset,
    WorkloadDataset,
    read_query_file,
)

if "../non_ml_index_evaluation/hcl" not in sys.path:
    sys.path.append(os.path.abspath("../non_ml_index_evaluation/hcl"))
from hcl_index import ContractionIndex, ContractionLabel, FlatCutIndex
from evaluate import parse_hcl_file

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate query workloads for shortest-distance datasets.")
parser.add_argument('--data_name', type=str, default='Surat_subgraph', help='Name of the dataset to use')
parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing datasets')
parser.add_argument('--query_strategy', type=str, default='query_perturbation', help='Strategy for generating query workload')
parser.add_argument('--query_dir', type=str, default='real_workload', help='Directory for reading queries within data directory')
parser.add_argument('--save_dir', type=str, default='default', help='Directory for saving queries within data directory')
parser.add_argument('--num_landmarks', type=int, default=20, help='Number of landmarks (if using landmark strategy)')
parser.add_argument('--num_random_pairs', type=int, default=1_000_000, help='Number of random pairs (if using random strategy)')
parser.add_argument('--perturb_k', type=int, default=4, help='Number of neighbor perturbations to use when augmenting queries')
parser.add_argument('--k_hop', type=int, default=1, help='Number of hops to consider for k-hop neighbors')
parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to generate (if applicable)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

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
    print("Creating train and test datasets using all pairs strategy...")
    train_dataset = AllPairsDataset(G)
    test_dataset = train_dataset  # Since we are using all pairs, train and test datasets are the same
elif QUERY_WORKLOAD_STRATEGY == "landmark":
    print("Creating train and test datasets using landmark strategy...")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234)  ## Using different seeds compared to model seeds
    test_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+2345)
elif QUERY_WORKLOAD_STRATEGY == "landmark_split":
    print("Creating train and test datasets using landmark split (80%-20%) strategy...")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234)  ## Using different seeds compared to model seeds
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])  # 80-20 split
elif QUERY_WORKLOAD_STRATEGY == "random":
    print("Creating train and test datasets using random pairs strategy...")
    train_dataset = RandomPairsDataset(G, k=NUM_RANDOM_PAIRS, seed=SEED+3456)
    test_dataset = RandomPairsDataset(G, k=NUM_RANDOM_PAIRS, seed=SEED+4567)
elif QUERY_WORKLOAD_STRATEGY == "query_aware_landmark":
    print(f"Using query-aware landmark strategy with {NUM_LANDMARKS} landmarks...")
    query_file_name = os.path.join(query_dir, f"{DATA_NAME}.queries")
    full_data = read_query_file(query_file_name, delimiter=None, comment='#', drop_duplicates=True)
    unique_nodes = np.unique(np.array(full_data)[:, :2]).tolist()
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
    print(f"Initial no. of unique nodes from queries: {len(unique_nodes)} ({len(unique_nodes)/G.number_of_nodes()*100:.2f}%)")
    unique_nodes = augment_nodes(G, unique_nodes, k=k)
    print(f"No. of unique nodes after adding neighbors: {len(unique_nodes)} ({len(unique_nodes)/G.number_of_nodes()*100:.2f}%)")
    train_dataset = LandmarkPairsDataset(G, l=NUM_LANDMARKS, seed=SEED+1234, subset_nodes=unique_nodes)
    print(f"Total {len(train_dataset)} queries have {len(unique_nodes)} unique nodes.")
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])  # 80-20 split
elif QUERY_WORKLOAD_STRATEGY == "query_perturbation":
    print(f"Using query perturbation strategy...")
    query_file_name = os.path.join(query_dir, f"{DATA_NAME}.queries")
    data = read_query_file(query_file_name, delimiter=None, comment='#', drop_duplicates=True)

    # Augment queries by making slight perturbations to the source and target nodes
    augmented_queries = augment_queries(G, data, perturb_k=PERTURB_K, k_hop=K_HOP, drop_duplicates=False)

    # Limit number of queries, using random sampling
    if MAX_QUERIES is not None and len(augmented_queries) > MAX_QUERIES:
        print("Total augmented queries before limiting: ", len(augmented_queries))
        np.random.seed(SEED+5678)
        sampled_indices = np.random.choice(len(augmented_queries), size=MAX_QUERIES, replace=False)
        augmented_queries = [augmented_queries[i] for i in sampled_indices]
        print("Total augmented queries after limiting: ", len(augmented_queries))
        print("No. of unique queries in augmented queries:", len(set([(src, dst) for src, dst, _ in augmented_queries])))

    # Load index
    index_path = f"../non_ml_index_evaluation/hcl/saved_models/{DATA_NAME}.hl"
    print(f"Processing labeling file: {index_path}")
    label_list = parse_hcl_file(index_path)
    hci = ContractionIndex(label_list)
    print("Index loaded.")

    # Compute true distances for augmented queries
    augmented_queries_with_distances = []
    for src, dst, _ in augmented_queries:
        src = int(src)
        dst = int(dst)
        true_distance = hci.get_distance(src+1, dst+1)  # Right-shift node id by 1 for HCL index
        augmented_queries_with_distances.append((src, dst, true_distance))

    # Create Workload Dataset
    train_dataset = WorkloadDataset(augmented_queries_with_distances)
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])  # 80-20 split
else:
    raise ValueError(f"Invalid QUERY_WORKLOAD_STRATEGY: {QUERY_WORKLOAD_STRATEGY}")

# TODO: Shall we limit the dataset size to 1M pairs?
# e.g., train_dataset = Subset(train_dataset, 1_000_000)

# Save dataset
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))
print("Sample from train dataset:", train_dataset[0])
print("Sample from test dataset:", test_dataset[0])
save_dataset(train_dataset, test_dataset, save_dir, DATA_NAME)

train_dataset1, test_dataset1 = load_dataset(save_dir)
print("Loaded train dataset size:", len(train_dataset1))
print("Loaded test dataset size:", len(test_dataset1))
print("Sample from loaded train dataset:", train_dataset1[0])
print("Sample from loaded test dataset:", test_dataset1[0])
