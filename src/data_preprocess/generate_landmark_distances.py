## Automated script:
#   bash generate_landmark_distances.sh

## Standalone Usage:
#   python generate_landmark_distances.py --data_name Surat --num_landmarks 61

#   python generate_landmark_distances.py --data_name W_Jinan    --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Shenzhen --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Chengdu  --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Beijing  --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_NewYork  --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Chicago  --num_landmarks 61

#   python generate_landmark_distances.py --data_name FLA  --num_landmarks 61
#   python generate_landmark_distances.py --data_name E    --num_landmarks 61
#   python generate_landmark_distances.py --data_name W    --num_landmarks 61
#   python generate_landmark_distances.py --data_name CTR  --num_landmarks 61
#   python generate_landmark_distances.py --data_name USA  --num_landmarks 61


import os
import sys
import argparse
import numpy as np

sys.path.insert(0, "..")
from utils.data_utils import (
    load_graph,
    print_summary_stats,
    print_green,
    print_warning,
    select_landmarks,
    compute_landmark_distances,
)


# Setup argument parser
parser = argparse.ArgumentParser(description="Generate landmark distance embeddings for shortest-distance datasets.")
parser.add_argument('--data_dir', type=str, default='../../data', help='Directory containing datasets')
parser.add_argument('--data_name', type=str, default='Surat', help='Name of the dataset to use')
parser.add_argument('--num_landmarks', type=int, default=61, help='Number of landmarks (if using landmark strategy)')
args = parser.parse_args()

data_dir = os.path.join(args.data_dir, args.data_name)

# Load graph
G = load_graph(data_dir)
print_summary_stats(G)

# Select landmarks and compute distances
landmarks = select_landmarks(G, args.num_landmarks, strategy="random")
dist_matrix = compute_landmark_distances(G, landmarks)
print(f"Computed landmark distance matrix with shape: {dist_matrix.shape}")

# Save landmark distance embeddings
node_attr_path = os.path.join(data_dir, f"landmark_dim{args.num_landmarks}.embeddings.npz")
print_green(f"Saving nodes: {node_attr_path}")
print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")
node_ids = np.arange(1, len(dist_matrix) + 1).reshape(-1, 1)
data = np.hstack([node_ids, dist_matrix])  # (n_nodes, 1 + n_landmarks): col 0 = node_id, rest = distances
np.savez_compressed(
    node_attr_path,
    comment=np.array("# Format: node_id features"),
    data=data,
)
