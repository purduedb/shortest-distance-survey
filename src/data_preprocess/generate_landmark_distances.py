# Usage:
#   python generate_landmark_distances.py --data_name W_Jinan --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Shenzhen --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Chengdu --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Beijing --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_NewYork --num_landmarks 61
#   python generate_landmark_distances.py --data_name W_Chicago --num_landmarks 61

import os
import argparse

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
parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing datasets')
parser.add_argument('--data_name', type=str, default='Surat_subgraph', help='Name of the dataset to use')
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
node_attr_path = os.path.join(data_dir, f"landmark_dim{args.num_landmarks}.embeddings")
print_green(f"Saving nodes: {node_attr_path}")
print_warning("Warning: The node ids are right-shifted by 1 (i.e., node ids start from `1 to n` instead of `0 to n-1`) in the saved files.")
comment = "#"
delimiter = " "
with open(node_attr_path, 'w') as f:
    f.write(f"{comment} Format: node_id features\n")
    for node, data in enumerate(dist_matrix):
        f.write(f"{node+1}{delimiter}{delimiter.join(map(str, data))}\n")  # Right-shift node id by 1
