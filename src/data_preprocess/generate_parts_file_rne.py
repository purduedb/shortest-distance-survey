## Automated script:
#   bash generate_parts_file_rne.sh

## Standalone Usage:
#   python generate_parts_file_rne.py --data_name Surat

#   python generate_parts_file_rne.py --data_name W_Jinan
#   python generate_parts_file_rne.py --data_name W_Shenzhen
#   python generate_parts_file_rne.py --data_name W_Chengdu
#   python generate_parts_file_rne.py --data_name W_Beijing
#   python generate_parts_file_rne.py --data_name W_Shanghai
#   python generate_parts_file_rne.py --data_name W_NewYork
#   python generate_parts_file_rne.py --data_name W_Chicago

#   python generate_parts_file_rne.py --data_name FLA
#   python generate_parts_file_rne.py --data_name E
#   python generate_parts_file_rne.py --data_name W
#   python generate_parts_file_rne.py --data_name CTR
#   python generate_parts_file_rne.py --data_name USA


import os
import sys
import argparse

import numpy as np
import pymetis

sys.path.insert(0, "..")
from utils.data_utils import load_graph
from utils.data_utils import print_green


def igraph_to_pymetis_format(G, weight_key=None):
    n = G.vcount()

    if weight_key is None:
        print("Using uniform weights (w=1) for partitioning")
    else:
        print(f"Using edge weights with key='{weight_key}' for partitioning")

    # Build undirected adjacency: each edge duplicated for both directions
    edgelist = np.array(G.get_edgelist())
    sources, targets = edgelist[:, 0], edgelist[:, 1]
    all_src = np.concatenate([sources, targets])
    all_dst = np.concatenate([targets, sources])

    # Sort by source node to produce CSR adjacency list
    sort_idx = np.argsort(all_src, kind='stable')
    adjncy = all_dst[sort_idx]

    # xadj: cumulative degree per node
    counts = np.bincount(all_src, minlength=n)
    xadj = np.zeros(n + 1)
    xadj[1:] = np.cumsum(counts)

    # Edge weights (default weight = 1 if attribute missing or weight_key is None)
    if weight_key and weight_key in G.edge_attributes():
        raw = np.array(G.es[weight_key])
        eweights = np.concatenate([raw, raw])[sort_idx]
    else:
        eweights = np.ones(len(adjncy))  # to get partitions of similar size, use uniform weights

    return xadj, adjncy, eweights


def generate_hierarchical_partitions(xadj, adjncy, eweights, nlevels, recursive=False, seed=42):
    n = xadj.shape[0] - 1
    parts = np.ones((n, len(nlevels)), dtype=np.int64)  # all nodes in one partition
    opts = pymetis.Options(seed=seed)
    adjacency = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
    print(f"Graph has {n} nodes")
    print(f"Using recursive bisection: {recursive}")
    print(f"nlevels: {nlevels}")
    print(f"METIS seed: {seed}")
    for _ in range(len(nlevels)):
        print(f"Partitioning level {_} with {nlevels[_]} partitions")
        assert nlevels[_] <= n, f"Number of partitions {nlevels[_]} cannot exceed number of nodes {n}"

        if nlevels[_] == n:
            # Last level, no partitioning
            print("Last level reached, assigning each node to its own partition")
            parts[:, _] = np.arange(0, n)
            continue

        # Use recursive bisection for better partitioning
        edgecuts, membership = pymetis.part_graph(nlevels[_], adjacency=adjacency, eweights=eweights, recursive=recursive, options=opts)

        parts[:, _] = np.array(membership).reshape(-1)

    return parts


def generate_parts(data_name, data_dir, seed=42):
    # Load graph
    G = load_graph(data_dir)

    # Convert to pymetis format
    # xadj, adjncy, eweights = igraph_to_pymetis_format(G, weight_key='weight')
    xadj, adjncy, eweights = igraph_to_pymetis_format(G, weight_key=None)
    print(f"Graph has {xadj.shape[0]-1} nodes and {len(adjncy)//2} edges")
    print(f"xadj.shape: {xadj.shape}, adjncy.shape: {adjncy.shape}, eweights.shape: {eweights.shape}")

    ## Deprecated: Hardcoded levels
    # N_LEVELS=[256, 1024, 4096, 16384, 65536, xadj.shape[0] - 1]
    # # Filter out levels that are too high for the graph size
    # nlevels = [n for n in N_LEVELS if n <= xadj.shape[0] - 1]

    # Create levels
    if xadj.shape[0] - 1 <= 1e6:
        # For smaller graphs, use finer levels (powers of 4) to get more hierarchical levels
        base = 4
    else:
        # For larger graphs, use coarser levels (powers of 8) to avoid too many levels and long partitioning times
        base = 8
    nlevels = [base**i for i in range(2, 20) if base**i <= xadj.shape[0] - 1]
    nlevels.append(xadj.shape[0] - 1)  # Ensure the last level is the number of nodes
    print(f"Using hierarchical levels: {nlevels}")

    # Generate hierarchical partitions
    # NOTE: PyMetis requires positive integers as edge weights and xadj as integers
    xadj = xadj.astype(np.int64)
    eweights = eweights.astype(np.int64)
    parts_ = generate_hierarchical_partitions(xadj, adjncy, eweights, nlevels, recursive=False, seed=seed)

    # (Optional) Print some statistics about the partitions
    for level_i in range(len(nlevels)):
        membership = parts_[:, level_i]
        print(f"Level {level_i}: min index={np.min(membership)}, max index={np.max(membership)}, unique partitions={len(np.unique(membership))}")
    print(f"Parts shape: {parts_.shape}")

    # (Optional) Check containment property: each node in level i+1 maps to only one node in level i
    for level in range(len(nlevels) - 1, 0, -1):
        mapping = {}
        for node_idx in range(parts_.shape[0]):
            higher = parts_[node_idx, level]
            lower = parts_[node_idx, level - 1]
            if higher not in mapping:
                mapping[higher] = set()
            elif lower not in mapping[higher]:
                print("Containment property violated:")
                print(f"Idx: {node_idx}, Level: {level}, Lower part: {lower}, Higher part: {higher}")
                print(f"Node id: {higher} maps to multiple parents: {mapping[higher]} and {lower}")
                print("Moving on to next level check...")
                break
            mapping[higher].add(lower)

    # Save the parts to a file
    file_name = os.path.join(data_dir, data_name + ".parts.npz")
    print_green(f"Saving parts: {file_name}")
    np.savez_compressed(file_name, data=parts_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hierarchical partitions for RNE datasets')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Directory containing datasets')
    parser.add_argument('--data_name', type=str, default='W_Jinan', help='Name of the dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for METIS partitioning')

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")

    data_name = args.data_name
    data_dir = os.path.join(args.data_dir, data_name)

    generate_parts(data_name, data_dir, seed=args.seed)
