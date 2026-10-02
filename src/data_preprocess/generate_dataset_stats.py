# Computes stats (#nodes, #edges, max/avg degree, max diameter, etc.) for a single dataset.
#
# Usage:
#   conda run -n myenv python generate_dataset_stats.py --data_name FLA

import os
import sys
import random
import argparse
import json

sys.path.insert(0, "..")
from utils.data_utils import load_graph, seed_everything

DATA_DIR = "./../../data"


def double_sweep_diameter(G, num_sweeps):
    """Approximate the diameter of a graph using the double-sweep method."""
    n = G.vcount()
    best = 0.0
    for _ in range(num_sweeps):
        # Randomly select a starting node
        start = random.randrange(n)

        # Compute distances from the starting node to all other nodes
        dist1 = G.distances(source=[start], weights="weight")[0]

        # Select the farthest node from the starting node
        a = max(range(n), key=lambda i: dist1[i])

        # Compute distances from the farthest node to all other nodes
        dist2 = G.distances(source=[a], weights="weight")[0]

        # Select the farthest node from node 'a'
        b = max(range(n), key=lambda i: dist2[i])

        # Update the best diameter found so far
        best = max(best, dist2[b])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_sweeps", type=int, default=10,
                        help="Double-sweep trials for approximate diameter")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed_everything(args.seed)

    # Load graph
    dir_name = os.path.join(DATA_DIR, args.data_name)
    G = load_graph(dir_name)

    # Compute basic graph statistics
    degrees = G.degree()
    min_node_degree = min(degrees)
    max_node_degree = max(degrees)
    avg_node_degree = sum(degrees) / len(degrees)

    weights = G.es["weight"]
    min_edge_weight = min(weights)
    max_edge_weight = max(weights)
    avg_edge_weight = sum(weights) / len(weights)

    # Compute approximate diameter using double-sweep method
    diameter_m = double_sweep_diameter(G, args.num_sweeps)
    diameter_km = diameter_m / 1000.0

    result = {
        "data_name": args.data_name,
        "nodes": G.vcount(),
        "edges": G.ecount(),
        "min_node_degree": min_node_degree,
        "max_node_degree": max_node_degree,
        "avg_node_degree": round(avg_node_degree, 2),
        "min_edge_weight": round(min_edge_weight, 2),
        "max_edge_weight": round(max_edge_weight, 2),
        "avg_edge_weight": round(avg_edge_weight, 2),
        "diameter_km": round(diameter_km, 2),
    }
    print(json.dumps(result))

    out_path = os.path.join(dir_name, "stats.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
