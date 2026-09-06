## Automated script:
#   ...

## Standalone Usage:
#   python generate_node2vec_embeds_pecanpy.py --data_name Surat

#   python generate_node2vec_embeds_pecanpy.py --data_name W_Jinan
#   python generate_node2vec_embeds_pecanpy.py --data_name W_Shenzhen
#   python generate_node2vec_embeds_pecanpy.py --data_name W_Chengdu
#   python generate_node2vec_embeds_pecanpy.py --data_name W_Beijing
#   python generate_node2vec_embeds_pecanpy.py --data_name W_Shanghai
#   python generate_node2vec_embeds_pecanpy.py --data_name W_NewYork
#   python generate_node2vec_embeds_pecanpy.py --data_name W_Chicago

#   python generate_node2vec_embeds_pecanpy.py --data_name FLA
#   python generate_node2vec_embeds_pecanpy.py --data_name E
#   python generate_node2vec_embeds_pecanpy.py --data_name W
#   python generate_node2vec_embeds_pecanpy.py --data_name CTR
#   python generate_node2vec_embeds_pecanpy.py --data_name USA

## Approximate Runtimes (1 epoch, dim=64, CPU 16 workers via sinteractive):
#   sinteractive -N1 -n1 -c16 --gres=gpu:1 --partition=training --mem=128G --account=csit --qos training --time 12:00:00
#
#   Surat           ~5s
#
#   W_Jinan         ~11s
#   W_Shenzhen      ~13s
#   W_Chengdu       ~17s
#   W_Beijing       ~1m
#   W_Shanghai      ~1m
#   W_NewYork       ~7m
#   W_Chicago       ~8m
#
#   FLA             ~29m
#   E               ~1h28m
#   W               ~2h43m
#   CTR             OOM (walks generated, Word2Vec training killed)
#   USA             OOM (walks generated, Word2Vec training killed)

import os
import sys
import glob
import argparse

import numpy as np

import logging
class _W2VFilter(logging.Filter):
    def filter(self, record):
        return "lifecycle event" not in record.getMessage()

import numba
from gensim.models import Word2Vec
from pecanpy import pecanpy

sys.path.insert(0, "..")
from utils.data_utils import print_green, print_warning


def generate_node2vec(data_dir, dimensions=64, walk_length=80, num_walks=10,
                      window_size=10, epochs=1, workers=0, seed=None):
    # Resolve workers: 0 means use all available threads
    if workers == 0:
        workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
    numba.set_num_threads(workers)
    print(f"Using {workers} workers")

    # Find .edges file
    edge_files = glob.glob(os.path.join(data_dir, "*.edges"))
    assert len(edge_files) > 0, f"No .edges file found in {data_dir}"
    edge_path = edge_files[0]
    print_green(f"Reading graph: {edge_path}")

    # Load graph using pecanpy (FirstOrderUnweighted: optimal for p=q=1, unweighted graphs)
    g = pecanpy.FirstOrderUnweighted(p=1, q=1, workers=workers, verbose=True,
                                     extend=False, gamma=0, random_state=seed)
    g.read_edg(edge_path, weighted=False, directed=False, delimiter=',')
    print(f"Graph loaded: {g.num_nodes} nodes")

    # Generate random walks
    print("Preprocessing transition probabilities...")
    g.preprocess_transition_probs()
    print("Generating random walks...")
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)

    # Train Word2Vec (skip-gram) — enable gensim logging for iteration-level progress
    print("Training Word2Vec...")
    _gensim_log = logging.getLogger("gensim")
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("  %(message)s"))
    _handler.addFilter(_W2VFilter())
    _gensim_log.addHandler(_handler)
    _gensim_log.setLevel(logging.INFO)
    _gensim_log.propagate = False

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    _gensim_log.removeHandler(_handler)
    _gensim_log.propagate = True

    # Save in the same format as landmark embeddings:
    # (n_nodes, 1 + dim) array where col 0 = 1-indexed node ID, rest = embedding vectors.
    # This is directly compatible with read_embedding_file in data_utils.py.
    out_path = os.path.join(data_dir, f"node2vec_dim{dimensions}_epochs{epochs}_unweighted_pecanpy.embeddings.npz")
    print_green(f"Saving embeddings: {out_path}")
    print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")

    ids = np.array([int(k) for k in model.wv.index_to_key])
    vectors = model.wv.vectors.astype(np.float32)
    sort_order = np.argsort(ids)
    data = np.hstack([ids[sort_order].reshape(-1, 1), vectors[sort_order]])
    np.savez_compressed(out_path, data=data)
    print(f"Saved embeddings with shape: {data.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate node2vec embeddings using PecanPy')
    parser.add_argument('--data_dir',    type=str, default='../../data',    help='Directory containing datasets')
    parser.add_argument('--data_name',   type=str, default='W_Jinan',       help='Name of the dataset to use')
    parser.add_argument('--dimensions',  type=int, default=64,              help='Embedding dimensions')
    parser.add_argument('--walk_length', type=int, default=80,              help='Length of each random walk')
    parser.add_argument('--num_walks',   type=int, default=10,              help='Number of walks per node')
    parser.add_argument('--window_size', type=int, default=10,              help='Context window size for Word2Vec')
    parser.add_argument('--epochs',      type=int, default=1,               help='Word2Vec training epochs')
    parser.add_argument('--workers',     type=int, default=0,               help='Parallel workers (0 = all available threads)')
    parser.add_argument('--seed',        type=int, default=42,              help='Random seed for reproducibility')

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")

    data_dir = os.path.join(args.data_dir, args.data_name)

    generate_node2vec(
        data_dir=data_dir,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
