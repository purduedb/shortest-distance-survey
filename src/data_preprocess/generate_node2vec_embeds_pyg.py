## GPU-accelerated node2vec using PyG's Node2Vec (random_walk op provided by pyg-lib).

## Automated script:
#   bash generate_node2vec_embeds_pyg.sh

## Standalone Usage:
#   python generate_node2vec_embeds_pyg.py --data_name Surat

#   python generate_node2vec_embeds_pyg.py --data_name W_Jinan
#   python generate_node2vec_embeds_pyg.py --data_name W_Shenzhen
#   python generate_node2vec_embeds_pyg.py --data_name W_Chengdu
#   python generate_node2vec_embeds_pyg.py --data_name W_Beijing
#   python generate_node2vec_embeds_pyg.py --data_name W_Shanghai
#   python generate_node2vec_embeds_pyg.py --data_name W_NewYork
#   python generate_node2vec_embeds_pyg.py --data_name W_Chicago

#   python generate_node2vec_embeds_pyg.py --data_name FLA
#   python generate_node2vec_embeds_pyg.py --data_name E
#   python generate_node2vec_embeds_pyg.py --data_name W
#   python generate_node2vec_embeds_pyg.py --data_name CTR
#   python generate_node2vec_embeds_pyg.py --data_name USA

## Approximate Runtimes (1 epoch, dim=64, A100 GPU):
#   sinteractive -N1 -n1 -c16 --gres=gpu:1 --partition=training --mem=128G --account=csit --qos training --time 12:00:00
#
#   Surat           ~6s
#
#   W_Jinan         ~11s
#   W_Shenzhen      ~11s
#   W_Chengdu       ~12s
#   W_Beijing       ~22s
#   W_Shanghai      ~21s
#   W_NewYork       ~1m17s
#   W_Chicago       ~1m26s
#
#   FLA             ~4m
#   E               ~13m
#   W               ~23m
#   CTR             ~52m
#   USA             ~1h29m

import os
import sys
import time
import argparse
from tqdm import tqdm

import numpy as np

sys.path.insert(0, "..")
from utils.data_utils import load_graph, get_num_cores, print_green, print_warning

# NOTE: Import torch after igraph (here via data_utils) to avoid `ImportError: ... CXXABI_1.3.15 not found`
import torch
from torch_geometric.nn import Node2Vec

def generate_node2vec(data_dir, dimensions=64, walk_length=80, num_walks=10,
                      window_size=10, epochs=1, p=1.0, q=1.0, negative_samples=5, batch_size=256,
                      lr=0.025, min_lr=0.0001, num_workers=None, seed=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)

    # load_graph handles delimiter detection, 0-indexing (subtracts 1), and deduplication
    G = load_graph(data_dir)
    num_nodes = G.vcount()
    print(f"Graph loaded: {num_nodes} nodes, {G.ecount()} edges")

    # Build COO edge_index — igraph stores undirected edges once; add both directions for PyG
    edges = np.array(G.get_edgelist(), dtype=np.int64)  # already 0-indexed
    src, dst = edges[:, 0], edges[:, 1]
    edge_index = torch.tensor(
        np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
        dtype=torch.long,
    )

    # Node2Vec — sparse=True + SparseAdam avoids materializing a dense gradient for
    # the full embedding table, which matters at 1M+ nodes.
    model = Node2Vec(
        edge_index,
        embedding_dim=dimensions,
        walk_length=walk_length,
        context_size=window_size,
        walks_per_node=num_walks,
        num_negative_samples=negative_samples,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_nodes,
    ).to(device)

    num_workers = num_workers if num_workers is not None else get_num_cores()
    if sys.platform == "darwin":
        num_workers = 0  # often faster on macOS
        print_warning("macOS detected: Setting num_workers=0 for DataLoader to avoid multiprocessing issues.")
    use_cuda = device.type == 'cuda'
    print(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}, pin_memory={use_cuda}")
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=use_cuda, persistent_workers=(num_workers > 0))
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    print(f"LR: {lr} → {min_lr} (linear decay, matching gensim alpha/min_alpha)")

    t0 = time.time()
    total_batches = len(loader) * epochs
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Linear LR decay over all batches — mirrors gensim's alpha annealing
            current_lr = lr - (lr - min_lr) * (global_step / total_batches)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device, non_blocking=use_cuda), neg_rw.to(device, non_blocking=use_cuda))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
        elapsed_so_far = (time.time() - t0) / 60
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {total_loss / len(loader):.4f}, lr: {current_lr:.6f}, elapsed: {elapsed_so_far:.2f} min")

    elapsed = (time.time() - t0) / 60

    # Extract embeddings — model() returns (num_nodes, dim) ordered by 0-index,
    # which corresponds to all_nodes[0..N-1] (sorted original IDs).
    model.eval()
    with torch.no_grad():
        emb = model().cpu().numpy().astype(np.float32)

    # Save in same format as pecanpy version:
    # (n_nodes, 1 + dim) array — col 0 = original node ID as float32, rest = embedding.
    # load_graph shifts IDs to 0-indexed, so original file IDs = 0-index + 1.
    orig_ids = np.arange(1, num_nodes + 1)
    data = np.hstack([orig_ids.reshape(-1, 1), emb])

    out_path = os.path.join(data_dir, f"node2vec_dim{dimensions}_epochs{epochs}_unweighted_pyg.embeddings.npz")
    print_green(f"Saving embeddings: {out_path}")
    print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")
    np.savez_compressed(out_path, data=data)
    print(f"Saved embeddings with shape: {data.shape}")
    print(f"# precomputation time={elapsed:.2f} minutes, num_nodes={num_nodes}, vector_size={dimensions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate node2vec embeddings using PyG (GPU-accelerated)')
    parser.add_argument('--data_dir',    type=str,   default='../../data', help='Directory containing datasets')
    parser.add_argument('--data_name',   type=str,   default='W_Jinan',    help='Name of the dataset to use')
    parser.add_argument('--dimensions',  type=int,   default=64,           help='Embedding dimensions')
    parser.add_argument('--walk_length', type=int,   default=80,           help='Length of each random walk')
    parser.add_argument('--num_walks',   type=int,   default=10,           help='Number of walks per node')
    parser.add_argument('--window_size', type=int,   default=10,           help='Context window size')
    parser.add_argument('--epochs',      type=int,   default=1,            help='Training epochs')
    parser.add_argument('--p',           type=float, default=1.0,          help='Return parameter (1.0 = unbiased)')
    parser.add_argument('--q',           type=float, default=1.0,          help='In-out parameter (1.0 = unbiased)')
    parser.add_argument('--negative_samples', type=int, default=5,         help='Negative samples per positive (gensim default is 5)')
    parser.add_argument('--batch_size',  type=int,   default=256,          help='Nodes per batch')
    parser.add_argument('--lr',          type=float, default=0.025,        help='Initial learning rate (gensim alpha default: 0.025)')
    parser.add_argument('--min_lr',      type=float, default=0.0001,       help='Final learning rate after linear decay (gensim min_alpha default: 0.0001)')
    parser.add_argument('--num_workers', type=int,   default=None,         help='DataLoader workers for walk generation (default: auto via get_num_cores())')
    parser.add_argument('--seed',        type=int,   default=42,           help='Random seed')

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
        p=args.p,
        q=args.q,
        negative_samples=args.negative_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )
