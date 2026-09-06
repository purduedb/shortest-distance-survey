"""
Latency + throughput benchmark for CatBoost (CPU-only, no GPU path).

Usage:
    python catboost_inference.py --model_path <path>.pt \
        --queries <path>.queries.npz --batch_sizes 1000000
"""
import os
import time
import argparse
import resource

import numpy as np

import torch

from utils.torch_utils import save_metrics_json
from utils.data_utils import get_num_cores

DEFAULT_OUTPUT_DIR = "../results/default/saved_catboost_metrics"
DEFAULT_MAX_BATCHES = 100  # Limit the number of batches for latency/throughput evaluation


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_queries(path, n=None):
    """Accepts .queries.npz or plain-text .queries (comma-separated src,dst[,dist], 1-indexed)."""
    if path.endswith(".npz"):
        d = np.load(path)
        src, dst = d["src"].astype(np.int64), d["dst"].astype(np.int64)
        gt = d["dist"].astype(np.float32) if "dist" in d.files else None
    else:
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        src = raw[:, 0].astype(np.int64)
        dst = raw[:, 1].astype(np.int64)
        gt = raw[:, 2].astype(np.float32) if raw.shape[1] >= 3 else None

    if n is not None:
        src, dst = src[:n], dst[:n]
        if gt is not None:
            gt = gt[:n]
    src -= 1  # 1-indexed on disk -> 0-indexed for the model
    dst -= 1
    return src, dst, gt


def mae_mre(preds, gt):
    diff = np.abs(preds - gt)
    mae = float(diff.mean())
    mre = float((diff / np.maximum(gt, 1e-6)).mean())
    return mae, mre


def encode(src, dst, coord_embs, landmark_embs):
    """Matches models/catboostmodel.py's encode()."""
    x1_coord, x2_coord = coord_embs[src], coord_embs[dst]
    x1_land, x2_land = landmark_embs[src], landmark_embs[dst]

    num = np.sum(x1_land * x2_land, axis=-1)
    denom = np.clip(np.linalg.norm(x1_land, axis=-1) * np.linalg.norm(x2_land, axis=-1), 1e-8, None)
    cosine_sim = (num / denom)[:, None]

    euclidean_dist = np.abs(x1_coord - x2_coord).sum(axis=-1, keepdims=True)

    return np.concatenate(
        [x1_land, x2_land, x1_coord, x2_coord, cosine_sim, euclidean_dist], axis=-1
    ).astype(np.float32)


def mean_std(run_times, keep_last=5):
    """Mean/std over the last `keep_last` eval runs (drops early runs -- warmup noise)."""
    t = np.array(run_times[-keep_last:])
    mean = float(t.mean())
    std = float(t.std(ddof=1)) if len(t) > 1 else 0.0
    return mean, std


def print_dataloader_table(dataloader_results):
    col_w = 10
    group_w = 2 * col_w + 2  # two sub-columns + the "  " gap between them
    print("\n=== Benchmark: Dataloader (vary batch_size) ===")
    print(f"{'':<14}  {'Latency':^{group_w}}  {'Throughput':^{group_w}}")
    print(f"{'':<14}  {'-' * group_w}  {'-' * group_w}")
    print(f"{'batch_size':<14}  {'avg us/q':>{col_w}}  {'std us/q':>{col_w}}  {'avg M q/s':>{col_w}}  {'std M q/s':>{col_w}}")
    print(f"{'-----------':<14}  {'--------':>{col_w}}  {'--------':>{col_w}}  {'---------':>{col_w}}  {'---------':>{col_w}}")
    for r in dataloader_results:
        print(f"{r['batch_size']:<14}  {r['latency_avg_us_per_query']:>{col_w}.6f}  {r['latency_std_us_per_query']:>{col_w}.6f}  "
              f"{r['throughput_avg_M_queries_per_sec']:>{col_w}.2f}  {r['throughput_std_M_queries_per_sec']:>{col_w}.2f}")
    print()


# ---------------------------------------------------------------------------
# CatBoost benchmark
# ---------------------------------------------------------------------------
def latency(model, features, batch_size, eval_runs, n_threads, seed=42):
    """Times forward (predict()) over the full batch sweep (CPU-only -- no H2D/D2H)."""
    n = min(features.shape[0], DEFAULT_MAX_BATCHES * batch_size)
    rng = np.random.default_rng(seed)

    latencies = []
    throughputs = []
    for _ in range(eval_runs):
        # Reshuffle order each eval run -- avoids caching effects
        perm = rng.permutation(features.shape[0])[:n]
        features_r = features[perm]

        # Start timing
        start_time = time.perf_counter()

        for start in range(0, n, batch_size):
            # Prepare batch
            end = min(start + batch_size, n)
            x = features_r[start:end]

            # Model inference
            _ = model.predict(x, thread_count=n_threads)

        # End timing
        elapsed_time = (time.perf_counter() - start_time)

        # Collect results
        latencies.append(elapsed_time / n)
        throughputs.append(n / elapsed_time)

    return latencies, throughputs


def compute_accuracy(model, features, dist, batch_size, n_threads):
    n = features.shape[0]
    preds = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds[start:end] = model.predict(features[start:end], thread_count=n_threads)
    return mae_mre(preds, dist)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="CatBoost_<Dataset>_<Query>.pt checkpoint")
    parser.add_argument("--queries", required=True, help="src,dst[,dist] queries file (comma-separated, 1-indexed)")
    parser.add_argument("--n_queries", type=int, default=None, help="Limit queries loaded (default: all)")
    parser.add_argument("--batch_sizes", default="1000,10000,100000,1000000")
    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--n_threads", type=int, default=None, help="The number of threads to use (default: all available)")
    parser.add_argument("--device", default="cpu", help="Accepted for CLI parity; CatBoost predict() is CPU-only")
    parser.add_argument("--precision", default="fp32", help="Accepted for CLI parity; no effect")
    parser.add_argument("--optimize", action="store_true", help="Accepted for CLI parity; no effect")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="metrics output dir (default: %(default)s)")
    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg:<12}: {value}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # --- Load checkpoint: node/landmark embeddings + CatBoost model, all in one file ---
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    coord_embs = ckpt["model_state_dict"]["coordinate_embs.weight"].numpy()
    landmark_embs = ckpt["model_state_dict"]["landmark_embs.weight"].numpy()
    model = ckpt["catboost_model"]
    n_threads = args.n_threads if args.n_threads is not None else get_num_cores()
    print(f"CatBoost predict() thread_count: {n_threads}")

    print("Loading queries...")
    src, dst, dist = load_queries(args.queries, args.n_queries)
    n_queries = len(src)
    has_accuracy = dist is not None
    print(f"  - No. of queries: {n_queries:,}")
    print(f"  - Has ground truth: {has_accuracy}")

    print(f"Building features for {n_queries} queries...")
    features = encode(src, dst, coord_embs, landmark_embs)

    # Warmup
    _ = model.predict(features[:min(1000, n_queries)], thread_count=n_threads)

    print("Running latency + throughput benchmark...")
    dataloader_results = []
    for bs in batch_sizes:
        print(f"  - bs={bs:<10}")
        # Benchmark latency + throughput
        raw_latencies, raw_throughputs = latency(model, features, bs, args.eval_runs, n_threads)
        lat_avg, lat_std = mean_std(raw_latencies, keep_last=5)
        thr_avg, thr_std = mean_std(raw_throughputs, keep_last=5)
        print(f"      - Latencies (us/q): {(np.array(raw_latencies) * 1e6).round(4)}")
        print(f"      - Throughputs (Mq/s): {(np.array(raw_throughputs) / 1e6).round(4)}")

        # Collect results
        dataloader_results.append({
            "batch_size": bs,
            "latency_raw_us_per_query": (np.array(raw_latencies) * 1e6).tolist(),
            "latency_avg_us_per_query": lat_avg * 1e6,
            "latency_std_us_per_query": lat_std * 1e6,
            "throughput_raw_M_queries_per_sec": (np.array(raw_throughputs) / 1e6).tolist(),
            "throughput_avg_M_queries_per_sec": thr_avg / 1e6,
            "throughput_std_M_queries_per_sec": thr_std / 1e6,
        })
    print_dataloader_table(dataloader_results)

    # Compute Accuracy (MAE/MRE)
    mae = mre = 0.0
    if has_accuracy:
        print("Computing accuracy...")
        mae, mre = compute_accuracy(model, features, dist, batch_sizes[-1], n_threads)
        print(f"  - MAE: {mae:.2f}")
        print(f"  - MRE: {mre * 100.0:.2f}%")
    else:
        print("(No ground-truth column — skipping MAE/MRE)")

    # --- Peak CPU memory (matches getrusage(RUSAGE_SELF).ru_maxrss, KB on Linux) ---
    cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    print(f"Peak CPU memory: {cpu_gb:.2f} GB")

    metrics = {
        "backend": "catboost",
        "dataloader": dataloader_results,
        "device": "cpu",
        "eval_runs": args.eval_runs,
        "gpu_name": "",
        "has_accuracy": has_accuracy,
        "mae": mae,
        "model_path": args.model_path,
        "model_size_mb": os.path.getsize(args.model_path) / (1024 * 1024),
        "mre_pct": mre * 100.0,
        "n_queries": n_queries,
        "optimize": False,
        "peak_cpu_memory_gb": cpu_gb,
        "peak_gpu_memory_gb": 0.0,
        "precision": "fp32",
        "query_path": args.queries,
        "environment": "python",
    }

    model_name = os.path.basename(args.model_path).removesuffix(".pt")
    output_filename = f"metrics_{model_name}.json"
    save_metrics_json(metrics, file_name=output_filename, dir_name=args.output_dir)


if __name__ == "__main__":
    main()
