"""
Latency + throughput benchmark.

Usage:
    TensorRT (default):
        python inference.py --backend tensorrt --model_path <path>.onnx \
            --queries <path>.queries.npz --batch_sizes 1000000
    LibTorch:
        python inference.py --backend libtorch --model_path <path>.jit.pt \
            --queries <path>.queries.npz --batch_sizes 1000000
"""
import os
import sys
import argparse
import time
import resource
import threading

import numpy as np

## Only for TensorRT backend
# Without this, the TensorRT EP fails to load and ORT silently falls back to
# CUDA (no error, just a warning) -- prepend tensorrt_libs/cudnn to LD_LIBRARY_PATH.
try:
    import tensorrt_libs
    _trt_lib_dir = os.path.dirname(tensorrt_libs.__file__)
    _cudnn_lib_dir = os.path.join(os.path.dirname(_trt_lib_dir), "nvidia", "cudnn", "lib")
    for _d in (_trt_lib_dir, _cudnn_lib_dir):
        if _d not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = _d + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
except ImportError:
    pass  # tensorrt_libs not installed -- --backend tensorrt will fail its own check below

import onnxruntime as ort
import torch

from utils.data_utils import (
    get_num_cores,
    print_green,
    print_warning
)
from utils.torch_utils import save_metrics_json

DEFAULT_OUTPUT_DIR = "../results/default/saved_inference_metrics"
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


def mean_std(run_times, keep_last=5):
    """Mean/std over the last `keep_last` eval runs (drops early runs -- warmup noise)."""
    t = np.array(run_times[-keep_last:])
    mean = float(t.mean())
    std = float(t.std(ddof=1)) if len(t) > 1 else 0.0
    return mean, std


def start_gpu_sampler(device, interval_s=0.2):
    """Samples GPU util (%) and memory (GB) via NVML on a background thread."""
    def _sample():
        util = torch.cuda.utilization(device)
        free, total = torch.cuda.mem_get_info(device)
        return util, (total - free) / 1024**3

    samples = [_sample()]  # guarantee >=1 sample even if the loop finishes before the thread is scheduled
    stop_event = threading.Event()

    def _poll():
        while not stop_event.wait(interval_s):
            samples.append(_sample())

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        thread.join()
        return samples

    return stop


def print_dataloader_table(dataloader_results):
    col_w = 10

    # Each group: (header, [(sub_label, key, fmt), ...])
    groups = [
        ("Latency (us/q)", [("avg", "latency_avg_us_per_query", ".6f"), ("std", "latency_std_us_per_query", ".6f")]),
        ("Throughput (M q/s)", [("avg", "throughput_avg_M_queries_per_sec", ".2f"), ("std", "throughput_std_M_queries_per_sec", ".2f")]),
    ]
    groups.append(("GPU Util (%)", [("median", "gpu_util_median_pct", ".2f")]))
    if any("gpu_memory_median_gb" in r for r in dataloader_results):
        groups.append(("GPU mem (GB)", [("median", "gpu_memory_median_gb", ".3f")]))

    # Each group's field spans its sub-columns (col_w each, "  " gaps between) -- at least as wide as the header itself
    group_widths = [max(len(header), len(sub_cols) * col_w + (len(sub_cols) - 1) * 2) for header, sub_cols in groups]

    header_row = f"{'':<14}"
    dash_row = f"{'':<14}"
    sub_header_row = f"{'batch_size':<14}"
    sub_dash_row = f"{'-----------':<14}"
    for (header, sub_cols), w in zip(groups, group_widths):
        align = "^" if len(sub_cols) > 1 else ">"
        header_row += f"  {header:{align}{w}}"
        dash_row += f"  {'-' * len(header):{align}{w}}"
        for sub_label, _, _ in sub_cols:
            sub_header_row += f"  {sub_label:>{col_w}}"
            sub_dash_row += f"  {'-' * max(len(sub_label), 8):>{col_w}}"

    print("\n=== Benchmark: Dataloader (vary batch_size) ===")
    print(header_row)
    print(dash_row)
    print(sub_header_row)
    print(sub_dash_row)
    for r in dataloader_results:
        line = f"{r['batch_size']:<14}"
        for _, sub_cols in groups:
            for _, key, fmt in sub_cols:
                line += f"  {r.get(key, 0.0):>{col_w}{fmt}}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# LibTorch backend
# ---------------------------------------------------------------------------
def libtorch_load(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model = torch.jit.freeze(model)
    model = torch.jit.optimize_for_inference(model)
    return model


def libtorch_accuracy(model, src, dst, gt, device, batch_size):
    n = len(gt)
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    preds = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            i = src_t[start:end].to(device)
            j = dst_t[start:end].to(device)
            out = model(i, j).to("cpu").numpy().reshape(-1)
            preds[start:end] = out
    return mae_mre(preds, gt)


def libtorch_latency(model, src, dst, device, batch_size, eval_runs, seed=0):
    """Times H2D + forward + D2H over the full batch sweep (mirrors TensorRT's fused latency())."""
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    n = min(src_t.shape[0], DEFAULT_MAX_BATCHES * batch_size)
    rng = np.random.default_rng(seed)

    latencies = []
    throughputs = []
    with torch.no_grad():
        for _ in range(eval_runs):
            # Reshuffle order each eval run -- avoids caching effects
            perm = torch.from_numpy(rng.permutation(src_t.shape[0])[:n])
            src_r, dst_r = src_t[perm], dst_t[perm]

            # Start timing
            start_time = time.perf_counter()

            for start in range(0, n, batch_size):
                # Prepare batch
                end = min(start + batch_size, n)
                i_cpu = src_r[start:end]
                j_cpu = dst_r[start:end]

                # Move data to device
                i = i_cpu.to(device, non_blocking=True)
                j = j_cpu.to(device, non_blocking=True)

                # Model inference
                out = model(i, j)
                out = out.to("cpu", non_blocking=True)

            # Sync cuda computation for time profiling
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # End timing
            elapsed_time = (time.perf_counter() - start_time)

            # Collect results
            latencies.append(elapsed_time / n)
            throughputs.append(n / elapsed_time)

    return latencies, throughputs


# ---------------------------------------------------------------------------
# TensorRT (ONNX Runtime) backend
# ---------------------------------------------------------------------------
def load_session(model_path, sess_options, trt_opts, device="cuda"):
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    actual_providers = sess.get_providers()
    if device == "cpu":
        print(f"CPUExecutionProvider active (--device cpu).")
    elif actual_providers[0] != "TensorrtExecutionProvider":
        print(f"⚠️ WARNING: TensorRT EP is not active, falling back to {actual_providers[0]}. "
              f"Results will not reflect TensorRT performance.", file=sys.stderr)
    else:
        print(f"✅ TensorRT EP active. opts={trt_opts}")
    return sess, actual_providers


def compute_accuracy(sess, src, dst, gt, batch_size):
    input_names = [i.name for i in sess.get_inputs()]
    n = len(gt)
    preds = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        i, j = src[start:end], dst[start:end]
        raw = sess.run(None, {input_names[0]: i, input_names[1]: j})
        if len(raw) == 0:
            raise RuntimeError(f"Model returned no output for batch [{start}:{end}]")
        out = raw[0]
        preds[start:end] = out.reshape(-1)
    return mae_mre(preds, gt)


def latency(sess, src, dst, batch_size, eval_runs, seed=42, io_binding=False):
    """Times H2D + forward + D2H via sess.run() (default) or via IOBinding API."""
    n = min(src.shape[0], DEFAULT_MAX_BATCHES * batch_size)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    rng = np.random.default_rng(seed)

    latencies = []
    throughputs = []
    for _ in range(eval_runs):
        # Reshuffle order each eval run -- avoids caching effects
        perm = rng.permutation(src.shape[0])[:n]
        src_r, dst_r = src[perm], dst[perm]

        if io_binding:
            # NOTE: per-batch IOBinding object creation is likely slower
            # than sess.run() due to Python-side call overhead.
            io_bind = sess.io_binding()

            # Start timing
            start_time = time.perf_counter()

            for start in range(0, n, batch_size):
                # Prepare batch
                end = min(start + batch_size, n)
                i, j = src_r[start:end], dst_r[start:end]

                # H2D
                io_bind.bind_cpu_input(input_names[0], i)
                io_bind.bind_cpu_input(input_names[1], j)
                io_bind.bind_output(output_names[0])

                # Forward
                sess.run_with_iobinding(io_bind)
                io_bind.synchronize_outputs()

                # D2H
                _ = io_bind.copy_outputs_to_cpu()

            # End timing
            elapsed_time = (time.perf_counter() - start_time)
        else:
            # Start timing
            start_time = time.perf_counter()

            for start in range(0, n, batch_size):
                # Prepare batch
                end = min(start + batch_size, n)
                i, j = src_r[start:end], dst_r[start:end]

                # Model inference (H2D + forward + D2H handled inside sess.run)
                _ = sess.run(output_names, {input_names[0]: i, input_names[1]: j})

            # End timing
            elapsed_time = (time.perf_counter() - start_time)

        # Collect results
        latencies.append(elapsed_time / n)
        throughputs.append(n / elapsed_time)

    return latencies, throughputs


def gpu_compute_latency(sess, src, dst, batch_size, eval_runs, seed=42):
    """Times forward-only compute via IOBinding with CUDA-resident inputs/outputs."""
    n = min(src.shape[0], DEFAULT_MAX_BATCHES * batch_size)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    rng = np.random.default_rng(seed)

    latencies = []
    throughputs = []
    for _ in range(eval_runs):
        # Reshuffle order each eval run -- avoids caching effects
        perm = rng.permutation(src.shape[0])[:n]

        # H2D -- move data to GPU once
        src_r = torch.as_tensor(src[perm], device="cuda")
        dst_r = torch.as_tensor(dst[perm], device="cuda")

        io_bind = sess.io_binding()

        # Start timing
        start_time = time.perf_counter()

        for start in range(0, n, batch_size):
            # Prepare batch (view, no copy)
            end = min(start + batch_size, n)
            i_dev = ort.OrtValue.from_dlpack(src_r[start:end].__dlpack__())
            j_dev = ort.OrtValue.from_dlpack(dst_r[start:end].__dlpack__())
            io_bind.bind_ortvalue_input(input_names[0], i_dev)
            io_bind.bind_ortvalue_input(input_names[1], j_dev)
            io_bind.bind_output(output_names[0], "cuda", 0)

            # Model Inference (forward only)
            sess.run_with_iobinding(io_bind)
            io_bind.synchronize_outputs()

        # End timing
        elapsed_time = (time.perf_counter() - start_time)

        # Collect results
        latencies.append(elapsed_time / n)
        throughputs.append(n / elapsed_time)

    return latencies, throughputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, default="tensorrt", choices=["tensorrt", "libtorch"])
    parser.add_argument("--model_path", required=True,
                        help=".onnx for --backend tensorrt (export via export_onnx.py first); "
                          ".jit.pt directly for --backend libtorch")
    parser.add_argument("--queries", required=True, help=".queries.npz or plain-text .queries")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (e.g., 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--batch_sizes", default="1000000")
    parser.add_argument("--eval_runs", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=None, help="Limit queries loaded (default: all)")
    parser.add_argument("--n_threads", type=int, default=None, help="The number of threads to use (default: all available)")
    parser.add_argument("--fp16", action="store_true",
                        help="[tensorrt only] WARNING: broke coordinate-embedding models at FLA scale -- always check accuracy output")
    parser.add_argument("--io_binding", action="store_true",
                        help="[tensorrt only] time via ONNX Runtime's IOBinding API instead of sess.run() (default)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="metrics output dir (default: %(default)s)")
    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg:<12}: {value}")

    print("Loading queries...")
    src, dst, gt = load_queries(args.queries, args.n_queries)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    print(f"  - No. of queries: {len(src):,}")
    print(f"  - Has ground truth: {gt is not None}")
    has_accuracy = gt is not None
    n_threads = args.n_threads if args.n_threads is not None else get_num_cores()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    results = {
        "backend": args.backend,
        "model_path": args.model_path,
        "model_size_mb": os.path.getsize(args.model_path) / (1024 * 1024),
        "query_path": args.queries,
        "gpu_name": gpu_name,
        "precision": "fp16" if args.fp16 else "fp32",
        "optimize": True,   # libtorch: freeze + optimize_for_inference; tensorrt: ORT_ENABLE_ALL graph opts
        "n_queries": len(src),
        "eval_runs": args.eval_runs,
        "has_accuracy": has_accuracy,
        "io_binding": args.io_binding,
        "dataloader": [],
    }

    if args.fp16:
        print_warning("fp16 accuracy check: compare MAE/MRE against the fp32 baseline before trusting speedups")

    device = torch.device(args.device)
    results["device"] = args.device

    if args.backend == "tensorrt":
        # ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = n_threads

        # TensorRT EP options
        trt_opts = {
            "trt_fp16_enable": args.fp16,
        }

        results["settings"] = trt_opts

        print("Running latency + throughput benchmark...")
        for bs in batch_sizes:
            print(f"  - bs={bs:<10}")
            # Create session
            print_green(f"Loading TensorRT session: {args.model_path}")
            sess, actual_providers = load_session(args.model_path, sess_options, trt_opts, device=args.device)
            print(f"ONNX Runtime intra_op_num_threads: {sess.get_session_options().intra_op_num_threads}")
            results["actual_providers"] = actual_providers

            # Start GPU stats sampler (NVML) -- TensorRT allocates outside PyTorch's allocator
            stop_sampler = start_gpu_sampler(args.device) if device.type == "cuda" else None

            # Benchmark latency
            raw_latencies, raw_throughputs = latency(sess, src, dst, bs, args.eval_runs, io_binding=args.io_binding)
            lat_avg, lat_std = mean_std(raw_latencies, keep_last=5)
            thr_avg, thr_std = mean_std(raw_throughputs, keep_last=5)
            print(f"      - Latencies (us/q): {(np.array(raw_latencies) * 1e6).round(4)}")
            print(f"      - Throughputs (Mq/s): {(np.array(raw_throughputs) / 1e6).round(4)}")

            # Stop sampler and collect GPU stats
            gpu_samples = stop_sampler() if stop_sampler else []
            gpu_util_raw, gpu_mem_raw = zip(*gpu_samples) if gpu_samples else ([], [])
            print(f"      - [Sampled] GPU util (%): {np.round(gpu_util_raw, 2)}")
            print(f"      - [Sampled] GPU mem (GB): {np.round(gpu_mem_raw, 3)}")

            if device.type == "cuda":
                # Benchmark GPU compute-only latency (excludes H2D/D2H)
                raw_gpu_latencies, raw_gpu_throughputs = gpu_compute_latency(sess, src, dst, bs, args.eval_runs)
                gpu_lat_avg, gpu_lat_std = mean_std(raw_gpu_latencies, keep_last=5)
                gpu_thr_avg, gpu_thr_std = mean_std(raw_gpu_throughputs, keep_last=5)
                print(f"      - GPU compute latencies (us/q): {(np.array(raw_gpu_latencies) * 1e6).round(4)}")
                print(f"      - GPU compute throughputs (Mq/s): {(np.array(raw_gpu_throughputs) / 1e6).round(4)}")

            # Collect results
            results["dataloader"].append({
                "batch_size": bs,
                "latency_raw_us_per_query": (np.array(raw_latencies) * 1e6).tolist(),
                "latency_avg_us_per_query": lat_avg * 1e6,
                "latency_std_us_per_query": lat_std * 1e6,
                "throughput_raw_M_queries_per_sec": (np.array(raw_throughputs) / 1e6).tolist(),
                "throughput_avg_M_queries_per_sec": thr_avg / 1e6,
                "throughput_std_M_queries_per_sec": thr_std / 1e6,
                **({
                    "gpu_compute_latency_raw_us_per_query": (np.array(raw_gpu_latencies) * 1e6).tolist(),
                    "gpu_compute_latency_avg_us_per_query": gpu_lat_avg * 1e6,
                    "gpu_compute_latency_std_us_per_query": gpu_lat_std * 1e6,
                    "gpu_compute_throughput_raw_M_queries_per_sec": (np.array(raw_gpu_throughputs) / 1e6).tolist(),
                    "gpu_compute_throughput_avg_M_queries_per_sec": gpu_thr_avg / 1e6,
                    "gpu_compute_throughput_std_M_queries_per_sec": gpu_thr_std / 1e6,
                } if device.type == "cuda" else {}),
                "gpu_util_raw_pct": list(gpu_util_raw),
                "gpu_util_median_pct": float(np.median(gpu_util_raw)) if gpu_util_raw else 0.0,
                "gpu_memory_raw_gb": list(gpu_mem_raw),
                "gpu_memory_median_gb": float(np.median(gpu_mem_raw)) if gpu_mem_raw else 0.0,
            })
        print_dataloader_table(results["dataloader"])

        if has_accuracy:
            print("Computing accuracy...")
            accuracy_sess, _ = load_session(args.model_path, sess_options, trt_opts, device=args.device)
            mae, mre = compute_accuracy(accuracy_sess, src, dst, gt, batch_size=batch_sizes[-1])
            print(f"  - MAE: {mae:.2f}")
            print(f"  - MRE: {mre*100:.2f}%")
            results["mae"] = mae
            results["mre_pct"] = mre * 100

        peak_gpu_gb = max(d["gpu_memory_median_gb"] for d in results["dataloader"])
        peak_cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2

    elif args.backend == "libtorch":
        torch.set_num_threads(n_threads)
        print(f"LibTorch torch.get_num_threads(): {torch.get_num_threads()}")
        print_green(f"Loading LibTorch model: {args.model_path}")
        model = libtorch_load(args.model_path, device)

        print("Running latency + throughput benchmark...")
        for bs in batch_sizes:
            print(f"  - bs={bs:<10}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            # Start GPU stats sampler (NVML)
            stop_sampler = start_gpu_sampler(device) if device.type == "cuda" else None

            # Benchmark latency + throughput
            raw_latencies, raw_throughputs = libtorch_latency(model, src, dst, device, bs, args.eval_runs)
            lat_avg, lat_std = mean_std(raw_latencies, keep_last=5)
            thr_avg, thr_std = mean_std(raw_throughputs, keep_last=5)
            print(f"      - Latencies (us/q): {(np.array(raw_latencies) * 1e6).round(4)}")
            print(f"      - Throughputs (Mq/s): {(np.array(raw_throughputs) / 1e6).round(4)}")

            # Stop sampler and collect GPU stats
            gpu_samples = stop_sampler() if stop_sampler else []
            gpu_util_raw, gpu_mem_raw = zip(*gpu_samples) if gpu_samples else ([], [])
            print(f"      - [Sampled] GPU util (%): {np.round(gpu_util_raw, 2)}")
            print(f"      - [Sampled] GPU mem (GB): {np.round(gpu_mem_raw, 3)}")

            # Collect results
            results["dataloader"].append({
                "batch_size": bs,
                "latency_raw_us_per_query": (np.array(raw_latencies) * 1e6).tolist(),
                "latency_avg_us_per_query": lat_avg * 1e6,
                "latency_std_us_per_query": lat_std * 1e6,
                "throughput_raw_M_queries_per_sec": (np.array(raw_throughputs) / 1e6).tolist(),
                "throughput_avg_M_queries_per_sec": thr_avg / 1e6,
                "throughput_std_M_queries_per_sec": thr_std / 1e6,
                "gpu_util_raw_pct": list(gpu_util_raw),
                "gpu_util_median_pct": float(np.median(gpu_util_raw)) if gpu_util_raw else 0.0,
                "gpu_memory_raw_gb": list(gpu_mem_raw),
                "gpu_memory_median_gb": float(np.median(gpu_mem_raw)) if gpu_mem_raw else 0.0,
            })
        print_dataloader_table(results["dataloader"])

        if has_accuracy:
            print("Computing accuracy...")
            mae, mre = libtorch_accuracy(model, src, dst, gt, device, batch_size=batch_sizes[-1])
            print(f"  - MAE: {mae:.2f}")
            print(f"  - MRE: {mre*100:.2f}%")
            results["mae"] = mae
            results["mre_pct"] = mre * 100

        peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        peak_cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    all_gpu_util = [u for d in results["dataloader"] for u in d.get("gpu_util_raw_pct", [])]
    all_gpu_mem = [m for d in results["dataloader"] for m in d.get("gpu_memory_raw_gb", [])]
    peak_gpu_util_pct = max(all_gpu_util) if all_gpu_util else 0.0

    print(f"Peak GPU Utilization (sampled): {peak_gpu_util_pct:.2f} %")
    print(f"Peak GPU Memory (sampled): {max(all_gpu_mem) if all_gpu_mem else peak_gpu_gb:.3f} GB")
    print(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")
    print(f"Peak CPU memory: {peak_cpu_gb:.2f} GB")
    results["peak_gpu_util_pct"] = peak_gpu_util_pct
    results["peak_gpu_memory_gb"] = peak_gpu_gb
    results["peak_cpu_memory_gb"] = peak_cpu_gb

    model_name = os.path.basename(args.model_path).removesuffix(".jit.pt").removesuffix(".onnx")
    output_filename = f"metrics_{model_name}.json"
    save_metrics_json(results, file_name=output_filename, dir_name=args.output_dir)


if __name__ == "__main__":
    main()
