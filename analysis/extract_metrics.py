"""Extracts metrics from training/inference runs and saves them as JSON or import as python dicts.

Usage:
    python extract_metrics.py
    open metrics_data.json
    or
    from extract_metrics import METRICS
"""

import json, glob, os, sys, statistics

# Utility to print colored text in terminal
IS_TTY = sys.stdout.isatty()
def print_green(text):
    """Print text in green color."""
    text = f"\033[92m{text}\033[0m" if IS_TTY else text
    print(text)

def print_warning(text):
    """Print warning text in yellow color."""
    text = f"\033[93m{text}\033[0m" if IS_TTY else text
    print(text)

# REPO base path
REPO = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------
# Path to saved metrics
# ----------------------------------------------------
# METRICS_DIRS ('training' / 'inference' / 'inference_bs_sweep' keys).
METRICS_DIRS = {
    'training': os.path.join(REPO, 'results/v9-camera-training/saved_metrics'),
    'inference': {
        'gpu':          os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/trt_cuda_100k_1m_training'),
        'cpu_trt':      os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/trt_cpu_100k_1m_training'),
        'cpu_catboost': os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/catboost_cpu_100k_1m_training'),
    },
    'inference_bs_sweep': {
        'gpu':          os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/bs_sweep_training/trt_cuda'),
        'cpu_trt':      os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/bs_sweep_training/trt_cpu'),
        'cpu_catboost': os.path.join(REPO, 'results/v9-camera-training/saved_inference_metrics/bs_sweep_training/catboost_cpu'),
    },
}


def sanitize_paths(obj):
    """Recursively rewrite all paths to be repo-root-relative."""
    if isinstance(obj, str):
        if obj.startswith('/') and os.path.realpath(obj).startswith(REPO + os.sep):
            return os.path.relpath(os.path.realpath(obj), REPO)
        if obj.startswith('/'):
            anchor = f'/{os.path.basename(REPO)}/'
            idx = obj.find(anchor)
            if idx != -1:
                return obj[idx + len(anchor):]
        if obj.startswith('../'):
            return obj[len('../'):]
        return obj
    if isinstance(obj, list):
        return [sanitize_paths(v) for v in obj]
    if isinstance(obj, dict):
        return {k: sanitize_paths(v) for k, v in obj.items()}
    return obj

## Function to load json metrics from a directory
def load_dir(dir_name):
    """{key: value} for every metrics_*.json in relpath."""
    out = {}

    # Return empty dict if the directory doesn't exist
    if not os.path.isdir(dir_name):
        return out

    # 'metrics_ANEDA_CTR_landmark_30M.json' -> 'ANEDA_CTR_landmark_30M'
    strip_key = lambda f: f.removeprefix('metrics_').removesuffix('.json')

    # Load all metrics_*.json files in the directory
    print(f"Loading metrics: {dir_name}/")
    for f in sorted(glob.glob(os.path.join(dir_name, '*.json'))):
        out[strip_key(os.path.basename(f))] = sanitize_paths(json.load(open(f)))
    return out


# ----------------------------------------------------
# Build METRICS: keyed by '<model>_<dataset>_<query_dir>'
# ----------------------------------------------------
def load_inference_dirs(dirs):
    """dirs: {'gpu':path, 'cpu_trt':path, 'cpu_catboost':path}. Returns {'gpu':.., 'cpu':..} per-filename dicts."""
    loaded = {name: load_dir(path) for name, path in dirs.items()}
    filenames = set(loaded['gpu']) | set(loaded['cpu_trt']) | set(loaded['cpu_catboost'])
    return {
        filename: {
            'gpu': loaded['gpu'].get(filename),
            'cpu': loaded['cpu_trt'].get(filename) or loaded['cpu_catboost'].get(filename),
        }
        for filename in sorted(filenames)
    }


_train = load_dir(METRICS_DIRS['training'])
_inference = load_inference_dirs(METRICS_DIRS['inference'])
_inference_bs_sweep = load_inference_dirs(METRICS_DIRS['inference_bs_sweep'])

_all_filenames = set(_train) | set(_inference) | set(_inference_bs_sweep)
METRICS = {
    filename: {
        'training': _train.get(filename),
        'inference': _inference.get(filename, {'gpu': None, 'cpu': None}),
        'inference_bs_sweep': _inference_bs_sweep.get(filename, {'gpu': None, 'cpu': None}),
    }
    for filename in sorted(_all_filenames)
}

# Fallback to the cached metrics_data.json dump
if not METRICS:
    _cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics_data.json')
    if os.path.isfile(_cache_path):
        print_warning(f"Warning: {METRICS_DIRS['training']} not found. Falling back to cached {_cache_path}")
        with open(_cache_path) as f:
            METRICS = json.load(f)
    else:
        print_warning(f"Warning: {METRICS_DIRS['training']} not found and no cached metrics_data.json to fall back to.")


# ----------------------------------------------------
# Add derived metrics
# ----------------------------------------------------
def get_gpu_peak_memory(d):
    """Peak GPU mem across train/eval_train/eval_test phases (CatBoost has none -> None)."""
    vals = [d.get('max_gpu_memory_train'), d.get('max_gpu_memory_eval_train'), d.get('max_gpu_memory_eval_test')]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def get_per_epoch_time_sec(d):
    """Mean of first-diffs of time_elapsed_history (min -> sec)."""
    h = d.get('time_elapsed_history')
    if not h or len(h) < 2:
        return None
    return statistics.mean((h[i] - h[i - 1]) * 60.0 for i in range(1, len(h)))


def get_per_epoch_time_per_sample_sec(d):
    """Per-epoch time divided by training set size."""
    per_epoch_time_sec = d.get('per_epoch_time_sec')
    train_size = d.get('train_data_size')
    if per_epoch_time_sec is None or not train_size:
        return None
    return per_epoch_time_sec / train_size


for filename, metrics in METRICS.items():
    training_metrics = metrics['training']

    # gpu_peak_memory
    training_metrics['gpu_peak_memory'] = get_gpu_peak_memory(training_metrics)

    # per_epoch_time_sec
    training_metrics['per_epoch_time_sec'] = get_per_epoch_time_sec(training_metrics)

    # per_epoch_time_per_sample_sec
    training_metrics['per_epoch_time_per_sample_sec'] = get_per_epoch_time_per_sample_sec(training_metrics)


# ----------------------------------------------------
# Important CONSTANTS for analysis
# ----------------------------------------------------
METHODS = {
    'Manhattan':              {'label': 'Manhattan'},
    'Landmark_random_subset': {'label': 'Landmark_rn'},
    'Landmark_kmeans_subset': {'label': 'Landmark_km'},
    'GeoDNN':                 {'label': 'GeoDNN'},
    'DistanceNN_sub':         {'label': 'DistanceNN'},
    'EmbeddingNN_mean':       {'label': 'EmbedNN'},
    'Vdist2vec':              {'label': 'Vdist2vec'},
    'Ndist2vec':              {'label': 'Ndist2vec'},
    'CatBoostNN':             {'label': 'LandmarkNN'},
    'GCN':                    {'label': 'RGCNdist2vec'},
    'SAGE':                   {'label': 'RSAGEdist2vec'},
    'GAT':                    {'label': 'RGATdist2vec'},
    'Path2vec':               {'label': 'Path2vec'},
    'ANEDA':                  {'label': 'ANEDA'},
    'RNE':                    {'label': 'RNE'},
    'CatBoost':               {'label': 'CatBoost'},
}

METHOD_GROUPS = {
    'Baselines':  ['Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset'],
    'NNs':        ['GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN'],
    'GNNs':       ['GCN', 'SAGE', 'GAT'],
    'Functional': ['Path2vec', 'ANEDA', 'RNE'],
    'Tree':       ['CatBoost'],
}

DATASETS = {
    'Surat':      {'label': 'SU',  'nodes': 2_508,      'query_dir': 'all_pairs'},
    'W_Jinan':    {'label': 'JN',  'nodes': 8_908,      'query_dir': 'real_workload_perturb_500k'},
    'W_Shenzhen': {'label': 'SZ',  'nodes': 11_933,     'query_dir': 'real_workload_perturb_500k'},
    'W_Chengdu':  {'label': 'CD',  'nodes': 17_567,     'query_dir': 'real_workload_perturb_500k'},
    'W_Beijing':  {'label': 'BJ',  'nodes': 74_383,     'query_dir': 'real_workload_perturb_500k'},
    'W_Shanghai': {'label': 'SH',  'nodes': 74_903,     'query_dir': 'real_workload_perturb_500k'},
    'W_NewYork':  {'label': 'NY',  'nodes': 334_930,    'query_dir': 'real_workload_perturb_500k'},
    'W_Chicago':  {'label': 'CH',  'nodes': 386_533,    'query_dir': 'real_workload_perturb_500k'},
    'FLA':        {'label': 'FLA', 'nodes': 1_070_376,  'query_dir': 'landmark_30M'},
    'E':          {'label': 'E',   'nodes': 3_598_623,  'query_dir': 'landmark_30M'},
    'W':          {'label': 'W',   'nodes': 6_262_104,  'query_dir': 'landmark_30M'},
    'CTR':        {'label': 'CTR', 'nodes': 14_081_816, 'query_dir': 'landmark_30M'},
    'USA':        {'label': 'USA', 'nodes': 23_947_347, 'query_dir': 'landmark_30M'},
}

DATASET_GROUPS = {
    'AllPairs':   ['Surat'],
    'Workload':   ['W_Jinan', 'W_Shenzhen', 'W_Chengdu', 'W_Beijing', 'W_Shanghai', 'W_NewYork', 'W_Chicago'],
    'DIMACS':     ['FLA', 'E', 'W', 'CTR', 'USA'],
}

def get_key(model, dataset):
    """Return the key used in METRICS for a given model and dataset."""
    return f'{model}_{dataset}_{DATASETS[dataset]["query_dir"]}'


def training(k):
    """METRICS[k]['training'], or None if the run doesn't exist."""
    return METRICS[k]['training'] if k in METRICS else None


def inference(device):
    """Returns a getter METRICS[k]['inference'][device], for device in ('gpu', 'cpu')."""
    return lambda k: METRICS[k]['inference'][device] if k in METRICS else None


def inference_bs_sweep(device):
    """Returns a getter METRICS[k]['inference_bs_sweep'][device], for device in ('gpu', 'cpu')."""
    return lambda k: METRICS[k]['inference_bs_sweep'][device] if k in METRICS else None


# ----------------------------------------------------
# Metric definitions: metric_name -> {get_value_fn, decimals, key, label, units, desc}
# ----------------------------------------------------
percentile = lambda field, p: lambda d: d.get(field, {}).get(p) if d else None

TRAINING_METRICS = {
    'test_mre': {
        'get_value_fn': lambda d: d.get('test_mre') if d else None,
        'decimals': 2, 'key': 'test_mre', 'label': 'Test MRE', 'units': '%',
        'desc': "Mean relative error on the test split.",
    },
    'test_mae': {
        'get_value_fn': lambda d: d.get('test_mae') if d else None,
        'decimals': 1, 'key': 'test_mae', 'label': 'Test MAE', 'units': 'm',
        'desc': "Mean absolute error on the test split.",
    },
    'test_mre_p80': {
        'get_value_fn': percentile('test_mre_percentiles', 'p80'),
        'decimals': 2, 'key': 'test_mre_percentiles.p80', 'label': 'P80 Test MRE', 'units': '%',
        'desc': "MRE below which 80% of test queries fall.",
    },
    'test_mre_p90': {
        'get_value_fn': percentile('test_mre_percentiles', 'p90'),
        'decimals': 2, 'key': 'test_mre_percentiles.p90', 'label': 'P90 Test MRE', 'units': '%',
        'desc': "MRE below which 90% of test queries fall.",
    },
    'test_mae_p80': {
        'get_value_fn': percentile('test_mae_percentiles', 'p80'),
        'decimals': 2, 'key': 'test_mae_percentiles.p80', 'label': 'P80 Test MAE', 'units': 'm',
        'desc': "MAE below which 80% of test queries fall.",
    },
    'test_mae_p90': {
        'get_value_fn': percentile('test_mae_percentiles', 'p90'),
        'decimals': 2, 'key': 'test_mae_percentiles.p90', 'label': 'P90 Test MAE', 'units': 'm',
        'desc': "MAE below which 90% of test queries fall.",
    },
    'peak_cpu_mem': {
        'get_value_fn': lambda d: d.get('max_cpu_memory') if d else None,
        'decimals': 2, 'key': 'max_cpu_memory', 'label': 'CPU Peak Memory', 'units': 'GB',
        'desc': "Peak CPU memory during training+eval.",
    },
    'peak_gpu_mem': {
        'get_value_fn': lambda d: d.get('gpu_peak_memory') if d else None,
        'decimals': 3, 'key': 'gpu_peak_memory', 'label': 'GPU Peak Memory', 'units': 'GB',
        'desc': "Peak GPU memory across train/eval phases. CatBoost is CPU-only (dash).",
    },
    'model_size': {
        'get_value_fn': lambda d: d.get('model_size') if d else None,
        'decimals': 2, 'key': 'model_size', 'label': 'Model Size', 'units': 'MB',
        'desc': "Saved .pt checkpoint size.",
    },
    'params': {
        'get_value_fn': lambda d: d.get('model_params') / 1e6 if d and d.get('model_params') is not None else None,
        'decimals': 3, 'key': 'model_params', 'label': 'Total Params', 'units': 'M',
        'desc': "Trainable parameter count, in millions.",
    },
    'jit_size': {
        'get_value_fn': lambda d: d.get('jit_model_size') if d else None,
        'decimals': 2, 'key': 'jit_model_size', 'label': 'JIT Model Size', 'units': 'MB',
        'desc': "Saved .jit.pt (TorchScript) checkpoint size.",
    },
    'test_query_time': {
        'get_value_fn': lambda d: d.get('test_query_time') if d else None,
        'decimals': 3, 'key': 'test_query_time', 'label': 'Query Time', 'units': 'us/sample, test',
        'desc': "Per-sample inference time on the test split.",
    },
    'num_epochs': {
        'get_value_fn': lambda d: d.get('last_train_epoch') if d else None,
        'decimals': 0, 'key': 'last_train_epoch', 'label': 'Epochs Reached', 'units': 'count',
        'desc': "Epochs completed before stopping. Baselines/CatBoost have no epoch loop (dash).",
    },
    'per_epoch_time': {
        'get_value_fn': lambda d: d.get('per_epoch_time_sec') if d else None,
        'decimals': 2, 'key': 'per_epoch_time_sec', 'label': 'Per-Epoch Time', 'units': 'sec',
        'desc': "Mean wall-clock time per epoch.",
    },
    'per_epoch_time_per_sample': {
        'get_value_fn': lambda d: d.get('per_epoch_time_per_sample_sec') if d else None,
        'decimals': 8, 'key': 'per_epoch_time_per_sample_sec', 'label': 'Per-Epoch Time per Sample', 'units': 'sec',
        'desc': "Per-epoch time divided by training set size.",
    },
    'precomp_time': {
        'get_value_fn': lambda d: d.get('precomputation_time') if d else None,
        'decimals': 2, 'key': 'precomputation_time', 'label': 'Precomputation/Training Time', 'units': 'min',
        'desc': "Total training wall-clock time.",
    },
}


def dataloader_metric(d, batch_size, metric_key):
    """Value of `metric_key` in the dataloader-sweep row matching `batch_size`, or None."""
    if d is None:
        return None
    for row in d.get('dataloader', []):
        if row['batch_size'] == batch_size:
            return row.get(metric_key)
    return None


# get_value_fn signature is (d, batch_size); `devices` lists which device(s) actually sample this metric.
INFERENCE_METRICS = {
    'latency': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'latency_avg_us_per_query'),
        'decimals': 4, 'key': 'latency_avg_us_per_query', 'label': 'Latency', 'units': 'us/query',
        'desc': 'Latency per query', 'devices': ('gpu', 'cpu'),
    },
    'throughput': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'throughput_avg_M_queries_per_sec'),
        'decimals': 2, 'key': 'throughput_avg_M_queries_per_sec', 'label': 'Throughput', 'units': 'M queries/sec',
        'desc': 'Throughput per query', 'devices': ('gpu', 'cpu'),
    },
    'gpu_compute_latency': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'gpu_compute_latency_avg_us_per_query'),
        'decimals': 5, 'key': 'gpu_compute_latency_avg_us_per_query', 'label': 'GPU Compute Latency', 'units': 'us/query',
        'desc': 'Compute-only latency per query (excludes H2D/D2H transfer)', 'devices': ('gpu',),
    },
    'gpu_compute_throughput': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'gpu_compute_throughput_avg_M_queries_per_sec'),
        'decimals': 2, 'key': 'gpu_compute_throughput_avg_M_queries_per_sec', 'label': 'GPU Compute Throughput', 'units': 'M queries/sec',
        'desc': 'Compute-only throughput per query (excludes H2D/D2H transfer)', 'devices': ('gpu',),
    },
    'gpu_util': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'gpu_util_median_pct'),
        'decimals': 2, 'key': 'gpu_util_median_pct', 'label': 'GPU Utilization (median, sampled)', 'units': '%',
        'desc': 'Median GPU utilization (sampled)', 'devices': ('gpu',),
    },
    'gpu_mem': {
        'get_value_fn': lambda d, bs: dataloader_metric(d, bs, 'gpu_memory_median_gb'),
        'decimals': 3, 'key': 'gpu_memory_median_gb', 'label': 'GPU Memory (median, sampled)', 'units': 'GB',
        'desc': 'Median GPU memory (sampled)', 'devices': ('gpu',),
    },
}


def batch_sizes_seen(device):
    """All distinct batch sizes present in any dataloader sweep for this device."""
    sizes = set()
    for entry in METRICS.values():
        d = entry['inference'][device]
        if d:
            sizes.update(row['batch_size'] for row in d.get('dataloader', []))
    return sorted(sizes)


def batch_sizes_seen_bs_sweep(device):
    """All distinct batch sizes present in any METRICS[k]['inference_bs_sweep'] dataloader sweep for this device."""
    sizes = set()
    for entry in METRICS.values():
        d = entry['inference_bs_sweep'][device]
        if d:
            sizes.update(row['batch_size'] for row in d.get('dataloader', []))
    return sorted(sizes)


if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics_data.json')
    print_green(f"Saving metrics: {out_path}")
    with open(out_path, 'w') as f:
        json.dump(METRICS, f, indent=2)
