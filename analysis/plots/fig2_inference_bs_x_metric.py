"""Inference benchmark: one figure per device (GPU, CPU), rows = batch_size, cols = latency|throughput, x-axis = model.

Usage:
    python fig2_inference_bs_x_metric.py
    open fig2_inference_bs_x_metric_gpu.pdf fig2_inference_bs_x_metric_cpu.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FIGURES_DIR,
    batch_sizes_seen,
    vertical_group_spans,
    inference_mean_std,
    tex_safe,
)
from extract_metrics import METHODS

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE = 1.0   # knob: local multiplier on plot_config's base font sizes
FS_AXIS, FS_TICK = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK))

BENCH_METRICS = {
    'latency':    ('latency_avg_us_per_query', 'Latency (us/query)'),
    'throughput': ('throughput_avg_M_queries_per_sec', 'Throughput (M queries/sec)'),
}

# Models to plot
MODEL_ORDER = [
    'Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset',           # Baselines
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]
MODEL_LABELS = [METHODS[m]['label'] for m in MODEL_ORDER]

def fmt_bs(bs):
    if bs >= 1_000_000:
        return f'{bs // 1_000_000}M'
    if bs >= 1_000:
        return f'{bs // 1_000}k'
    return str(bs)

x = np.arange(len(MODEL_ORDER))
group_spans = vertical_group_spans(MODEL_ORDER)  # shared across both figures


def make_figure(device, device_label, out_name):
    all_bs = batch_sizes_seen(device)
    n_rows = len(all_bs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.9 * n_rows), sharex=False)
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row_i, bs in enumerate(all_bs):
        for col_i, (mkey, (field, ylabel)) in enumerate(BENCH_METRICS.items()):
            ax = axes[row_i, col_i]
            means, stds = [], []
            for m in MODEL_ORDER:
                mean, std = inference_mean_std(device, m, bs, field)
                means.append(mean if mean is not None else np.nan)
                stds.append(std if std is not None else 0.0)
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)
            mask = ~np.isnan(means)
            if mask.any():
                ax.errorbar(x[mask], means[mask], yerr=stds[mask], fmt='o-',
                            color='#4E79A7', linewidth=1.1, markersize=3,
                            capsize=2, elinewidth=0.7, alpha=0.9)

            ax.set_yscale('log')
            ax.set_ylabel(ylabel, fontsize=FS_AXIS - 0.5)
            ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.6)
            ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=FS_TICK)

        axes[row_i, 0].set_title(f'bs={fmt_bs(bs)} — Latency', fontsize=FS_AXIS, loc='left', pad=3)
        axes[row_i, 1].set_title(f'bs={fmt_bs(bs)} — Throughput', fontsize=FS_AXIS, loc='left', pad=3)

    for row_i in range(n_rows):
        for col_i in range(2):
            ax = axes[row_i, col_i]
            for gname, x0, x1 in group_spans:
                ax.axvspan(x0, x1, color=METHOD_GROUP_COLORS[gname], alpha=0.07, zorder=0, linewidth=0)
            ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_LABELS, rotation=60, ha='right', fontsize=FS_TICK)

    fig.suptitle(tex_safe(f'Inference Benchmark ({device_label}) — Latency & Throughput by Model, Batch Size\n'
                          '(mean ± std across 13 datasets per cell)'),
                 fontsize=FS_AXIS + 2, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    out = FIGURES_DIR / out_name
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}')


make_figure('gpu', 'GPU', 'fig2_inference_bs_x_metric_gpu.pdf')
make_figure('cpu', 'CPU', 'fig2_inference_bs_x_metric_cpu.pdf')
