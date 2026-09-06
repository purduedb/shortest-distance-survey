"""Horizontal dumbbell/dot plot: one row per model, CPU vs GPU, side-by-side latency|throughput panels.

Usage:
    python fig3a_latency_throughput.py
    open fig3a_latency_throughput.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import (
    METHOD_GROUP_COLORS as GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FS_LABEL,
    FIGURES_DIR,
    plot_label,
    inference_mean_std,
)
from extract_metrics import METHODS, METHOD_GROUPS

FIG_W = 1.8 * 1.6 * 2   # two side-by-side panels, each ~ as wide as the original single-panel figure

# ── Tunable display flags ──────────────────────────────────────────────────
FONT_SCALE               = 1.0    # knob: local multiplier on plot_config's base font sizes (crowded model-name column)
FS_AXIS, FS_TICK, FS_LEG, FS_LABEL = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG, FS_LABEL))
SHOW_GROUP_LABELS_RIGHT  = False  # colored method-group names on a right-hand twin y-axis (throughput panel)
COLOR_MODEL_LABEL_TEXT   = False  # left y-tick model names: font tinted by group color
COLOR_MODEL_LABEL_BG     = False  # left y-tick model names: background tinted by group color
LEGEND_TOP               = 0.85   # knob: axes top margin (subplots_adjust `top`) -- lower = more gap under the legend
LEGEND_Y                 = 0.98   # knob: legend's bbox_to_anchor y -- how close the legend sits to the figure top

# Which rows get a speedup annotation when SHOW_SPEEDUPS is on
SHOW_SPEEDUPS            = True   # CPU->GPU speedup arrows/labels (latency panel only)
SPEEDUP_MODELS = ['CatBoostNN', 'RNE']

SHOW_GPU_SPEEDUPS  = True         # GPU (End-to-End) -> GPU (Compute-only) speedup arrows/labels (latency panel only)
GPU_SPEEDUP_MODELS = ['Manhattan', 'RNE']

CPU_BS = 1_000_000                # batch_size = 1M for CPU inference
GPU_BS = 1_000_000                # batch_size = 1M for GPU inference
LAT_FIELD  = 'latency_avg_us_per_query'
THR_FIELD  = 'throughput_avg_M_queries_per_sec'
GPU_COMPUTE_LAT_FIELD = 'gpu_compute_latency_avg_us_per_query'
GPU_COMPUTE_THR_FIELD = 'gpu_compute_throughput_avg_M_queries_per_sec'

LATENCY_LABELS = {
    0.0001: '0.1 ns', 0.001: '1 ns', 0.01: '10 ns', 0.1: '100 ns',
    1: '1 µs', 10: '10 µs', 100: '100 µs',
}
THROUGHPUT_LABELS = {
    0.1: '100K', 1: '1M', 10: '10M', 100: '100M', 1000: '1B', 10000: '10B',
}

def _fmt_latency(v, _pos=None):
    return LATENCY_LABELS.get(v, '')

def _fmt_throughput(v, _pos=None):
    return THROUGHPUT_LABELS.get(v, '')

# Models to plot
MODEL_ORDER = [
    'Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset',           # Baselines
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]
_MODEL_GROUP_LOOKUP = {m: g for g, models in METHOD_GROUPS.items() for m in models}
_MODEL_KEY_OF_LABEL = {METHODS[mk]['label']: mk for mk in MODEL_ORDER}

# ── Build data dicts: {label: (cpu_mean, cpu_std, gpu_mean, gpu_compute_mean)} ─
lat_data, thr_data = {}, {}
for mk in MODEL_ORDER:
    label = METHODS[mk]['label']
    cpu_lat_m, cpu_lat_s = inference_mean_std('cpu', mk, CPU_BS, LAT_FIELD)
    gpu_lat_m, _ = inference_mean_std('gpu', mk, GPU_BS, LAT_FIELD)
    gpu_compute_lat_m, _ = inference_mean_std('gpu', mk, GPU_BS, GPU_COMPUTE_LAT_FIELD)
    lat_data[label] = (cpu_lat_m, cpu_lat_s, gpu_lat_m, gpu_compute_lat_m)

    cpu_thr_m, cpu_thr_s = inference_mean_std('cpu', mk, CPU_BS, THR_FIELD)
    gpu_thr_m, _ = inference_mean_std('gpu', mk, GPU_BS, THR_FIELD)
    gpu_compute_thr_m, _ = inference_mean_std('gpu', mk, GPU_BS, GPU_COMPUTE_THR_FIELD)
    thr_data[label] = (cpu_thr_m, cpu_thr_s, gpu_thr_m, gpu_compute_thr_m)

# Rebuild group->labels from MODEL_ORDER
groups = {}
for mk in MODEL_ORDER:
    g = _MODEL_GROUP_LOOKUP[mk]
    groups.setdefault(g, []).append(METHODS[mk]['label'])

GROUP_DISPLAY_NAME = {
    'Baselines':  'Baselines',
    'NNs':        'Neural Network Methods',
    'GNNs':       'Graph Neural Network Methods',
    'Functional': 'Functional Methods',
    'Tree':       'Tree-based Gradient Boosting Methods',
}

# ── Build ordered method list (table order = top-to-bottom in plot) ───────────
methods = []
method_color = {}
for g, ms in groups.items():
    for m in ms:
        methods.append(m)
        method_color[m] = GROUP_COLORS[g]

n = len(methods)

# ── Layout parameters ────────────────────────────────────────────────────────
ROW_SPACING      = 0.4
SPEEDUP_TEXT_GAP = 0.13
SPEEDUP_Y_SHIFT  = -0.15  # knob: shift the speedup arrow + text down (negative) / up (positive), in row-spacing units
FIG_H            = 1.2 + n * ROW_SPACING * 0.29

y = ROW_SPACING * np.arange(n - 1, -1, -1)
y_of = {m: y[i] for i, m in enumerate(methods)}

OFF_CPU =  0.15 * ROW_SPACING
OFF_GPU = -0.15 * ROW_SPACING


def draw_group_shading(ax):
    for g, ms in groups.items():
        ys = [y_of[m] for m in ms]
        ax.axhspan(min(ys) - 0.5 * ROW_SPACING, max(ys) + 0.5 * ROW_SPACING,
                   color=GROUP_COLORS[g], alpha=0.07, zorder=0)


def draw_points(ax, panel_data):
    """panel_data: {label: (cpu_mean, cpu_std, gpu_mean, gpu_compute_mean)}.
    Hollow circle = CPU, solid circle = GPU (end-to-end), solid triangle = GPU (compute-only)."""
    for m in methods:
        cpu_m, cpu_s, gpu_m, gpu_compute_m = panel_data[m]
        c = method_color[m]
        yi = y_of[m]
        if cpu_m is None and gpu_m is None and gpu_compute_m is None:
            ax.text(ax.get_xlim()[0], yi, '—  no data', va='center', ha='left',
                    fontsize=FS_LABEL - 0.5, color=c, style='italic', zorder=4)
            continue
        if cpu_m is not None:
            ax.errorbar(cpu_m, yi + OFF_CPU, xerr=cpu_s,
                        fmt='o', color=c, lw=0.8, capsize=2, capthick=0.8, zorder=4,
                        ms=4, markerfacecolor='white', markeredgewidth=1.0)
        if gpu_m is not None:
            ax.plot(gpu_m, yi + OFF_GPU, 'o', color=c, ms=4, zorder=4)
        if gpu_compute_m is not None:
            ax.plot(gpu_compute_m, yi + OFF_GPU, '^', color=c, ms=4, zorder=4)


LEGEND_HANDLES = [
    Line2D([0], [0], marker='^', color='gray', ms=4, lw=0, label='GPU (Compute only, no data transfer)'),
    Line2D([0], [0], marker='o', color='gray', ms=4, lw=0, label='GPU (End-to-End)'),
    Line2D([0], [0], marker='o', color='gray', ms=4, lw=0,
           markerfacecolor='white', markeredgewidth=1.0, label=r'CPU ($\mu \pm \sigma$)'),
]


def draw(fig, gs):
    """Draw the latency|throughput panel pair into a 2-cell gridspec slice. Returns (ax_lat, ax_thr)."""
    ax_lat = fig.add_subplot(gs[0])
    ax_thr = fig.add_subplot(gs[1], sharey=ax_lat)

    draw_group_shading(ax_lat)
    draw_points(ax_lat, lat_data)

    draw_group_shading(ax_thr)
    draw_points(ax_thr, thr_data)

    # ── Speedup lines (CPU->GPU), latency panel only, curated subset ────────
    if SHOW_SPEEDUPS:
        speedup_labels = {METHODS[mk]['label'] for mk in SPEEDUP_MODELS}
        for i, m in enumerate(methods):
            if m not in speedup_labels:
                continue
            cpu_m, cpu_s, gpu_m, _ = lat_data[m]
            if cpu_m is None or gpu_m is None or gpu_m <= 0:
                continue
            yi = y_of[m] + SPEEDUP_Y_SHIFT * ROW_SPACING
            speedup = cpu_m / gpu_m
            ax_lat.annotate('', xy=(cpu_m, yi), xytext=(gpu_m, yi),
                             arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.1,
                                             mutation_scale=8))
            label_below = (i % 2 == 0)
            dy = -SPEEDUP_TEXT_GAP * ROW_SPACING if label_below else SPEEDUP_TEXT_GAP * ROW_SPACING
            va = 'top' if label_below else 'bottom'
            ax_lat.text(np.sqrt(gpu_m * cpu_m), yi + dy,
                        f'{speedup:.0f}x', va=va, ha='center',
                        fontsize=10, color='#555555', zorder=5)

    # ── Speedup lines (GPU End-to-End -> GPU Compute-only), latency panel only, curated subset ─
    if SHOW_GPU_SPEEDUPS:
        gpu_speedup_labels = {METHODS[mk]['label'] for mk in GPU_SPEEDUP_MODELS}
        for i, m in enumerate(methods):
            if m not in gpu_speedup_labels:
                continue
            _, _, gpu_m, gpu_compute_m = lat_data[m]
            if gpu_m is None or gpu_compute_m is None or gpu_compute_m <= 0:
                continue
            yi = y_of[m] + SPEEDUP_Y_SHIFT * ROW_SPACING
            speedup = gpu_m / gpu_compute_m
            ax_lat.annotate('', xy=(gpu_compute_m, yi), xytext=(gpu_m, yi),
                             arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.1,
                                             mutation_scale=8))
            label_below = (i % 2 == 0)
            dy = -SPEEDUP_TEXT_GAP * ROW_SPACING if label_below else SPEEDUP_TEXT_GAP * ROW_SPACING
            va = 'top' if label_below else 'bottom'
            ax_lat.text(np.sqrt(gpu_m * gpu_compute_m), yi + dy,
                        f'{speedup:.0f}x', va=va, ha='center',
                        fontsize=10, color='#555555', zorder=5)

    # ── Latency panel axes ───────────────────────────────────────────────────
    ax_lat.set_xscale('log')
    ax_lat.set_xlim(left=0.0001)  # 0.1 ns; right stays at matplotlib's autoscaled default
    ax_lat.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax_lat.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_latency))
    ax_lat.xaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10), numticks=20))
    ax_lat.tick_params(axis='x', labelsize=FS_TICK)
    ax_lat.tick_params(axis='x', which='minor', length=2)
    ax_lat.set_xlabel('Query Latency (s)', fontsize=FS_AXIS)
    ax_lat.grid(axis='x', which='major', linestyle=':', linewidth=0.5, alpha=0.6)
    ax_lat.grid(axis='x', which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

    # ── Throughput panel axes ────────────────────────────────────────────────
    # Data is stored as M queries/sec (millions); formatter relabels ticks in
    # absolute queries/sec (1M, 10M, ...).
    ax_thr.set_xscale('log')
    ax_thr.set_xlim(right=10000)  # 10B; left stays at matplotlib's autoscaled default
    ax_thr.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax_thr.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_throughput))
    ax_thr.xaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10), numticks=20))
    ax_thr.tick_params(axis='x', labelsize=FS_TICK)
    ax_thr.tick_params(axis='x', which='minor', length=2)
    ax_thr.set_xlabel('Throughput (queries/s)', fontsize=FS_AXIS)
    ax_thr.grid(axis='x', which='major', linestyle=':', linewidth=0.5, alpha=0.6)
    ax_thr.grid(axis='x', which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
    if SHOW_GROUP_LABELS_RIGHT:
        ax_thr.spines['right'].set_visible(False)  # twin axis (below) draws its own boundary instead

    # ── Shared y-axis (model names, left panel only) ────────────────────────
    display_labels = [plot_label(_MODEL_KEY_OF_LABEL[m]) for m in methods]
    ax_lat.set_yticks(list(y_of.values()))
    ax_lat.set_yticklabels(display_labels, fontsize=FS_LABEL)
    ax_lat.set_ylim(-0.6 * ROW_SPACING, ROW_SPACING * (n - 1) + 0.6 * ROW_SPACING)
    plt.setp(ax_thr.get_yticklabels(), visible=False)

    # ── Group labels on right y-axis of the throughput panel (optional) ─────
    if SHOW_GROUP_LABELS_RIGHT:
        ax2 = ax_thr.twinx()
        ax2.set_ylim(ax_thr.get_ylim())
        ax2.set_yticks([np.mean([y_of[m] for m in ms]) for ms in groups.values()])
        ax2.set_yticklabels([GROUP_DISPLAY_NAME[g] for g in groups.keys()], fontsize=FS_LABEL)
        for tick, g in zip(ax2.get_yticklabels(), groups.keys()):
            tick.set_color(GROUP_COLORS[g])
        ax2.tick_params(axis='y', length=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)

    # ── Colorize left-axis model-name labels by group ───────────────────────
    if COLOR_MODEL_LABEL_TEXT or COLOR_MODEL_LABEL_BG:
        for tick, m in zip(ax_lat.get_yticklabels(), methods):
            c = method_color[m]
            if COLOR_MODEL_LABEL_TEXT:
                tick.set_color(c)
            if COLOR_MODEL_LABEL_BG:
                tick.set_bbox(dict(facecolor=c, alpha=0.15, edgecolor='none', pad=1.5))

    return ax_lat, ax_thr


if __name__ == '__main__':
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    ax_lat, ax_thr = draw(fig, gs)

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(top=LEGEND_TOP, wspace=0.08)  # wspace: horizontal gap between the two panels

    fig.legend(handles=LEGEND_HANDLES, fontsize=FS_LEG, loc='upper center',
               bbox_to_anchor=(0.5, LEGEND_Y), ncol=len(LEGEND_HANDLES),
               frameon=False, handlelength=1.2, labelspacing=0.3,
               borderpad=0.4, handletextpad=0.5, columnspacing=1.5)

    OUT = FIGURES_DIR / 'fig3a_latency_throughput.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=200)
    print(f'Saved {OUT}')
