"""3-panel line plot: query latency (bs=1M) vs dataset, GPU (Compute only) | GPU (End-to-End) | CPU.

Usage:
    python fig5a_latency_vs_datasets.py
    open fig5a_latency_vs_datasets.pdf
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FIGURES_DIR,
    plot_label,
    MODEL_GROUP_OF,
    group_shades,
)
from extract_metrics import get_key, DATASETS, inference, dataloader_metric

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE = 1.3    # knob: local multiplier on plot_config's base font sizes
FS_AXIS, FS_TICK, FS_LEG = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG))
TITLE_MARKER_SIZE = 7  # knob: marker glyph size in each subplot title (bigger than in-plot markers)

BS = 1_000_000        # batch_size = 1M, all panels
LAT_FIELD              = 'latency_avg_us_per_query'
LAT_STD_FIELD          = 'latency_std_us_per_query'
GPU_COMPUTE_LAT_FIELD     = 'gpu_compute_latency_avg_us_per_query'
GPU_COMPUTE_LAT_STD_FIELD = 'gpu_compute_latency_std_us_per_query'

# fig3a-style latency tick labels (values are in us, log-spaced powers of ten).
LATENCY_LABELS = {
    0.0001: '0.1 ns', 0.001: '1 ns', 0.01: '10 ns', 0.1: '100 ns',
    1: r'1 $\mu$s', 10: r'10 $\mu$s', 100: r'100 $\mu$s', 1000: '1 ms', 10000: '10 ms',
}

def _fmt_latency(v, _pos=None):
    return LATENCY_LABELS.get(v, '')

FIG_H = 3.9    # taller than fig1a's FIG_H, to fit the 16-model legend at FONT_SCALE=1.3
PANEL_W = 3.0
FIG_W = PANEL_W * 3

# Datasets left to right, already in increasing node-count order (dict order in extract_metrics).
DATASET_ORDER = list(DATASETS.keys())
DATASET_LABELS = [DATASETS[dk]['label'] for dk in DATASET_ORDER]

# Panels: (device, latency field, std field, marker, hollow?, title).
PANELS = [
    ('gpu', GPU_COMPUTE_LAT_FIELD, GPU_COMPUTE_LAT_STD_FIELD, '^', False, 'GPU (Compute only, no data transfer)'),
    ('gpu', LAT_FIELD,             LAT_STD_FIELD,             'o', False, 'GPU (End-to-End)'),
    ('cpu', LAT_FIELD,             LAT_STD_FIELD,             'o', True,  'CPU'),
]

# Models to plot (same roster as fig1a/fig3a).
MODEL_ORDER = [
    'Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset',           # Baselines
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]

# Per-model line color: same group hue family, distinct shade per model within a group (fig1a convention).
_groups_order = {}
for mk in MODEL_ORDER:
    _groups_order.setdefault(MODEL_GROUP_OF[mk], []).append(mk)

MODEL_COLOR = {}
for g, ms in _groups_order.items():
    for mk, c in zip(ms, group_shades(METHOD_GROUP_COLORS[g], len(ms))):
        MODEL_COLOR[mk] = c


def draw(fig, gs):
    """Draw the GPU-compute | GPU-e2e | CPU latency-vs-dataset row. Returns the list of Axes."""
    axes = []
    ax_prev = None
    for i, (device, field, std_field, marker, hollow, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i], sharey=ax_prev)
        axes.append(ax)
        ax_prev = ax

        get_d = inference(device)
        for mk in MODEL_ORDER:
            xs, ys, yerrs = [], [], []
            for xi, dk in enumerate(DATASET_ORDER):
                d = get_d(get_key(mk, dk))
                v = dataloader_metric(d, BS, field)
                if v is None:
                    continue
                xs.append(xi)
                ys.append(v)
                yerrs.append(dataloader_metric(d, BS, std_field) or 0.0)
            if not xs:
                continue
            c = MODEL_COLOR[mk]
            mfc = 'white' if hollow else c
            ax.errorbar(xs, ys, yerr=yerrs, color=c, lw=1.3, marker=marker,
                        markersize=5, markerfacecolor=mfc, markeredgewidth=0.8,
                        markeredgecolor=c, capsize=1.5, elinewidth=0.5, alpha=0.9,
                        label=plot_label(mk), zorder=3)

        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_latency))
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10), numticks=20))
        ax.tick_params(axis='y', which='minor', length=2)
        ax.set_xticks(range(len(DATASET_ORDER)))
        sparse_labels = [lbl if xi % 2 == 0 else '' for xi, lbl in enumerate(DATASET_LABELS)]
        ax.set_xticklabels(sparse_labels, rotation=0, ha='center')
        ax.set_xlabel('Datasets', fontsize=FS_AXIS)
        ax.set_title(title, fontsize=FS_AXIS, pad=6)
        ax._title_marker_spec = (marker, 'white' if hollow else 'black')  # placed post-draw, once title extent is known
        ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax.tick_params(axis='both', labelsize=FS_TICK)

        if i == 0:
            ax.set_ylabel('Query Latency (s)', fontsize=FS_AXIS)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    return axes


def _build_legend_handles():
    handles = []
    for g, ms in _groups_order.items():
        for mk in ms:
            handles.append(Line2D([0], [0], color=MODEL_COLOR[mk], lw=1.6, label=plot_label(mk)))
    return handles


def _place_title_markers(fig, axes):
    """Draw each panel's marker glyph just left of its (already-rendered) title text."""
    fig.canvas.draw()  # finalize title layout so bboxes are measurable
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        marker, mfc = ax._title_marker_spec
        bbox = ax.title.get_window_extent(renderer=renderer)
        x_disp = bbox.x0 - 16  # small gap left of the title's left edge, in display (pixel) coords
        y_disp = (bbox.y0 + bbox.y1) / 2
        x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        fig.add_artist(Line2D([x_fig], [y_fig], marker=marker, color='black',
                               markersize=TITLE_MARKER_SIZE, markerfacecolor=mfc,
                               markeredgewidth=1.0, transform=fig.transFigure,
                               clip_on=False, zorder=6))


def make_figure():
    LEGEND_W_FRAC = 0.19  # fraction of total figure width reserved on the left for the legend
    wspace = 0.08

    fig = plt.figure(figsize=(FIG_W / (1 - LEGEND_W_FRAC), FIG_H))
    gs = fig.add_gridspec(1, 3, wspace=wspace)
    axes = draw(fig, gs)

    fig.subplots_adjust(left=LEGEND_W_FRAC + 0.02, right=0.99, top=0.90, bottom=0.28, wspace=wspace)
    _place_title_markers(fig, axes)

    handles = _build_legend_handles()
    leg = fig.legend(handles=handles, fontsize=FS_LEG, loc='center left',
               bbox_to_anchor=(0.01, 0.54), bbox_transform=fig.transFigure,
               title='Models', title_fontsize=FS_LEG,
               frameon=False,
               handlelength=1.5, labelspacing=0.4,
               borderpad=0.6, handletextpad=0.5)
    leg.get_title().set_fontweight('bold')

    OUT = FIGURES_DIR / 'fig5a_latency_vs_datasets.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    make_figure()
