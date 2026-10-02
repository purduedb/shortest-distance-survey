"""Square single-panel plot: x-axis = model, bars = per-epoch/sample time, line = avg epochs reached.

Usage:
    python fig4_complexity.py
    open fig4_complexity.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FIGURES_DIR,
    plot_label,
    vertical_group_spans,
    training_mean_std,
)

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE      = 1.0    # knob: local multiplier on plot_config's base font sizes
FS_AXIS, FS_TICK, FS_LEG = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG))
LOG_SCALE       = False  # knob: log-scale both y-axes (else linear)
SHOW_ERROR_BARS = False  # knob: draw std-across-datasets error bars

C_TIME  = '#4D4D4D'  # gray/black bars, matches fig1b_distance_error.py's MRE bars
C_EPOCH = '#B0402B'  # line, matches fig1b_distance_error.py's MAE line color

# Models to plot, explicit and in x-axis order (matches other per-model plots
# in this dir -- excludes Euclidean).
MODEL_ORDER = [
    'Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset',           # Baselines
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]
MODEL_LABELS = [plot_label(m) for m in MODEL_ORDER]

time_mean, time_std, epoch_mean, epoch_std = [], [], [], []
for m in MODEL_ORDER:
    tm, ts = training_mean_std(m, lambda d: d.get('per_epoch_time_per_sample_sec'))
    time_mean.append(tm * 1e6 if not np.isnan(tm) else np.nan); time_std.append(ts * 1e6)  # -> microseconds
    em, es = training_mean_std(m, lambda d: d.get('last_train_epoch'))
    epoch_mean.append(em); epoch_std.append(es)

time_mean = np.array(time_mean); time_std = np.array(time_std)
epoch_mean = np.array(epoch_mean); epoch_std = np.array(epoch_std)

x = np.arange(len(MODEL_ORDER))
BAR_W = 0.55

FIG_W, FIG_H = 3.5, 3.1  # square, matches fig3b_storage.py

group_spans = vertical_group_spans(MODEL_ORDER)


def draw(fig, gs):
    """Draw the time/epoch/sample-bars / epochs-reached-line panel. Returns (ax_time, ax_epoch)."""
    ax1 = fig.add_subplot(gs)
    ax2 = ax1.twinx()

    for gname, x0, x1 in group_spans:
        ax1.axvspan(x0, x1, color=METHOD_GROUP_COLORS[gname], alpha=0.07, zorder=0, linewidth=0)

    mask = ~np.isnan(time_mean)
    b1 = ax1.bar(x[mask], time_mean[mask], BAR_W,
                 yerr=time_std[mask] if SHOW_ERROR_BARS else None,
                 color=C_TIME, edgecolor='white', linewidth=0.4, capsize=2.5,
                 error_kw=dict(elinewidth=0.8, ecolor='#222222'),
                 zorder=3, label='Epoch Time per Sample')

    mask2 = ~np.isnan(epoch_mean)
    l1, = ax2.plot(x[mask2], epoch_mean[mask2], color=C_EPOCH, marker='o', ms=4,
                    lw=1.6, zorder=4, label='Avg Epochs Reached')
    if SHOW_ERROR_BARS:
        ax2.errorbar(x[mask2], epoch_mean[mask2], yerr=epoch_std[mask2], fmt='none',
                      ecolor=C_EPOCH, elinewidth=0.8, capsize=2.5, zorder=4)

    if LOG_SCALE:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    ax1.set_ylabel('Time (µs)', fontsize=FS_AXIS, color=C_TIME)
    ax2.set_ylabel('Epochs', fontsize=FS_AXIS, color=C_EPOCH)
    ax1.tick_params(axis='y', labelcolor=C_TIME, labelsize=FS_TICK)
    ax2.tick_params(axis='y', labelcolor=C_EPOCH, labelsize=FS_TICK)

    ax1.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS, rotation=60, ha='right', fontsize=FS_TICK)

    ax1.grid(axis='y', which='major', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax2.spines['top'].set_visible(True)  # twinx draws the shared box; keep it closed on top

    return ax1, ax2


if __name__ == '__main__':
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 1)
    ax1, ax2 = draw(fig, gs[0, 0])

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.82)  # headroom for outside-top legend

    handles = [
        Patch(facecolor=C_TIME, edgecolor='white', label='Epoch Time per Sample'),
        Line2D([0], [0], color=C_EPOCH, marker='o', ms=4, lw=1.6, label='Avg Epochs Reached'),
    ]
    fig.legend(handles=handles, fontsize=FS_LEG, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=len(handles),
               frameon=False, handlelength=1.5, labelspacing=0.3,
               borderpad=0.4, handletextpad=0.5, columnspacing=1.5)

    OUT = FIGURES_DIR / 'fig4_complexity.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'Saved {OUT}')
