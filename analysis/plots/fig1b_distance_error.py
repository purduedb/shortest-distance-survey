"""Square single-panel plot: x-axis = model, bars = P90 test MRE, line = P90 test MAE.

Usage:
    python fig1b_distance_error.py
    open fig1b_distance_error.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FIGURES_DIR,
    plot_label,
    MODEL_GROUP_OF,
    group_shades,
    vertical_group_spans,
    training_mean_std,
    tex_safe,
)

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE         = 1.0    # knob: local multiplier on plot_config's base font sizes
FS_AXIS, FS_TICK, FS_LEG = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG))
LOG_SCALE          = True   # knob: log-scale both y-axes (else linear)
SHOW_ERROR_BARS     = False  # knob: draw std-across-datasets error bars
SHOW_GROUP_SHADING  = False  # knob: pastel axvspan bands behind bars (redundant with bar color, off by default)

C_MAE = '#17A398'   # teal line -- distinct hue from all five METHOD_GROUP_COLORS (blue/orange/green/purple/red)

# Models to plot
MODEL_ORDER = [
    'Manhattan', 'Landmark_random_subset', 'Landmark_kmeans_subset',           # Baselines
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]
MODEL_LABELS = [plot_label(m) for m in MODEL_ORDER]

# Per-model bar color: same group hue family as the background shading, but
# each model within a group gets a distinct shade.
_groups_in_order = {}
for mk in MODEL_ORDER:
    _groups_in_order.setdefault(MODEL_GROUP_OF[mk], []).append(mk)

MODEL_COLOR = {}
for g, ms in _groups_in_order.items():
    for mk, c in zip(ms, group_shades(METHOD_GROUP_COLORS[g], len(ms))):
        MODEL_COLOR[mk] = c


mre_mean, mre_std, mae_mean, mae_std = [], [], [], []
for m in MODEL_ORDER:
    mm, ms = training_mean_std(m, lambda d: d.get('test_mre_percentiles', {}).get('p90'))
    mre_mean.append(mm); mre_std.append(ms)
    am, as_ = training_mean_std(m, lambda d: d.get('test_mae_percentiles', {}).get('p90'))
    mae_mean.append(am); mae_std.append(as_)

mre_mean = np.array(mre_mean); mre_std = np.array(mre_std)
mae_mean = np.array(mae_mean); mae_std = np.array(mae_std)

x = np.arange(len(MODEL_ORDER))
BAR_W = 0.55

FIG_W, FIG_H = 3.5, 3.1  # square, matches fig3b_storage.py

group_spans = vertical_group_spans(MODEL_ORDER)


def draw(fig, gs):
    """Draw the P90-MRE-bars / P90-MAE-line panel into a gridspec cell. Returns (ax_mre, ax_mae)."""
    ax1 = fig.add_subplot(gs)
    ax2 = ax1.twinx()
    # ax2 (line) must render on top of ax1 (bars)
    ax2.set_zorder(ax1.get_zorder() + 1)
    ax2.patch.set_visible(False)

    if SHOW_GROUP_SHADING:
        for gname, x0, x1 in group_spans:
            ax1.axvspan(x0, x1, color=METHOD_GROUP_COLORS[gname], alpha=0.07, zorder=0, linewidth=0)

    mask = ~np.isnan(mre_mean)
    bar_colors = [MODEL_COLOR[m] for m in np.array(MODEL_ORDER)[mask]]
    b1 = ax1.bar(x[mask], mre_mean[mask], BAR_W,
                 yerr=mre_std[mask] if SHOW_ERROR_BARS else None,
                 color=bar_colors, edgecolor='white', linewidth=0.4, capsize=2.5,
                 error_kw=dict(elinewidth=0.8, ecolor='#222222'),
                 zorder=3, label='Test MRE (p90)')

    mask2 = ~np.isnan(mae_mean)
    l1, = ax2.plot(x[mask2], mae_mean[mask2], color=C_MAE, marker='o', ms=4,
                    lw=1.6, zorder=4, label='Test MAE (p90)')
    if SHOW_ERROR_BARS:
        ax2.errorbar(x[mask2], mae_mean[mask2], yerr=mae_std[mask2], fmt='none',
                      ecolor=C_MAE, elinewidth=0.8, capsize=2.5, zorder=4)

    if LOG_SCALE:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    ax1.set_ylabel(tex_safe('Test MRE (%)'), fontsize=FS_AXIS)
    ax2.set_ylabel('Test MAE (km)', fontsize=FS_AXIS, color=C_MAE)
    ax1.tick_params(axis='y', labelsize=FS_TICK)
    ax2.tick_params(axis='y', labelcolor=C_MAE, labelsize=FS_TICK)
    # data stays in meters; ticks display km in power-of-10 form (10^0, 10^1, ...)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: ticker.LogFormatterSciNotation()(v / 1000)))

    ax1.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS, rotation=60, ha='right', fontsize=FS_TICK)

    ax1.grid(axis='y', which='major', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    return ax1, ax2


LEGEND_HANDLES = [
    Patch(facecolor='#888888', edgecolor='white', label='Test MRE (p90)'),
    Line2D([0], [0], color=C_MAE, marker='o', ms=4, lw=1.6, label='Test MAE (p90)'),
]


if __name__ == '__main__':
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 1)
    draw(fig, gs[0, 0])

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.82)  # headroom for suptitle + outside-top legend

    fig.suptitle('Distance Error (90th Percentile)', fontsize=FS_AXIS + 1, y=0.99)

    fig.legend(handles=LEGEND_HANDLES, fontsize=FS_LEG, loc='upper center',
               bbox_to_anchor=(0.5, 0.93), ncol=len(LEGEND_HANDLES),
               frameon=False, handlelength=1.5, labelspacing=0.3,
               borderpad=0.4, handletextpad=0.5, columnspacing=1.5)

    OUT = FIGURES_DIR / 'fig1b_distance_error.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'Saved {OUT}')
