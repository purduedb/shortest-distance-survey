"""3-column learning-curve panel (Surat, W_Shanghai, W): validation MRE vs training progress, one line per model.

Usage:
    python fig1a_learning_curves.py
    open fig1a_learning_curves_time.pdf fig1a_learning_curves_epoch.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FIGURES_DIR,
    plot_label,
    training,
    MODEL_GROUP_OF,
    group_shades,
    tex_safe,
)
from extract_metrics import get_key, METHOD_GROUPS

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE          = 1.0    # knob: local multiplier on plot_config's base font sizes (this fig's labels are crowded)
FS_AXIS, FS_TICK, FS_LEG = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG))
LOG_SCALE           = True   # knob: log-scale the y-axis (validation MRE) on all panels
X_AXIS_MODE         = 'time'  # knob: 'time' (wall-clock minutes) or 'epoch' (epoch index)
SHOW_BEST_BASELINE  = True   # knob: dashed black hline at the best (lowest) baseline's test MRE per panel
SHARE_Y_AXIS        = False   # knob: share the y-axis (validation MRE scale) across all 3 panels

X_FIELD = {'time': 'time_elapsed_history', 'epoch': 'epoch_history'}
X_LABEL = {'time': 'Precomputation Time (min)', 'epoch': 'Epoch'}

FIG_H = 3.1  # matches fig1b_distance_error.py's FIG_H, for side-by-side stacking
PANEL_W = 2.9  # tuned so each panel's plotted axes box (after title/label margins) reads square
FIG_W = PANEL_W * 3

# Datasets to plot, left to right.
PANEL_DATASETS = ['Surat', 'W_Shanghai', 'W']
PANEL_TITLES = {
    'Surat':      'Surat (SU)',
    'W_Shanghai': 'Shanghai (SH)',
    'W':          'Western USA (W)',
}

# Models to plot -- excludes Manhattan/Landmark_* (no epoch/val_mre history).
MODEL_ORDER = [
    'GeoDNN', 'DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec', 'CatBoostNN',  # NNs
    'GCN', 'SAGE', 'GAT',                                                      # GNNs
    'Path2vec', 'ANEDA', 'RNE',                                                # Functional
    'CatBoost',                                                                # Tree
]

# Per-model line color: same group hue family as bars/shading elsewhere, but
# each model within a group gets a distinct shade so overlapping lines stay
# distinguishable (tab10-style lightness ramp within each group's hue).
_groups_order = {}
for mk in MODEL_ORDER:
    _groups_order.setdefault(MODEL_GROUP_OF[mk], []).append(mk)

MODEL_COLOR = {}
for g, ms in _groups_order.items():
    for mk, c in zip(ms, group_shades(METHOD_GROUP_COLORS[g], len(ms))):
        MODEL_COLOR[mk] = c


def best_baseline_mre(dk):
    """Lowest test_mre (%) across the baseline models for one dataset."""
    vals = []
    for mk in METHOD_GROUPS['Baselines']:
        d = training(get_key(mk, dk))
        if d is not None:
            v = d.get('test_mre')
            if v is not None:
                vals.append(v)
    return min(vals) if vals else None


def draw(fig, gs, x_axis_mode=None):
    """Draw the 3-panel learning-curve grid into a gridspec slice. Returns the list of Axes."""
    x_axis_mode = x_axis_mode or X_AXIS_MODE
    x_field = X_FIELD[x_axis_mode]

    axes = []
    ax_prev = None
    for i, dk in enumerate(PANEL_DATASETS):
        ax = fig.add_subplot(gs[i], sharey=ax_prev if SHARE_Y_AXIS else None)
        axes.append(ax)
        ax_prev = ax if SHARE_Y_AXIS else None

        for mk in MODEL_ORDER:
            d = training(get_key(mk, dk))
            if d is None:
                continue
            xvals = d.get(x_field)
            val_mre = d.get('val_mre_history')
            if not xvals or not val_mre:
                continue
            ax.plot(xvals, val_mre, color=MODEL_COLOR[mk], lw=1.3,
                    label=plot_label(mk), zorder=3)

        if SHOW_BEST_BASELINE:
            baseline = best_baseline_mre(dk)
            if baseline is not None:
                ax.axhline(baseline, color='black', linestyle='--', lw=1.1,
                           zorder=5, label='Best Baseline')

        if LOG_SCALE:
            ax.set_yscale('log')
        ax.set_xlabel(X_LABEL[x_axis_mode], fontsize=FS_AXIS)
        ax.set_title(PANEL_TITLES[dk], fontsize=FS_AXIS, pad=6)
        ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax.tick_params(axis='both', labelsize=FS_TICK)

        if i == 0:
            ax.set_ylabel(tex_safe('Validation MRE (%)'), fontsize=FS_AXIS)
        elif SHARE_Y_AXIS:
            plt.setp(ax.get_yticklabels(), visible=False)

    return axes


def _build_legend_handles():
    from matplotlib.lines import Line2D
    handles = []
    for g, ms in _groups_order.items():
        for mk in ms:
            handles.append(Line2D([0], [0], color=MODEL_COLOR[mk], lw=1.6, label=plot_label(mk)))
    if SHOW_BEST_BASELINE:
        handles.append(Line2D([0], [0], color='black', linestyle='--', lw=1.1, label='Best Baseline'))
    return handles


def make_figure(x_axis_mode):
    LEGEND_W_FRAC = 0.19  # fraction of total figure width reserved on the left for the legend
    wspace = 0.05 if SHARE_Y_AXIS else 0.22  # extra room for each panel's own y-tick labels when unshared

    fig = plt.figure(figsize=(FIG_W / (1 - LEGEND_W_FRAC), FIG_H))
    gs = fig.add_gridspec(1, 3, wspace=wspace)
    draw(fig, gs, x_axis_mode=x_axis_mode)

    fig.subplots_adjust(left=LEGEND_W_FRAC + 0.02, right=0.99, top=0.90, bottom=0.18, wspace=wspace)

    handles = _build_legend_handles()
    leg = fig.legend(handles=handles, fontsize=FS_LEG, loc='center left',
               bbox_to_anchor=(0.01, 0.5), bbox_transform=fig.transFigure,
               title='Models', title_fontsize=FS_LEG,
               frameon=True, framealpha=0.95, edgecolor='#888888',
               handlelength=1.5, labelspacing=0.4,
               borderpad=0.6, handletextpad=0.5)
    leg.get_title().set_fontweight('bold')

    OUT = FIGURES_DIR / f'fig1a_learning_curves_{x_axis_mode}.pdf'
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    # Always generate both x-axis variants -- X_AXIS_MODE only controls the
    # default used by draw() when composed into another figure.
    make_figure('time')
    make_figure('epoch')
