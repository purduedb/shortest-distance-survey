"""Square single-panel plot: index size vs graph size, one band per model family.

Usage:
    python fig3b_storage.py
    open fig3b_storage.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import (
    METHOD_GROUP_COLORS,
    FS_AXIS,
    FS_TICK,
    FS_LEG,
    FS_ANN,
    FIGURES_DIR,
    LW,
    MS,
    training,
    tex_safe,
)
from extract_metrics import get_key, METHODS, DATASETS

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE = 1.0   # knob: local multiplier on plot_config's base font sizes
FS_AXIS, FS_TICK, FS_LEG, FS_ANN = (f * FONT_SCALE for f in (FS_AXIS, FS_TICK, FS_LEG, FS_ANN))
LEGEND_LABELSPACING = 0.12  # knob: vertical gap between the in-axes legend's entries

C_COORD = METHOD_GROUP_COLORS['Baselines']
C_GNN   = METHOD_GROUP_COLORS['GNNs']
C_NN    = METHOD_GROUP_COLORS['NNs']
C_FUNC  = METHOD_GROUP_COLORS['Functional']
C_TREE  = METHOD_GROUP_COLORS['Tree']

ds_nodes = np.array([meta['nodes'] for meta in DATASETS.values()])

# Extract model size data from METRICS (keyed by '<model>_<dataset>_<query_dir>')
plot_data = {}
for model_key, model_value in METHODS.items():
    vals = []
    for data_key in DATASETS:
        metrics = training(get_key(model_key, data_key))
        vals.append(metrics.get('model_size') if metrics else np.nan)
    plot_data[model_value['label']] = np.array(vals, dtype=float)

def band(models):
    s = np.stack([plot_data[METHODS[m]['label']] for m in models])
    return np.nanmedian(s, axis=0), np.nanmin(s, axis=0), np.nanmax(s, axis=0)

coord_med, coord_lo, coord_hi = band(['Manhattan', 'GeoDNN'])
gnn_med,   gnn_lo,   gnn_hi   = band(['GCN', 'SAGE', 'GAT'])
nn_med,    nn_lo,    nn_hi    = band(['DistanceNN_sub', 'EmbeddingNN_mean', 'Vdist2vec', 'Ndist2vec'])
func_med,  func_lo,  func_hi  = band(['Path2vec', 'ANEDA', 'RNE', 'Landmark_random_subset', 'Landmark_kmeans_subset'])
landmarknn = plot_data[METHODS['CatBoostNN']['label']]
catboost   = plot_data[METHODS['CatBoost']['label']]

x = ds_nodes
mask_nn   = ~np.isnan(nn_med)
mask_func = ~np.isnan(func_med)

FIG_W, FIG_H = 3.5, 3.1

ANN_GAP = 0.9  # gap between line and label
ANN_X_SHIFT = 2.1   # anchor x multiplier — increase to push labels right
ANN_IDX = 11  # index of x-coord to label


def line_angle(ax, xa, ya, xb, yb):
    pa = ax.transData.transform((xa, ya))
    pb = ax.transData.transform((xb, yb))
    return np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))


def label_on_line(ax, xarr, yarr, idx, text, color, above=True, fs=FS_ANN, angle=None):
    i0 = max(0, idx - 1)
    i1 = min(len(xarr) - 1, idx + 1)
    if angle is None:
        angle = line_angle(ax, xarr[i0], yarr[i0], xarr[i1], yarr[i1])
    factor = ANN_GAP if above else (1.0 / ANN_GAP)
    ax.text(xarr[idx] * ANN_X_SHIFT, yarr[idx] * factor, text,
            fontsize=fs, color=color, rotation=angle,
            rotation_mode='anchor', ha='right', va='center', zorder=6)


def draw(fig, gs, y_labels_right=False):
    """Draw the index-size-vs-graph-size panel into a gridspec cell. Returns the Axes."""
    ax = fig.add_subplot(gs)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.fill_between(x, coord_lo, coord_hi, color=C_COORD, alpha=0.15)
    l_coord, = ax.plot(x, coord_med, color=C_COORD, lw=LW, marker='o', ms=MS, zorder=4,
                  label='Manhattan, GeoDNN')

    ax.fill_between(x, gnn_lo, gnn_hi, color=C_GNN, alpha=0.15)
    l_gnn, = ax.plot(x, gnn_med, color=C_GNN, lw=LW, marker='o', ms=MS, zorder=4,
                  label='GNN')

    ax.fill_between(x[mask_nn],   nn_lo[mask_nn],   nn_hi[mask_nn],   color=C_NN,   alpha=0.15)
    l_nn, = ax.plot(x[mask_nn], nn_med[mask_nn], color=C_NN, lw=LW, marker='o', ms=MS, zorder=4,
                  label='NN')
    ax.fill_between(x[mask_func], func_lo[mask_func], func_hi[mask_func], color=C_FUNC, alpha=0.15)
    l_func, = ax.plot(x[mask_func], func_med[mask_func], color=C_FUNC, lw=LW, marker='o', ms=MS, zorder=4,
                  label='Functional, Landmark$_{rn,km}$')

    l_landmarknn, = ax.plot(x, landmarknn, color=C_NN, lw=LW - 0.4, marker='o', ms=MS, zorder=4,
                  linestyle=(0, (2, 2)), label='LandmarkNN')

    l_catboost, = ax.plot(x, catboost, color=C_TREE, lw=LW, marker='o', ms=MS, zorder=4,
                  label='Tree-based (CatBoost)')

    # ── Axis setup ────────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f'{v/1e6:.0f}M' if v >= 1e6 else (f'{v/1e3:.0f}K' if v >= 1e3 else f'{v:.0f}')
    ))
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10)))
    ax.tick_params(axis='x', which='minor', length=2)
    ax.tick_params(axis='x', labelsize=FS_TICK)

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    key_ds = {DATASETS[dk]['label']: DATASETS[dk]['nodes'] for dk in ['Surat', 'W_Jinan', 'W_Shanghai', 'W_NewYork', 'FLA', 'W', 'USA']}
    ax2.set_xticks(list(key_ds.values()))
    ax2.set_xticklabels(list(key_ds.keys()), fontsize=FS_TICK)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    ax2.tick_params(axis='x', length=3)

    def _fmt_mb(v, _pos=None):
        if v >= 1000:
            return f'{v/1000:g} GB'
        if v >= 1:
            return f'{v:g} MB'
        return f'{v*1000:g} KB'

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_mb))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.1, 1, 10, 100, 1000, 10000]))  # 10KB tick added at 0.01
    ax.set_ylim(bottom=0.008)  # lowered slightly so the 10KB tick is visible below the data's min (~30KB)
    if y_labels_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.spines['top'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_xlabel(tex_safe('Graph Size (#nodes)'), fontsize=FS_AXIS)
    ax.set_ylabel('Index Size', fontsize=FS_AXIS)

    ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

    ax.legend(handles=[l_coord, l_nn, l_landmarknn, l_gnn, l_func, l_catboost], fontsize=FS_LEG, loc='lower right',
              framealpha=0.88, handlelength=1.8, labelspacing=LEGEND_LABELSPACING,
              borderpad=0.4, handletextpad=0.5)

    # ── On-line angled labels ────────────────────────────────────────────
    fig.canvas.draw()
    idx = min(ANN_IDX, len(x) - 1)
    shared_angle = line_angle(ax, x[idx - 1], coord_med[idx - 1],
                               x[min(idx + 1, len(x) - 1)], coord_med[min(idx + 1, len(x) - 1)])
    label_on_line(ax, x, coord_med,          idx, 'coords (d=2)',   C_COORD, above=False, fs=FS_ANN, angle=shared_angle)
    label_on_line(ax, x, gnn_med,            idx, 'coords+edges',   C_GNN,   above=False, fs=FS_ANN, angle=shared_angle)
    label_on_line(ax, x[mask_nn], nn_med[mask_nn], idx, 'embeds (d=64)', C_NN, above=False, fs=FS_ANN, angle=shared_angle)

    return ax


if __name__ == '__main__':
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 1)
    draw(fig, gs[0, 0])

    fig.tight_layout(pad=0.3)

    OUT = FIGURES_DIR / 'fig3b_storage.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'Saved {OUT}')
