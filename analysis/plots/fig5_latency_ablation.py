"""Meta plot: composes fig5a_latency_vs_datasets.py + fig5b_latency_vs_bs.py stacked vertically.

Usage:
    python fig5_latency_ablation.py
    open fig5_latency_ablation.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/

from plot_config import FIGURES_DIR
import fig5a_latency_vs_datasets as vs_dataset
import fig5b_latency_vs_bs as vs_bs

# ── Tunable display flags ───────────────────────────────────────────────────
LEGEND_W_FRAC = 0.22  # fraction of total figure width reserved on the left for the legend (matches fig5a/fig5b)
wspace = 0.08
hspace = 0.35  # vertical gap between the two rows (extra room for the bottom row's x-axis labels/title)
HEIGHT_SCALE = 0.85  # knob: shrink combined figure height relative to fig5a+fig5b's native heights

FIG_H = (vs_dataset.FIG_H + vs_bs.FIG_H) * HEIGHT_SCALE
FIG_W = vs_dataset.FIG_W

fig = plt.figure(figsize=(FIG_W / (1 - LEGEND_W_FRAC), FIG_H))
gs = fig.add_gridspec(2, 1, hspace=hspace)

gs_top = gs[0].subgridspec(1, 3, wspace=wspace)
gs_bottom = gs[1].subgridspec(1, 3, wspace=wspace)
axes_top = vs_dataset.draw(fig, gs_top)
axes_bottom = vs_bs.draw(fig, gs_bottom)

fig.subplots_adjust(left=LEGEND_W_FRAC + 0.04, right=0.99, top=0.95, bottom=0.08, wspace=wspace, hspace=hspace)
vs_dataset._place_title_markers(fig, axes_top)
vs_bs._place_title_markers(fig, axes_bottom)

# Override the default legend fontsize to match the panels' y-axis label size (not plot_config's FS_LEG).
handles = vs_dataset._build_legend_handles()
leg = fig.legend(handles=handles, fontsize=vs_dataset.FS_AXIS, loc='center left',
           bbox_to_anchor=(0.01, 0.5), bbox_transform=fig.transFigure,
           title='Models', title_fontsize=vs_dataset.FS_AXIS,
           frameon=False,
           handlelength=1.5, labelspacing=0.4,
           borderpad=0.6, handletextpad=0.5)
leg.get_title().set_fontweight('bold')

OUT = FIGURES_DIR / 'fig5_latency_ablation.pdf'
fig.savefig(OUT, bbox_inches='tight', dpi=300)
print(f'Saved {OUT}')
