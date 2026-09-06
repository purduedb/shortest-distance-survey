"""Meta plot: composes fig3a_latency_throughput.py + fig3b_storage.py side by side.

Usage:
    python fig3_efficiency.py
    open fig3_efficiency.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import FS_LEG, FIGURES_DIR
import fig3a_latency_throughput as lt
import fig3b_storage as sto

# ── Tunable display flags ───────────────────────────────────────────────────
LEGEND_TOP = 0.88  # knob: axes top margin (subplots_adjust `top`) -- lower = more gap under the legend
FONT_SCALE = 1.1   # knob: multiplier on this file's legend text AND lt/sto's FS_* (on top of their own FONT_SCALE)
FS_LEG = FS_LEG * FONT_SCALE
lt.FS_AXIS *= FONT_SCALE
lt.FS_TICK *= FONT_SCALE
lt.FS_LEG  *= FONT_SCALE
lt.FS_LABEL *= FONT_SCALE
sto.FS_AXIS *= FONT_SCALE
sto.FS_TICK *= FONT_SCALE
sto.FS_LEG  *= FONT_SCALE
sto.FS_ANN  *= FONT_SCALE

FIG_H = lt.FIG_H
FIG_W = lt.FIG_W + sto.FIG_W
WSPACE = 0.08  # same tight gap on both sides of the throughput|storage boundary

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, sto.FIG_W / (lt.FIG_W / 2)], wspace=WSPACE)

gs_lt = gs[0, 0:2].subgridspec(1, 2, wspace=WSPACE)
lt.draw(fig, gs_lt)
sto.draw(fig, gs[0, 2], y_labels_right=True)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(top=LEGEND_TOP, wspace=WSPACE)

fig.legend(handles=lt.LEGEND_HANDLES, fontsize=FS_LEG, loc='upper center',
           bbox_to_anchor=(0.32, lt.LEGEND_Y), ncol=len(lt.LEGEND_HANDLES),
           frameon=False, handlelength=1.2, labelspacing=0.3,
           borderpad=0.4, handletextpad=0.5, columnspacing=1.5)

OUT = FIGURES_DIR / 'fig3_efficiency.pdf'
fig.savefig(OUT, bbox_inches='tight', dpi=200)
print(f'Saved {OUT}')
