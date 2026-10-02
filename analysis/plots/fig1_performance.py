"""Meta plot: composes fig1a_learning_curves.py + fig1b_distance_error.py side by side.

Usage:
    python fig1_performance.py
    open fig1_performance_time.pdf fig1_performance_epoch.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> v7_camera_expts/

from plot_config import FS_TICK, FS_AXIS, FIGURES_DIR
import fig1b_distance_error as de
import fig1a_learning_curves as lc

FIG_H = de.FIG_H  # == lc.FIG_H

# ── Tunable display flags ───────────────────────────────────────────────────
FONT_SCALE           = 1.5  # knob: multiplier on this file's legend/title text AND lc/de's FS_* (on top of their own FONT_SCALE)
FS_TICK, FS_AXIS     = (f * FONT_SCALE for f in (FS_TICK, FS_AXIS))
for _mod in (lc, de):
    _mod.FS_AXIS *= FONT_SCALE
    _mod.FS_TICK *= FONT_SCALE
    _mod.FS_LEG  *= FONT_SCALE
LEGEND_LABELSPACING  = 0.3  # knob: vertical gap between legend entries (fig.legend's labelspacing)
LEGEND_GAP           = 0.5  # knob: extra inches between the legend column and the first panel's y-label


def make_figure(x_axis_mode):
    LEGEND_W = 2.1  # inches reserved on the far left for the unboxed legend column
    fig_w = LEGEND_W + LEGEND_GAP + lc.FIG_W + de.FIG_W + 0.3
    fig = plt.figure(figsize=(fig_w, FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[lc.FIG_W, de.FIG_W], wspace=0.11,
                           left=(LEGEND_W + LEGEND_GAP) / fig_w, right=0.99)

    gs_lc = gs[0, 0].subgridspec(1, 3, wspace=0.05 if lc.SHARE_Y_AXIS else 0.16)
    lc.draw(fig, gs_lc, x_axis_mode=x_axis_mode)

    ax_de, _ = de.draw(fig, gs[0, 1])
    ax_de.set_title('Distance Error (90th Percentile)', fontsize=FS_AXIS, pad=6)

    fig.subplots_adjust(bottom=0.18, top=0.90)

    # Top-aligned to the axes
    handles = lc._build_legend_handles() + de.LEGEND_HANDLES
    fig.legend(handles=handles, fontsize=FS_TICK, loc='upper left',
               bbox_to_anchor=(0.01, 1.02), bbox_transform=fig.transFigure,
               frameon=False, handlelength=1.5, labelspacing=LEGEND_LABELSPACING,
               borderpad=0.4, handletextpad=0.5)

    OUT = FIGURES_DIR / f'fig1_performance_{x_axis_mode}.pdf'
    fig.savefig(OUT, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    make_figure('time')
    make_figure('epoch')
