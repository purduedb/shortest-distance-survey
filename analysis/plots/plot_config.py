# ── Shared plotting config ────────────────────────────────────────────────────
# Import this in all plot scripts for consistency.
import sys
import os
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/

from extract_metrics import (
    METHODS, METHOD_GROUPS, DATASETS, get_key,
    training, inference, dataloader_metric, batch_sizes_seen,
)

FIGURES_DIR = Path(__file__).parent

# ── METRICS lookup helpers ───────────────────────────────────────────────────
def training_mean_std(model_key, field_fn):
    """Mean/std of field_fn(training_metrics) across all datasets for one model."""
    vals = []
    for dk in DATASETS:
        d = training(get_key(model_key, dk))
        if d is None:
            continue
        v = field_fn(d)
        if v is not None:
            vals.append(v)
    if not vals:
        return np.nan, 0.0
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def inference_mean_std(device, model_key, batch_size, field):
    """Mean/std of a dataloader field across all datasets, for one (model, bs) on device."""
    vals = []
    for dk in DATASETS:
        d = inference(device)(get_key(model_key, dk))
        v = dataloader_metric(d, batch_size, field) if d else None
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


# ── Per-model-group shading/color helpers for bar/line plots ────────────────
MODEL_GROUP_OF = {m: g for g, models in METHOD_GROUPS.items() for m in models}

def group_shades(base_hex, n):
    """n distinct shades of base_hex (lighter to slightly darker), for a group's rows."""
    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    if n == 1:
        return [base]
    # spread from lighter to slightly darker than base, base sits mid-ramp
    fracs = np.linspace(0.55, -0.25, n)
    out = []
    for f in fracs:
        if f < 0:
            out.append(base * (1 + f) + np.array([0, 0, 0]) * (-f))
        else:
            out.append(base * (1 - f) + white * f)
    return out


def vertical_group_spans(model_order):
    """(group_name, x_start, x_end) axvspan bounds for each contiguous group run in model_order."""
    spans = []
    start_i = 0
    last_group = MODEL_GROUP_OF[model_order[0]]
    for i, m in enumerate(model_order):
        g = MODEL_GROUP_OF[m]
        if g != last_group:
            spans.append((last_group, start_i - 0.5, i - 0.5))
            start_i = i
            last_group = g
    spans.append((last_group, start_i - 0.5, len(model_order) - 0.5))
    return spans

# ── Text helpers (safe under both usetex and non-usetex font styles) ────────
# LaTeX treats # $ % & _ { } ~ ^ \ as special even in text mode; usetex=True crashes or mis-renders
_TEX_SPECIAL = {
    '\\': r'\textbackslash{}',
    '#': r'\#', '$': r'\$', '%': r'\%', '&': r'\&',
    '_': r'\_', '{': r'\{', '}': r'\}',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
}

def tex_safe(text):
    """Escape LaTeX-special characters in `text` when usetex is active; no-op otherwise."""
    if not matplotlib.rcParams['text.usetex']:
        return text
    return ''.join(_TEX_SPECIAL.get(c, c) for c in text)

# ── LaTeX-flavored method labels ─────────────────────────────────────────────
LATEX_LABEL_OVERRIDES = {
    'Landmark_random_subset': 'Landmark$_{rn}$',
    'Landmark_kmeans_subset': 'Landmark$_{km}$',
}

def plot_label(model_key):
    return LATEX_LABEL_OVERRIDES.get(model_key, METHODS[model_key]['label'])

# ── Colors ────────────────────────────────────────────────────────────────────
METHOD_GROUP_COLORS = {
    'Baselines':  '#4E79A7',   # blue
    'NNs':        '#F28E2B',   # orange
    'GNNs':       '#59A14F',   # green
    'Functional': '#B07AA1',   # purple
    'Tree':       '#E15759',   # red
}

# ── Font sizes ────────────────────────────────────────────────────────────────
# Base sizes for all figures.
FS_SCALE = 1.2
FS_AXIS  = 9*FS_SCALE
FS_TICK  = 7.5*FS_SCALE
FS_LABEL = 7.5*FS_SCALE
FS_LEG   = 7*FS_SCALE
FS_ANN   = 7.5*FS_SCALE

# ── Figure dimensions & line/marker defaults ──────────────────────────────────
FIG_W = 3.8   # column width
LW    = 1.6   # default line width
MS    = 3     # default marker size

# ── Font presets ──────────────────────────────────────────────────────────────
# Call set_font(style) once at the top of each plot script.
#   'default'   — matplotlib sans-serif text; CM mathtext for $...$
#   'cm'        — usetex, full Computer Modern (text + math via LaTeX)
#   'libertine' — usetex, Linux Libertine text + newtxmath (matches ACM SIGCONF)
def set_font(style='default'):
    rc = matplotlib.rcParams
    if style == 'libertine':
        rc['text.usetex'] = True
        rc['text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}'
        rc['font.family'] = 'serif'
    elif style == 'cm':
        rc['text.usetex'] = True
        rc['font.family'] = 'serif'
    elif style == 'default':
        rc['text.usetex'] = False
        rc['font.family'] = 'sans-serif'
        rc['mathtext.fontset'] = 'cm'   # CM only for $...$ math expressions
        rc['mathtext.rm']      = 'serif'
    else:
        raise ValueError(f"Unknown font style '{style}'")

set_font('libertine')  # change to 'libertine' for publication-quality ACM SIGCONF style (requires LaTeX installed)
