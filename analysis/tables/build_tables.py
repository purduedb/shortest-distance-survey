"""Builds markdown tables of metrics extracted from the training/inference runs.

Usage:
    python build_tables.py
    open all_tables.txt
"""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/
from extract_metrics import (
    METHODS, METHOD_GROUPS, DATASETS, DATASET_GROUPS,
    TRAINING_METRICS, INFERENCE_METRICS,
    get_key, batch_sizes_seen, training, inference,
)

################
# Table layout config
################

METHOD_GROUP_LABELS = {
    'Baselines':  'Baselines',
    'NNs':        'Neural Network Methods',
    'GNNs':       'Graph Neural Network Methods',
    'Functional': 'Functional Methods',
    'Tree':       'Tree-based Gradient Boosting Methods',
}

DATASET_GROUP_LABELS = {
    'AllPairs': 'All',
    'Workload': 'Workload',
    'DIMACS':   'DIMACS',
}

# Column config
DATASET_COLS = [(dk, meta['label']) for dk, meta in DATASETS.items()]
SW_COLS, DIMACS_COLS = DATASET_COLS[:8], DATASET_COLS[8:]  # SU + 7 workload | 5 dimacs
MODEL_W = 17
SW_LABEL_W = max(len(l) for _, l in SW_COLS)
DIMACS_LABEL_W = max(len(l) for _, l in DIMACS_COLS)

################
# Helpers
################

pad_left = lambda s, w: s.rjust(w)

def avgstd(vals, decimals):
    """'mean±std' across a row's values, or a single value / dash if <2."""
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.{decimals}f}'
    return f'{statistics.mean(vals):.{decimals}f}±{statistics.pstdev(vals):.{decimals}f}'


def max_col_width(get_value_fn, fmt_fn, cols, label_w, src):
    """Widest formatted cell across all models/datasets in `cols` (>= label width)."""
    w = label_w
    for models in METHOD_GROUPS.values():
        for m in models:
            for dk, _ in cols:
                v = get_value_fn(src(get_key(m, dk)))
                w = max(w, len(fmt_fn(v)))
    return w

################
# Table builder
################

def build_table(get_value_fn, fmt_fn, decimals, src=training, sw_w=None, dimacs_w=None):
    """Renders one markdown table: Model | SU | 7 Workload cols | 5 DIMACS cols | avg ± std."""
    if sw_w is None:
        sw_w = max_col_width(get_value_fn, fmt_fn, SW_COLS, SW_LABEL_W, src)
    if dimacs_w is None:
        dimacs_w = max_col_width(get_value_fn, fmt_fn, DIMACS_COLS, DIMACS_LABEL_W, src)

    def row_values(m):
        """(formatted cells, raw values) across all dataset columns for model m."""
        fmt_vals, raw_vals = [], []
        for dk, _ in DATASET_COLS:
            v = get_value_fn(src(get_key(m, dk)))
            fmt_vals.append(fmt_fn(v))
            if v is not None:
                raw_vals.append(v)
        return fmt_vals, raw_vals

    # avg±std column width needs every row's string precomputed first
    all_rows = [(m, *row_values(m)) for models in METHOD_GROUPS.values() for m in models]
    extra_header = 'avg ± std'
    extra_w = max(len(extra_header), *(len(avgstd(raw, decimals)) for _, _, raw in all_rows))

    lines = []
    sw_labels = [l for _, l in SW_COLS]
    dimacs_labels = [l for _, l in DIMACS_COLS]
    header = ('| ' + 'Model'.ljust(MODEL_W) + ' | '
              + ' | '.join(pad_left(l, sw_w) for l in sw_labels) + ' | '
              + ' | '.join(pad_left(l, dimacs_w) for l in dimacs_labels) + ' |'
              + ' ' + pad_left(extra_header, extra_w) + ' |')
    table_width = len(header)
    row_divider = '|' + '-' * (table_width - 2) + '|'

    # dataset-group headers ("All" / "Workload" / "DIMACS")
    pipe_pos = [i for i, c in enumerate(header) if c == '|']
    col_count = 1  # pipe_pos index of the Model|<dataset cols> boundary
    boundaries = [pipe_pos[col_count]]
    for group_models in DATASET_GROUPS.values():
        col_count += len(group_models)
        boundaries.append(pipe_pos[col_count])

    span = lambda text, a, b: text.center(b - a - 1)
    group_hdr = list(' ' * table_width)
    group_hdr[0] = '|'
    group_hdr[pipe_pos[-1]] = '|'  # closes the trailing avg±std column
    for p in boundaries:
        group_hdr[p] = '|'
    for i, gname in enumerate(DATASET_GROUPS):
        a, b = boundaries[i], boundaries[i + 1]
        for j, c in enumerate(span(DATASET_GROUP_LABELS[gname], a, b)):
            group_hdr[a + 1 + j] = c
    lines.append(row_divider)
    lines.append(''.join(group_hdr))
    lines.append(header)

    sep = ('|:' + '-' * (MODEL_W + 1) + '|'
           + '|'.join('-' * (sw_w + 1) + ':' for _ in sw_labels) + '|'
           + '|'.join('-' * (dimacs_w + 1) + ':' for _ in dimacs_labels) + '|'
           + '-' * (extra_w + 1) + ':|')
    lines.append(sep)

    for gi, (gname, models) in enumerate(METHOD_GROUPS.items()):
        if gi > 0:
            lines.append(row_divider)
        group_label = f'**{METHOD_GROUP_LABELS[gname]}**'
        lines.append('|' + group_label.center(table_width - 2) + '|')
        for m, fmt_vals, raw_vals in all_rows:
            if m not in models:
                continue
            sw_cells = [pad_left(v, sw_w) for v in fmt_vals[:8]]
            dimacs_cells = [pad_left(v, dimacs_w) for v in fmt_vals[8:]]
            row = ('| ' + METHODS[m]['label'].ljust(MODEL_W) + ' | ' + ' | '.join(sw_cells)
                   + ' | ' + ' | '.join(dimacs_cells) + ' | ' + pad_left(avgstd(raw_vals, decimals), extra_w) + ' |')
            lines.append(row)
    lines.append(row_divider)
    return '\n'.join(lines)

################
# Format helpers
################

fmt = lambda v, decimals: f'{v:.{decimals}f}' if v is not None else '—'
int_fmt = lambda v: f'{int(v)}' if v is not None else '—'

################
# Build sections: (section_name, key, desc, table_str), training then inference.
################

sections = []

for m in TRAINING_METRICS.values():
    decimals = m['decimals']
    fmt_fn = int_fmt if decimals == 0 else (lambda v, decimals=decimals: fmt(v, decimals))
    table = build_table(m['get_value_fn'], fmt_fn, decimals=decimals)
    section_name = f"{m['label'].upper()} ({m['units']})"
    key = f"training.{m['key']}"
    sections.append((section_name, key, m['desc'], table))

for device in ('gpu', 'cpu'):
    for bs in batch_sizes_seen(device):
        for metric_name, m in INFERENCE_METRICS.items():
            if device not in m['devices']:
                continue
            decimals = m['decimals']
            get_val = lambda d, bs=bs, fn=m['get_value_fn']: fn(d, bs)
            fmt_val = lambda v, decimals=decimals: fmt(v, decimals)
            table = build_table(get_val, fmt_val, decimals=decimals, src=inference(device))
            section_name = f"{device.upper()} bs={bs} {metric_name.upper()} ({m['units']})"
            key = f"inference.{device}.dataloader[batch_size={bs}].{m['key']}"
            desc = f"{m['desc']}, batch_size={bs}, device={device}. Dash = no run for this model/dataset/device."
            sections.append((section_name, key, desc, table))

################
# Write output
################

if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_tables.txt')
    with open(out_path, 'w') as f:
        for section_name, key, desc, table in sections:
            f.write(f'=== {section_name} ===\n')
            f.write(f'# key: {key}\n')
            f.write(f'# desc: {desc}\n')
            f.write(table + '\n\n')

    print(f'done, written to {out_path}')
    print(f'{len(sections)} tables written ({len(TRAINING_METRICS)} training + {len(sections) - len(TRAINING_METRICS)} inference-benchmark)')
