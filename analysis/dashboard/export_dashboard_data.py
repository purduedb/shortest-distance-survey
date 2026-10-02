"""Exports the same metrics build_tables.py prints, as JSON for dashboard.html

Usage:
    python export_dashboard_data.py
    open dashboard.html
"""
import sys, os, json
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # '..' -> analysis/
from extract_metrics import (
    METHODS, METHOD_GROUPS, DATASETS, DATASET_GROUPS,
    TRAINING_METRICS, INFERENCE_METRICS,
    get_key, training, inference,
    batch_sizes_seen,
    print_green
)

################
# Layout config
################

METHOD_GROUP_COLORS = {
    'Baselines':  '#4E79A7',   # blue
    'NNs':        '#F28E2B',   # orange
    'GNNs':       '#59A14F',   # green
    'Functional': '#B07AA1',   # purple
    'Tree':       '#E15759',   # red
}

################
# Helpers
################

def shades(hex_color, n):
    """n distinct shades of hex_color, darkest to lightest, for a method group's rows."""
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(hex_color))
    if n == 1:
        return [mcolors.to_hex(mcolors.hsv_to_rgb((h, s, v)))]
    return [mcolors.to_hex(mcolors.hsv_to_rgb((h, max(0.35, s * (1 - 0.55 * i / (n - 1))),
                                                min(0.95, v * (1 + 0.35 * i / (n - 1))))))
            for i in range(n)]


################
# Datasets / models (rows)
################

dk_to_group = {dk: g for g, dks in DATASET_GROUPS.items() for dk in dks}
datasets_out = [
    {'key': dk, 'label': meta['label'], 'nodes': meta['nodes'], 'group': dk_to_group[dk]}
    for dk, meta in DATASETS.items()
]

row_color = {}
for gname, models in METHOD_GROUPS.items():
    for m, shade in zip(models, shades(METHOD_GROUP_COLORS[gname], len(models))):
        row_color[m] = shade

models_out = [
    {'label': METHODS[m]['label'], 'group': gname, 'color': row_color[m], 'dashed': False}
    for gname, models in METHOD_GROUPS.items() for m in models
]

################
# Build metrics_out / values
################

metrics_out = []
values = {}  # metric_id -> {model_label: {dataset_key: value_or_None}}

def add_metric(mkey, title, ylabel, get_value_fn, src=training):
    metrics_out.append({'id': mkey, 'title': title, 'ylabel': ylabel})
    values[mkey] = {
        METHODS[m]['label']: {dk: get_value_fn(src(get_key(m, dk))) for dk in DATASETS}
        for models in METHOD_GROUPS.values() for m in models
    }

for metric_name, m in TRAINING_METRICS.items():
    add_metric(metric_name, m['label'], m['units'], m['get_value_fn'])

for device in ('gpu', 'cpu'):
    for bs in batch_sizes_seen(device):
        for metric_name, m in INFERENCE_METRICS.items():
            if device not in m['devices']:
                continue
            mkey = f'bench_{device}_bs{bs}_{metric_name}'
            title = f"{device.upper()} bs={bs} {m['label']}"
            get_val = lambda d, bs=bs, fn=m['get_value_fn']: fn(d, bs)
            add_metric(mkey, title, m['units'], get_value_fn=get_val, src=inference(device))

################
# Write output
################

payload = {
    'datasets': datasets_out,
    'datasetGroups': list(DATASET_GROUPS.keys()),
    'models': models_out,
    'modelGroups': list(METHOD_GROUPS.keys()),
    'metrics': metrics_out,
    'values': values,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data.js')
with open(out_path, 'w') as f:
    f.write('const DASHBOARD_DATA = ')
    json.dump(payload, f)
    f.write(';\n')

print_green(f'Saved {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)')
print(f'{len(metrics_out)} metrics total ({len(TRAINING_METRICS)} training + {len(metrics_out)-len(TRAINING_METRICS)} benchmark)')
