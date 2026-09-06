#!/bin/bash
# Usage: bash generate_node2vec_embeds_openne.sh

DATASETS=(
    ## Figshare Datasets
    Surat           # 2.5k, ~12s

    # ## Workload-driven Datasets
    # W_Jinan         # 8.9k, ~35s
    # W_Shenzhen      # 11.9k, ~34s
    # W_Chengdu       # 17.5k, ~47s
    # W_Beijing       # 74.3k, ~3m16s
    # W_Shanghai      # 74.9k, ~3m16s
    # W_NewYork       # 334.9k, ~16m08s
    # W_Chicago       # 386.5k, ~22m14s
)

DIM=64
EPOCHS=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENNE_SRC="$(cd "$REPO_ROOT/archive/OpenNE/src" && pwd)"
DATA_DIR="$(cd "$REPO_ROOT/data" && pwd)"

# Auto-detect available CPU cores
#   Linux+SLURM : sched_getaffinity returns only the SLURM-allocated cores.
#   Linux/macOS : falls back to multiprocessing.cpu_count().
#   No Python   : tries nproc (Linux), then sysctl (macOS), then hardcodes 4.
WORKERS=$(python -c "import os,multiprocessing; \
           print(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else multiprocessing.cpu_count())" \
    2>/dev/null \
    || nproc 2>/dev/null \
    || sysctl -n hw.logicalcpu 2>/dev/null \
    || echo 4)

cd "$OPENNE_SRC"

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    OUT="$DATA_DIR/$name/node2vec_dim${DIM}_epochs${EPOCHS}_unweighted_openne.embeddings"
    python -m openne \
        --input "$DATA_DIR/$name/$name.edges" \
        --method node2vec --representation-size $DIM --workers $WORKERS --epochs $EPOCHS \
        --output "$OUT"

    printf "Converting $OUT to $OUT.npz...\n"
    python -c "
import numpy as np
raw = np.loadtxt('$OUT', delimiter=',', comments='#')
ids = raw[:, 0].astype(np.int64)
vectors = raw[:, 1:].astype(np.float32)
order = np.argsort(ids)
data = np.hstack([ids[order].reshape(-1, 1), vectors[order]])
np.savez_compressed('$OUT.npz', data=data)
print(f'Saved {\"$OUT\"}.npz with shape {data.shape}')
"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
