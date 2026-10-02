#!/bin/bash
# Usage: bash generate_dataset_stats.sh

DATASETS=(
    ## Figshare Datasets
    Surat           # 2.5k, ~4s

    ## Workload-driven Datasets
    W_Jinan         # 8.9k, ~5s
    W_Shenzhen      # 11.9k, ~4s
    W_Chengdu       # 17.5k, ~4s
    W_Beijing       # 74.3k, ~6s
    W_Shanghai      # 74.9k, ~5s
    W_NewYork       # 334.9k, ~8s
    W_Chicago       # 386.5k, ~10s

    ## DIMACS Datasets
    FLA             # 1.07M, ~18s
    E               # 3.60M, ~59s
    W               # 6.26M, ~1m41s
    CTR             # 14.1M, ~4m32s
    USA             # 23.9M, ~6m36s
)

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_dataset_stats.py --data_name "$name"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
