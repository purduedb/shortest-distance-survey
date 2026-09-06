#!/bin/bash
# Usage: bash generate_parts_file_rne.sh

DATASETS=(
    ## Figshare Datasets
    Surat           # 2.5k, ~2s

    ## Workload-driven Datasets
    W_Jinan         # 8.9k, ~3s
    W_Shenzhen      # 11.9k, ~3s
    W_Chengdu       # 17.5k, ~4s
    W_Beijing       # 74.3k, ~9s
    W_Shanghai      # 74.9k, ~8s
    W_NewYork       # 334.9k, ~31s
    W_Chicago       # 386.5k, ~37s

    ## Dimacs Datasets
    FLA             # 1.07M, ~57s
    E               # 3.60M, ~4m12s
    W               # 6.26M, ~7m27s
    CTR             # 14.1M, ~12m53s
    USA             # 23.9M, ~35m10s
)

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_parts_file_rne.py --data_name "$name"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
