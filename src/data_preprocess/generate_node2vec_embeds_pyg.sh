#!/bin/bash
# Usage: bash generate_node2vec_embeds_pyg.sh

DATASETS=(
    ## Figshare Datasets
    Surat           # 2.5k, ~8s

    ## Workload-driven Datasets
    W_Jinan         # 8.9k, ~11s
    W_Shenzhen      # 11.9k, ~11s
    W_Chengdu       # 17.5k, ~14s
    W_Beijing       # 74.3k, ~26s
    W_Shanghai      # 74.9k, ~28s
    W_NewYork       # 334.9k, ~1m42s
    W_Chicago       # 386.5k, ~1m57s

    ## Dimacs Datasets
    FLA             # 1.07M, ~5m17s
    E               # 3.60M, ~17m54s
    W               # 6.26M, ~31m18s
    CTR             # 14.1M, ~1h12m24s
    USA             # 23.9M, TODO: ~1h30m (yet to be confirmed, current OOMs on 32gb RAM)
)

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_node2vec_embeds_pyg.py --data_name "$name"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
