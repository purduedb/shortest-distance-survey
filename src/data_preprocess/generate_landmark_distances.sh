#!/bin/bash
# Usage: bash generate_landmark_distances.sh

NUM_LANDMARKS=61

DATASETS=(
    ## Figshare Datasets
    Surat           # 2.5k, ~3s

    ## Workload-driven Datasets
    W_Jinan         # 8.9k, ~3s
    W_Shenzhen      # 11.9k, ~4s
    W_Chengdu       # 17.5k, ~4s
    W_Beijing       # 74.3k, ~6s
    W_NewYork       # 334.9k, ~20s
    W_Chicago       # 386.5k, ~23s
    W_Shanghai      # 74.9k, ~7s

    ## Dimacs Datasets
    FLA             # 1.07M, ~49s
    E               # 3.60M, ~2m49s
    W               # 6.26M, ~4m56s
    CTR             # 14.1M, ~11m41s
    USA             # 23.9M, ~18m50s
)

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_landmark_distances.py --data_name "$name" --num_landmarks $NUM_LANDMARKS

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
