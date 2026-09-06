#!/bin/bash
# Usage: bash generate_query_data_synthetic.sh

## ARGS presets — uncomment one:
## Random pairs 500k
ARGS="--query_strategy random_hl --save_dir random_500k --num_random_pairs 500000"

## Random pairs 30M
# ARGS="--query_strategy random_hl --save_dir random_30M --num_random_pairs 30000000"

## All pairs
# ARGS="--query_strategy all --save_dir all_pairs"

## landmark split 80/20 (100 landmarks, 1M pairs)
# ARGS="--query_strategy landmark_split --query_dir query_landmark_100_split_80_20 --save_dir query_landmark_100_split_80_20 --num_landmarks 100 --num_random_pairs 1000000"

DATASETS=(
    # ## Figshare Datasets
    Surat       # 2.5k

    ## Workload-driven Datasets
    W_Jinan     # 8.9k
    W_Shenzhen  # 11.9k
    W_Chengdu   # 17.5k
    W_Beijing   # 74.3k
    W_Shanghai  # 74.9k
    W_NewYork   # 334.9k
    W_Chicago   # 386.5k

    ## DIMACS Datasets
    FLA         # 1.07M
    E           # 3.60M
    W           # 6.26M
    CTR         # 14.1M
    USA         # 23.9M
)

for name in "${DATASETS[@]}"; do
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_query_data.py --data_name "$name" $ARGS

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
done
