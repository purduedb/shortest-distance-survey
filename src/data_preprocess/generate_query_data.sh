#!/bin/bash
# Usage: bash generate_query_data.sh

## Real workload query_perturbation (perturb_k / k_hop are dataset-specific)
## Requires data/<name>/real_workload/<name>.queries.npz seed queries.
PERTURB_DATASETS=(
    W_Jinan     # 8.9k, ~5m09s
    W_Shenzhen  # 11.9k, ~4m12s
    W_Chengdu   # 17.5k, ~30s
    W_Beijing   # 74.3k, ~29s
    W_Shanghai  # 74.9k, ~33s
    W_NewYork   # 334.9k, ~1m06s
    W_Chicago   # 386.5k, ~33s
)
declare -A PERTURB_ARGS=(
    [W_Jinan]="--query_dir real_workload    --query_strategy query_perturbation --perturb_k 60  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_Shenzhen]="--query_dir real_workload --query_strategy query_perturbation --perturb_k 60  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_Chengdu]="--query_dir real_workload  --query_strategy query_perturbation --perturb_k 30  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_Beijing]="--query_dir real_workload  --query_strategy query_perturbation --perturb_k 50  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_Shanghai]="--query_dir real_workload --query_strategy query_perturbation --perturb_k 10  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_NewYork]="--query_dir real_workload  --query_strategy query_perturbation --perturb_k 10  --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
    [W_Chicago]="--query_dir real_workload  --query_strategy query_perturbation --perturb_k 120 --k_hop 8 --max_queries 500000 --save_dir real_workload_perturb_500k"
)

## landmark_hl 30M (num_landmarks is dataset-specific)
LANDMARK_DATASETS=(
    FLA  # 1.07M, ~135s
    E    # 3.60M, ~156s
    W    # 6.26M, ~149s
    CTR  # 14.1M, ~188s
    USA  # 23.9M, ~214s   NOTE: USA.hl index is ~56GB; needs a high-memory job (128GB confirmed to succeed, 64GB OOMs/SIGKILLs)
)
declare -A LANDMARK_ARGS=(
    [FLA]="--query_strategy landmark_hl --num_landmarks 30 --save_dir landmark_30M"
    [E]="--query_strategy landmark_hl   --num_landmarks 10 --save_dir landmark_30M"
    [W]="--query_strategy landmark_hl   --num_landmarks 5  --save_dir landmark_30M"
    [CTR]="--query_strategy landmark_hl --num_landmarks 2  --save_dir landmark_30M"
    [USA]="--query_strategy landmark_hl --num_landmarks 1  --save_dir landmark_30M"
)

## All pairs
ALL_PAIRS_DATASETS=(
    Surat  # 2.5k, ~30s
)
ALL_PAIRS_ARGS="--query_strategy all --save_dir all_pairs"

run_one() {
    local name="$1"
    shift
    printf "[%s] Starting...\n" "$name"
    start_time=$SECONDS

    python generate_query_data.py --data_name "$name" "$@"

    elapsed_time=$(( SECONDS - start_time ))
    printf "[%s] Done in %02d:%02d:%02d\n\n" "$name" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60))
}

# ~2.5 min total
for name in "${PERTURB_DATASETS[@]}"; do
    run_one "$name" ${PERTURB_ARGS[$name]}
done

# ~14 min total (135+156+149+188+214s on 128GB job; USA OOMs on 64GB)
for name in "${LANDMARK_DATASETS[@]}"; do
    run_one "$name" ${LANDMARK_ARGS[$name]}
done

# ~30s total (Surat only)
for name in "${ALL_PAIRS_DATASETS[@]}"; do
    run_one "$name" $ALL_PAIRS_ARGS
done
