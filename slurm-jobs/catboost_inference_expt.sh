#!/bin/bash
# Run the native CatBoost (GBM) query latency benchmark for every dataset.
# Minimal adaptation of run_cpp_benchmark.sh: single "model" (CatBoost, no
# JIT variant exists), src/catboost_inference.py in place
# of the LibTorch binary, run directly against each CatBoost_<Dataset>_<Query>.pt
# checkpoint (already contains node/landmark embeddings + the CatBoost model).
#
# Usage:
#   bash catboost_inference_expt.sh --saved_models_dir <dir>              # dry run
#   bash catboost_inference_expt.sh --saved_models_dir <dir> -e           # run locally
#   bash catboost_inference_expt.sh --saved_models_dir <dir> -s           # submit to SLURM (dry run)
#   bash catboost_inference_expt.sh --saved_models_dir <dir> -s -e        # submit to SLURM and execute

###################
## Parse arguments
###################
EXECUTE=false                           # Set EXECUTE to false by default
SLURM=false                             # Set SLURM to false by default
SAVED_MODELS_DIR_ARG=""                 # Directory containing CatBoost_<Dataset>_<Query>.pt (required)
N_QUERIES_ARG=10000000                  # Number of queries to generate/use (default: 10M)
BATCH_SIZES_ARG="1000,10000,100000,1000000"  # Comma-separated batch sizes for dataloader benchmark
EVAL_RUNS_ARG=10                        # Number of timing runs (stats from last 5)
FILTER_DATASETS=""                      # Optional: comma-separated datasets to run, e.g. W_Jinan,W_Beijing
OUTPUT_DIR_ARG=""                       # Optional: --output_dir passthrough (default: saved_catboost_metrics/ sibling of saved_models_dir)
JOB_TIME_ARG=""                         # Optional: overrides sbatch --time (default: 4:00:00)
ACCOUNT_ARG=""                          # Optional: overrides sbatch --account (default: csit)
PARTITION_ARG=""                        # Optional: overrides sbatch --partition (default: training)
QOS_ARG=""                              # Optional: overrides sbatch --qos (default: training)
MEM_ARG=""                              # Optional: overrides sbatch --mem (default: 32G)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --execute|-e)         EXECUTE=true ;;
        --slurm|-s)           SLURM=true ;;
        --saved_models_dir)   SAVED_MODELS_DIR_ARG="$2"; shift ;;
        --n_queries)          N_QUERIES_ARG="$2";     shift ;;
        --batch_sizes)        BATCH_SIZES_ARG="$2";   shift ;;
        --eval_runs)          EVAL_RUNS_ARG="$2";     shift ;;
        --filter_datasets)    FILTER_DATASETS="$2";   shift ;;
        --output_dir)         OUTPUT_DIR_ARG="$2";    shift ;;
        --job_time)           JOB_TIME_ARG="$2";      shift ;;
        --account)            ACCOUNT_ARG="$2";       shift ;;
        --partition)          PARTITION_ARG="$2";     shift ;;
        --qos)                QOS_ARG="$2";           shift ;;
        --mem)                MEM_ARG="$2";           shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$SAVED_MODELS_DIR_ARG" ]]; then
    echo "Error: --saved_models_dir is required"
    exit 1
fi

SAVED_MODELS_DIR_ARG="$(realpath "$SAVED_MODELS_DIR_ARG")"

## Change to the script's directory
SCRIPT_PATH="$(realpath "$BASH_SOURCE")"
cd "$(dirname "$SCRIPT_PATH")"

###################
## Configuration
###################
REPO_ROOT="$HOME/scratch/shortest-distance-survey"
SAVED_MODELS_DIR="$SAVED_MODELS_DIR_ARG"
DATA_DIR="$REPO_ROOT/data"
CATBOOST_SCRIPT="$REPO_ROOT/src/catboost_inference.py"
JOB_TIME="4:00:00"    # default sbatch --time; override via --job_time
ACCOUNT="csit"        # default sbatch --account; override via --account
PARTITION="training"  # default sbatch --partition; override via --partition
QOS="training"        # default sbatch --qos; override via --qos
MEM="32G"             # default sbatch --mem; override via --mem
[[ -n "$JOB_TIME_ARG" ]]  && JOB_TIME="$JOB_TIME_ARG"
[[ -n "$ACCOUNT_ARG" ]]   && ACCOUNT="$ACCOUNT_ARG"
[[ -n "$PARTITION_ARG" ]] && PARTITION="$PARTITION_ARG"
[[ -n "$QOS_ARG" ]]       && QOS="$QOS_ARG"
[[ -n "$MEM_ARG" ]]       && MEM="$MEM_ARG"
N_QUERIES="$N_QUERIES_ARG"

# Infer a human-readable suffix for query subdir name (e.g. 1000000 → 1M, 500000 → 500k)
if (( N_QUERIES >= 1000000 && N_QUERIES % 1000000 == 0 )); then
    QUERY_SUFFIX="$((N_QUERIES / 1000000))M"
elif (( N_QUERIES >= 1000 && N_QUERIES % 1000 == 0 )); then
    QUERY_SUFFIX="$((N_QUERIES / 1000))k"
else
    QUERY_SUFFIX="$N_QUERIES"
fi
QUERY_SUBDIR="benchmark_queries_${QUERY_SUFFIX}"

RUN_NAME="catboost_inference_jobs"
LOG_DIR="$(dirname "$SAVED_MODELS_DIR")/$RUN_NAME"
if [[ -n "$SLURM_JOBID" ]]; then
    MASTER_LOG="$LOG_DIR/catboost_inference-master-${SLURM_JOBID}.log"
else
    MASTER_LOG="$LOG_DIR/catboost_inference-master.log"
fi

if [[ -z "$OUTPUT_DIR_ARG" ]]; then
    OUTPUT_DIR_ARG="$(dirname "$SAVED_MODELS_DIR")/saved_catboost_metrics"
fi

# Node counts — used for random query generation
declare -A N_NODES
## FIGSHARE Datasets ##
N_NODES["Surat"]=2508
## Workload Datasets ##
N_NODES["W_Jinan"]=8908
N_NODES["W_Shenzhen"]=11933
N_NODES["W_Chengdu"]=17567
N_NODES["W_Beijing"]=74383
N_NODES["W_Shanghai"]=74903
N_NODES["W_NewYork"]=334930
N_NODES["W_Chicago"]=386533
## DIMACS Datasets ##
N_NODES["FLA"]=1070376
N_NODES["E"]=3598623
N_NODES["W"]=6262104
N_NODES["CTR"]=14081816
N_NODES["USA"]=23947347

# Known query-dir strings in checkpoint filenames, CatBoost_<Dataset>_<QueryDir>.pt,
# to avoid ambiguity with dataset names that are prefixes of others (e.g. "W" vs "W_Beijing").
QUERY_DIRS=(
    'all_pairs'
    'landmark_30M'
    'real_workload_perturb_500k'
)

DATASETS=(
    ## FIGSHARE Datasets ##
    'Surat'         # 2.5k

    ## Workload Datasets ##
    'W_Jinan'       # 8.9k
    'W_Shenzhen'    # 11.9k
    'W_Chengdu'     # 17.5k
    'W_Beijing'     # 74.3k
    'W_Shanghai'    # 74.9k
    'W_NewYork'     # 334.9k
    'W_Chicago'     # 386.5k

    ## DIMACS Datasets ##
    'FLA'           # 1.07M
    'E'             # 3.60M
    'W'             # 6.26M
    'CTR'           # 14.1M
    'USA'           # 23.9M
)

if [[ -n "$FILTER_DATASETS" ]]; then
    IFS=',' read -ra DATASETS <<< "$FILTER_DATASETS"
fi

echo "----------------------------------------"
echo "EXECUTE mode      : $EXECUTE"
echo "SLURM mode        : $SLURM"
echo "SAVED_MODELS_DIR  : $SAVED_MODELS_DIR"
echo "DATASETS          : ${DATASETS[*]}"
echo "N_QUERIES         : $QUERY_SUFFIX ($N_QUERIES)"
echo "QUERY_SUBDIR       : $QUERY_SUBDIR"
echo "BATCH_SIZES       : $BATCH_SIZES_ARG"
echo "EVAL_RUNS         : $EVAL_RUNS_ARG"
echo "OUTPUT_DIR        : $OUTPUT_DIR_ARG"
echo "JOB_TIME          : $JOB_TIME"
echo "ACCOUNT           : $ACCOUNT"
echo "PARTITION         : $PARTITION"
echo "QOS               : $QOS"
echo "MEM               : $MEM"
echo "LOG_DIR           : $LOG_DIR"
echo "MASTER_LOG        : $MASTER_LOG"
echo "----------------------------------------"

## If SLURM is true, submit job and exit ##
if [ "$SLURM" = true ]; then
    MASTER_LOGFILE="$LOG_DIR/catboost_inference-master-%j.log"
    SBATCH_COMMAND="sbatch \
        --account $ACCOUNT \
        --partition $PARTITION \
        --qos $QOS \
        -N 1 \
        -n 1 \
        -c 16 \
        --mem $MEM \
        --gres gpu:1 \
        --time $JOB_TIME \
        --job-name $RUN_NAME \
        --output $MASTER_LOGFILE \
        $SCRIPT_PATH"
    SBATCH_COMMAND=$(echo "$SBATCH_COMMAND" | tr -s ' ')

    [ "$EXECUTE" = true ]      && SBATCH_COMMAND="$SBATCH_COMMAND --execute"
    SBATCH_COMMAND="$SBATCH_COMMAND --saved_models_dir $SAVED_MODELS_DIR_ARG"
    SBATCH_COMMAND="$SBATCH_COMMAND --n_queries $N_QUERIES_ARG"
    SBATCH_COMMAND="$SBATCH_COMMAND --batch_sizes $BATCH_SIZES_ARG"
    SBATCH_COMMAND="$SBATCH_COMMAND --eval_runs $EVAL_RUNS_ARG"
    [ -n "$FILTER_DATASETS" ] && SBATCH_COMMAND="$SBATCH_COMMAND --filter_datasets $FILTER_DATASETS"
    SBATCH_COMMAND="$SBATCH_COMMAND --output_dir $OUTPUT_DIR_ARG"

    echo "SBATCH_COMMAND: $SBATCH_COMMAND"
    mkdir -p "$LOG_DIR"
    eval ${SBATCH_COMMAND}
    echo "MASTER_LOGFILE: $MASTER_LOGFILE"
    echo "----------------------------------------"
    exit 0
fi

## Environment Setup ##
module load conda
conda activate myenv

# Pin CWD with an absolute path — sbatch does not inherit the submitting
# shell's CWD on this cluster (jobs start in /var/spool/slurm/job<ID>/), and
# relative paths like --output_dir below depend on CWD being right here.
cd ~/scratch/shortest-distance-survey/slurm-jobs

echo "========================================"
echo "Run started: $(date)"
echo "========================================"

# ---------------------------------------------------------------------------
# generate_queries: emit N random (src, dst) pairs (1-indexed) to stdout.
# ---------------------------------------------------------------------------
generate_queries() {
    local n_nodes=$1
    local n_queries=$2
    python - <<PYEOF
import numpy as np, sys
rng = np.random.default_rng(42)
n = ${n_nodes}
q = ${n_queries}
src = rng.integers(1, n + 1, size=q)
dst = rng.integers(1, n + 1, size=q)
out = "\n".join(f"{s},{d}" for s, d in zip(src.tolist(), dst.tolist()))
sys.stdout.write(out + "\n")
PYEOF
}

# ---------------------------------------------------------------------------
# Ensure every dataset has a query file under benchmark_queries_{suffix}/
# ---------------------------------------------------------------------------
echo "=== Preparing query files ($QUERY_SUFFIX queries each) ==="
for dataset in "${DATASETS[@]}"; do
    qdir="$DATA_DIR/$dataset/$QUERY_SUBDIR"
    qfile="$qdir/${dataset}_${QUERY_SUBDIR}.queries"
    if [ -f "$qfile" ]; then
        echo "  $dataset: $qfile already exists — skipping"
    else
        echo "  $dataset: generating $qfile"
        if [ "$EXECUTE" = true ]; then
            mkdir -p "$qdir"
            generate_queries "${N_NODES[$dataset]}" "$N_QUERIES" > "$qfile"
            echo "  $dataset: done ($(wc -l < "$qfile") lines)"
        else
            echo "  $dataset: (dry run — would generate ${N_NODES[$dataset]}-node, $QUERY_SUFFIX queries)"
        fi
    fi
done
echo "----------------------------------------"

# ---------------------------------------------------------------------------
# Run benchmark for every dataset (single model: CatBoost)
# ---------------------------------------------------------------------------
SECONDS=0
COUNTER=0
SKIPPED=0
TOTAL=${#DATASETS[@]}

echo "Running $TOTAL dataset(s) against SAVED_MODELS_DIR=$SAVED_MODELS_DIR"
echo "----------------------------------------"

for dataset in "${DATASETS[@]}"; do
    COUNTER=$((COUNTER + 1))
    echo "Task: $COUNTER / $TOTAL"

    # Find the CatBoost_<Dataset>_<QueryDir>.pt checkpoint for this dataset
    pt_path=""
    for qd in "${QUERY_DIRS[@]}"; do
        candidate="$SAVED_MODELS_DIR/CatBoost_${dataset}_${qd}.pt"
        [ -f "$candidate" ] && { pt_path="$candidate"; break; }
    done
    if [ -z "$pt_path" ]; then
        echo "  SKIP: no CatBoost checkpoint for dataset=$dataset"
        echo "----------------------------------------"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    qfile="$DATA_DIR/$dataset/$QUERY_SUBDIR/${dataset}_${QUERY_SUBDIR}.queries"

    echo "  Dataset    : $dataset"
    echo "  Checkpoint : $pt_path"
    echo "  Queries    : $qfile"

    if [ ! -f "$qfile" ]; then
        echo "  SKIP: query file not found: $qfile"
        echo "----------------------------------------"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    CMD="python $CATBOOST_SCRIPT \
        --model_path $pt_path \
        --queries $qfile \
        --n_queries $N_QUERIES \
        --batch_sizes $BATCH_SIZES_ARG \
        --eval_runs $EVAL_RUNS_ARG \
        --output_dir $OUTPUT_DIR_ARG"
    CMD=$(echo "$CMD" | tr -s ' ')
    echo "  CMD: $CMD"

    if [ "$EXECUTE" = false ]; then
        echo "  EXECUTE: false (not executing)"
        echo "----------------------------------------"
    else
        echo "  EXECUTE: true"
        /usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
            $CMD
        echo "  Exit code: $?"
        echo "----------------------------------------"
    fi
done

DONE=$((TOTAL - SKIPPED))
duration=$SECONDS

echo
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
printf "Ran %d / %d benchmarks\n" $DONE $TOTAL
printf "Total Time Elapsed: %d-%02d:%02d:%02d\n" $((duration/86400)) $(( (duration%86400)/3600 )) $(( (duration%3600)/60 )) $(( duration%60 ))
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "MASTER_LOG=$MASTER_LOG"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
