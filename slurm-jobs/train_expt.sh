#!/bin/bash
# Usage: bash train_expt.sh [--execute | -e] [--slurm | -s]
# Dry run locally:      bash train_expt.sh
# Run locally:          bash train_expt.sh -e
# Dry run on SLURM:     bash train_expt.sh -s
# Run on SLURM:         bash train_expt.sh -s -e

###################
## Parse arguments
###################
EXECUTE=false       # Set EXECUTE to false by default
SLURM=false         # Set SLURM to false by default
EXPT_NAME_ARG="slurm_default"  # Experiment name → results saved to ../results/<EXPT_NAME>
N2V_BACKEND_ARG=""             # Optional: overrides N2V_BACKEND config
QUERY_DIR_ARG=""               # Optional: overrides QUERY_DIR config (choices: real_workload_perturb_500k, random_1M, all_pairs, random_500k)
TIME_LIMIT_ARG=""              # Optional: overrides TIME_LIMIT config (in minutes)
JOB_TIME_ARG=""                # Optional: overrides sbatch --time (SLURM wall-clock, format HH:MM:SS)
ACCOUNT_ARG=""                 # Optional: overrides sbatch --account (default: csit)
PARTITION_ARG=""               # Optional: overrides sbatch --partition (default: training)
QOS_ARG=""                     # Optional: overrides sbatch --qos (default: training)
MEM_ARG=""                     # Optional: overrides sbatch --mem (default: 64G)
FILTER_DATASETS=""             # Optional: comma-separated datasets to run, e.g. W_Jinan,W_Beijing
FILTER_MODELS=""               # Optional: comma-separated model categories, e.g. gnn,baselines
                               #           choices: baselines, nn, gnn, functional, tree
EXTRA_PYTHON_ARGS=""           # Optional: raw string appended to every PYTHON_COMMAND, e.g. "--sparse_embedding"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --execute|-e)      EXECUTE=true ;;
        --slurm|-s)        SLURM=true ;;
        --expt_name)       EXPT_NAME_ARG="$2"; shift ;;
        --n2v_backend)     N2V_BACKEND_ARG="$2"; shift ;;
        --query_dir)       QUERY_DIR_ARG="$2";   shift ;;
        --time_limit)      TIME_LIMIT_ARG="$2";  shift ;;
        --job_time)        JOB_TIME_ARG="$2";    shift ;;
        --account)         ACCOUNT_ARG="$2";     shift ;;
        --partition)       PARTITION_ARG="$2";   shift ;;
        --qos)             QOS_ARG="$2";         shift ;;
        --mem)             MEM_ARG="$2";         shift ;;
        --filter_datasets) FILTER_DATASETS="$2"; shift ;;
        --filter_models)   FILTER_MODELS="$2";   shift ;;
        --extra_python_args) EXTRA_PYTHON_ARGS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

## Change to the script's directory
SCRIPT_PATH="$(realpath "$BASH_SOURCE")"
cd "$(dirname "$SCRIPT_PATH")"

###################
## Configuration
###################
# EXPT_NAME="v36-real_workload_perturb_500k"      ## epochs=20
# EXPT_NAME="v37-real_workload_perturb_500k"      ## time_limit=2 mins, epochs=1000
# EXPT_NAME="v38-real_workload_perturb_500k"      ## eval_runs=5, device=cpu, time_limit=0.5 mins
# ==============================================================================================================================
# EXPT_NAME="v39-real_workload_perturb_500k"      ## eval_runs=0, device=cuda, time_limit=5 mins, validate! (default lr = 0.01)
# EXPT_NAME="v40-real_workload_perturb_500k"      ## same as v39 but default learning rate as 0.003
# EXPT_NAME="v41-real_workload_perturb_500k"      ## same as v39 but default learning rate as 0.001
# EXPT_NAME="v61-gpu-v1"                          ## same as v39 but eval_runs=10
# EXPT_NAME="v62-gpu-v1"                          ## same as v39 but eval_runs=10 and high precision matmul in torch
# EXPT_NAME="v3-sigspatial-gnn"                     ## GNN-only re-run; batched precompute_embeddings fix
# EXPT_NAME="v3-sigspatial"                         ## Surat all_pairs, half models per job
# ==============================================================================================================================
EXPT_NAME="$EXPT_NAME_ARG"
# QUERY_DIR="random_1M"
# QUERY_DIR="all_pairs"
QUERY_DIR="real_workload_perturb_500k"
# QUERY_DIR="random_500k"
TIME_LIMIT=5            # default; override per group via --time_limit
JOB_TIME="4:00:00"      # default sbatch wall-clock; override via --job_time
ACCOUNT="csit"          # default sbatch --account; override via --account
PARTITION="a100-40gb"   # default sbatch --partition; override via --partition
QOS="standby"           # default sbatch --qos; override via --qos
MEM="32G"               # default sbatch --mem; override via --mem
LOG_DIR="../results/$EXPT_NAME"
N2V_BACKEND="openne"  # {pyg/pecanpy/openne}
[[ -n "$N2V_BACKEND_ARG" ]] && N2V_BACKEND="$N2V_BACKEND_ARG"
[[ -n "$QUERY_DIR_ARG" ]] && QUERY_DIR="$QUERY_DIR_ARG"
[[ -n "$TIME_LIMIT_ARG" ]] && TIME_LIMIT="$TIME_LIMIT_ARG"
[[ -n "$JOB_TIME_ARG" ]] && JOB_TIME="$JOB_TIME_ARG"
[[ -n "$ACCOUNT_ARG" ]] && ACCOUNT="$ACCOUNT_ARG"
[[ -n "$PARTITION_ARG" ]] && PARTITION="$PARTITION_ARG"
[[ -n "$QOS_ARG" ]] && QOS="$QOS_ARG"
[[ -n "$MEM_ARG" ]] && MEM="$MEM_ARG"

DATASETS=(
    ## FIGSHARE Datasets ##
    'Surat'         # 2.5k

    # Workload datasets ##
    'W_Jinan'       # 8.9k
    'W_Shenzhen'    # 11.9k
    'W_Chengdu'     # 17.5k
    'W_Beijing'     # 74.3k
    'W_Shanghai'    # 74.9k
    'W_NewYork'     # 334.9k
    'W_Chicago'     # 386.5k

    # DIMACS Datasets ##
    'FLA'           # 1.07M
    'E'             # 3.60M
    'W'             # 6.26M
    'CTR'           # 14.1M
    'USA'           # 23.9M
)
# Override DATASETS from --filter_datasets if provided (comma-separated)
if [[ -n "$FILTER_DATASETS" ]]; then
    IFS=',' read -ra DATASETS <<< "$FILTER_DATASETS"
fi

MODELS=(
    # ## LpNorm
    "train.py --model_class lpnorm --model_name Manhattan --p_norm 1"
    # # "train.py --model_class lpnorm --model_name Euclidean --p_norm 2"

    # ## Landmark
    # "train.py --model_class landmark --model_name Landmark_random --landmark_selection random"
    "train.py --model_class landmark --model_name Landmark_random_subset --landmark_selection random --select_landmarks_from_train"
    # "train.py --model_class landmark --model_name Landmark_kmeans --landmark_selection kmeans"
    "train.py --model_class landmark --model_name Landmark_kmeans_subset --landmark_selection kmeans --select_landmarks_from_train"
    # # "train.py --model_class landmark --model_name Landmark_betweenness_high --landmark_selection betweenness_high"
    # # "train.py --model_class landmark --model_name Landmark_degree --landmark_selection degree"
    # # "train.py --model_class landmark --model_name Landmark_pagerank --landmark_selection pagerank"
    # # "train.py --model_class landmark --model_name Landmark_betweenness_low --landmark_selection betweenness_low"
    # # "train.py --model_class landmark --model_name Landmark_closeness_high --landmark_selection closeness_high"  ## Takes too long for large graphs (>10k nodes)
    # # "train.py --model_class landmark --model_name Landmark_closeness_low --landmark_selection closeness_low"  ## Takes too long for large graphs (>10k nodes)
    # # "train.py --model_class landmark --model_name Landmark_eigenvector --landmark_selection eigenvector"
    # # "train.py --model_class landmark --model_name Landmark_katz --landmark_selection katz"
    # # "train.py --model_class landmark --model_name Landmark_harmonic --landmark_selection harmonic"  ## Takes too long

    # GeoDNN
    "train.py --model_class geodnn --model_name GeoDNN"

    # Vdist2vec
    "train.py --model_class vdist2vec --model_name Vdist2vec"

    # Ndist2vec
    "train.py --model_class ndist2vec --model_name Ndist2vec"

    # GNN
    "train.py --model_class rgnndist2vec --model_name SAGE --gnn_layer sage --loss_function smoothl1"
    "train.py --model_class rgnndist2vec --model_name GAT --gnn_layer gat --loss_function smoothl1 --disable_edge_weight"
    "train.py --model_class rgnndist2vec --model_name GCN --gnn_layer gcn --loss_function smoothl1 --disable_edge_weight"

    # # EmbeddingNN
    # "train.py --model_class embeddingnn --model_name EmbeddingNN --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method concat"
    "train.py --model_class embeddingnn --model_name EmbeddingNN_mean --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method mean"
    # "train.py --model_class embeddingnn --model_name EmbeddingNN_sub --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method subtract"

    # # DistanceNN
    # "train.py --model_class distancenn --model_name DistanceNN --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method concat"
    # "train.py --model_class distancenn --model_name DistanceNN_mean --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method mean"
    "train.py --model_class distancenn --model_name DistanceNN_sub --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz --aggregation_method subtract"

    # #######
    # # ANEDA
    "train.py --model_class aneda --model_name ANEDA --embedding_filename node2vec_dim64_epochs1_unweighted_${N2V_BACKEND}.embeddings.npz"
    # "train.py --model_class aneda --model_name ANEDA_random"

    # Path2vec
    "train.py --model_class path2vec --model_name Path2vec"

    # RNE
    "train.py --model_class rne --model_name RNE"

    # CatBoost
    "train.py --model_class catboost --model_name CatBoost --embedding_filename landmark_dim61.embeddings.npz"

    # CatBoostNN
    "train.py --model_class catboostnn --model_name CatBoostNN --embedding_filename landmark_dim61.embeddings.npz"
)
# Filter models by category and/or individual model_class, comma-separated
if [[ -n "$FILTER_MODELS" ]]; then
    _all_classes=$(for _m in "${MODELS[@]}"; do echo "$_m" | sed 's/.*--model_class \([^ ]*\).*/\1/'; done | sort -u)
    _filtered=()
    IFS=',' read -ra _categories <<< "$FILTER_MODELS"
    for _cat in "${_categories[@]}"; do
        case "$_cat" in
            # Filter by category
            baselines)  _keep="lpnorm landmark" ;;
            nn)         _keep="geodnn distancenn embeddingnn vdist2vec ndist2vec catboostnn" ;;
            gnn)        _keep="rgnndist2vec" ;;
            functional) _keep="path2vec aneda rne" ;;
            tree)       _keep="catboost" ;;
            # Filter by individual model_class
            *)
                if grep -qx "$_cat" <<< "$_all_classes"; then
                    _keep="$_cat"
                else
                    echo "Unknown --filter_models value: $_cat (choices: baselines, nn, gnn, functional, tree, or any model_class: $(echo $_all_classes))"
                    exit 1
                fi
                ;;
        esac
        for _m in "${MODELS[@]}"; do
            _class=$(echo "$_m" | sed 's/.*--model_class \([^ ]*\).*/\1/')
            for _f in $_keep; do
                if [[ "$_class" == "$_f" ]]; then _filtered+=("$_m"); break; fi
            done
        done
    done
    MODELS=("${_filtered[@]}")
fi
# Define specific learning rates for model-dataset combinations
declare -A MODEL_DATASET_LR
# Format: MODEL_DATASET_LR["ModelName:DatasetName"]="learning_rate"
MODEL_DATASET_LR["RNE:W_NewYork"]="0.001"  # Slow learning rate for large road networks
MODEL_DATASET_LR["RNE:W_Chicago"]="0.001"
MODEL_DATASET_LR["RNE:FLA"]="0.001"
MODEL_DATASET_LR["RNE:E"]="0.001"
MODEL_DATASET_LR["RNE:W"]="0.001"
MODEL_DATASET_LR["RNE:CTR"]="0.0003"
MODEL_DATASET_LR["RNE:USA"]="0.0003"
# Define model specific learning rates
declare -A MODEL_LR
MODEL_LR["ANEDA"]="0.03"
MODEL_LR["Path2vec"]="0.03"
MODEL_LR["RNE"]="0.003"
MODEL_LR["CatBoost"]="0.3"
MODEL_LR["CatBoostNN"]="0.0003"

echo "----------------------------------------"
echo "EXECUTE mode    : $EXECUTE"
echo "SLURM mode      : $SLURM"
echo "EXPT_NAME       : $EXPT_NAME"
echo "DATASETS        : ${DATASETS[*]}"
echo "QUERY_DIR       : $QUERY_DIR"
echo "TIME_LIMIT      : $TIME_LIMIT min"
echo "JOB_TIME        : $JOB_TIME"
echo "ACCOUNT         : $ACCOUNT"
echo "PARTITION       : $PARTITION"
echo "QOS             : $QOS"
echo "MEM             : $MEM"
echo "MODELS          : $(echo "${MODELS[@]}" | grep -oP '(?<=--model_name )\S+' | tr '\n' ' ')"
echo "N2V_BACKEND     : $N2V_BACKEND"
echo "EXTRA_PYTHON_ARGS: $EXTRA_PYTHON_ARGS"
echo "----------------------------------------"

## If SLURM is true, submit job and exit ##
if [ "$SLURM" = true ]; then
    mkdir -p $LOG_DIR/train_jobs
    MASTER_LOGFILE="$LOG_DIR/train_jobs/train-master-%j.log"
    # Construct sbatch command
    # Stanford Blog "nodes vs tasks vs cpus vs cores": https://login.scg.stanford.edu/faqs/cores/
    # BSC Blog: https://www.bsc.es/supportkc/docs/CTE-POWER/slurm/
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
        --job-name ${EXPT_NAME} \
        --output $MASTER_LOGFILE \
        $SCRIPT_PATH"

    # Remove extra spaces from the command
    SBATCH_COMMAND=$(echo "$SBATCH_COMMAND" | tr -s ' ')

    # Append the --execute flag if originally set
    [ "$EXECUTE" = true ]       && SBATCH_COMMAND="$SBATCH_COMMAND --execute"
    SBATCH_COMMAND="$SBATCH_COMMAND --expt_name $EXPT_NAME_ARG"
    [ -n "$N2V_BACKEND_ARG" ]   && SBATCH_COMMAND="$SBATCH_COMMAND --n2v_backend $N2V_BACKEND_ARG"
    [ -n "$QUERY_DIR_ARG" ]     && SBATCH_COMMAND="$SBATCH_COMMAND --query_dir $QUERY_DIR_ARG"
    [ -n "$TIME_LIMIT_ARG" ]    && SBATCH_COMMAND="$SBATCH_COMMAND --time_limit $TIME_LIMIT_ARG"
    [ -n "$JOB_TIME_ARG" ]      && SBATCH_COMMAND="$SBATCH_COMMAND --job_time $JOB_TIME_ARG"
    [ -n "$ACCOUNT_ARG" ]       && SBATCH_COMMAND="$SBATCH_COMMAND --account $ACCOUNT_ARG"
    [ -n "$PARTITION_ARG" ]     && SBATCH_COMMAND="$SBATCH_COMMAND --partition $PARTITION_ARG"
    [ -n "$QOS_ARG" ]           && SBATCH_COMMAND="$SBATCH_COMMAND --qos $QOS_ARG"
    [ -n "$MEM_ARG" ]           && SBATCH_COMMAND="$SBATCH_COMMAND --mem $MEM_ARG"
    [ -n "$FILTER_DATASETS" ]   && SBATCH_COMMAND="$SBATCH_COMMAND --filter_datasets $FILTER_DATASETS"
    [ -n "$FILTER_MODELS" ]     && SBATCH_COMMAND="$SBATCH_COMMAND --filter_models $FILTER_MODELS"
    [ -n "$EXTRA_PYTHON_ARGS" ] && SBATCH_COMMAND="$SBATCH_COMMAND --extra_python_args \"$EXTRA_PYTHON_ARGS\""

    # Submit the job
    echo "SBATCH_COMMAND: $SBATCH_COMMAND"
    eval ${SBATCH_COMMAND}
    echo "MASTER_LOGFILE: $MASTER_LOGFILE"
    echo "----------------------------------------"

    # Exit the script
    exit 0
fi

## Environment Setup ##
# To create squash file: `mksquashfs ~/.conda/envs/myenv ~/scratch/myenv.sqsh -comp lz4 -processors 16`
# unsquashfs -d /tmp/myenv-sqsh ~/scratch/myenv.sqsh
module load conda
conda activate myenv
# conda activate /tmp/myenv-sqsh

## Run your job ##
cd ~/scratch/shortest-distance-survey/src

mkdir -p $LOG_DIR/train_jobs
mkdir -p $LOG_DIR/train_logs

## MAIN ##
SECONDS=0  # Timer for total duration
COUNTER=0  # Counter for number of commands
TOTAL_COMMANDS=$((${#DATASETS[@]} * ${#MODELS[@]}))  # Total number of commands to run
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Increment counter
        COUNTER=$((COUNTER + 1))
        echo "Task: $COUNTER / $TOTAL_COMMANDS"

        # Extract model_name from, e.g., "train.py --model_class landmark --model_name Landmark_random --landmark_selection random"
        model_name=$(echo "$model" | sed -n 's/.*--model_name \([^ ]*\).*/\1/p')

        # Determine learning rate
        if [ -n "${MODEL_DATASET_LR["${model_name}:${dataset}"]}" ]; then
            custom_lr="${MODEL_DATASET_LR["${model_name}:${dataset}"]}"
        elif [ -n "${MODEL_LR["${model_name}"]}" ]; then
            custom_lr="${MODEL_LR["${model_name}"]}"
        else
            custom_lr="0.01"  # default learning rate
        fi

        # Python command
        PYTHON_COMMAND="python ${model} \
            --data_dir ${dataset} \
            --query_dir ${QUERY_DIR} \
            --log_dir $LOG_DIR \
            --eval_runs 0 \
            --seed 1234 \
            --device cuda \
            --time_limit $TIME_LIMIT \
            --learning_rate ${custom_lr} \
            --epochs 120000 \
            --validate \
            ${EXTRA_PYTHON_ARGS}"

        # Remove extra spaces from the command
        PYTHON_COMMAND=$(echo "$PYTHON_COMMAND" | tr -s ' ')

        # Log file
        LOG_FILE="$LOG_DIR/train_logs/train_${model_name}_${dataset}_${QUERY_DIR}.log"

        # Prints
        echo "PYTHON_COMMAND: $PYTHON_COMMAND"
        echo "LOG_FILE: $LOG_FILE"

        # Run command if not in debug mode
        if [ "$EXECUTE" = false ]; then
            echo "EXECUTE: false (not executing the command)"
            echo "----------------------------------------"
        else
            echo "EXECUTE: true (executing the command)"
            # NOTE: using `> output.log` will only save stdout to the file,
            # and stderr will be shown in the terminal.
            /usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
            $PYTHON_COMMAND > $LOG_FILE
            echo "----------------------------------------"
        fi
    done
done
duration=$SECONDS  # Timer for total duration
echo "Finished."

echo
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
printf "Total Time Elapsed: %d-%02d:%02d:%02d\n" $((duration/86400)) $(( (duration%86400)/3600 )) $(( (duration%3600)/60 )) $(( duration%60 ))
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
