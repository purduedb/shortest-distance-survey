#!/bin/bash
# Usage: bash urban_expt.sh [--execute | -e] [--slurm | -s]
# Dry run locally:      bash urban_expt.sh
# Run locally:          bash urban_expt.sh -e
# Dry run on SLURM:     bash urban_expt.sh -s
# Run on SLURM:         bash urban_expt.sh -s -e

###################
## Parse arguments
###################
EXECUTE=false  # Set EXECUTE to false by default
SLURM=false  # Set SLURM to false by default
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --execute|-e) EXECUTE=true ;;  # Toggle EXECUTE to true if --execute or -e is provided
        --slurm|-s) SLURM=true ;;  # Toggle SLURM to true if --slurm or -s is provided
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift  # Move to the next argument
done
echo "EXECUTE mode: $EXECUTE"
echo "SLURM mode: $SLURM"
echo "----------------------------------------"

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
EXPT_NAME="v63-gpu-v1"                          ## same as v39 but eval_runs=10 and medium precision matmul in torch
QUERY_DIR="real_workload_perturb_500k"
LOG_DIR="../results/$EXPT_NAME"
DATASETS=(
    # Workload datasets ##
    'W_Jinan'       # 8.9k
    'W_Shenzhen'    # 11.9k
    'W_Chengdu'     # 17.5k
    'W_Beijing'     # 74.3k
    'W_Shanghai'    # 74.9k
    'W_NewYork'     # 334.9k
    'W_Chicago'     # 386.5k
)
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
    # "train.py --model_class embeddingnn --model_name EmbeddingNN --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method concat"
    "train.py --model_class embeddingnn --model_name EmbeddingNN_mean --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method mean"
    # "train.py --model_class embeddingnn --model_name EmbeddingNN_sub --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method subtract"

    # # DistanceNN
    # "train.py --model_class distancenn --model_name DistanceNN --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method concat"
    # "train.py --model_class distancenn --model_name DistanceNN_mean --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method mean"
    "train.py --model_class distancenn --model_name DistanceNN_sub --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings --aggregation_method subtract"

    # #######
    # # ANEDA
    "train.py --model_class aneda --model_name ANEDA --embedding_filename node2vec_dim64_epochs1_unweighted.embeddings"
    # "train.py --model_class aneda --model_name ANEDA_random"

    # Path2vec
    "train.py --model_class path2vec --model_name Path2vec"

    # RNE
    "train.py --model_class rne --model_name RNE"

    # CatBoost
    "train.py --model_class catboost --model_name CatBoost --embedding_filename landmark_dim61.embeddings"

    # CatBoostNN
    "train.py --model_class catboostnn --model_name CatBoostNN --embedding_filename landmark_dim61.embeddings"
)
# Define specific learning rates for model-dataset combinations
declare -A MODEL_DATASET_LR
# Format: MODEL_DATASET_LR["ModelName:DatasetName"]="learning_rate"
MODEL_DATASET_LR["RNE:W_NewYork"]="0.001"  # Slow learning rate for large road networks
MODEL_DATASET_LR["RNE:W_Chicago"]="0.001"
# Define model specific learning rates
declare -A MODEL_LR
MODEL_LR["ANEDA"]="0.03"
MODEL_LR["Path2vec"]="0.03"
MODEL_LR["RNE"]="0.003"
MODEL_LR["CatBoost"]="0.3"
MODEL_LR["CatBoostNN"]="0.0003"

## If SLURM is true, submit job and exit ##
if [ "$SLURM" = true ]; then
    MASTER_LOGFILE="$LOG_DIR/train-master-%j.log"
    # Construct sbatch command
    SBATCH_COMMAND="sbatch \
        --account csml \
        --partition a30 \
        -N 1 \
        -n 16 \
        --mem 32G \
        --gres gpu:1 \
        --time 16:00:00 \
        --job-name ${EXPT_NAME} \
        --output $MASTER_LOGFILE \
        $0"

    # Remove extra spaces from the command
    SBATCH_COMMAND=$(echo "$SBATCH_COMMAND" | tr -s ' ')

    # Append the --execute flag if originally set
    [ "$EXECUTE" = true ] && SBATCH_COMMAND="$SBATCH_COMMAND --execute"

    # Submit the job
    echo "SBATCH_COMMAND: $SBATCH_COMMAND"
    eval ${SBATCH_COMMAND}
    echo "MASTER_LOGFILE: $MASTER_LOGFILE"
    echo "----------------------------------------"

    # Exit the script
    exit 0
fi

## Environment Setup ##
module load conda
conda activate openne1

## Run your job ##
cd ~/scratch/shortest-distance-survey/src

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
            --eval_runs 10 \
            --seed 1234 \
            --device cuda \
            --time_limit 5 \
            --learning_rate ${custom_lr} \
            --epochs 12000 \
            --validate"

        # Remove extra spaces from the command
        PYTHON_COMMAND=$(echo "$PYTHON_COMMAND" | tr -s ' ')

        # Log file
        LOG_FILE="$LOG_DIR/train_${model_name}_${dataset}.log"

        # Prints
        echo "PYTHON_COMMAND: $PYTHON_COMMAND"
        echo "LOG_FILE: $LOG_FILE"

        # Run command if not in debug mode
        if [ "$EXECUTE" = false ]; then
            echo "EXECUTE: false (not executing the command)"
            echo "----------------------------------------"
        else
            echo "EXECUTE: true (executing the command)"
            mkdir -p $LOG_DIR
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
