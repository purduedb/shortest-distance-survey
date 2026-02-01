#!/bin/bash
# Usage: sbatch compute_embeddings.sh

## Resources needed ##
#SBATCH -A csml-b             # --account, account/queue name {debug, standby, csml-b}
#SBATCH -N 1                 # --nodes, number of nodes
#SBATCH -n 8                 # --ntasks, cores
#SBATCH --gpus=1             # --gpus, number of GPUs
#SBATCH -t 12:00:00          # --time, time limit
#SBATCH -J compute-embeddings          # --job-name, job name
#SBATCH -o ../results/v25-compute-embeddings-%j.log      # standard output, %x=job-name, %u=user-name, %j=job-id

## Environment Setup ##
module load conda
conda activate openne1

## Run you job ##
cd ~/scratch/shortest-distance/third_party/OpenNE/src

DATASETS=(
    ## Sample datasets ##
    "Surat_subgraph"

    ## Workload datasets ##
    'W_Beijing'
    'W_Chicago'
    'W_NewYork'
    'W_Jinan'
    'W_Shenzhen'
    'W_Chengdu'

    ## Small (<10k nodes) ##
    "Surat"  # 1843
    "Quanzhou"  # 3079
    "Dongguan"  # 4606
    "Fuzhou"  # 5955
    "Harbin"  # 6328
    "Qingdao"  # 7992
    "Dhaka"  # 8349
    "Ahmedabad"  # 8808
    "Shenyang"  # 9020
    "Chongqing"  # 9519

    ## Medium (10k-100k nodes) ##
    "Pune"  # 14542
    "Shenzhen"  # 21809
    "Mumbai"  # 25668
    "Delhi"  # 37971
    "Beijing"  # 45201
    "Jacarta"  # 50215
    "Naples"  # 63070
    "Hyderabad"  # 74357
    "Rome"  # 87739
    "Istambul"  # 98573

    # ## Large (100k-1M nodes) ##
    # "Atlanta"  # 113028
    # "Milan"  # 119638
    # "London"  # 150598
    # "Boston"  # 181666
    # "NewYork"  # 190368
    # "Phoenix"  # 200519
    # "LosAngeles"  # 228185
    # "Chicago"  # 259887
    # "Paris"  # 281031
    # "Moscow"  # 406505

    ## Extra Large (>1M nodes) ##
    # ...

    # ## DIMACS datasets ##
    # "FLA"
    # "E"
)

MODELS=(
    "node2vec"
    # "deepwalk"  # Uncomment to run deepwalk
    "line"  # Uncomment to run LINE
)
DIM=64  # Representation size
WORKERS=8  # Number of workers, applicable to node2vec and deepwalk
ORDER=2  # Order for LINE method
EPOCHS=1  # Number of epochs
WEIGHTED=0  # Use 0 for no weight, 1 for weighted edges

COUNTER=0  # Counter for number of commands
TOTAL_COMMANDS=$((${#DATASETS[@]} * ${#MODELS[@]}))  # Total number of commands to run
for dataset in "${DATASETS[@]}"; do
    for method in "${MODELS[@]}"; do
        # Increment counter
        COUNTER=$((COUNTER + 1))
        echo "Processing [$COUNTER/$TOTAL_COMMANDS]: Dataset=$dataset, Method=$method"

        if [ "$method" == "node2vec" ]; then
            ARGS="--method node2vec --representation-size $DIM --workers $WORKERS --epochs $EPOCHS"
            if [ $WEIGHTED -eq 1 ]; then
                ARGS="$ARGS --weighted"
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_weighted.embeddings"
            else
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_unweighted.embeddings"
            fi
        elif [ "$method" == "deepwalk" ]; then
            ARGS="--method deepwalk --representation-size $DIM --workers $WORKERS --epochs $EPOCHS"
            if [ $WEIGHTED -eq 1 ]; then
                ARGS="$ARGS --weighted"
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_weighted.embeddings"
            else
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_unweighted.embeddings"
            fi
        elif [ "$method" == "line" ]; then
            ARGS="--method line --representation-size $DIM --workers $WORKERS --epochs $EPOCHS --order $ORDER"
            if [ $WEIGHTED -eq 1 ]; then
                ARGS="$ARGS --weighted"
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_order${ORDER}_weighted.embeddings"
            else
                OUT="../../../data/$dataset/${method}_dim${DIM}_epochs${EPOCHS}_order${ORDER}_unweighted.embeddings"
            fi
        else
            echo "Unknown method: $method"
            exit 1
        fi

        ## Capture memory and time footprint ##
        /usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
        python -m openne --input "../../../data/${dataset}/${dataset}.edges" $ARGS --output "$OUT"
    done
done

echo "Finished."

## Usage ##
# sbatch job.sh
# squeue -u $USER
# scancel <job-id>
