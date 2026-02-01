#!/bin/bash
# Usage: bash generate_query_workload.sh [--execute | -e] [--slurm | -s]
# Dry run locally:      bash generate_query_workload.sh
# Run locally:          bash generate_query_workload.sh -e
# Dry run on SLURM:     bash generate_query_workload.sh -s
# Run on SLURM:         bash generate_query_workload.sh -s -e

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
EXPT_NAME="v25-query_landmark_100_split_80_20"
LOG_DIR="../results/$EXPT_NAME"
DATASETS=(
    ## Workload datasets ##
    'W_Jinan'  # 8.9k
    'W_Shenzhen'  # 11.9k
    'W_Chengdu'  # 17.5k
    'W_Beijing'  # 74.3k
    'W_NewYork'  # 334.9k
    'W_Chicago'  # 386.5k

    ## Small (<10k nodes) ##
    'Surat'  # 2.5k
    'Quanzhou'  # 5.2k
    'Dongguan'  # 7.7k
    'Harbin'  # 10.1k
    'Fuzhou'  # 10.6k
    'Ahmedabad'  # 12.7k
    'Shenyang'  # 12.8k
    'Qingdao'  # 13.2k
    'Dhaka'  # 14.7k
    'Chongqing'  # 20.6k

    # ## Medium (10k-100k nodes) ##
    'Pune'  # 27.8k
    'Shenzhen'  # 34.2k
    'Mumbai'  # 44.9k
    'Delhi'  # 52.6k
    'Jacarta'  # 73.0k
    'Beijing'  # 74.4k
    'Naples'  # 124.6k
    'Hyderabad'  # 127.5k
    'Istambul'  # 131.1k
    'Rome'  # 158.5k

    ## Large (100k-1M nodes) ##
    'Milan'  # 187.5k
    'London'  # 285.0k
    'Atlanta'  # 310.6k
    'NewYork'  # 334.9k
    'Phoenix'  # 340.6k
    'Boston'  # 350.9k
    'Chicago'  # 386.5k
    'LosAngeles'  # 398.6k
    'Paris'  # 461.5k
    'Moscow'  # 685.1k

    # ## DIMACS datasets ##
    # "FLA"  # 1.07M
    # "E"  # 3.5M
)

## If SLURM is true, submit job and exit ##
if [ "$SLURM" = true ]; then
    MASTER_LOGFILE="$LOG_DIR/query-master-%j.log"
    # Construct sbatch command
    SBATCH_COMMAND="sbatch \
        --account csml \
        --partition v100 \
        --qos standby \
        -N 1 \
        -n 8 \
        --mem 32G \
        --gres gpu:1 \
        --time 04:00:00 \
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

## Run you job ##
cd ~/scratch/shortest-distance/src

## MAIN ##
SECONDS=0  # Timer for total duration
COUNTER=0  # Counter for number of commands
TOTAL_COMMANDS=${#DATASETS[@]}  # Total number of commands to run
for dataset in "${DATASETS[@]}"; do
    # Increment counter
    COUNTER=$((COUNTER + 1))
    echo "Task: $COUNTER / $TOTAL_COMMANDS"

    # Python command
    PYTHON_COMMAND="python generate_query_workload.py \
        --data_name ${dataset} \
        --data_dir ../data \
        --query_strategy landmark_split \
        --query_dir query_landmark_100_split_80_20 \
        --num_landmarks 100 \
        --num_random_pairs 1000000 \
        --seed 42"

    # Remove extra spaces from the command
    PYTHON_COMMAND=$(echo "$PYTHON_COMMAND" | tr -s ' ')

    # Log file
    LOG_FILE="$LOG_DIR/query_${dataset}.log"

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
duration=$SECONDS  # Timer for total duration
echo "Finished."

echo
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
printf "Total Time Elapsed: %d-%02d:%02d:%02d\n" $((duration/86400)) $(( (duration%86400)/3600 )) $(( (duration%3600)/60 )) $(( duration%60 ))
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
