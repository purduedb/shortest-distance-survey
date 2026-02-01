#!/bin/bash
## Resources needed, if not specified in sbatch command line ##
#SBATCH -A debug             # --account, account/queue name {debug, standby, csml-b}
#SBATCH -N 1                 # --nodes, number of nodes
#SBATCH -n 16                 # --ntasks, cores
#SBATCH --gpus=1             # --gpus, number of GPUs
#SBATCH -t 00:30:00          # --time, time limit
#SBATCH -J dataset_summary          # --job-name, job name
#SBATCH -o ../results/logs/%x-%u-%j.out      # standard output, %x=job-name, %u=user-name, %j=job-id

## Environment Setup ##
module load conda
conda activate myenv

## Run you job ##
cd ~/scratch/shortest-distance-survey/src

# Check if the PYTHON_COMMAND variable is set, if not, use a default command
PYTHON_COMMAND=${PYTHON_COMMAND:-"python dataset_summary.py"}
echo "Running command: $PYTHON_COMMAND"

## Capture memory and time footprint ##
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
$PYTHON_COMMAND
