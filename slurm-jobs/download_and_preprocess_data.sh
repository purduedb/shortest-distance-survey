#!/bin/bash
# Usage: sbatch download_data.sh

## Resources needed ##
#SBATCH -A debug             # --account, account/queue name {debug, standby, csml-b}
#SBATCH -N 1                 # --nodes, number of nodes
#SBATCH -n 4                 # --ntasks, cores
#SBATCH --gpus=1             # --gpus, number of GPUs
#SBATCH -t 00:30:00          # --time, time limit
#SBATCH -J download_data          # --job-name, job name
#SBATCH -o ../results/logs/expt-21-download-master-%j.log      # standard output, %x=job-name, %u=user-name, %j=job-id

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo

## Environment Setup ##
module load conda
conda activate myenv

## Run you job ##
cd ~/scratch/shortest-distance/src

## Capture memory and time footprint ##
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python download_and_preprocess_data.py

echo "Finished."

## Usage ##
# sbatch job.sh
# squeue -u $USER
# scancel <job-id>
