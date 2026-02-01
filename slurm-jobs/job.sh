#!/bin/bash
# Usage: sbatch job.sh

## Resources needed ##
#SBATCH --account csml              # --account, account/queue name
#SBATCH --partition a30             # --partition, partition name
#SBATCH --qos standby               # --qos, quality of service
#SBATCH -N 1                        # --nodes, number of nodes
#SBATCH -n 4                        # --ntasks, cores
#SBATCH --mem 10G                   # --mem, memory per node
#SBATCH --gres gpu:1                # --gres, GPU resources
#SBATCH --time 00:02:00             # --time, time limit
#SBATCH -J job                      # --job-name, job name
#SBATCH -o ../results/default/%x-%u-%j.out      # standard output, %x=job-name, %u=user-name, %j=job-id

## Environment Setup ##
module load conda
conda activate myenv

## Run you job ##
cd ~/scratch/shortest-distance/src

## Capture memory and time footprint ##
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python -c "print('Hello World!')"

## Usage ##
# sbatch job.sh
# squeue -u $USER
# scancel <job-id>
