#!/bin/bash
# Usage: sbatch generate_query_workload_v1.sh

## Resources needed ##
#SBATCH --account csml              # --account, account/queue name
#SBATCH --partition v100            # --partition, partition name
#SBATCH --qos standby               # --qos, quality of service
#SBATCH -N 1                        # --nodes, number of nodes
#SBATCH -n 16                       # --ntasks, cores
#SBATCH --mem 32G                   # --mem, memory per node
#SBATCH --gres gpu:1                # --gres, GPU resources
#SBATCH --time 00:30:00             # --time, time limit
#SBATCH -J generate_queries         # --job-name, job name
#SBATCH -o ../results/v34-real_workload_perturb_500k/%x-%u-%j.out      # standard output, %x=job-name, %u=user-name, %j=job-id

## Environment Setup ##
module load conda
conda activate openne1

## Run you job ##
cd ~/scratch/shortest-distance-survey/src

python generate_query_workload.py --data_name W_Jinan    --query_dir real_workload --query_strategy query_perturbation --perturb_k 5   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_Shenzhen --query_dir real_workload --query_strategy query_perturbation --perturb_k 7   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_Chengdu  --query_dir real_workload --query_strategy query_perturbation --perturb_k 36  --k_hop 2 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_Beijing  --query_dir real_workload --query_strategy query_perturbation --perturb_k 64  --k_hop 3 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_Shanghai --query_dir real_workload --query_strategy query_perturbation --perturb_k 10  --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_NewYork  --query_dir real_workload --query_strategy query_perturbation --perturb_k 5   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k
python generate_query_workload.py --data_name W_Chicago  --query_dir real_workload --query_strategy query_perturbation --perturb_k 128 --k_hop 3 --max_queries 500000 --save_dir real_workload_perturb_500k

## Capture memory and time footprint ##
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python -c "print('Hello World!')"

## Usage ##
# sbatch job.sh
# squeue -u $USER
# scancel <job-id>
