#!/bin/bash
#############################################################
# Load slurm, if not present:           $ module load slurm
# Submit the job:                       $ sbatch job.sh
# Check status of all jobs:             $ squeue -u <username> (or `squeue --me`)
# Check specific job status (overview): $ jobinfo <job-id>
# Check specific job status (detailed): $ scontrol show job <job-id>
# Job output:                           $ cat <job-id>.out
# Cancel the job:                       $ scancel <job-id>
#############################################################

## Resources needed ##
#SBATCH --account csml              # --account, account/queue name
#SBATCH --partition a30             # --partition, partition name
#SBATCH --qos standby               # --qos, quality of service
#SBATCH -N 1                        # --nodes, number of nodes
#SBATCH -n 4                        # --ntasks, cores
#SBATCH --mem 10G                   # --mem, memory per node
#SBATCH --gres gpu:1                # --gres, GPU resources
#SBATCH --time 00:02:00             # --time, time limit
#SBATCH -J basic_job                # --job-name, job name
#SBATCH -o ../results/default/%x-%u-%j.out      # standard output, %x=job-name, %u=user-name, %j=job-id

# can skip this, and it will write error to the same file as output
# SBATCH -e %j.err            # standard error

## Environment Setup ##
module load conda
conda activate myenv

## Checks ##
module list
python -c "import torch; print('Torch Version: ', torch.__version__); print('Torch CUDA Version: ', torch.version.cuda); print('CUDA available: ', torch.cuda.is_available()); print('CUDA device: ', torch.cuda.current_device()); print('CUDA device name: ', torch.cuda.get_device_name())"
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"
python -c "import pandas, numpy, networkx, seaborn, tqdm, matplotlib; print('Done')"
conda info --envs
nvidia-smi
nvcc --version
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Hostname="$(/bin/hostname)
echo "SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME
echo "SLURM_SUBMIT_HOST="$SLURM_SUBMIT_HOST
echo "SLURM_JOB_PARTITION="$SLURM_JOB_PARTITION
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo

# ## Monitoring job - START ##
# module load utilities monitor
# # track per-code CPU load
# monitor cpu percent --all-cores >cpu-percent.log &
# CPU_PID=$!
# # track memory usage
# monitor cpu memory >cpu-memory.log &
# MEM_PID=$!

## Run you job ##
# change directory
cd ~/scratch/shortest-distance/src

## Capture memory and time footprint ##
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python -c "print('Hello World!')"

# ## Monitoring job - END ##
# ## shut down the resource monitors (to be used together with ) ##
# kill -s INT $CPU_PID $MEM_PID
