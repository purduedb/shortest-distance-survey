#!/bin/bash
#SBATCH --job-name=machine_info
#SBATCH --output=%x_%j_training.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=training
#SBATCH --account=csit
#SBATCH --qos=training
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G


echo "=== lscpu (summary) ===" > "machine_info.txt"
lscpu | grep -E "Model name|^CPU\(s\):|Socket|Core\(s\) per socket|Thread" >> "machine_info.txt"
echo "" >> "machine_info.txt"

echo "=== nvidia-smi (summary) ===" >> "machine_info.txt"
nvidia-smi --query-gpu=name,memory.total,pci.bus_id --format=csv >> "machine_info.txt"
echo "" >> "machine_info.txt"

echo "=== os-release ===" >> "machine_info.txt"
cat /etc/os-release >> "machine_info.txt"
echo "" >> "machine_info.txt"

echo "=== hostname ===" >> "machine_info.txt"
hostname >> "machine_info.txt"

echo "Done on $(hostname)"
