#!/bin/bash
# submit_array_ckpt.sh

#SBATCH -A tial 
#SBATCH -p ckpt
#SBATCH --nodes 1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH -o ./slurm/%x/%A_%a.out
#SBATCH -e ./slurm/%x/%A_%a.err
#SBATCH --requeue

cd `pwd`
bash $1/$SLURM_ARRAY_TASK_ID.sh
