#!/bin/bash
# Job name:
#SBATCH --job-name=train_10k
#
# Account:
#SBATCH --account=fc_caddatabase
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
## Command(s) to run (example):
module load python/3.8.8
#pip install --user wandb
module load cuda/10.2

#source activate /global/scratch/users/sarashonkwiler/3d
source activate /global/scratch/users/sarashonkwiler/3d

python one_model_regression_once_data_loaded.py >& one_model_regression_once_data_loaded.out