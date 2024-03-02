#!/bin/bash
# Job name:
#SBATCH --job-name=3dcnn_test_run_10k
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
#SBATCH --time=30:00:00
#
## Command(s) to run (example):
cd /global/home/users/changli824/CNN/U-Net
module load python
source activate /global/home/users/changli824/.conda/envs/3dcnn
python3 Savio_3D_CNN.py --file hyper.pkl
