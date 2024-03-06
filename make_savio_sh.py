import random

PREFACE = """#!/bin/bash
# Job name:
#SBATCH --job-name=%s
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
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
cd /global/home/users/changli824/CNN
module load python
source activate /global/scratch/users/changli824/conda/envs/3dcnn
"""
for kernel_size in [3, 5]:
    for activation_fn in ['ReLU', 'Sigmoid']:
        for epochs_choice in [5, 10, 15]:
            for learning_rate in [1e-4]:
                for batch_size in [4]:
                    rand_id = f'CNN_{kernel_size}_{activation_fn}_{epochs_choice}_{learning_rate}_{batch_size}'
                    with open(f'savio_scripts/{rand_id}.sh', 'w') as w:
                        w.write(PREFACE % rand_id)
                        w.write(f'./3DCNN/Savio_3D_CNN.py --kernel_size {kernel_size} --activation_fn {activation_fn} --epochs_choice {epochs_choice} --learning_rate {learning_rate} --batch_size {batch_size}')
