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
#SBATCH --time=00:10:00
#
## Command(s) to run (example):
cd /global/home/users/changli824/CNN
module load python
source activate /global/home/users/changli824/.conda/envs/3dcnn
"""
'''
for data in [10000]:
    for epoch in [10, 20]:
        for lr in [0.01, 0.001]:
            for batch_size in [16]:
                for nneighbor in [16, 64]:
                    for nblocks in [2, 4]:
                        for transformer_dim in [64, 256]:
                            rand_id = f"TSFM_{data}_n_{nblocks}_dim_{transformer_dim}_epoch{epoch}_neighb_{nneighbor}_lr_{lr}"
                            with open(f"{data}_gridsearch_{rand_id}.sh", "w") as w:
                                w.write(PREFACE % rand_id)
                                w.write(
                                    f"python scripts/trainPCE.py TSFM_{data}_n_{nblocks}_dim_{transformer_dim}_seeded {data} {epoch} {lr} {batch_size} {nneighbor} {nblocks} {transformer_dim} -savio\n"
                                )
'''

for kernel_size in [3]:
    for activation_fn in ['ReLU']:
        for epochs_choice in [5]:
            for learning_rate in [1e-4]:
                for batch_size in [4]:
                    rand_id = f'CNN_{kernel_size}_{activation_fn}_{epochs_choice}_{learning_rate}_{batch_size}'
                    with open(f'savio_scripts/{rand_id}.sh', 'w') as w:
                        w.write(PREFACE % rand_id)
                        w.write(f'./U-Net/Savio_3D_CNN.py --kernel_size {kernel_size} --activation_fn {activation_fn} --epochs_choice {epochs_choice} --learning_rate {learning_rate} --batch_size {batch_size}')
