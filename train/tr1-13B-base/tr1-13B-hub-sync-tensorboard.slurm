#!/bin/bash
#SBATCH --job-name=tr1-13B-hub-sync-tensorboard  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=40           # number of cores per task
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=prepost

module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard

tools/hub-sync.py --repo-path $TENSORBOARD_PATH --patterns '*tfevents*' -d
