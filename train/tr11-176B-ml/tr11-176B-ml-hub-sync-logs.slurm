#!/bin/bash
#SBATCH --job-name=tr11-176B-ml-hub-sync-logs
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil
#SBATCH --account=six@cpu

echo "START TIME: $(date)"

source $six_ALL_CCFRWORK/code/tr8b-104B/bigscience/train/tr8b-104B/start-tr8b-104B
module load git-lfs

BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/tr8b-104B/bigscience
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/tr11-176B-ml-logs

$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $DATA_OUTPUT_PATH --patterns 'tensorboard/*/*tfevents*' 'logs/*/*.txt' -d

echo "END TIME: $(date)"
