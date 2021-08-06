#!/bin/bash
#SBATCH --job-name=tr1-13B-hub-sync-logs  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=prepost

echo "START TIME: $(date)"

module load git-lfs

DATA_OUTPUT_PATH=$SCRATCH/synched_exps/tr3-1B3-full
LOGS_PATH=$DATA_OUTPUT_PATH/logs
BIG_SCIENCE_REPO_PATH=$SCRATCH/bigscience

$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $LOGS_PATH --patterns '*.out' -d

echo "END TIME: $(date)"
