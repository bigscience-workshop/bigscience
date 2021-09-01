#!/bin/bash
#SBATCH --job-name=oscar-to-jsonl    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

# 1. first run this script directly by bash (not sbatch) after commenting out the offline exports on
# a login instance as there is little processing needed, once the data is downloaded kill the script

# 2. after the download was done, then run this is a slurm with the exports enabled
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# use SCRATCH for building as it's much faster
cd $six_ALL_CCFRSCRATCH/datasets/oscar-small
$six_ALL_CCFRWORK/code/bigscience/data/oscar/oscar-to-jsonl.py

