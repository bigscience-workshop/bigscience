#!/bin/bash
#SBATCH --job-name=oscar-to-jsonl    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod

# must have already downloaded the data in a networked instance
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# use SCRATCH for building as it's much faster
cd $six_ALL_CCFRSCRATCH/datasets/oscar-small
$six_ALL_CCFRWORK/code/bigscience/data/oscar/oscar-to-jsonl.py
