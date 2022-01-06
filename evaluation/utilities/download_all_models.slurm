#!/bin/bash
#SBATCH --job-name=download_all_models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x.out           # output file name
#SBATCH --account=six@gpu
#SBATCH --partition=compil

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_lm_eval

# TODO: replace with local fork of bigscience
BIGSCIENCE_REPO=$WORK/code/big_science/bigscience/evaluation/results/tr3

pushd $BIGSCIENCE_REPO

# TODO: replace with experiment / steps
EXPERIMENTS=bigscience/tr3d-1B3-oscar-checkpoints,bigscience/tr3e-1B3-c4-checkpoints,bigscience/tr3m-1B3-pile-checkpoints
STEPS=$(python -c "print(\",\".join([str(i) for i in range(19500, 118500, 1500)]))")

python download_all_models.py --experiments $EXPERIMENTS --steps $STEPS
