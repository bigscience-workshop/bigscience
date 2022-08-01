#!/bin/bash
#SBATCH --job-name=prepare-xp3    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=compil

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_t_zero_evaluation

cd /gpfswork/rech/six/commun/bigscience-training/jsonls/xp3/train/
python /gpfswork/rech/six/commun/bigscience-training/jsonls/xp3/train/prepare_xp3.py
