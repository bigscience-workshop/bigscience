#!/bin/bash
#SBATCH --job-name=oscar-meg-gpt2-merge.slurm # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10           # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
DATA=$six_ALL_CCFRSCRATCH/datasets/oscar-small
/usr/bin/time -v python tools/merge_preprocessed_data.py \
    --datasets \
    $DATA/meg-gpt2-p1_text_document \
    $DATA/meg-gpt2-p2_text_document \
    $DATA/meg-gpt2-p3_text_document \
    $DATA/meg-gpt2-p4_text_document \
    --output-prefix $DATA/meg-gpt2_text_document
