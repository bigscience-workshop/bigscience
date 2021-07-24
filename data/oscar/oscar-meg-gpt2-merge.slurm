#!/bin/bash
#SBATCH --job-name=oscar-meg-gpt2-merge.slurm # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10           # number of cores per task
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@cpu

set -x -e

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
DATA=$six_ALL_CCFRSCRATCH/datasets/oscar-small
/usr/bin/time -v python tools/preprocess_data.py \
    --datasets \
    $DATA/meg-gpt2-p1_text_document \
    $DATA/meg-gpt2-p2_text_document \
    $DATA/meg-gpt2-p3_text_document \
    $DATA/meg-gpt2-p4_text_document \
    --output-prefix $DATA/meg-gpt2 \
    --merge-file data/gpt2-merges.txt \
    --vocab data/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer

