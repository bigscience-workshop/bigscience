#!/bin/bash
#SBATCH --job-name=jsonl-to-meg-t5   # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=std-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext/openwebtext.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-t5 \
       --vocab $six_ALL_CCFRWORK/datasets-custom/vocabs/bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers 8
