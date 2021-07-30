#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-gpt2 # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

input=$six_ALL_CCFRSCRATCH/datasets/oscar-small/oscar-en-shuffled-p1.jsonl
output=$six_ALL_CCFRSCRATCH/datasets/oscar-small/meg-gpt2-p1

cd $six_ALL_CCFRWORK/code/megatron-lm
/usr/bin/time -v python tools/preprocess_data.py \
       --input $input \
       --output-prefix $output \
       --vocab data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --workers 16

#echo "now copy the results to $six_ALL_CCFRWORK/datasets-custom/oscar/ from $six_ALL_CCFRSCRATCH/datasets/oscar-small/meg-gpt2"
