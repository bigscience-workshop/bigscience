#!/bin/bash
#SBATCH --job-name=xp3jsonl # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr13f-6B3-ml-t0/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/p31t0/p31t0_train.jsonl
OUTPUT=/gpfswork/rech/six/commun/bigscience-training/p31t0bos/p31t0_train
TOKENIZER_PATH="bigscience/tokenizer"
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-bos \
    --workers 35
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 35


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/p31t0/p31t0_validation.jsonl
OUTPUT=/gpfswork/rech/six/commun/bigscience-training/p31t0bos/p31t0_validation
TOKENIZER_PATH="bigscience/tokenizer"

python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-bos \
    --workers 35
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 35
