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

TOKENIZER_PATH="bigscience/tokenizer"

LANGS=(
ln
)


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/xp3/train
cd $DATA_PATH


for LANG in LANGS
do
    cd $LANG
    cat *.jsonl > merged_$LANG.jsonl
    OUTPUT=/gpfswork/rech/six/commun/bigscience-training/xp3/train/xp3_$LANG
    python tools/preprocess_data.py \
        --input $DATA_PATH \
        --output-prefix $OUTPUT \
        --dataset-impl mmap \
        --json-key inputs \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --workers 35
    python tools/preprocess_data.py \
        --input $DATA_PATH \
        --output-prefix $OUTPUT \
        --dataset-impl mmap \
        --json-key targets \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --append-eod \
        --prepend-space \
        --workers 35
done


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/xp3/validation
cd $DATA_PATH


for LANG in LANGS
do
    cd $LANG
    cat *.jsonl > merged_$LANG.jsonl
    OUTPUT=/gpfswork/rech/six/commun/bigscience-training/xp3/validation/xp3_$LANG
    python tools/preprocess_data.py \
        --input $DATA_PATH \
        --output-prefix $OUTPUT \
        --dataset-impl mmap \
        --json-key inputs \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --workers 35
    python tools/preprocess_data.py \
        --input $DATA_PATH \
        --output-prefix $OUTPUT \
        --dataset-impl mmap \
        --json-key targets \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $TOKENIZER_PATH \
        --append-eod \
        --prepend-space \
        --workers 35
done
