#!/bin/bash
#SBATCH --job-name=catalogue-jsonl-to-meg-ds # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/catalogue-jsonl-to-meg-ds/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-501
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

# ======= Generate meg-ds file ======

DATASET_PATHS=($(ls -d $six_ALL_CCFRWORK/bigscience-training/jsonl_v2/**/*.jsonl))
DATASET_PATH=${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

DATASET_NAME=$(basename $DATASET_PATH)

SAVE_MEG_DS_DATASET=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/meg-ds/"$DATASET_NAME"/meg_ds_"${TOKENIZER_NAME_OR_PATH//\//_}"

mkdir -p $(dirname $SAVE_MEG_DS_DATASET)

if [[ -f "$SAVE_MEG_DS_DATASET"_text_document.bin ]];
then
    echo "$SAVE_MEG_DS_DATASET exists."
    exit 0
fi

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
/usr/bin/time -v python tools/preprocess_data_many_cores.py \
       --input $DATASET_PATH \
       --output-prefix $SAVE_MEG_DS_DATASET \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
       --append-eod \
       --workers 40
