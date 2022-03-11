#!/bin/bash
#SBATCH --job-name=catalogue-jsonl-to-meg-ds # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/merge-meg-ds/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-48
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

# ======= Generate merged files ======

MEG_DS_REPO=$six_ALL_CCFRWORK/code/Megatron-DeepSpeed
pushd $MEG_DS_REPO

BASE_PATH=$six_ALL_CCFRSCRATCH/bigscience-datasets/meg-ds_v2
LANGUAGES=($(ls $BASE_PATH))
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

SAVE_PATH=$six_ALL_CCFRSCRATCH/bigscience-datasets/merged-meg-ds_v2/$LANG/"${TOKENIZER_NAME_OR_PATH//\//_}"_${LANG}_text_document
# fancy way of collecting all datasets within a folder
DATASETS=$(ls $six_ALL_CCFRSCRATCH/bigscience-datasets/meg-ds_v2/$LANG/**/*.bin | xargs -I {} python -c "print('{}'[:-4])")

mkdir -p $(dirname $SAVE_PATH)

/usr/bin/time -v python -m tools.merge_preprocessed_data \
    --datasets $DATASETS \
    --output-prefix $SAVE_PATH
