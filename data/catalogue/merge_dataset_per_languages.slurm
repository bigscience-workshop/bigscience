#!/bin/bash
#SBATCH --job-name=catalogue-jsonl-to-meg-ds # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/merge-meg-ds/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=compil

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

BATCH_ID=0
TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v2-dedup-lines-articles

# ======= Get dataset names ======

CATALOGUE_DATA_REPO=$WORK/code/big_science/catalogue_data
pushd $CATALOGUE_DATA_REPO

DATASET_PATH=$(python get_dataset_path.py --dataset-ratios-file training_dataset_ratios.json --index $SLURM_ARRAY_TASK_ID)
echo "Converting $DATASET_PATH to json"

if [[ $DATASET_PATH == /gpfswork/rech/six/uty16tp/dataset/tokenization/* ]]
then
    if [[ $DATASET_PATH == */data ]]
    then
        DATASET_NAME=${DATASET_PATH:48:-5}
    else
        DATASET_NAME=${DATASET_PATH:48}
    fi
else
    DATASET_NAME=$DATASET_PATH
fi

# ======= Generate language ratio file ======

BIGSCIENCE_REPO=$WORK/code/big_science/bigscience

LANGUAGE_RATIOS_PATH=$BIGSCIENCE_REPO/data/catalogue/training_dataset_ratios_batch_${BATCH_ID}_per_language.json
MEG_DS_DATASET_PREFIX=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/meg-ds-per-lang/{lang}/"${TOKENIZER_NAME_OR_PATH//\//_}"_batch_${BATCH_ID}_text_document

mkdir -p $(dirname $MEG_DS_DATASET_PREFIX)

python $BIGSCIENCE_REPO/data/catalogue/merge_dataset_per_language.py \
    --dataset-ratios-path $BIGSCIENCE_REPO/data/catalogue/training_dataset_ratios_batch_$BATCH_ID.json \
    --split train \
    --meg-ds-dataset-prefix MEG_DS_DATASET_PREFIX \
    --output-ratio-file $LANGUAGE_RATIOS_PATH

# ======= Generate merged files ======

MEG_DS_REPO=$WORK/code/big_science/Megatron-Deepspeed
pushd $MEG_DS_REPO
MERGE_ARGUMENTS=(
python -c "
import json

with open($LANGUAGE_RATIOS_PATH, 'r') as fi:
    data = json.load(fi)

print('\n'.join(
    [f\"--datasets {' '.join(elt['original_datasets'])} --output-prefix dataset['dataset_path']\" for lang, dataset in data.]
))
"
)

echo $MERGE_ARGUMENTS

for MERGE_ARGUMENT in "${MERGE_ARGUMENTS[@]}"
do
	/usr/bin/time -v python tools/merge_preprocessed_data.py \
        $MERGE_ARGUMENT
done
