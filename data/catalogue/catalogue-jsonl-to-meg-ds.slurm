#!/bin/bash
#SBATCH --job-name=catalogue-jsonl-to-meg-ds # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/catalogue-jsonl-to-meg-ds/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-509
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

# Get dataset names

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

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v2-dedup-lines-articles

input=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/jsonl_8_shards/$DATASET_NAME/data_0_8.jsonl
output=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/meg-ds/"$DATASET_NAME"/meg_ds_"${TOKENIZER_NAME_OR_PATH//\//_}"_shard_0_8

mkdir -p $(dirname $output)

if [[ -f "$output"_text_document.bin ]];
then
    echo "$output exists."
    exit 0
fi

echo "Downloading ${DATASET_NAME}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
/usr/bin/time -v python tools/preprocess_data_many_cores.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
       --append-eod \
       --workers 40
