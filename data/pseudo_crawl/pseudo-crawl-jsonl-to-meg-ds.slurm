#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-gpt2 # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=10:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-613
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling
pushd $DATA_TOOLING_REPO

SEED_ID=$(python cc_pseudo_crawl/python_scripts/load_all_seed_ids.py \
  --seed-paths "$DATA_TOOLING_REPO"/cc_pseudo_crawl/seeds_batch_1/sourcing_sheet_seeds/seeds.csv,"$DATA_TOOLING_REPO"/cc_pseudo_crawl/seeds_batch_2/sourcing_sheet_seeds/seeds.csv \
  --seed-index $SLURM_ARRAY_TASK_ID \
)

DATASET_NAME= # TODO: Set it correctly
input= # TODO: Set it correctly
output=$six_ALL_CCFRSCRATCH/bigscience-datasets/pseudo-crawl/meg-ds/$DATASET_NAME

mkdir -p $(dirname $output)

if [[ -f "$output"_text_document.bin ]];
then
    echo "$output exists."
    exit 0
fi

echo "Downloading ${DATASET_NAME}"
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TOKENIZER_NAME_OR_PATH= # TODO: Set it correctly

pushd $six_ALL_CCFRWORK/code/megatron-lm
/usr/bin/time -v python tools/preprocess_data.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
       --append-eod \
       --workers 40
