#!/bin/bash
#SBATCH --job-name=get_sizes # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/get_sizes/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-497
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

# common repo
BIGSCIENCE_REPO=/gpfswork/rech/six/commun/code/bigscience
cd $BIGSCIENCE_REPO
OUTPUT_FOLDER=$BIGSCIENCE_REPO/sizes_per_dataset/
mkdir -p $OUTPUT_FOLDER

DATASET_PATHS=($(ls -d $six_ALL_CCFRWORK/bigscience-training/jsonls/**/**/*.jsonl))
DATASET_PATH=${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}

DATASET_NAME_WITH_JSONL=$(basename $DATASET_PATH)
DATASET_NAME=${DATASET_NAME_WITH_JSONL:0:-6}

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/usr/bin/time -v python data/catalogue/get_sizes.py \
       --input-path $DATASET_PATH \
       --output-folder $OUTPUT_FOLDER
