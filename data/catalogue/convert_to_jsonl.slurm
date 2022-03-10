#!/bin/bash
#SBATCH --job-name=convert_datasets_to_jsonl # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/convert_to_jsonl/%x-%j.out          # output file name
#SBATCH --array=0-501
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

# ======= Generate json file ======

DATASET_PATHS=($(ls -d /gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/* /gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/*))
DATASET_PATH=${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}

BIGSCIENCE_REPO=$WORK/code/big_science/bigscience
SAVE_JSON_DATASET_PATH_PREFIX=$six_ALL_CCFRWORK/bigscience-training/jsonl_v2
mkdir -p $SAVE_JSON_DATASET_PATH_PREFIX

python $BIGSCIENCE_REPO/data/catalogue/sample_and_convert_to_jsonl.py \
    --dataset-path $DATASET_PATH\
    --save-json-dataset-path-prefix $SAVE_JSON_DATASET_PATH_PREFIX \
    --num-proc 10 \
    --batch-size 100
