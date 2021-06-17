#!/bin/bash
#SBATCH --job-name=preprocesslmt5
#SBATCH --partition=prepost
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --cpus-per-task=40            # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=10:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out               # output file name
#SBATCH --error=%x-%j.out                # error file name (same to watch just one file)
#SBATCH --account=six@gpu
#SBATCH --mail-type=ALL

set -x -e

source $six_ALL_CCFRWORK/start-prod
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET=openwebtext
LOGG_FREQUENCY=125
SAVE_FREQUENCY=250
EVAL_FREQUENCY=100000
SERIALIZATION_DIR=${eha_ALL_CCFRSCRATCH}/experiments/preprocesslmt5
LOGGING_DIR=${eha_ALL_CCFRSCRATCH}/tensorboard/preprocesslmt5

python ${six_ALL_CCFRWORK/code/bigscience/jz/scripts/run_text2text.py \
  --model_type t5 \
  --tokenizer_name t5-small \
  --config_name ${six_ALL_CCFRWORK/code/bigscience/jz/configs/lm_t5/lm_t5-tiny.json \
  --dataset_name ${DATASET} --block_size 512 \
  --preprocessing_num_workers 76 \
  --do_train --do_eval \
  --max_train_samples 1 --max_val_samples 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
  --report_to tensorboard \
  --logging_strategy steps --logging_first_step --logging_dir ${LOGGING_DIR} --logging_steps ${LOGG_FREQUENCY} \
  --eval_steps ${EVAL_FREQUENCY} --evaluation_strategy steps \
  --save_strategy steps --save_steps ${SAVE_FREQUENCY} --save_total_limit 200
