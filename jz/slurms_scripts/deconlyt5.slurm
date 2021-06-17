#!/bin/bash
#SBATCH --job-name=deconlyt5
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4              # t4 enables 100H trainings
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --gres=gpu:8                  # number of GPUs per node
#SBATCH --cpus-per-task=24            # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
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
EVAL_FREQUENCY=1000
SERIALIZATION_DIR=${eha_ALL_CCFRSCRATCH}/experiments/dec_only_t5-medium
LOGGING_DIR=${eha_ALL_CCFRSCRATCH}/tensorboard/dec_only_t5-medium

deepspeed ${six_ALL_CCFRWORK/code/bigscience/jz/scripts/run_clm.py \
  --deepspeed ${SCRATCH}/code/bigscience/jz/configs/deepspeed/ds_zero3.json \
  --model_type decoder_only_t5 \
  --tokenizer_name t5-small \
  --config_name ${six_ALL_CCFRWORK/code/bigscience/jz/configs/dec_only_t5/decoder_only_t5-medium.json \
  --dataset_name ${DATASET} --block_size 1024 \
  --preprocessing_num_workers 76 \
  --do_train --do_eval \
  --max_steps 34000 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --learning_rate 6e-4 \
  --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1 \
  --warmup_steps 800 \
  --max_grad_norm 1.0 \
  --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
  --report_to tensorboard \
  --logging_strategy steps --logging_first_step --logging_dir ${LOGGING_DIR} --logging_steps ${LOGG_FREQUENCY} \
  --eval_steps ${EVAL_FREQUENCY} --evaluation_strategy steps --max_val_samples 10000 \
  --save_strategy steps --save_steps ${SAVE_FREQUENCY} --save_total_limit 200
