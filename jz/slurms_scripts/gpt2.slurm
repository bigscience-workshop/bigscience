#!/bin/bash
#SBATCH --job-name=gpt2_repro_initial # job name
#SBATCH --partition=gpu_p13           # partition with 8 32GB gpu nodes
#SBATCH --qos=qos_gpu-t4              # t4 enables 100H trainings
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=10             # number of cores per tasks
#SBATCH --output=%j.out          # output file name
#SBATCH --error=%j.out           # error file name (same to watch just one file)
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
N_LAYER=3
N_EMBD=128
N_INNER=128
N_HEAD=8
LOG_FREQUENCY=10000
RUN_NAME=${N_LAYER}-${N_EMBD}-${N_INNER}
SERIALIZATION_DIR=${SCRATCH}/experiments/gpt2_repro/${RUN_NAME}
LOGGING_DIR=${SCRATCH}/tensorboard/gpt2_repro/${RUN_NAME}

deepspeed ${SCRATCH}/code/jay-z/scripts/run_clm.py \
  --deepspeed ${SCRATCH}/code/jay-z/configs/deepspeed/ds_zero2.json \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --dataset_name ${DATASET} --block_size 1024 \
  --cache_dir ${ALL_CCFRSCRATCH}/cache_dir \
  --preprocessing_num_workers 76 \
  --do_train --do_eval \
  --max_steps 15000 \
  --max_train_samples 10000000 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 8 \
  --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
  --report_to tensorboard \
  --logging_strategy steps --logging_first_step --logging_dir ${LOGGING_DIR} --logging_steps ${LOG_FREQUENCY} \
  --eval_steps ${LOG_FREQUENCY} --evaluation_strategy steps \
  --save_strategy steps --save_steps ${LOG_FREQUENCY} --save_total_limit 31 \
  --n_layer ${N_LAYER} --n_embd ${N_EMBD} --n_inner ${N_INNER} --n_head ${N_HEAD}
