#!/bin/bash
#SBATCH --job-name=gpt2_repro_initial # job name
#SBATCH --partition=gpu_p2l           # partition with 8 32GB gpu nodes
#SBATCH --qos=qos_gpu-t4              # t4 enables 100H trainings
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --gres=gpu:4                  # number of GPUs per node
#SBATCH --cpus-per-task=20             # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=20:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err            # error file name
#SBATCH --account=ajs@gpu
#SBATCH --mail-type=ALL

set -x -e

module load cuda/10.2

DATASET=openwebtext
SERIALIZATION_DIR=${ALL_CCFRWORK}/experiments/gpt2_repro
LOGGING_DIR=${ALL_CCFRWORK}/tensorboard/gpt2_repro

source ~/.bashrc
source ${WORK}/reckoner/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python ${WORK}/jay-z/scripts/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name ${ALL_CCFRSCRATCH}/datasets/${DATASET} --block_size 1024 \
    --cache_dir ${ALL_CCFRSCRATCH}/cache_dir \
    --preprocessing_num_workers 32 \
    --do_train --do_eval \
    --max_steps 15000 \
    --max_train_samples 10000000 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 8 \
    --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
    --report_to tensorboard \
    --logging_strategy steps --logging_first_step --logging_dir ${LOGGING_DIR} --logging_steps 20 \
    --eval_steps 250 --evaluation_strategy steps \
    --save_strategy steps --save_steps 500 --save_total_limit 31 \
    --n_layer 3 --n_embd 128 --n_inner 128 --n_head 8
