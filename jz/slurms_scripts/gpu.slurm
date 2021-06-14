#!/bin/bash
#SBATCH --job-name=second_lm_balanced_prompted # job name
#SBATCH --partition=gpu_p2l           # partition with 8 32GB gpu nodes
#SBATCH --qos=qos_gpu-t4              # t4 enables 100H trainings
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --gres=gpu:8                  # number of GPUs per node
#SBATCH --cpus-per-task=6             # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=50:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=std-%j.out           # output file name
#SBATCH --error=std-%j.err            # error file name
#SBATCH --account=ajs@gpu
#SBATCH --mail-type=ALL

set -x -e

module load cuda/10.2

DATASET=wiki_bk_prompted
SERIALIZATION_DIR=${ALL_CCFRSCRATCH}/experiments/second_lm_balanced_prompted

source ~/.bashrc
conda activate smallexps
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

deepspeed ${WORK}/jay-z/scripts/run_clm_prompted.py \
    --deepspeed ${WORK}/jay-z/configs/deepspeed/ds_zero2.json \
    --model_name_or_path gpt2-medium \
    --tokenizer_name gpt2 \
    --dataset_name ${ALL_CCFRSCRATCH}/datasets/${DATASET} --block_size 1024 \
    --preprocessing_num_workers 31 \
    --group_by_length --length_column_name length \
    --cache_dir ${CACHE_DIR} \
    --do_train --do_eval \
    --max_steps 15000 \
    --max_train_samples 10000000 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 8 \
    --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
    --report_to tensorboard \
    --logging_strategy steps --logging_first_step --logging_dir tb --logging_steps 20 \
    --eval_steps 250 --evaluation_strategy steps \
    --save_strategy steps --save_steps 500 --save_total_limit 31
