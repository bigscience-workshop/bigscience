#!/bin/bash
#SBATCH --job-name=eval-array         # job name
#SBATCH --qos=qos_gpu-t3              # t3 enables 20h jobs but on 512 GPUs
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --gres=gpu:4                  # number of GPUs per node
#SBATCH --cpus-per-task=40            # number of cores per tasks
#SBATCH -C v100-16g
#SBATCH --array=500-17000:1000%26     # array of values
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=04:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=std-eval-%A_%a.out   # output file name
#SBATCH --error=std-eval-%A_%a.out    # error file name
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
SERIALIZATION_DIR=${eha_ALL_CCFRSCRATCH}/experiments/dec_only_t5-tiny

python -m torch.distributed.launch --nproc_per_node 4 ${six_ALL_CCFRWORK/code/bigscience/jz/scripts/run_clm.py \
    --model_name_or_path ${SERIALIZATION_DIR}/checkpoint-${SLURM_ARRAY_TASK_ID} \
    --tokenizer_name t5-small \
    --dataset_name ${DATASET} --block_size 1024 \
    --preprocessing_num_workers 76 \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --output_dir ${SERIALIZATION_DIR}/checkpoint-${SLURM_ARRAY_TASK_ID} \
    --report_to tensorboard --logging_dir ${SERIALIZATION_DIR}/checkpoint-${SLURM_ARRAY_TASK_ID}
