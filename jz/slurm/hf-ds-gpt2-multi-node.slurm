#!/bin/bash
#SBATCH --job-name=hf_ds_gpt2_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=std-%j.out          # output file name
#SBATCH --error=std-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@gpu

GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/transformers
export PYTHONPATH=$six_ALL_CCFRWORK/code/transformers

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=13370

export LAUNCHER=" \
    python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

MODEL=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"

export CMD=" \
    `pwd`/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --fp16 \
    --report_to none \
    --deepspeed tests/deepspeed/ds_config_zero2.json \
    "

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# to debug - add echo (it exits and prints what it would have launched)
srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
