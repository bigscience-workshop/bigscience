#!/bin/bash
#SBATCH --job-name=bseval-tr13f-6B3
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --qos=qos_gpu-gc             # up to 100h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
conda activate muennighofflmeval

echo "START TIME: $(date)"

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Converted transformer checkpoint
#MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6B3-ml-t0-lmtoks168B-t0toks8b5
#MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6B3-ml-t0-lmtoks168B-t0toks0
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks168b-t0toks13b

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/lm-evaluation-harness

# GEM/wiki_lingua_es has 5 prompts
NUM_TASKS=5


# Use this fork of lm-eval: https://github.com/bigscience-workshop/lm-evaluation-harness/pull/109
python3 main.py --model hf-causal \
    --model_args pretrained=$MODEL_CKPT,use_accelerate=True,tokenizer=$MODEL_CKPT,dtype=float16 \
    --tasks GEM/wiki_lingua_es \
    --device cuda \
    --batch_size 16 \
    --no_cache \
    --no_tracking \
    --prompts $SLURM_ARRAY_TASK_ID \
    --num_fewshot 0

echo "END TIME: $(date)"
