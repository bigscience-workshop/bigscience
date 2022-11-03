#!/bin/bash
#SBATCH --job-name=run_trevalharness-tr11f-6b3-ml
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --reservation=hug

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
#conda activate muennighofflmevalgen
conda activate thomas_t_zero_evaluation

echo "START TIME: $(date)"

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Converted transformer checkpoint
#MODEL_CKPT=/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3/bloom-7b1

cd /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/bslmevaltransformers/lm-evaluation-harness


DATASETS_AND_CONFIGS=(
arc_challenge
arc_easy
)
#,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT
IFS=',' read dataset_name <<< "${DATASET_AND_CONFIG}"

# Use this fork of lm-eval: https://github.com/bigscience-workshop/lm-evaluation-harness/pull/109
python main.py \
    --model gpt2 \
    --model_args pretrained=$MODEL_CKPT \
    --batch_size 16 \
    --tasks $dataset_name \
    --output_path "${MODEL_CKPT}_{$dataset_name}.json" \
    --skip_tokenizer \
    --no_cache \
    --dtype=float16

echo "END TIME: $(date)"
