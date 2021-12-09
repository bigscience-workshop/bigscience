#!/bin/bash
#SBATCH --job-name=tr11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_lm_eval # Debug deepspeed temporarily

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EXPERIMENT_DIRECTORY=$six_ALL_CCFRSCRATCH/synched_exps/tr11-zero-shot-evaluation

ALL_CHECKPOINTS = (

)
CHECKPOINT=${ALL_CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}

export RESULTS_PATH=$EXPERIMENT_DIRECTORY/results/${CHECKPOINT}.txt
export LOGS_PATH=$EXPERIMENT_DIRECTORY/logs/${CHECKPOINT}_.out

# TODO: Fix conditions when we get checkpoints
if [[ $CHECKPOINT = *c4 || $CHECKPOINT = *oscar ]]
then
  TOKENIZER=gpt2
elif [[ $CHECKPOINT = *c4 ]]
then
  TOKENIZER=t5-small
else
  echo "invalid checkpoint, Got $CHECKPOINT"
  exit
fi

pushd $WORK/code/bigscience/lm-evaluation-harness

python main.py \
    --model gpt2 \
    --model_args tokenizer=$TOKENIZER,pretrained=$CHECKPOINT \
    --device cuda:0 \
    --output_path $RESULTS_PATH \
    --tasks arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
    2>&1 | tee $LOGS_PATH
