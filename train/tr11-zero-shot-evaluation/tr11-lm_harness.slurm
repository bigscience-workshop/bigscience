#!/bin/bash
#SBATCH --job-name=tr11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --array=1-33
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_lm_eval # Debug deepspeed temporarily

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EXPERIMENT_DIRECTORY=$six_ALL_CCFRSCRATCH/synched_exps/tr11-zero-shot-evaluation

ALL_CHECKPOINTS = (
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step19500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step28500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step37500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step48000
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step57000
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step66000
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step76500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step85500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step94500
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step105000
pretrained=bigscience/tr3d-1B3-oscar-checkpoints,revision=global_step114000
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step19500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step28500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step37500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step48000
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step57000
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step66000
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step76500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step85500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step94500
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step105000
pretrained=bigscience/tr3e-1B3-c4-checkpoints,revision=global_step114000
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step19500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step28500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step37500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step48000
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step57000
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step66000
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step76500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step85500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step94500
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step105000
pretrained=bigscience/tr3m-1B3-pile-checkpoints,revision=global_step114000
)
CHECKPOINT=${ALL_CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}

export RESULTS_PATH=$EXPERIMENT_DIRECTORY/results/${CHECKPOINT}.txt
export LOGS_PATH=$EXPERIMENT_DIRECTORY/logs/${CHECKPOINT}_.out
mkdir -p $(dirname $RESULTS_PATH)
mkdir -p $(dirname $LOGS_PATH)

# TODO: Fix conditions when we get checkpoints
if [[ $CHECKPOINT = *tr3d* || $CHECKPOINT = *tr3m* ]]
then
  TOKENIZER=gpt2
elif [[ $CHECKPOINT = *tr3e* ]]
then
  TOKENIZER=t5-small
else
  echo "invalid checkpoint, Got $CHECKPOINT"
  exit
fi

pushd $WORK/code/bigscience/lm-evaluation-harness

python main.py \
    --model gpt2 \
    --model_args tokenizer=$TOKENIZER,$CHECKPOINT \
    --device cuda \
    --output_path $RESULTS_PATH \
    --tasks arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
    2>&1 | tee $LOGS_PATH
