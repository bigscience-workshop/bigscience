#!/bin/bash
#SBATCH --job-name=eval-tr3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --array=0-197 # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_lm_eval # Debug deepspeed temporarily

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EXPERIMENT_DIRECTORY=$six_ALL_CCFRSCRATCH/synched_exps/eval-tr3

# TODO: Modify according to the model you want to evaluated
EXPERIMENTS=(
tr3d-1B3-oscar-checkpoints
tr3e-1B3-c4-checkpoints
tr3m-1B3-pile-checkpoints
)
CHECKPOINT_STEPS=($(python -c "print(\" \".join([str(i) for i in range(19500, 118500, 1500)]))"))

EXPERIMENTS_SIZE=${#EXPERIMENTS[@]}
CHECKPOINT_STEPS_SIZE=${#CHECKPOINT_STEPS[@]}

if (( SLURM_ARRAY_TASK_COUNT != EXPERIMENTS_SIZE * CHECKPOINT_STEPS_SIZE ))
then
echo "Please update the array size as the it doesn't correspond to the number of models we want to evaluate. Array size: $SLURM_ARRAY_TASK_COUNT, number of models: $((EXPERIMENTS_SIZE * CHECKPOINT_STEPS_SIZE))"
exit 1
fi

EXPERIMENT=${EXPERIMENTS[$(python -c "print($SLURM_ARRAY_TASK_ID % ${#EXPERIMENTS[@]})")]}
CHECKPOINT_STEP=${CHECKPOINT_STEPS[$(python -c "print($SLURM_ARRAY_TASK_ID // ${#EXPERIMENTS[@]})")]}
MODEL_ARGS="pretrained=bigscience/$EXPERIMENT,revision=global_step$CHECKPOINT_STEP"

CHECKPOINT="${EXPERIMENT}_${CHECKPOINT_STEP}"
export RESULTS_PATH=$EXPERIMENT_DIRECTORY/results/${CHECKPOINT}.json
export LOGS_PATH=$EXPERIMENT_DIRECTORY/logs/${CHECKPOINT}_.out
mkdir -p $(dirname $RESULTS_PATH)
mkdir -p $(dirname $LOGS_PATH)

pushd $WORK/code/big_science/lm-evaluation-harness

export TOKENIZERS_PARALLELISM=false

python main.py \
    --model gpt2 \
    --model_args $MODEL_ARGS \
    --device cuda \
    --output_path $RESULTS_PATH \
    --tasks arc_challenge,arc_easy,boolq,copa,headqa_en,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
    2>&1 | tee $LOGS_PATH
