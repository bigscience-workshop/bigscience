#!/bin/bash
#SBATCH --job-name=eai-eval-opt-175b
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/logs/%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-32

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
# Required in order to load the opt tokenizer
conda activate thomas_lm_eval

echo "START TIME: $(date)"


CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/opt/opt-175b-meg-ds
MEGATRON_DEEPSPEED_REPO=/gpfswork/rech/six/uty16tp/code/big_science/Megatron-DeepSpeed
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

cd $MEGATRON_DEEPSPEED_REPO

# Make sure you use the slow version of the tokenizer.
# Same tokenizer for 125m and 175b
TOKENIZER_NAME_OR_PATH=/gpfsscratch/rech/six/commun/opt/opt-125m

PP_SIZE=8
TP_SIZE=1

NHIDDEN=12288
NLAYERS=96
NHEADS=96
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2050

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1

MEGATRON_REQUIRED_ARGS="
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --fp16 \
    --relu \
    --seed 42 \
    --pad-vocab-size-to 50272 \
    --make-vocab-size-divisible-by 1\
    --no-bias-gelu-fusion\
"


ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 1,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

# --task_list arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
TASKS=(
arc_challenge
arc_easy
boolq
copa
headqa
hellaswag
lambada
logiqa
mathqa
mc_taco
mrpc
multirc
openbookqa
piqa
prost
pubmedqa
qnli
qqp
race
rte
sciq
sst
triviaqa
webqs
wic
winogrande
wnli
wsc
lambada_mt_en
lambada_mt_fr
lambada_mt_de
lambada_mt_it
lambada_mt_es
)

#if [ "${#TASKS[@]}" -ne "$SLURM_ARRAY_TASK_COUNT" ];
#    then
#    echo "Please update the array size as the it doesn't correspond to the number of models we want to evaluate. Array size: $SLURM_ARRAY_TASK_COUNT, number of models: ${#TASKS[@]}"
#    exit 1
#fi
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

OPT_FOLDER=$WORK/opt/opt-175b/$TASK
mkdir -p $OPT_FOLDER

CMD="./tasks/eval_harness/evaluate.py  \
    --load $CHECKPOINT_PATH \
    --results_path $OPT_FOLDER/eai_results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --task_list $TASK\
    --deepspeed \
    --deepspeed_config ds_config.json \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier 16 \
    --offloadearly \
    $MEGATRON_REQUIRED_ARGS \
    "

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CUDA_LAUNCH_BLOCKING=1

echo $LAUNCHER $CMD

export PYTHONPATH=$MEGATRON_DEEPSPEED_REPO

$LAUNCHER $CMD 2>&1 | tee $OPT_FOLDER/eval-harness.log
