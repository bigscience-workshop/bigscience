#!/bin/bash
#SBATCH --job-name=eai-eval-opt
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

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
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
TOKENIZER_NAME_OR_PATH=bigscience/opt

PP_SIZE=8
TP_SIZE=1
SEQ_LEN=2048

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1

#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1 \
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
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

OPT_FOLDER=$WORK/opt

CMD="./tasks/eval_harness/evaluate.py  \
    --load $CHECKPOINT_PATH \
    --results_path $OPT_FOLDER/eai_results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --relu \
    --no-load-rng \
    --fp16 \
    --inference \
    --seq-length $SEQ_LEN \
    --task_list arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier 4 \
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
