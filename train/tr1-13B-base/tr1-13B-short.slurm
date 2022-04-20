#!/bin/bash
#SBATCH --job-name=tr1-13B-short
#SBATCH --constraint=v100-32g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@v100



# This is the same as the main script but pre-configured for a small model that can be easily tested
# on 1 or 2 nodes, which is handy to verify everything works before using it on the main training.
#
# It's also useful for pre-building megatron CUDA kernels if and when things get borked and it gets stuck in building kernels.
#
# Change to NNODES=1 if needed
#
# to allocate (change to 2 for NNODES=2)
# salloc --constraint=v100-32g --account=six@v100 --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash

source $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B

set -x -e

echo "START TIME: $(date)"

#ROUND=3

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B-test
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon
LOGS_PATH=$DATA_OUTPUT_PATH/logs

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

cd $MEGATRON_DEEPSPEED_REPO

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=4
NNODES=2    # switch to 64
TP_SIZE=2    # always fixed to the size of a single node
PP_SIZE=2    # NLAYERS must be a multiple of PP_SIZE here
#DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE) # will get derived automatically by trainer

# GLOBAL_BATCH_SIZE has to be divisible by MICRO_BATCH_SIZE*DP_size
# GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$GAS*$DP_SIZE)) - GAS is auto-derived by deepspeed
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64

NLAYERS=8
NHIDDEN=512
NHEADS=8
FFN_HIDDEN_SIZE=2048
SEQ_LEN=512
VOCAB_SIZE=50257

SAVE_INTERVAL=2000


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 16 16 5_000_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 300_000_000 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    --seed 42
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 5 \
    --codecarbon-dir $CODECARBON_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
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


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

#    /usr/bin/env PYTHONPATH="." `pwd`/pretrain_gpt.py \
export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
#2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"

#
