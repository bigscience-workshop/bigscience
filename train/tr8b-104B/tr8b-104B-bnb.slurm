#!/bin/bash
#SBATCH --job-name=tr8b-104B-bnb
#SBATCH --constraint=v100-32g
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/code/tr8b-104B/bigscience/train/tr8b-104B/start-tr8b-104B

echo "START TIME: $(date)"

VARIANT=bnb

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr8b-104B/
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$VARIANT
REPO_PATH=$DATA_OUTPUT_PATH/tr8b-104B-logs/
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$VARIANT
LOGS_PATH=$REPO_PATH/logs/$VARIANT
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr8b-104B/Megatron-DeepSpeed-tr8b-104B

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

cd $MEGATRON_DEEPSPEED_REPO

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=4
NNODES=128   # switch to 128
TP_SIZE=4    # always fixed to the size of a single node
PP_SIZE=32   # NLAYERS must be a multiple of PP_SIZE here
#DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE) # will get derived automatically by trainer

# GLOBAL_BATCH_SIZE has to be divisible by MICRO_BATCH_SIZE*DP_size
# GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$GAS*$DP_SIZE)) - GAS is auto-derived by deepspeed
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2048

NLAYERS=64
NHIDDEN=11600
NHEADS=80
SEQ_LEN=2048
VOCAB_SIZE=50257

SAVE_INTERVAL=300

OPTIMIZER_ARGS=" \
    --use-bnb-optimizer \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
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
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 16 16 6_000_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 300_000_000 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 12 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --seed 43 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 5 \
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
  "zero_allow_untested_optimizer": true,
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
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"
