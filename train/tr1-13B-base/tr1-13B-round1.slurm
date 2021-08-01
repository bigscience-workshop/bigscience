#!/bin/bash
#SBATCH --job-name=tr1-13B-round3.slurm
#SBATCH --constraint=v100-32g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e

echo "START TIME: $(date)"

#ROUND=3

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/Megatron-DeepSpeed

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

source $six_ALL_CCFRWORK/start-prod
cd $MEGATRON_DEEPSPEED_REPO

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

GPUS_PER_NODE=4
NNODES=64    # switch to 64
TP_SIZE=2    # always fixed to the size of a single node
PP_SIZE=4    # NLAYERS must be a multiple of PP_SIZE here
#DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE) # will get derived automatically by trainer

# GLOBAL_BATCH_SIZE has to be divisible by MICRO_BATCH_SIZE*DP_size
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

NLAYERS=40
NHIDDEN=5120
NHEADS=32
#FFN_HIDDEN_SIZE=20480
SEQ_LEN=2048
VOCAB_SIZE=50257


SAVE_INTERVAL=1500

#if   [[ ${ROUND} == 1 ]]; then EXIT_INTERVAL=100  SAVE_INTERVAL=10
#elif [[ ${ROUND} == 2 ]]; then EXIT_INTERVAL=1000 SAVE_INTERVAL=18
#elif [[ ${ROUND} == 3 ]]; then                    SAVE_INTERVAL=1500
#else echo "invalid ROUND: $ROUND"
#fi

# if   [[ ${ROUND} == 1 ]]; then SAVE_INTERVAL=10
# elif [[ ${ROUND} == 2 ]]; then SAVE_INTERVAL=18
# elif [[ ${ROUND} == 3 ]]; then SAVE_INTERVAL=1500
# else echo "invalid ROUND: $ROUND"
# fi

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

# XXX: fails to get this option? but it's fine, as the default is correct
#    --ffn_hidden_size $FFN_HIDDEN_SIZE \

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

#     --exit-interval $EXIT_INTERVAL \

# GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$GAS*$DP_SIZE))
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
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

# codecarbon isn't ready, will enable once it's working
#    --codecarbon-dir $CODECARBON_PATH \
OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 100 \
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
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee tr1-13B-round1.$SLURM_JOBID.out

echo "END TIME: $(date)"

#
