#!/bin/bash
#SBATCH --job-name=tr1-13B-round1.slurm
#SBATCH --constraint=v100-32g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 00:20:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e


ROUND=1
TESTING=1

SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/
MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/Megatron-DeepSpeed
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt

if [[ ${TESTING} == 1 ]]; then
    # testing on 10k
    DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k/meg-gpt2_text_document
else
    # production on full 304M records
    DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

fi

source $six_ALL_CCFRWORK/start-prod
cd $MEGATRON_DEEPSPEED_REPO

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

# adjust depending on the number of the nodes

# XXX: edit me
GPUS_PER_NODE=4
NNODES=16
PP_SIZE=8 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=4 # always fixed to the size of a single node
DP_SIZE=$NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE) # will get derived automatically by trainer

MICRO_BATCH_SIZE=2
GAS=256 # PP chunks

EXIT_EARLY=0
#EXIT_EARLY=1

NLAYERS=40
NHIDDEN=5120
NHEADS=32
FFN_HIDDEN_SIZE=20480
SEQ_LEN=1024
VOCAB_SIZE=50257

if   [[ ${ROUND} == 1 ]]; then TRAIN_ITER=100    SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then TRAIN_ITER=1000   SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then TRAIN_ITER=300000 SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

# --exit-duration-in-mins
# --exit-interval - exit after this many iterations
if [[ ${EXIT_EARLY} == 1 ]]; then EXIT_OPTS="--exit-interval 2"
else EXIT_OPTS=""
fi

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 432_640 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

# GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$GAS*$DP_SIZE))
GLOBAL_BATCH_SIZE=2048
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 32 32 10_000_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

SAVE_INTERVAL
OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    "

ZERO_STAGE=1

config_json="./ds_config.$SLURM_JOBID.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $GAS,
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
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "


# # clear old checkpoint as it'd mismatch while we sort things out
# if   [[ ${MYENV} == stas ]];    then
#     rm -rf $six_CCFRSCRATCH/checkpoints/gpt2-meg-ds/*
# else
#     rm -rf $six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds/*
# fi


echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee tr1-13B-round1.$SLURM_JOBID.out
