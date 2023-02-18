#!/bin/bash
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 04:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x.out           # output file name
#SBATCH --partition=gpu_p5
#SBATCH --account=ajs@a100
#SBATCH -C a100

set -x -e

#source $ajs_ALL_CCFRWORK/start-py38-pt110
#source $ajs_ALL_CCFRWORK/start-py38-pt111
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml

echo "START TIME: $(date)"

variant=main

DATA_PATH=$ajs_ALL_CCFRSCRATCH/datasets/c4/gpt2tok_c4_text_document
DATA_OUTPUT_PATH=$ajs_ALL_CCFRSCRATCH/checkpoints/tr14-39M-lr$1-init$2-inpm$3-outm$4-atnm$5-mup
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr14-39M-lr$1-init$2-inpm$3-outm$4-atnm$5-mup-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=$ajs_ALL_CCFRWORK/code/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO

TOKENIZER_NAME_OR_PATH=gpt2

# defining the right environment variables
export TRANSFORMERS_CACHE=$ajs_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$ajs_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$ajs_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$ajs_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# testing for potential faulty nodes
# srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=512

NLAYERS=32
NHIDDEN=256
NHEADS=8
SEQ_LEN=2048

SAVE_INTERVAL=1000

TRAIN_SAMPLES=1_953_125  # 40B tokens
LR_DECAY_SAMPLES=1_953_125  # Decay in the same amount
LR_WARMUP_SAMPLES=183_105  # 375M tokens


MUP_ARGS=" \
    --lr $1 \
    --min-lr `bc <<< "scale=3; $1/10"` \
    --init-method-std $2 \
    --mup \
    --mup-input-mult $3 \
    --mup-output-mult $4 \
    --mup-attn-mult $5 \
"


OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990
EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

GPT_ARGS=" \
    --pp-partition-method 'type:transformer' \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --pad-vocab-size-to 51200 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

# TODO: decide on efficient eval-interval + eval-iters

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

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $MUP_ARGS \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

echo $CMD

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

clear; srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"
