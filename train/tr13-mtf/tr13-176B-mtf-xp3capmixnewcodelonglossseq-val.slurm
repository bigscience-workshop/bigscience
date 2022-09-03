#!/bin/bash
#SBATCH --job-name=val-tr13-176B-ml
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --qos=qos_gpu-gc             # up to 100h
#SBATCH --nodes=36
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
echo "START TIME: $(date)"

variant=xp3capmixnewcodelonglossseq

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr13-176B-ml-t0
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr13-176B-ml-t0-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH
mkdir -p $TENSORBOARD_PATH

MEGATRON_DEEPSPEED_REPO=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0/megdslossseqnew/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO

KILL_SWITCH_PATH=$MEGATRON_DEEPSPEED_REPO/kill-switch-tr13-176B-mtf

TRAIN_DATA_PATH=$six_ALL_CCFRWORK/code/tr13f-6B3-ml-t0/Megatron-DeepSpeed/data/xp3capmixnewcodelong_train.txt
VALID_DATA_PATH=$six_ALL_CCFRWORK/code/tr13f-6B3-ml-t0/Megatron-DeepSpeed/data/xp3capmixnewcodelong_validation_pretr_only.txt
TOKENIZER_NAME_OR_PATH=bigscience/tokenizer

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# testing for potential faulty nodes
# srun --jobid $SLURM_JOB_ID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6001

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# TP=1/PP=72/MBS=1/Nodes=36
PP_SIZE=72
TP_SIZE=1

# T0 paper:
# ...truncate input and target sequences to 1024 and 256 tokens...
# ...use a batch size of 1024 sequences ... 2^20 total input tokens per batch...
# We use 2048 total tokens and 512 batch size = 2**20
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2048

NHIDDEN=14336
NLAYERS=70
NHEADS=112
SEQ_LEN=2048

# After CKPT is saved; stop training; change to desired SAVE_INTERNAL & remove no-load-optim & remove universal ckpt
SAVE_INTERVAL=5

TRAIN_SAMPLES=6_348_800

# T0 paper:
# "...we use a learning rate of 1e-3..."
# However, they use Adafactor, which adapts the LR
# For Adam we likely want a lower one
# FLAN:
# "...decay of 1e-4..""

# Uncomment for the first step
# --no-load-optim \
# --reset-progress \
OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --lr-decay-style constant \
    --lr-warmup-samples 0 \
    --clip-grad 1.0 \
    --weight-decay 1e-4 \
    --no-load-optim \
    --norm-target-loss \
    "
# for 20h 1190, for 100h 5990
#    --exit-duration-in-mins 1190 \
EXIT_OPTS=" \
    --exit-duration-in-mins 5990 \
    "

GPT_ARGS=" \
    --pp-partition-method 'type:transformer|embedding' \
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
    --init-method-std 0.0048 \
    --embed-layernorm \
    --sync-tp-duplicated-parameters \
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --kill-switch-path $KILL_SWITCH_PATH \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 250 \
    --eval-iters 10 \
    --eval-only True \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent

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
  "bf16": {
    "enabled": true
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

#    --universal-checkpoint \
export CMD=" \
    `pwd`/finetune_t0.py \
    --universal-checkpoint \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $TRAIN_DATA_PATH \
    --valid-weighted-split-paths-path $VALID_DATA_PATH \
    --dataloader-type single \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

echo $CMD

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"
