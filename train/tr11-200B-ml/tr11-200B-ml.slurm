#!/bin/bash
#SBATCH --job-name=tr11-200B-ml
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS). TODO: update to 100 hours
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

#source $six_ALL_CCFRWORK/code/tr8b-104B/bigscience/train/tr8b-104B/start-tr8b-104B
#source $six_ALL_CCFRWORK/start-py38-pt110
source $six_ALL_CCFRWORK/start-py38-pt111

echo "START TIME: $(date)"

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr11-200B-ml
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
REPO_PATH=$DATA_OUTPUT_PATH/tr11-200B-ml
TENSORBOARD_PATH=$REPO_PATH/tensorboard
LOGS_PATH=$REPO_PATH/logs
mkdir -p $LOGS_PATH

KILL_SWITCH_PATH=/tmp/kill-switch-tr11-200B-exp1

MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/code/tr8b-104B/Megatron-DeepSpeed-tr8b-104B
cd $MEGATRON_DEEPSPEED_REPO

# TODO: point to the actual dataset path
DATA_PATH=$six_ALL_CCFRSCRATCH/synched_exps/tr5c-1B3-multilingual-alpha-alibi
TRAIN_DATA_PATH=$(cat $DATA_PATH/sampling_probs/train_data_string.0.3.txt)
VALID_DATA_PATH=$(cat $DATA_PATH/sampling_probs/valid_data_string.0.3.txt)
TEST_DATA_PATH=$(cat $DATA_PATH/sampling_probs/test_data_string.0.3.txt)

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# testing for potential faulty nodes
# srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'


# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# Expected Throughput ~ 151.64 TFLOPS, Speed ~ 107.12 Sec/it
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES  # Should be 48

PP_SIZE=12
TP_SIZE=4

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=2048  # 4.2M tokens. It is larger than the initial plan of 3.2M tokens to get higher throughput

NLAYERS=78
NHIDDEN=13824  # NHIDDEN / NLAYERS = 177
NHEADS=96  # NHIDDEN / NHEADS =144
SEQ_LEN=2048

SAVE_INTERVAL=2

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens. We used to use 216_320, not clear why


# Notes:
# "--adam-beta1 0.9  --adam-beta2 0.95  --adam-eps 1e-8"  are from the 104B experiment

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

# TODO: fix
#    --init-method-std \
#    --bf16 \
#    --tokenizer-name-or-path bigscience/oscar_13_languages_alpha_weight \

# Notes:
# --rampup-batch-size 16 16 9_765_625  # starts with gbs=16 then linearly increase by 16 until reaching gbs=1568 over
#                                      # 9_765_625 samples x 2048 = 20B tokens
#                                      # Use 20B tokens (not 12B) because we are using a large GLOBAL_BATCH_SIZE than prior work


GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 16 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path bigscience/oscar_13_languages_alpha_weight \
    --init-method-std 0.005 \
    --loss-scale 12 \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --kill-switch-path $KILL_SWITCH_PATH \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 10000 \
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
    "stage": $ZERO_STAGE,
    "move_params_to_cpu_during_init": false
  },
  "bf16": {
    "enabled": false
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "checkpoint_comm_enabled": false,
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
    "

#export LAUNCHER="python -u -m torch.distributed.launch \
#    --nproc_per_node $GPUS_PER_NODE \
#    --nnodes $NNODES \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT \
#    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths $TRAIN_DATA_PATH \
    --valid-weighted-split-paths $VALID_DATA_PATH \
    --test-weighted-split-paths $TEST_DATA_PATH \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "


# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH


echo $CMD

# for debug
#export CUDA_LAUNCH_BLOCKING=1

#srun --jobid $SLURM_JOBID bash -c 'free -m'
#srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID /gpfswork/rech/six/commun/code/tr8b-104B/bigscience/train/tr11-200B-ml/test.py" 2>&1 | tee -a $LOGS_PATH/main_log.txt
#asdlkjds

# XXX: remove for main training!!!
#export CUDA_LAUNCH_BLOCKING=1

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_log.txt
