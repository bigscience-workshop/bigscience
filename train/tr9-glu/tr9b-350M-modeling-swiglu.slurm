#!/bin/bash
#SBATCH --job-name=350M-swiglu.slurm
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --error=logs/%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e

# TODO: modify these for your training setup, just Ctrl-F replace <YOUR_TRAINING_NAME>
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr9b-350M-swiglu
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
REPO_PATH=$DATA_OUTPUT_PATH/tr9b-350M-swiglu-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard
CODECARBON_PATH=$REPO_PATH/codecarbon
LOGS_PATH=$REPO_PATH/logs
MEGATRON_DEEPSPEED_REPO=$DATA_OUTPUT_PATH/code/Megatron-DeepSpeed

# TODO: you may change the dataset, some examples are at tr3-1B3-baseline (tr3 = c4 + t5-tokenizer, tr3m = the Pile)
VOCAB_FILE=$DATA_OUTPUT_PATH/data/gpt2-vocab.json
MERGE_FILE=$DATA_OUTPUT_PATH/data/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document

# defining the right environment variables
source $six_ALL_CCFRWORK/start-prod

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $MEGATRON_DEEPSPEED_REPO

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# TODO: this is our base config for 1B3, edit PP/TP/batch size/model config if smaller or bigger
GPUS_PER_NODE=4
NNODES=16
PP_SIZE=4 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=4 # always fixed to the size of a single node
DP_SIZE=$((NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE))) # will get derived automatically by trainer

MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=512
TRAIN_ITER=73_242_187

NLAYERS=24
NHIDDEN=1024
NHEADS=16
# 2728 = (4096 * (2/3)) // 8 * 8
#   - 2/3 is linked to glu;
#   - //8 * 8 is to have multiples of 8 so that it's fast. Something about tensorcores
#     https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
FFN_HIDDEN_SIZE=2728
SEQ_LEN=2048

SAVE_INTERVAL=1500

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 2e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 73_242_187 \
    --lr-warmup-samples 183_105 \
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
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --rampup-batch-size 32 32 2_000_000 \

    --no-bias-gelu-fusion \
    --glu-activation swiglu \

    --train-samples $TRAIN_ITER \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

OUTPUT_ARGS=" \
    --log-interval 200 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 100 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
# TODO: Add --codecarbon-dir $CODECARBON_PATH \ if you want to use codecarbon, not adding it for now to make the current
# series of experiments consistent, especially speed-wise. Adding it once Tr6 and Tr7 are done

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


# # clear old checkpoint as it'd mismatch while we sort things out
#     rm -rf $SAVE_CHECKPOINT_PATH


echo $CMD

# We create the folder where the logs and codecarbon will be stored.
mkdir -p $LOGS_PATH
# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a $LOGS_PATH/main_log.txt
