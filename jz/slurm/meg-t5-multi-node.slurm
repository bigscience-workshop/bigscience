#!/bin/bash
#SBATCH --job-name=meg_t5_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@gpu

GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/megatron-lm


VOCAB_FILE=$six_ALL_CCFRWORK/datasets-custom/vocabs/bert-large-uncased-vocab.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-t5_text_sentence
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRWORK/checkpoints/t5

MASTER_ADDR=`hostname`
MASTER_PORT=13370

#    --train-iters 100000 \
#    --lr-decay-iters 320000 \
# from t5 training:
#   --global-batch-size 2048 \
GPT_ARGS=" \
   --num-layers 12 \
   --hidden-size 768 \
   --num-attention-heads 12 \
   --kv-channels 64 \
   --ffn-hidden-size 3072 \
   --encoder-seq-length 512 \
   --decoder-seq-length 128 \
   --micro-batch-size 16 \
   --max-position-embeddings 512 \
   --train-iters 1000000 \
   --lr-decay-iters 1000000 \
   --vocab-file $VOCAB_FILE \
   --vocab-extra-ids 100 \
   --lr 0.0001 \
   --min-lr 0.00001 \
   --lr-decay-style linear \
   --lr-warmup-fraction .01 \
   --weight-decay 1e-2 \
   --clip-grad 1.0 \
   --fp16 \
    --checkpoint-activations \
   "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export CMD=" \
    `pwd`/pretrain_t5.py \
    --tensor-model-parallel-size 2 \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    "

# to debug - add echo (it exits and prints what it would have launched)
srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
