# This version I haven't quite figured out - the job hangs on the master host - probably misconfigured megatron-lm launching command
# this script I found here https://www.glue.umd.edu/hpcc/help/software/pytorch.html
# did some mods to it

#!/bin/bash
#SBATCH --job-name=megatron_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:4
#SBATCH --time 00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --output=std-%j.out
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/megatron-lm

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release/
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=data/checkpoints

GPUS_PER_NODE=4
NNODES=2

MASTER_ADDR=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER_ADDR $SLAVES"

MASTER_PORT=12345
#`ss -tan | awk '{print $4}' | cut -d':' -f2 | \
#        grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | \
#        sort | uniq | shuf | head -1`

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


#    --train-iters 100000 \
#    --lr-decay-iters 320000 \
GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --finetune \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves
NODE_RANK=0
for node in $HOSTLIST; do
    ssh -q $node \
        python -m torch.distributed.launch \
            --nproc_per_node $GPUS_PER_NODE \
            --nnodes $NNODES \
            --node_rank $NODE_RANK \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            `pwd`/pretrain_gpt.py \
            --tensor-model-parallel-size 2 \
            --pipeline-model-parallel-size 2 \
            $GPT_ARGS \
            $OUTPUT_ARGS \
            --save $SAVE_CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl
    NODE_RANK=$((NODE_RANK+1))
done
wait
