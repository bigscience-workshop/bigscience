# gpt2 experiments

Logs of GPT2 experiments on JZ.

## Megatron-LM

- `TP_SIZE` = tensor parallel
- `PP_SIZE` = pipeline parallel
- `DP_SIZE` = data parallel is derived automatically from `WORLD_SIZE / (TP_SIZE * PP_SIZE)`


### 4-Node 4x v100 16GB TP=4 PP=4 DP=1

Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

add `-C v100-32g` for 32gb nodes.

```
salloc --nodes=4 --ntasks=4 --cpus-per-task=32 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $ALL_CCFRSCRATCH/start-prod
```

The biggest model we can fit with `micro-batch-size`=1: **7.5B**

```

cd base/code/megatron-lm/

CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$eha_ALL_CCFRSCRATCH/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=4

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=4096
NLAYERS=36
SEQ_LEN=512

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

TP_SIZE=4
PP_SIZE=4
DP_SIZE=1

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
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
    "

# clear old checkpoint as it'd mismatch while we sort things out
rm -rf $eha_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:
```
iteration      200/    1000 | consumed samples:          800 | elapsed time per iteration (ms): 430.0 | learning rate: 1.342E-04 | global batch size:     4 | lm loss: 7.399415E+00 | loss scale: 16384.0 | grad norm: 2.918 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 53.79 | forward-recv: 38.38 | backward-compute: 138.84 | backward-send: 0.28 | backward-send-forward-recv: 2.60 | backward-params-all-reduce: 10.17 | backward-embedding-all-reduce: 135.50 | optimizer-copy-to-main-grad: 4.48 | optimizer-unscale-and-check-inf: 12.63 | optimizer-clip-main-grad: 8.62 | optimizer-copy-main-to-model-params: 4.42 | optimizer: 49.42 | batch-generator: 2.35

```
gpus:
```
| ============================================================================= |
| 0   N/A  N/A     59019      C   .../conda/hf-prod/bin/python    14659MiB      |
| 1   N/A  N/A     59020      C   .../conda/hf-prod/bin/python    14691MiB      |
| 2   N/A  N/A     59021      C   .../conda/hf-prod/bin/python    14659MiB      |
| 3   N/A  N/A     59022      C   .../conda/hf-prod/bin/python    14643MiB      |
```

### 16-Node 4x v100 16GB TP=8 PP=8 DP=1


Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=16 --ntasks=16 --cpus-per-task=32 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $ALL_CCFRSCRATCH/start-prod
```

The biggest model we can fit with `micro-batch-size`=1: slightly less than **39B**

but OOMed quickly - so fitting < 39B

```
CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$eha_ALL_CCFRSCRATCH/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=16

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=8192
NLAYERS=48
SEQ_LEN=1024

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

TP_SIZE=8
PP_SIZE=8
DP_SIZE=1

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
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
    "

# clear old checkpoint as it'd mismatch while we sort things out
rm -rf $eha_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:

```
 iteration       10/    1000 | consumed samples:           40 | elapsed time per iteration (ms): 3974.1 | learning rate: 0.000E+00 | global batch size:     4 | loss scale: 8388608.0 | number of skipped iterations:  10 | number of nan iterations:   0 |
time (ms) | forward-compute: 363.26 | forward-recv: 1346.01 | backward-compute: 632.88 | backward-send: 64.84 | backward-send-forward-recv: 172.89 | backward-params-all-reduce: 14.16 | backward-embedding-all-reduce: 1300.71 | optimizer-copy-to-main-grad: 5.60 | optimizer-unscale-and-check-inf: 72.41 | optimizer: 78.14 | batch-generator: 3.70
```
