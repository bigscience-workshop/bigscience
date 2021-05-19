# gpt2 experiments

Log experiments here

## Megatron-LM

`MP_SIZE` = tensor parallel
`PP_SIZE` = pipeline parallel
`DP_SIZE` = data parallel is derived automatically from `WORLD_SIZE / (MP_SIZE * PP_SIZE)`


### 4-Node 4x v100 32GB MP=4 PP=4 DP=1

```
salloc  -C v100-32g --nodes=4 --ntasks=4 --cpus-per-task=32 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $ALL_CCFRSCRATCH/start-prod
```

The biggest model we can fit with micro-batch-size=1: 18B

seqlen
- 512 works
- 1024 almost fits - OOMs at times
- 2048 OOMs


```
CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$eha_ALL_CCFRSCRATCH/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=4

MASTER_ADDR=`perl -le 'print((split /,/, $ENV{"SLURM_JOB_NODELIST"})[0])'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=6144
NLAYERS=40
SEQ_LEN=512

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

MP_SIZE=4
PP_SIZE=4

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS))
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
    --tensor-model-parallel-size $MP_SIZE \
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
 iteration      110/    1000 | consumed samples:          440 | elapsed time per iteration (ms): 974.9 | learning rate: 1.462E-04 | global batch size:     4 | lm loss: 8.111837E+00 | loss scale: 16384.0 | grad norm: 6.753 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 211.64 | forward-recv: 79.05 | backward-compute: 293.57 | backward-send: 0.37 | backward-send-forward-recv: 3.48 | backward-params-all-reduce: 23.43 | backward-embedding-all-reduce: 257.82 | optimizer-copy-to-main-grad: 11.07 | optimizer-unscale-and-check-inf: 14.57 | optimizer-clip-main-grad: 19.51 | optimizer-copy-main-to-model-params: 11.37 | optimizer: 103.82 | batch-generator: 5.14

```
