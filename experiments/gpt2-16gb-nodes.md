# GPT2 Experiments

Scripts and logs of GPT2 experiments on Jean Zay HPC.

Using 4x VT100 16GB nodes.

For now can't really allocate many 32gb nodes so can't do any serious evaluation there.
(add `-C v100-32g` for 32gb nodes.)

## Megatron-LM

Constants:

- `TP_SIZE` = tensor parallel
- `PP_SIZE` = pipeline parallel
- `DP_SIZE` = data parallel is derived automatically from `WORLD_SIZE / (TP_SIZE * PP_SIZE)`
- `WORLD_SIZE` = total number of GPUs

According to Megatron-LM paper the highest degree of TP we can use is 4 for 4-gpu nodes - crossing nodes would slow things down a lot. So max `TP_SIZE=4`. So the full 4 gpu node is used only for tensor parallel dimension.


### Summary

This section summarizes the numbers from the experiment sections below:

**Megatron**:

Not yet optimized with NVIDIA team!

| GPUs | Size | Micro-BS | PP Chunks |  DP | PP | Throughput |
| ---: | ---: | -------: | --------: | --: | -: | ---------: |
|   16 | 7.5B |        1 |         4 |   1 |  4 |  661ms     |
|   64 |  30B |        1 |         4 |   1 | 16 | 1439ms     |
|  128 |  50B |        1 |         4 |   1 | 32 | 2124ms     |
|  256 |  78B |        1 |         4 |   1 | 64 | 2953ms     |
|  256 |  22B |        1 |         4 |   4 | 16 | 1826ms     |
|      |      |          |           |     |    |            |


- `TP=4` in all of entries
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * pp_chunks * dp_size`
- PP chunks is the number of PP stages, so each pipeline handles `micro-batch-size * pp_chunks`

**Megatron + Deepspeed ZeRO**:

Not yet optimized with Deepspeed team!

| GPUs | Size | Micro-BS | PP Chunks | DP  | PP | Throughput |
| ---: | ---: | -------: | --------: | --: | -: | ---------: |
| 64   | 30B  | 1        | 4         | 1   | 16 | 28716ms    |
|      |      |          |           |     |    |            |




### Nodes=4 DP=1 TP=4 PP=4

Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=4 --ntasks=4 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

The biggest model we can fit with `micro-batch-size=1`: **7.5B**

```

cd ~/prod/code/megatron-lm/

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=4

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=4096
NLAYERS=36
SEQ_LEN=1024
VOCAB_SIZE=50257

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

PP_SIZE=4
DP_SIZE=1
TP_SIZE=4

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
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:

```
iteration 50/ 1000 | consumed samples: 200 | elapsed time per iteration (ms): 661.3 | learning rate:
1.497E-04 | global batch size: 4 | lm loss: 8.238016E+00 | loss scale: 16384.0 | grad norm: 2.555 |
number of skipped iterations: 0 | number of nan iterations: 0 | time (ms) | forward-compute: 92.25 |
forward-recv: 65.68 | backward-compute: 239.82 | backward-send: 0.54 | backward-send-forward-recv:
4.29 | backward-params-all-reduce: 10.50 | backward-embedding-all-reduce: 204.76 |
optimizer-copy-to-main-grad: 4.47 | optimizer-unscale-and-check-inf: 5.68 |
optimizer-clip-main-grad: 8.56 | optimizer-copy-main-to-model-params: 4.41 | optimizer: 42.31 |
batch-generator: 2.70
```


### Nodes=16 DP=1 TP=4 PP=16


Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

The biggest model we can fit with `micro-batch-size=1`: barely **30B**

(30B is not in paper's table - took 39B model and reduced NHIDDEN=7168 to overcome OOM) but it still OOM'ed after 60 steps so was a bit too much.

```

cd ~/prod/code/megatron-lm/

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=16

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=7168
NLAYERS=48
SEQ_LEN=1024

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

PP_SIZE=16
DP_SIZE=1
TP_SIZE=4

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
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:



```
iteration 30/ 1000 | consumed samples: 120 | elapsed time per iteration (ms): 1439.3 | learning
rate: 1.500E-04 | global batch size: 4 | lm loss: 2.667133E+01 | loss scale: 16384.0 | grad norm:
73.338 | number of skipped iterations: 1 | number of nan iterations: 0 | time (ms) |
forward-compute: 77.94 | forward-recv: 285.81 | backward-compute: 203.21 | backward-send: 0.91 |
backward-send-forward-recv: 5.44 | backward-params-all-reduce: 10.38 |
backward-embedding-all-reduce: 811.34 | optimizer-copy-to-main-grad: 4.61 |
optimizer-unscale-and-check-inf: 7.90 | optimizer-clip-main-grad: 7.91 |
optimizer-copy-main-to-model-params: 3.95 | optimizer: 43.19 | batch-generator: 2.64
```

### Nodes=32 DP=1 TP=4 PP=32

Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=32 --ntasks=32 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

The biggest model we can fit with `micro-batch-size=1`: **50B**

(50B is not in paper's table - took 76B model - had to change to nlayer=64 for it to work and reduced NHIDDEN=8192 to overcome OOM) but it still OOM'ed after 60 steps so was a bit too much.

```
perl -le 'print( (120*402780160+8*514977792)>>20)'
50023
```

```

cd ~/prod/code/megatron-lm/

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=32

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=8192
NLAYERS=64
SEQ_LEN=1024

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

PP_SIZE=32
DP_SIZE=1
TP_SIZE=4

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
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:

```
iteration 50/ 1000 | consumed samples: 200 | elapsed time per iteration (ms): 2124.0 | learning
rate: 1.497E-04 | global batch size: 4 | lm loss: 1.038553E+01 | loss scale: 16384.0 | grad norm:
14.954 | number of skipped iterations: 0 | number of nan iterations: 0 | time (ms) |
forward-compute: 68.08 | forward-recv: 485.51 | backward-compute: 175.50 | backward-send: 0.85 |
backward-send-forward-recv: 5.63 | backward-params-all-reduce: 9.54 | backward-embedding-all-reduce:
1321.49 | optimizer-copy-to-main-grad: 4.19 | optimizer-unscale-and-check-inf: 21.21 |
optimizer-clip-main-grad: 8.04 | optimizer-copy-main-to-model-params: 3.98 | optimizer: 56.47 |
batch-generator: 2.72

```





### Nodes=64 DP=1 TP=4 PP=64

Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=64 --ntasks=64 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

The biggest model we can fit with `micro-batch-size=1`: **78B**

(78B is not in paper's table - took 76B model - had to change to nlayers=64 for it to work)

```
perl -le 'print( (248*314652160+8*454899200)>>20)'
77889
```

```

cd ~/prod/code/megatron-lm/

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=64

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=10240
NLAYERS=64
SEQ_LEN=1024

MICRO_BATCH_SIZE=1
PP_CHUNKS=4

PP_SIZE=64
DP_SIZE=1
TP_SIZE=4

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
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

```

Stats:

```
iteration 30/ 1000 | consumed samples: 120 | elapsed time per iteration (ms): 2953.3 | learning
rate: 1.500E-04 | global batch size: 4 | lm loss: 3.785040E+01 | loss scale: 16384.0 | grad norm:
47.681 | number of skipped iterations: 1 | number of nan iterations: 0 | time (ms) |
forward-compute: 53.67 | forward-recv: 746.59 | backward-compute: 134.74 | backward-send: 1.01 |
backward-send-forward-recv: 6.49 | backward-params-all-reduce: 8.29 | backward-embedding-all-reduce:
1964.85 | optimizer-copy-to-main-grad: 3.64 | optimizer-unscale-and-check-inf: 8.68 |
optimizer-clip-main-grad: 6.34 | optimizer-copy-main-to-model-params: 3.10 | optimizer: 36.80 |
batch-generator: 2.52
```



### Nodes=64 DP=4 TP=4 PP=16

Let's try a smaller model with a larger batch size.

Pre-allocate so that we can run experiments immediately and not wait for slurm to grant us resources:

```
salloc --nodes=64 --ntasks=64 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

The biggest model we can fit with `micro-batch-size=1` + D4: **22B**

```
perl -le 'print( (48*402780160+8*514977792)>>20)'
22366
```

```

cd ~/prod/code/megatron-lm/

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-1-node

GPUS_PER_NODE=4
NNODES=64

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=8192
NLAYERS=32
SEQ_LEN=1024

MICRO_BATCH_SIZE=1
PP_CHUNKS=4
GAS=$PP_CHUNKS

PP_SIZE=16
DP_SIZE=4
TP_SIZE=4

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --gas $GAS \
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
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:

```
 iteration 40/ 1000 | consumed samples: 640 | elapsed time per iteration (ms): 1826.3 | learning
 rate: 1.499E-04 | global batch size: 16 | lm loss: 1.290925E+01 | loss scale: 16384.0 | grad norm:
 7.607 | number of skipped iterations: 0 | number of nan iterations: 0 |
time (ms) | forward-compute: 80.84 | forward-recv: 225.57 | backward-compute: 172.26 |
backward-send: 0.86 | backward-send-forward-recv: 5.76 | backward-params-all-reduce: 307.62 |
backward-embedding-all-reduce: 746.14 | optimizer-copy-to-main-grad: 4.20 |
optimizer-unscale-and-check-inf: 250.90 | optimizer-clip-main-grad: 8.06 |
optimizer-copy-main-to-model-params: 3.99 | optimizer: 286.27 | batch-generator: 2.72


```




## Megatron + Deepspeed ZeRO

**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3` is not in sync with M-LM master - so several config args don't match.

Status: Unoptimized

### Nodes=16


```
salloc --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Todo:

46B experiment:
NHEADS=32
NHIDDEN=9216
NLAYERS=48
SEQ_LEN=1024
VOCAB_SIZE=50257


```

cd ~/prod/code/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

GPUS_PER_NODE=4
NNODES=16

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=7168
NLAYERS=48
SEQ_LEN=1024
VOCAB_SIZE=50257

MICRO_BATCH_SIZE=16
PP_CHUNKS=4

PP_SIZE=16
DP_SIZE=2
TP_SIZE=2

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#    --micro-batch-size $MICRO_BATCH_SIZE \
#    --lr-warmup-fraction .01 \
#    --global-batch-size $GLOBAL_BATCH_SIZE
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --batch-size $MICRO_BATCH_SIZE \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

#ZeRO Configs
gradient_accumulation_steps=1
reduce_bucket_size=$(($NHIDDEN*$NHIDDEN))
stage3_prefetch_bucket_size=$(($NHIDDEN*$NHIDDEN*9/10))
stage3_param_persistence_threshold=$((10*$NHIDDEN))

# Here it is different from the other setup
train_batch_size=$(($WORLD_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))

config_json="./ds_zero_stage_3_config.json"

#  "train_batch_size": $train_batch_size,

cat <<EOT > $config_json
{
  "gradient_accumulation_steps": $gradient_accumulation_steps,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": $stage3_prefetch_bucket_size,
    "stage3_param_persitence_threshold": $stage3_param_persistence_threshold,
    "reduce_bucket_size": $reduce_bucket_size,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 10,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": false
}
EOT

MP_SIZE=$TP_SIZE

stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# TiledLinear splits, 0 is disable
TILED_LINEAR="false"
TILE_DIM=1


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${stage} \
    --zero-reduce-bucket-size ${rbs} \
    --zero-allgather-bucket-size ${agbs} \
    "

if [ "${contigious_gradients}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-reduce-scatter"
fi

CHKP_ARGS=" \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

#    --tensor-model-parallel-size $TP_SIZE \
#    --pipeline-model-parallel-size $PP_SIZE \
export CMD=" \
    `pwd`/pretrain_gpt2.py \
    --model-parallel-size $TP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
     $CHKP_ARGS \
    "

rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-meg-ds

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

```

Stats:

```
iteration 20/ 1000 | elapsed time per iteration (ms): 28716.0 | learning rate: 1.500E-04 | lm loss:
2.324108E+01 | loss scale: 1024.0 | number of skipped iterations: 0 | number of nan iterations: 0 |
time (ms) | forward: 5495.35 | backward: 22976.72 | backward-backward: 22976.69 |
backward-allreduce: 0.00 | optimizer: 243.03 | batch generator: 1.00 Effective Tera Flops per GPU:
0.21 and total parameters 29.998 B
```


## Megatron + Deepspeed 3D Parallelism

**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism` is not in sync with M-LM master - so several config args don't match.

Status: Unoptimized

### Nodes=16


```
salloc --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


```

cd ~/prod/code/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

GPUS_PER_NODE=4
NNODES=16

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
NODE_RANK=0

NHEADS=32
NHIDDEN=7168
NLAYERS=48
SEQ_LEN=1024
VOCAB_SIZE=50257

MICRO_BATCH_SIZE=1
PP_CHUNKS=4
GAS=$PP_CHUNKS

PP_SIZE=16
DP_SIZE=1
TP_SIZE=4

GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$PP_CHUNKS*$DP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#    --micro-batch-size $MICRO_BATCH_SIZE \
#    --lr-warmup-fraction .01 \
#    --global-batch-size $GLOBAL_BATCH_SIZE
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --batch-size $MICRO_BATCH_SIZE \
    --gas $GAS \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

#ZeRO Configs
gradient_accumulation_steps=1
reduce_bucket_size=$(($NHIDDEN*$NHIDDEN))
stage3_prefetch_bucket_size=$(($NHIDDEN*$NHIDDEN*9/10))
stage3_param_persistence_threshold=$((10*$NHIDDEN))
train_batch_size=$(($DP_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))

config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_batch_size": $train_batch_size,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $gradient_accumulation_steps,
  "steps_per_print": 10,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 10,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": false
}
EOT

MP_SIZE=$TP_SIZE

stage=0
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${stage} \
    --zero-reduce-bucket-size ${rbs} \
    --zero-allgather-bucket-size ${agbs} \
    "

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${stage} \
    --zero-reduce-bucket-size ${rbs} \
    --zero-allgather-bucket-size ${agbs} \
    "

if [ "${contigious_gradients}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-reduce-scatter"
fi

CHKP_ARGS=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --profile-backward"
fi

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

#    --tensor-model-parallel-size $TP_SIZE \
#    --pipeline-model-parallel-size $PP_SIZE \
export CMD=" \
    `pwd`/pretrain_gpt2.py \
    --model-parallel-size $TP_SIZE \
    --pipe-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
     $CHKP_ARGS \
    "

rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-meg-ds

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

# can't figure out how to launch from salloc
#
# r10i5n[5-6],r10i6n[4-5,7-8],r10i7n[0,4-5],r11i3n[3-6],r13i1n[2-4]
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround
while ($ENV{"SLURM_JOB_NODELIST"} =~ m/(\w+)(?:\[([\d-,]+)\])?,?/msg) {
$b=$1; $s=$2||q[""]; $s=~s/-/../g;
print map { "$b$_ slots=$slots\n" } eval $s }'
}
makehostfile > hostfile
#
#
# srun --jobid $SLURM_JOBID deepspeed -H `pwd`/hostfile --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} $CMD
#

# to kill hanging python processes on all nodes at once
# srun pkill python

```

Stats:
```
iteration 650/ 1000 | elapsed time per iteration (ms): 1210.1 | learning rate: 1.450E-05 | lm loss:
7.287670E+00 | loss scale: 8192.0 | number of skipped iterations: 0 | number of nan iterations: 0 |
time (ms) | forward: 0.00 | backward: 0.00 | optimizer: 0.00 | batch generator: 0.00

```

```
| N/A   50C    P0   181W / 300W |  13236MiB / 32510MiB |     99%      Default |
|    0   N/A  N/A     72371      C   .../conda/hf-prod/bin/python    13233MiB |
|    1   N/A  N/A     72372      C   .../conda/hf-prod/bin/python    13193MiB |
|    2   N/A  N/A     72373      C   .../conda/hf-prod/bin/python    13161MiB |
|    3   N/A  N/A     72374      C   .../conda/hf-prod/bin/python    13169MiB |
```

## HF + Deepspeed ZeRO

### Nodes=16 ZeRO-2


```
salloc --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

32GB nodes

This works - at about 25GB / gpus - very slow 20s/it

Model size: 3.5B

Higher model the 40GB/gpu limit is passed and processes get killed.

We don't have zero.Init() here so the whole model is loaded onto each process - not possible to scale.

This memory gets released afterwards, but we don't have enough to bypass that hump.

```

# use custom PR branch to handle the model creation on the fly
cd ~/prod/code/transformers-clm-any-model-config/

export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

MODEL=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"

GPUS_PER_NODE=4
NNODES=16

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

NHEADS=32
NHIDDEN=3072
NLAYERS=30
SEQ_LEN=1024
VOCAB_SIZE=50257

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "


config_json="./ds_z2_no_offload.json"
cat <<EOT > $config_json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true,
        "cpu_offload": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT

export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0

#    deepspeed  -H `pwd`/hostfile-exp2 --num_nodes $NNODES --num_gpus $GPUS_PER_NODE \
export CMD=" \
    examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --config_overrides "n_embd=$NHIDDEN,n_head=$NHEADS,n_layer=$NLAYERS,n_positions=$SEQ_LEN" \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 10000 \
    --max_eval_samples 1000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --fp16 \
    --report_to none \
    --deepspeed $config_json \
    "

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'


```

Stats:

```


```


## Node16 ZeRO-3 + CPU Offload

32GB nodes

Model size: 7B


```

# use custom PR branch to handle the model creation on the fly
cd ~/prod/code/transformers-clm-any-model-config/

export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

MODEL=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"

GPUS_PER_NODE=4
NNODES=2

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

NHEADS=32
NHIDDEN=1024
NLAYERS=10
SEQ_LEN=1024
VOCAB_SIZE=50257

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "


config_json="./ds_z3_cpu_offload.json"
cat <<EOT > $config_json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e14,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT

export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0

#    deepspeed  -H `pwd`/hostfile-exp2 --num_nodes $NNODES --num_gpus $GPUS_PER_NODE \
export CMD=" \
    examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --config_overrides "n_embd=$NHIDDEN,n_head=$NHEADS,n_layer=$NLAYERS,n_positions=$SEQ_LEN" \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 10000 \
    --max_eval_samples 1000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --fp16 \
    --report_to none \
    --deepspeed $config_json \
    "

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

```


Stats:

```

```


### Trying deepspeed launcher again


```

#!/bin/bash
#SBATCH --job-name=hf_ds_gpt2_multi_node_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@gpu

# use custom PR branch to handle the model creation on the fly
cd ~/prod/code/transformers-clm-any-model-config/

source $six_ALL_CCFRWORK/start-prod

export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

MODEL=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"

GPUS_PER_NODE=4
NNODES=2

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

NHEADS=32
NHIDDEN=1024
NLAYERS=10
SEQ_LEN=1024
VOCAB_SIZE=50257

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "


config_json="./ds_z3_cpu_offload.json"
cat <<EOT > $config_json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e14,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT

export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0

export CMD=" \
    deepspeed --num_nodes $NNODES --num_gpus $GPUS_PER_NODE \
    examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --config_overrides "n_embd=$NHIDDEN,n_head=$NHEADS,n_layer=$NLAYERS,n_positions=$SEQ_LEN" \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 10000 \
    --max_eval_samples 1000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --fp16 \
    --report_to none \
    --deepspeed $config_json \
    "

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

#srun --jobid $SLURM_JOBID bash -c '$CMD'
srun --jobid $SLURM_JOBID bash -c '$CMD'



```
