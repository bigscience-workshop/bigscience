# GPT2 Experiments

Scripts and logs of GPT2 experiments on Jean Zay HPC.

Using 4x VT100 32GB nodes.

(add `-C v100-32g` for 32gb nodes.)

## Megatron-LM

Constants:

- `TP_SIZE` = tensor parallel
- `PP_SIZE` = pipeline parallel
- `DP_SIZE` = data parallel is derived automatically from `WORLD_SIZE / (TP_SIZE * PP_SIZE)`
- `WORLD_SIZE` = total number of GPUs

According to Megatron-LM paper the highest degree of TP we can use is 4 for 4-gpu nodes - crossing nodes would slow things down a lot. So max `TP_SIZE=4`. So the full 4 gpu node is used only for tensor parallel dimension.

## Metrics

TFlops: `model_size_in_B * 4 * 2 * seq * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

The factor of 4 is when used with activation check-pointing,
otherwise it will be 3, but for 200B model, activation check-pointing will always be on.

The peak of V100 32gb gpu is about 125 TFlops/sec [spec](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf). But we cannot get the peak. The max achievable performance will be 30-60TFlops depending on the model size. So if you see low 20s, the model is not tuned well, if you see, over 100 then there is a bug in the calculation. ￼

For v100 16gb gpus the max spec is 120 TFlops/sec.


### Summary

This section summarizes the numbers from the experiment sections below:

#### Megatron

The full slurm scripts and log files are at [`gpt2-meg`](./gpt2-meg):
- scripts starting with `meg_gpt2_base_` are for getting the baseline with tiny BS
- scripts starting with `meg_gpt2_perf_` are for smaller model, and tuned for high performance

Not yet optimized with NVIDIA team!

**Max model size**

These first results are all about how big of a model can be fit into the given the hardware on the smallest batch size, disregarding throughput.

16GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 | 7.5B |  1 |  4 |         4 |      1 |       4 | 0.661s     |        |
|   64 | 30B  |  1 | 16 |         4 |      1 |       4 | 1.439s     |        |
|  128 | 50B  |  1 | 32 |         4 |      1 |       4 | 2.124s     |        |
|  256 | 78B  |  1 | 64 |         4 |      1 |       4 | 2.953s     |        |
|  256 | 22B  |  4 | 16 |         4 |      1 |       4 | 1.826s     |        |
|      |      |    |    |           |        |         |            |        |

32GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 | 18B  |  1 |  4 |         4 |      1 |       4 | 1.381s     | 26.693 |
|   32 | 28B  |  1 |  8 |         4 |      1 |       4 | 1.618s     | 17.720 |
|   64 | 61B  |  1 | 16 |         4 |      1 |       4 | 2.738s     | 11.406 |
|  128 | 109B |  1 | 32 |         4 |      1 |       4 | 4.234s     |  6.590 |
|  256 | 193B |  1 | 64 |         4 |      1 |       4 | 6.736s     |  3.667 |
|      |      |    |    |           |        |         |            |        |

The TFLops are very low because there are too few PP chunks/micro-batches (4) (gradient accumulation size / GAS) and so the bubble takes a lot of overhead, increasing PP chunks should dramatically improve performance but also need to lower the max model size to have memory to hold those chunks in.

**Performance**

These experiments are to try a lower model size, but much higher TFlops performance

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: | ----: |
|   16 | 18B  |  1 |  8 |        64 |      4 |     256 | 90.5s      |   26.1 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      4 |     512 | 177s       |   26.5 | 05-26 |
|   16 | 18B  |  1 |  8 |       256 |      4 |    1024 | 356s       |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   16 | 18B  |  1 |  4 |       128 |      4 |     512 | 179s       |   26.3 | 05-26 |
|   16 | 18B  |  1 |  4 |       128 |      6 |     768 | 262s       |   27.0 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      6 |     768 | 259s       |   27.3 | 05-26 |
|   16 | 18B  |  1 |  8 |        32 |      8 |     256 | 89s        |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   32 | 36B  |  1 |  8 |       128 |      4 |     512 | 82s        |   57.5 | 05-26 |
|   32 | 36B  |  1 |  8 |       128 |      6 |     768 | 123s       |   57.5 | 05-26 |
|   32 | 36B  |  1 |  8 |       256 |      6 |    1536 | 241s       |   58.7 | 05-26 |
|   32 | 36B  |  1 |  8 |       512 |      6 |    3072 | 478s       |   59.2 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 48B  |  1 | 16 |       256 |      4 |    1024 | 129s       |   48.7 | 05-25 |
|   64 | 48B  |  1 | 16 |       256 |      4 |    1024 | 217s       |   29.0 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 48B  |  1 | 16 |       256 |      6 |    1536 | 328s       |   28.7 | 05-26 |
|   64 | 48B  |  1 | 16 |       256 |      8 |    2048 | 435s       |   28.9 | 05-26 |
|   64 | 48B  |  1 | 16 |       512 |      6 |    3072 | 650s       |   29.0 | 05-26 |
|   64 | 48B  |  1 | 16 |       512 |      8 |    4096 | 870s       |   28.9 | 05-26 |
|   64 | 48B  |  1 | 32 |       256 |      4 |    1024 | 220s       |   28.6 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |


data:
- Size = Model Size
- `TP=4` in all of entries
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * pp_chunks * dp_size`
- PP chunks is the number of PP stages, so each pipeline handles `micro-batch-size * pp_chunks`
- Seq length is 1024

notes:
- 32gpus had a very snag fit for gpu memory for 36B model (others were in ~75%) so it might be a bit too risky to OOM-borderline


```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=127; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```


#### Megatron + Deepspeed 3D

Uses 3D:
- TP: tensor parallelism
- PP: pipeline parallelism
- DP: data parallelism

same features as Megatron's native, but improved by Deepspeed

**Performance**

| GPUs | Size | DP | PP | PP chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: | ------: | ---------: | -----: |
| 64   | 48B  | 1  | 16 | 256       | 4      | 1024    | 244s       | 25     |
|      |      |    |    |           |        |         |            |        |


- GAS = Gradient Accumulation size (same as PP_chunks / number of PP stages)
- Global_bs = pp_chunks*micro_bs*dp_size
- `TP_SIZE=4` (size of the node)

```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=244; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```




#### Megatron + Deepspeed ZeRO

See scripts and logs under [gpt2-meg-ds-zero](./gpt2-meg-ds-zero).

This one uses only TP from Megatron (no PP)

Not yet optimized with Deepspeed team!

**With Offload off**

**Performance**
| GPUs | Size  | DP | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ----: | -: | ---:   | -----:  | ---------: | -----: |
| 64   | 48B   | 16 | 48     | 768     | 122s       | 38.67  |
|      |       |    |        |         |            |        |
|      |       |    |        |         |            |        |



```
perl -le '$ng=64; $ms=48; $gbs=768; $sp=122; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```
- Seq length is 1024
- `TP=4` in all of entries
- `DP` is number of nodes here
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * nnodes`

- tried w/ and w/o Tiling once but saw no difference - perhaps would be more important on larger collections

**With full cpu offload**

| GPUs | Size | DP | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -----: | ------: | ---------: | -----: |
| 64   | 48B  | 16 | 64     | 1024    | 171s       | 39.71  |
|      |      |    |        |         |            |        |
|      |      |    |        |         |            |        |

XXX: Don't seem to be able to enlarge the global bs here - either using up more than 40GB RAM/gpu or OOMing


**With full optim cpu offload**




#### HF + Deepspeed Zero 3 + Full Offload

See scripts and logs under [gpt2-hf-ds](./gpt2-hf-ds).

Not yet optimized with Deepspeed team!

**Max model size**

| GPUs | Size  | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ----: | -----: | ------: | ---------: | -----: |
|   16 | 23B   |      4 |      64 |  58.72s    |  25.66 |
|   32 | 48B   |      4 |     128 | 114.56s    |  13.72 |
|   64 | 91B   |      4 |     256 | 222.72s    |  13.38 |
|      |       |        |         |            |        |


**Performance**

| GPUs | Size  | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ----: | -----: | ------: | ---------: | -----: |
| 64   | 48B   | 8      | 512     | 139.52s    | 22.54  |
|      |       |        |         |            |        |

Don't seem to be able to enlarge the global bs here - OOMing

- gradient checkpointing activated

- DP=1

- global bs = micro bs * DP

- Throughput reported by HF Trainer is samples_per_second per node - So total throughput in the table is `glob_bs/samples_per_second`

- TFlops: `model_size_in_B * 4 * 2 * seq * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`
```
perl -le '$ng=64; $ms=48; $gbs=512; $sp=139.52; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
22
```




## Megatron + Deepspeed ZeRO

**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3` is not in sync with M-LM master - so several config args don't match.

Status: Unoptimized

See scripts and logs under [gpt2-meg-ds-zero](./gpt2-meg-ds-zero).


## Megatron + Deepspeed 3D Parallelism

**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism` is not in sync with M-LM master - so several config args don't match.

Status: Unoptimized

See scripts and logs under [gpt2-meg-ds-3d](./gpt2-meg-ds-3d).

### Nodes=16


```
salloc -C v100-32g --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $ALL_CCFRSCRATCH/start-prod
```


```

cd ~/base/code/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism

CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$eha_ALL_CCFRSCRATCH/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$eha_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

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

rm -rf $eha_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

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
salloc -C v100-32g --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $ALL_CCFRSCRATCH/start-prod
```

32GB nodes

This works - at about 25GB / gpus - very slow 20s/it

Model size: 3.5B

Higher model the 40GB/gpu limit is passed and processes get killed.

We don't have zero.Init() here so the whole model is loaded onto each process - not possible to scale.

This memory gets released afterwards, but we don't have enough to bypass that hump.

```

# use custom PR branch to handle the model creation on the fly
cd ~/base/code/transformers-clm-any-model-config/

export HF_DATASETS_CACHE=$eha_ALL_CCFRSCRATCH/datasets
export HF_MODULES_CACHE=$eha_ALL_CCFRSCRATCH/modules
export HF_METRICS_CACHE=$eha_ALL_CCFRSCRATCH/metrics

MODEL=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron-gpt2-345m
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
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 2**30 :.0f}B')"

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
cd ~/base/code/transformers-clm-any-model-config/

export HF_DATASETS_CACHE=$eha_ALL_CCFRSCRATCH/datasets
export HF_MODULES_CACHE=$eha_ALL_CCFRSCRATCH/modules
export HF_METRICS_CACHE=$eha_ALL_CCFRSCRATCH/metrics

MODEL=$eha_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000


NNODES=16

# succeeded:
# MSIZE=7
# MSIZE=14
MSIZE=18 # maximum @32

# to try:
#MSIZE=48
#MSIZE=75


if [[ ${MSIZE} == 7 ]];  then
    NHIDDEN=4096; NLAYERS=36
elif [[ ${MSIZE} == 14 ]];  then
    NHIDDEN=6144; NLAYERS=32
elif [[ ${MSIZE} == 18 ]];  then
    NHIDDEN=6144; NLAYERS=40
elif [[ ${MSIZE} == 23 ]];  then
    NHIDDEN=7168; NLAYERS=40
elif [[ ${MSIZE} == 28 ]];  then
    NHIDDEN=7168; NLAYERS=48
elif [[ ${MSIZE} == 39 ]];  then
    NHIDDEN=8192; NLAYERS=48
elif [[ ${MSIZE} == 48 ]];  then
    NHIDDEN=8192; NLAYERS=64
elif [[ ${MSIZE} == 61 ]];  then
    NHIDDEN=9216; NLAYERS=64
elif [[ ${MSIZE} == 75 ]];  then
    NHIDDEN=10240; NLAYERS=64
elif [[ ${MSIZE} == 91 ]];  then
    NHIDDEN=11264; NLAYERS=64
elif [[ ${MSIZE} == 109 ]];  then
    NHIDDEN=12288; NLAYERS=64
elif [[ ${MSIZE} == 127 ]];  then
    NHIDDEN=13312; NLAYERS=64
elif [[ ${MSIZE} == 148 ]];  then
    NHIDDEN=14336; NLAYERS=64
elif [[ ${MSIZE} == 169 ]];  then
    NHIDDEN=15360; NLAYERS=64
elif [[ ${MSIZE} == 193 ]];  then
    NHIDDEN=16384; NLAYERS=64
else
    echo "invalid MSIZE: $MSIZE"
fi


GPUS_PER_NODE=4

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
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 2**30 :.0f}B')"

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

```


Stats:

```

```
