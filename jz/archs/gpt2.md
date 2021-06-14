# GPT2 Comparisons

## SLURM


1 nodes / 4 gpus:

```
srun --pty --nodes=1 --ntasks=4 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread --time=60 bash
```

For multi-node versions of these scripts please see `$six_ALL_CCFRWORK/code/jay-z/slurm`.


## Data

Using OpenWebText https://huggingface.co/datasets/openwebtext

```
from datasets import load_dataset
dataset = load_dataset("openwebtext", split='train')
dataset = load_dataset("stas/openwebtext-10k", split='train')
```

Ready datasets:

1. HF datasets use:

   * `openwebtext` - 8M records `--dataset_name "openwebtext"`
   * `stas/openwebtext-10k` - 10K records `--dataset_name "stas/openwebtext-10k"`

2. Jsonlines (derived):

   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext/openwebtext.jsonl`
   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/openwebtext-10k.jsonl`

3. Megatron-preprocessed datasets (derived):

   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-gpt2_text_document.*`
   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document.*`




#### How the above was done

To convert to jsonlines for Megatron

run on a beefy cpu instance (but firewalled), e.g.:
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:0 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Get vocabs:
```
cd $six_ALL_CCFRWORK/datasets-custom/vocabs
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

small
```
mkdir -p $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k
cd $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k
$six_ALL_CCFRWORK/code/jay-z/data/openwebtext-to-jsonl.py -10k
```

full (needs lots or RAM)
```
mkdir -p $six_ALL_CCFRWORK/datasets-custom/openwebtext
cd $six_ALL_CCFRWORK/datasets-custom/openwebtext
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $six_ALL_CCFRWORK/code/jay-z/data/openwebtext-to-jsonl.py
```

To prep a 10k-sample for megatron
```
cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/openwebtext-10k.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2 \
       --vocab $six_ALL_CCFRWORK/datasets-custom/vocabs/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $six_ALL_CCFRWORK/datasets-custom/vocabs/gpt2-merges.txt \
       --append-eod \
       --workers 8
```

To prep a full dataset for megatron
```
cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext/openwebtext.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-gpt2 \
       --vocab $six_ALL_CCFRWORK/datasets-custom/vocabs/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $six_ALL_CCFRWORK/datasets-custom/vocabs/gpt2-merges.txt \
       --append-eod \
       --workers 8
```
as it should take a few hours to convert, use `slurm/jsonl-to-meg-gpt2.slurm` job to complete it
```
sbatch jsonl-to-meg-gpt2.slurm
```


## Model


Ready pretrained models: GPT2 megatron_lm_345m

1. HF

* `$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m`

2. Megatron

* `$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release`


#### How the above was done

**Megatron model prep**


1. Download nvidia checkpoint:
```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
```
2.
```
unzip megatron_lm_345m_v0.0.zip
```


**HF transformers model prep**


prep HF model - it's not avaliable on the hub

1. Download nvidia checkpoint:
```
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
```

2. Convert:
```
python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py megatron_lm_345m_v0.0.zip
```

3. Fetch missing files
```
git clone https://huggingface.co/nvidia/megatron-gpt2-345m/
```

4. Move the converted files into the cloned model dir
```
mv config.json pytorch_model.bin megatron-gpt2-345m/
```

5. megatron-gpt2-345m dir should now have all the files which can be passed as  `--model_name_or_path megatron-gpt2-345m`


XXX: may be will use some small samples for testing - need .txt and .json for megatron-lm

```
    #--train_file {data_dir}/sample_text.txt \
    #--validation_file {data_dir}/sample_text.txt \
```


## Training

### Megatron-LM

running native https://github.com/NVIDIA/Megatron-LM

```
cd $six_ALL_CCFRWORK/code
git clone https://github.com/NVIDIA/megatron-lm
cd megatron-lm
```


### Megatron: finetuning on a single GPU


Setup: 1 node / 1 gpu
```
srun --pty --nodes=1 --ntasks=4 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Launch training:

adding `--finetune` to work with existing checkpoint, remove to train from scratch
```
CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRWORK/checkpoints/gpt2

#    --train-samples 200 \
#    --lr-decay-samples 150 \
#    --train-iters 100000 \
#    --lr-decay-iters 320000 \
GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr-warmup-fraction .01 \
    --finetune \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --fp16 \
    --checkpoint-activations \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

python pretrain_gpt.py \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH
```

Speed: 0.637s / iteration



### Megatron: finetune distributed with MP

2 types of parallelism supported:

- `--tensor-model-parallel-size`
- `--pipeline-model-parallel-size`

To get the average throughput have to process the logfile:

```
perl -nle 'use List::Util qw/sum/; m|elapsed time per iteration .ms.: ([\d\.]+)| && push @x, $1; END { print sum(@x)/+@x }' std-1611136.out
```

Setup: 1 node / 4 gpus
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Launch training:
```
CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRWORK/checkpoints/gpt2

GPUS_PER_NODE=4
NNODES=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS=" \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

NLAYERS=24
NHIDDEN=1024
BATCHSIZE=4

#    --train-iters 100000 \
#    --lr-decay-iters 320000 \
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
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
    --checkpoint-activations \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

python -m torch.distributed.launch \
    $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
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
```


Speed: 0.560s / iteration


### Megatron: finetune distributed with MP - multi-node


Use `jay-z/slurm/meg-gpt2-multi-node.slurm`.

Speed: 0.560s / iteration


### Megatron-LM+Deepspeed: w/ deepspeed Pipeline

This is the version with Deepspeed's pipeline

https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh



Setup: 1 node / 4 gpus
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


```

cd ~/base/code/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism


CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRWORK/checkpoints/gpt2

GPUS_PER_NODE=4
NNODES=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

config_json="./ds_config.json"


# Megatron Model Parallelism
mp_size=2
# DeepSpeed Pipeline parallelism
pp_size=2

NLAYERS=24
NHIDDEN=1024
BATCHSIZE=4
NUM_ATTN_HEADS=16


LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"

GAS=16

#ZeRO Configs
stage=0
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

GPT_ARGS=" \
    --model-parallel-size ${mp_size} \
    --pipe-parallel-size ${pp_size} \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NUM_ATTN_HEADS \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --batch-size $BATCHSIZE \
    --gas $GAS \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --fp16 \
    "
    #--tensorboard-dir ${LOGDIR}

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
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

full_options="${GPT_ARGS} ${OUTPUT_ARGS} ${DEEPSPEED_ARGS} ${CHKP_ARGS}"

run_cmd="deepspeed --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

```


### Megatron-LM+Deepspeed: w/ deepspeed zero3/inf

This is the version with Deepspeed's Zero3/inf

https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/examples/ds_pretrain_gpt2-zero3.sh



Setup: 1 node / 4 gpus

```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


```

cd ~/base/code/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3


# Change for multinode config
MP_SIZE=1

GPUS_PER_NODE=4
NNODES=1

DLTS_NUM_WORKER=$NNODES
DLTS_NUM_GPU_PER_WORKER=$GPUS_PER_NODE

NUM_WORKERS=${DLTS_NUM_WORKER}
NUM_GPUS_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER}
HIDDEN_SIZE=1024
NUM_LAYERS=24
BATCHSIZE=4
NUM_ATTN_HEADS=16

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRWORK/checkpoints/gpt2

config_json="./ds_zero_stage_3_config.json"

#ZeRO Configs
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


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


GPT_ARGS=" \
    --model-parallel-size ${MP_SIZE} \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --batch-size $BATCHSIZE \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --fp16 \
    --scattered-embeddings \
    --split-transformers \
    "
    #--tensorboard-dir ${LOGDIR}

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
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


full_options="${GPT_ARGS} ${OUTPUT_ARGS} ${DEEPSPEED_ARGS} ${CHKP_ARGS}"

run_cmd="deepspeed --num_nodes ${NNODES} --num_gpus ${GPUS_PER_NODE} pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

```


### HF transformers distributed

Have to run once on a non-gpu instance which has network to retrieve the model and data files and get those cached.


```
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
```

```
MODEL=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron-gpt2-345m
DATASET="stas/openwebtext-10k"
```

```
cd $six_ALL_CCFRWORK/code/transformers
#git clone https://github.com/huggingface/transformers
#cd transformers
```

```
source $six_ALL_CCFRWORK/start-prod

```


first run on networked instance to get the dataset et, al.
```
PYTHONPATH="src" \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 160 \
    --max_eval_samples 160 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --report_to none
```


2nd run on gpu instance w/o network
```
PYTHONPATH="src" \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m torch.distributed.launch --nproc_per_node=4 \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --fp16 \
    --report_to none
```

Speed:

train_samples_per_second   =      5.043


let's do multi-node:

Setup: 2 nodes / 4 gpus
```
srun --pty --nodes=2 --ntasks=8 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Launch training:

```
PYTHONPATH="src" \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=4 \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --fp16 \
    --report_to none
```

### HF transformers + Deepspeed + zero2



```
PYTHONPATH="src" \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed --num_nodes 1 --num_gpus 4 \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --fp16 \
    --report_to none \
    --deepspeed tests/deepspeed/ds_config_zero2.json
```

Speed:

train_samples_per_second   =       2.14

### HF transformers + Deepspeed + zero3

probably should test w/o offload

```
PYTHONPATH="src" \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed --num_nodes 1 --num_gpus 4 \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_train_samples 1000 \
    --max_eval_samples 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --block_size 64 \
    --fp16 \
    --report_to none \
    --deepspeed tests/deepspeed/ds_config_zero3.json
```

Speed:

train_samples_per_second   =      0.952



### HF transformers + Deepspeed + zero2 - multi-node


Use `jay-z/slurm/hf-ds-gpt2-multi-node.slurm`.

Speed:  / iteration
