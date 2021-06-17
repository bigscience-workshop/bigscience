

# GPT2 Comparisons on EnWiki

This is a back up copy of the work in progress notes when it was started using Enwiki.

It's currently not being kept up-to-date

For now we moved to openwebtext so the main README.md doc is now using that.

## SLURM


1 nodes / 4 gpus:

```
srun --pty --nodes=1 --ntasks=4 --cpus-per-task=10 --gres=gpu:4 --hint=nomultithread --time=60 bash
```



## Data



### Enwiki

data prep  https://github.com/NVIDIA/Megatron-LM#collecting-wikipedia-training-data

Megatron-LM's training is based on enwiki
huge dataset - but it's not needed for sample run, see short sample below
```
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
pip install git+https://github.com/attardi/wikiextractor
wikiextractor --json enwiki-latest-pages-articles.xml.bz2
```


short sample
```
cd data
wget https://dumps.wikimedia.org/enwiki/20210501/enwiki-20210501-pages-articles-multistream1.xml-p1p41242.bz2
wikiextractor --json enwiki-20210501-pages-articles-multistream1.xml-p1p41242.bz2
mv text text-short
cd -
python tools/preprocess_data.py \
       --input data/text-short/AD/wiki_29 \
       --output-prefix my-gpt2 \
       --vocab data/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file data/gpt2-merges.txt \
       --append-eod
```

### OpenWebText

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

   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-gpt2_*` (still churning)
   * `$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_*`


#### How the above was done

To convert to jsonlines for Megatron

run on a beefy cpu instance (but firewalled), e.g.:
```
srun --pty --nodes=1 --ntasks=4 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --time=60 bash
```

small
```
mkdir -p $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k
cd $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k
$six_ALL_CCFRWORK/code/bigscience/data/megatron/openwebtext-to-jsonl.py -10k
```

full (needs lots or RAM)
```
mkdir -p $six_ALL_CCFRWORK/datasets-custom/openwebtext
cd $six_ALL_CCFRWORK/datasets-custom/openwebtext
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $six_ALL_CCFRWORK/code/bigscience/data/megatron/openwebtext-to-jsonl.py
```



To prep for megatron 10k-sample
```
cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/openwebtext-10k.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2 \
       --vocab data/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file data/gpt2-merges.txt \
       --append-eod
```

To prep for megatron full dataset
```
cd $six_ALL_CCFRWORK/code/megatron-lm
python tools/preprocess_data.py \
       --input $six_ALL_CCFRWORK/datasets-custom/openwebtext/openwebtext.jsonl \
       --output-prefix $six_ALL_CCFRWORK/datasets-custom/openwebtext/meg-gpt2 \
       --vocab data/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file data/gpt2-merges.txt \
       --append-eod
```
as it should take about 11h to convert use `gpt2/jsonl-to-meg.slurm` job to complete it



## Model


### HF transformers model prep


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

### finetuning on a single GPU


adding --finetune to work with existing checkpoint
```
CHECKPOINT_PATH=checkpoints/megatron_lm_345m_v0.0/release
SAVE_CHECKPOINT_PATH=data/checkpoints
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

#          --train-samples 200 \
#          --lr-decay-samples 150 \
#         --train-iters 100000 \
#         --lr-decay-iters 320000 \
GPT_ARGS="--num-layers 24 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 4 \
          --global-batch-size 8 \
          --lr 0.00015 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --finetune \
          --train-iters 1000 \
          --lr-decay-iters 800 \
          --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $SAVE_CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH
```


### finetune distributed with MP


```
OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=my-gpt2_text_document
CHECKPOINT_PATH=checkpoints/megatron_lm_345m_v0.0/release
SAVE_CHECKPOINT_PATH=data/checkpoints

GPUS_PER_NODE=4
NNODES=1

#Change for multinode config

MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#         --train-iters 100000 \
#         --lr-decay-iters 320000 \

python -m torch.distributed.launch \
       $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --save $SAVE_CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       $OUTPUT_ARGS \
       --train-samples 5000 \
       --lr-decay-samples 4000 \
       --finetune \
       --fp16
```


### stats ###

```
16gb v100:
nodes=1, gpus=4 => 560 ms / iteration
nodes=1, gpus=1 => 628 ms / iteration
```


### Megatron-LM+Deepspeed: w/ deepspeed Pipeline

This is the version with Deepspeed's pipeline

https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh



### Megatron-LM+Deepspeed: w/ deepspeed zero3/inf

This is the version with Deepspeed's Zero3/inf

https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-ZeRO3/examples/ds_pretrain_gpt2-zero3.sh



### HF transformers distributed

Have to run once on a non-gpu instance which has network to retrieve the model and data files and get those cached.


```
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
```

```
MODEL=$WORK/hf/megatron-lm/checkpoints/megatron-gpt2-345m
DATASET1=" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1"

DATASET=" \
    --dataset_name openwebtext"
```

first run on networked instance to get the dataset et, al.
```
PYTHONPATH="src" \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    $DATASET \
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
    $DATASET \
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
    --fp16 \
    --report_to none
```



### HF transformers + Deepspeed

probably should test zero2 and zero3

```
PYTHONPATH="src" \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed --num_nodes 1 --num_gpus 4 \
examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $WORK/hf/megatron-lm/checkpoints/megatron-gpt2-345m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
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
    --fp16 \
    --report_to none \
    --deepspeed tests/deepspeed/ds_config_zero3.json

```
