# Train 1 - 13B - unmodified Megatron gpt2 - baseline


## Task

Auto-regressive objective using regular Megatron-LM GPT2 language model



## Architecture

40 layers | 40 heads (128d each) | hid size 5120 | ffn size 20480


config:
```
NLAYERS=40
NHIDDEN=5120
NHEADS=32
FFN_HIDDEN_SIZE=20480

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    [...]
    "
```



## Optimizer

- AdamW,  β1=0.9, β2=0.95 eps=1e−8
- learning rate: peak=1e-4, warmup over 2000 steps
- clipping by global norm of 1 (as in GPT-3)
- weight decay of 0.1

```
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr_warmup_iters 2000
    --clip-grad 1.0 \
    --weight-decay 1e-1 \

```


## Sequence Length

Default Megatron-LM LM with 1024 tokens

All dataset samples are at least 1024 tokens long (filtered via `transformers`'s `GPT2TokenizerFast`).

```
SEQ_LEN=1024

    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \

```


## Global batch size

use a schedule:

- start from 32k tokens
- increase linearly to 2048k over 10K steps (for a total of ~10B tokens = 10M samples)
- then continue at 2048k for 290K steps (290B tokens = 290M samples)

Total: 300B tokens (300M steps)

syntax:
```
--rampup-batch-size <start batch size>  <batch size increment> <ramp-up samples>
```

At seqlen 1024 (1k tokens is bs=1), we get:

XXX: can we use scientific notation in args?

```
    --rampup-batch-size 32 32 10e6 \
    --global-batch-size 2048 \
```

This means it will start with global batch size 32 and over 63 (`(2048-32)/32`) intervals will increase the
batch size by 32 linearly to 2048. Each interval is ~160 steps (`10000/63`).

Ramp-Up samples is calculated to be ~10M. First 160 steps at bs=32, next 160 steps at `bs=64=2*32`, next 160 steps at `bs=192=3*32`, ..., finally last 160 steps at `bs=2016=63*32`, all summed up gives 10,321,920 from `32*160*(1+2+3+4+...+63)` or `5120*63*(1+63)/2`.


## Checkpoints

We need the checkpoints:

1. in order to be able to resume the training when the training is prematurely stopped for whatever reason.
2. In addition a special saving schedule has been requested by the interpretabity group.

Because there are 3 different schedules, and Megatron-LM has only fixed checkpoint saving schedule, we will need 3 different run scripts, to be launched in a sequence, each starting once the previous has finished.

1. steps 1-100 - 10 checkpoints, interval 10 steps
2. steps 101-1000 - 50 checkpoints, interval 18 steps
3. steps 1001-300K - 100+ checkpoints, interval 1500 steps
4. if still needed, can continue with schedule 3

note: the interoperability study doesn't care for checkpoints in the range of 1k-20k, so we only save those to be able to restart the training.

1.
```
    --train-iter 100 \
    --save-interval 10 \
```

2.
```
    --train-iter 1000 \
    --save-interval 18 \
```

3.
```
    --train-iter 300000 \
    --save-interval 1500 \
```

XXX: I assume that each run get a different input at the beginning - i.e. dataloader doesn't have a fixed seed?

Because it'd be potentially too demanding to export TBs of data and the intended users might not be even able to download all that data, most likely we will need to run the interpretabity post-analysis experiments on JZ and send the reports to those who need the reports.

XXX: figure out how Megatron-LM resumes from checkpoint. Does it need the exact path or does it auto-discover the latest checkpoint by default.

```
--load path_to_check_point \
```



## Logging

XXX: Tensorboard config?

XXX: need to export to HF model hub for collaborators

XXX: need to find how to best export from JZ

XXX: need to figure out how we emulate crontab on JZ via self-respawning slurm job (since JZ has no crontab)


## Dataset

We are using english-only OSCAR with full documents (*not* individual sentences).

After filtering 1024K+ token-long documents we have ~70M shuffled records in 900GB of jsonl text which was then pre-processed into Megatron-LM format.

A rough estimate, considering average 4.5chars/word, 900GB is roughly ~200B tokens. But we are only going to use 1024 tokens from each record, so we really have only 70B tokens for epoch 1, and then for subsequent epochs other slices of the document will be taken randomly in documents longer than 1024 tokens.

For more information on the pre-processing process see: [OSCAR](../../data/oscar/README.md)

We aren't doing of the following - next training perhaps?

- ask data tooling WG and @Julien Launay about filtering - otherwise our model will generate… naughty stuff :)  @Julien Launay wrote: "In my experience with raw OSCAR, you end up with a very naughty model. This might be OK for a first test run. We used CCNet (https://github.com/facebookresearch/cc_net) for our French GPT-2, and it increased generation quality and reduced NSFW content a lot"
- tokenization / subword:
   -@mryab will ask the tokenization WG for best practices



## Estimated run time

Best case scenario:
```
$ python -c 'Btokens=300; Bmodel=13; n_gpus=256; Tflops=45; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
31.35 days
```

You will find the detailed explanation of the estimation formula [here](../../math/README.md#estimate-model-training-time).

Of course, the training will be much slower in the first 10k steps because of the batch size rampup, where the pipeline will be very inefficient.



## Exports

- GCS https://console.cloud.google.com/storage/browser/bigscience
- The Hub https://huggingface.co/bigscience


## Extras
