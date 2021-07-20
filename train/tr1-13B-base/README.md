# Train 1 - 13B - unmodified Megatron gpt2 - baseline

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

## task

regular transformer language model with prefix-LM objective

## global batch size

use a schedule

- start from 32k tokens
- increase linearly to 2048k over 10,000 steps (for a total of ~10B tokens)
- example schedule: increase batch size by 32k tokens every 160 steps


XXX: sort this out
```
    --rampup-batch-size
    --global-batch-size

```


## optimizer: AdamW,  β1=0.9, β2=0.95 eps=1e−8

- learning rate: peak=1e-4, warmup over 2000 steps, 
- clipping by global norm of 1 (as in GPT-3)
- weight decay of 0.1

```
    --lr 1e-4 \
    --lr_warmup_iters 2000
    --clip-grad 1.0 \
    --weight-decay 1e-1 \

```


## sequence length

prompt 512 tokens + autoregressive part 512 tokens

XXX: sort this out

- loss is computed over the last 512 tokens _only_
- both prompt and autoregressive part count toward batch size
- pre-padding: sample minibatches, where prompt is (partially or fully) empty - align prompts to the right, use special padding to pad from the left
- no post-padding: always fill the entire 1024 tokens with text, - if a document is too short, join multiple documents over EOS token (same as in GPT3)

## dataset:

- by default, use OSCAR:
   - use version with full documents (*not* individual sentences) 
   - this version is *not* public, ask tokenization WG for access
- ask data tooling WG and @Julien Launay about filtering - otherwise our model will generate… naughty stuff :)  @Julien Launay wrote: "In my experience with raw OSCAR, you end up with a very naughty model. This might be OK for a first test run. We used CCNet (https://github.com/facebookresearch/cc_net) for our French GPT-2, and it increased generation quality and reduced NSFW content a lot"
- tokenization / subword:
   -@mryab will ask the tokenization WG for best practices

## extras

- need to save intermediate checkpoints (prioritizing early steps) for experiments on training dynamics (@Hendrik Strobelt)

- would be great to have some public dashboard (e.g. tensorboard) for collaborators
