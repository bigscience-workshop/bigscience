# Train 1 - 13B - unmodified Megatron gpt2 - baseline

## Architecture

40 layers | 40 heads (128d each) | hid size 5120 | ffn size 20480

## task

regular transformer language model with prefix-LM objective

## global batch size: use a schedule

- start from 32k tokens
- increase linearly to 2048k over 10,000 steps (for a total of ~10B tokens)
- example schedule: increase batch size by 32k tokens every 160 steps

## optimizer: AdamW,  β1=0.9, β2=0.95 eps=1e−8

- learning rate: peak=1e-4, warmup over 2000 steps, 
- clipping by global norm of 1 (as in GPT-3)
- weight decay of 0.1

## sequence length: prompt 512 tokens + autoregressive part 512 tokens

- loss is computed over the last 512 tokens _only_
- both prompt and autoregressive part count toward batch size
- pre-padding: sample minibatches, where prompt is (partially or fully) empty - align prompts to the right, use special padding to pad from the left
- no post-padding: always fill the entire 1024 tokens with text, - if a document is too short, join multiple documents over EOS token (same as in GPT3)

## dataset:

- by default, use OSCAR:
   - use version with full documents (*not* individual sentences) 
   - this version is *not* public, ask tokenization WG for access
- ask data tooling WG and @Julien Launay about filtering - otherwise our model will generate… naughty stuff :)
- tokenization / subword:
   -@mryab will ask the tokenization WG for best practices

## extras

- need to save intermediate checkpoints (prioritizing early steps) for experiments on training dynamics (@Hendrik Strobelt)

- would be great to have some public dashboard (e.g. tensorboard) for collaborators
