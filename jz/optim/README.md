# Optimizer

2 considerations:

1. AdamW - robust, but memory-hungry
2. Adafactor - more lean, but more difficult to figure out to converge - more likely to be used if the model is t5-like


## HF

default AdamW

## Deepspeed

default AdamW


## Megatron

Has `--optimizer adam` via `apex`

To add a new optimizer need to add a new option [here](https://github.com/NVIDIA/Megatron-LM/blob/aed2f75e209e525c842aec7c044af7acae2a4614/megatron/optimizer/__init__.py#L50) and import that new optimizer.
