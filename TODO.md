# Things to do

## Carbon Footprint Tracking

Instrument multi-node carbon footprint tracking. https://github.com/mlco2/codecarbon
Seems like we only need to add about 2 lines in our code.

The decision is to run it on one node (gpu?), and then the results will be multiplied by the number of nodes. It generates a csv results. Need to figure out where to broadcast it to from JZ.

Blocking event: no codebase to yet to add it to - need to fork https://github.com/microsoft/Megatron-DeepSpeed once it's ready.


## Weights-Only checkpoints

Contributors that have no access to JZ will want to have intermediary checkpoints to work with. It'll be very slow to scp full checkpoints. Would it be possible to either post-process the Deepspeed PP checkpoints and extract just the model weights before copying those from JZ?

The current DS PP format saves each layer's state dict in its own file, and they're named differently than the optimizer states. Could be as simple as pattern matching the scp. The pipeline engine selectively loads the files based on pipeline rank, so no need to merge them.

But users outside of JZ will very likely have a different HW setup, so these will need to be re-shaped to match a new PP-degree.

- Also wrote checkpoint-shrinker, but not sure how to test it doesn't break anything. I need a solid checkpoint to compare eval on. Probably can use one of the recent 13B checkpoints and check the loss is the same - need just 2 nodes to run the test.

update: tested it to work correctly on one of the recent checkponts

- probably need to write layers-to-single weights file converter as well. Need to find a mapping from `params_to_save` saved in the checkpoint to the megatron-style native checkpoint.

## conda packages

currently each one of us has a copy of the same conda packages:

```
conda config --show pkgs_dirs envs_dirs and the output is:
pkgs_dirs:
  - /gpfslocalsup/pub/anaconda-py3/2020.02/pkgs
  - /linkhome/rech/genhug01/uue59kq/.conda/pkgs
envs_dirs:
  - /gpfswork/rech/six/commun/conda
  - /linkhome/rech/genhug01/uue59kq/.conda/envs
  - /gpfslocalsup/pub/anaconda-py3/2020.02/envs
```

we should aggregate them under the same dir.

probably need to find out the right env var (best) or ~/.condarc (less good) and point it to the shared conda env.

- also document in the getting started docs to make sure new users don't end up with ~/.conda dir which uses up their HOME dir to 100%.


# TODO

general:
- need a watchdog to test we aren't close to running out of disc space - especially SCRATCH and WORK

- don't generate 'latest' file when under deepspeed - otherwise it's hard to tell which is the real tagging file
- update chronicles
- check if --jobid=$SLURM_JOB is actually needed in the slurm script - especially when doing it interactively
- Set up a basic MLflow setup when Canwen installs ML server https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/87
- add alerts for loss spikes
- check that my syncing script doesn't sync deleted files, should SCRATCH wipe something out that is already on the hub!
- figure out what's missing in the last record here with time:
```
 iteration    49010/  311541 | consumed samples:     31167824 | elapsed time per iteration (ms): 80703.5 | learning rate: 8.745E-05 | global batch size:  1024 | lm loss: 2.597475E+00 | loss scale: 131072.0 | grad norm: 16142.646 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
```
- possible recover TFLOPS reports per iteration that were there in DSE before Megatron-Deepspeed MSFT repo was created. It got lost in the shuffle. Or may be we just need to enable tflops in DS config file?

- update deepspeed_to_transformers.py to require a specific version once a new version of transformers is released and then update the doc https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base#checkpoint-conversion-and-upload

- add codecarbon validation - emissions.csv generated to the test suite
- add codecarbon>=2.0.0 to requirements.txt
- adjust Meg-DS to use the correct init_method with pt-1.9+
https://github.com/pytorch/pytorch/issues/63874#issuecomment-904899656
- see if can speed up the meg cuda kernels building
https://huggingface.slack.com/archives/C01NHER1JLS/p1630520151064500?thread_ts=1630473623.060700&cid=C01NHER1JLS

- since we are starting to tweak the seed, we should start logging the ranges of iteration for each seed, so that down the road we could reproduce the data.

data:
- extract the source text based on a range of sample ids: created an issue:
https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/56


# new training

- test wide shallow model 100B - run benchmark - same as 13B
model size 100B
attention heads / divisable by 8 (tensor cores) and tp/pp divisable
number of layers
slightly larger than gpt3
larger GBS 3.2M token - perhaps 4M tokens
keep rampup





- test too deep model same layers as in 200B but smaller - 96 layers

sysadmin:
-
