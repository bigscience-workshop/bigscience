# Things to do


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

- check if --jobid=$SLURM_JOB is actually needed in the slurm script - especially when doing it interactively

- add alerts for loss spikes

- check that my syncing script doesn't sync deleted files, should SCRATCH wipe something out that is already on the hub!

- update deepspeed_to_transformers.py to require a specific version once a new version of transformers is released and then update the doc https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base#checkpoint-conversion-and-upload

- adjust Meg-DS to use the correct init_method with pt-1.9+
https://github.com/pytorch/pytorch/issues/63874#issuecomment-904899656
- see if can speed up the meg cuda kernels building
https://huggingface.slack.com/archives/C01NHER1JLS/p1630520151064500?thread_ts=1630473623.060700&cid=C01NHER1JLS

- since we are starting to tweak the seed, we should start logging the ranges of iteration for each seed, so that down the road we could reproduce the data.


sysadmin:
-
