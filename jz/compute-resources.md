# Compute Resources

## Login Instance

This is the shell you get into when ssh'ng from outside

- Networked (except ssh to outside)
- 1 core per user
- 5 GB of RAM per user
- 30 min of CPU time per process

## Pre/post processing Instance

Activated with `--partition=prepost`

- Networked
- only 4 nodes
- 2 to 20 hours
- No limitations of the login shell
- I think there 1x 16gb gpu there, but it's there w/o asking

to request:
```
srun --pty --partition=prepost --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --time=1:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

## GPU Instances

- No network to outside world
- 160 GB of usable memory. The memory allocation is 4 GB per reserved CPU core if hyperthreading is deactivated (`--hint=nomultithread`). So max per node is `--cpus-per-task=40`


## Quotas

Group/project (`six`):

- `$six_ALL_CCFRSCRATCH` - no quota fastest (full SSD),  →  files removed after 30 days without access
- `$six_ALL_CCFRWORK` - 5TB / 500k inodes (slower than SCRATCH) → sources, constantly used input/output files
- `$six_ALL_CCFRSTORE` - 50TB / 100k inodes (slow) → for long term storage in tar files (very few inodes!)

Personal:

- `$HOME` - 3GB / 150k inodes (for small files)
- `$SCRATCH` - fastest (full SSD), no quota, files removed after 30 days without access
- `$WORK` - Shared with the `$six_ALL_CCFRWORK` quota, that is `du -sh $six_ALL_CCFRWORK/..`
- `$STORE` - Shared with the  `$six_ALL_CCFRSTORE` quota, that is `du -sh $six_ALL_CCFRSTORE/..`

Note that WORK and STORE group quotas of the project include all project's users' WORK and STORE usage correspondingly.

Checking usage:
```
idrquota -m # $HOME @ user
idrquota -s -p six # $STORE @ shared (this is updated every 30min)
idrquota -w -p six # $WORK @ shared
```


if you prefer it the easy way here is an alias to add to `~/.bashrc`:
```
alias dfi=" \
echo \"*** Total (six) ***\"; \
idrquota -w -p six; \
idrquota -s -p six; \
echo; \
echo \"*** Personal ***\"; \
idrquota -m; \
echo WORK: `du -hs $WORK | cut -f1`; \
echo WORK: `du -hs --inodes $WORK| cut -f1` inodes; \
echo STORE: `du -hs $STORE | cut -f1`; \
echo STORE: `du -hs --inodes $STORE| cut -f1` inodes; \
echo SCRATCH: `du -hs $SCRATCH | cut -f1`; \
echo SCRATCH: `du -hs --inodes $SCRATCH| cut -f1` inodes; \
"
```
This includes the report on usage of personal WORK and SCRATCH partitions.



## Diagnosing the Lack of Disc Space

To help diagnose the situations when we are short of disc space here are some tools:

Useful commands:

* Get current dir's sub-dir usage breakdown sorted by highest usage first:
```
du -ahd1 | sort -rh
```

* Check that users don't consume too much of their personal `$WORK` space, which goes towards the total WORK space limit.

```
du -ahd1 $six_ALL_CCFRWORK/.. | sort -rh
```




## Directories

- `$six_ALL_CCFRSCRATCH` - for checkpoints - make sure to copy important ones to WORK or tarball to STORE
- `$six_ALL_CCFRWORK` - for everything else
- `$six_ALL_CCFRSTORE` - for long term storage in tar files (very few inodes!)

More specifically:

- `$six_ALL_CCFRWORK/cache_dir` - `CACHE_DIR` points here
- `$six_ALL_CCFRWORK/checkpoints` - symlink to `$six_ALL_CCFRWORK/checkpoints` - point slurm scripts here
- `$six_ALL_CCFRWORK/code` - clones of repos we use as source (`transformers`, `megatron-lm`, etc.)
- `$six_ALL_CCFRWORK/conda` - our production conda environment
- `$six_ALL_CCFRWORK/datasets` - cached datasets (normally under `~/.cache/huggingface/datasets`)
- `$six_ALL_CCFRWORK/datasets-custom` - Manually created datasets are here (do not delete these - some take many hours to build):
- `$six_ALL_CCFRWORK/downloads` -  (normally under `~/.cache/huggingface/downloads`)
- `$six_ALL_CCFRWORK/envs` - custom scripts to create easy to use environments
- `$six_ALL_CCFRWORK/models-custom` - manually created or converted models
- `$six_ALL_CCFRWORK/modules` -  (normally under `~/.cache/huggingface/modules`)
