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
- 1x V100-16GB
- The computing hours are not deducted from your allocation

to request:
```
srun --pty --partition=prepost --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --time=1:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

or to work interactively there, could ssh directly to that partition via:
```
ssh jean-zay-pp          # from inside
ssh jean-zay-pp.idris.fr # from outside
```
There are 4 boxes, so `jean-zay-pp1`, ..., `jean-zay-pp4`. It's possible that larger numbers have less users, but not necessarily.

In this case there is no need to do SLURM,


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

[Detailed information](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-calculateurs-disques-eng.html)

Checking usage:
```
idrquota -m # $HOME @ user
idrquota -s -p six # $STORE @ shared (this is updated every 30min)
idrquota -w -p six # $WORK @ shared
```


if you prefer it the easy way here is an alias to add to `~/.bashrc`:
```
alias dfi=' \
echo \"*** Total \(six\) ***\"; \
idrquota -w -p six; \
idrquota -s -p six; \
echo WORKSF: $(du -hs /gpfsssd/worksf/projects/rech/six/commun | cut -f1); \
echo WORKSF: $(du -hs --inodes /gpfsssd/worksf/projects/rech/six/commun | cut -f1) inodes; \
echo; \
echo \"*** Personal ***\"; \
idrquota -m; \
echo WORK: $(du -hs $WORK | cut -f1); \
echo WORK: $(du -hs --inodes $WORK | cut -f1) inodes; \
echo STORE: $(du -hs $STORE | cut -f1); \
echo STORE: $(du -hs --inodes $STORE | cut -f1) inodes; \
echo SCRATCH: $(du -hs $SCRATCH | cut -f1); \
echo SCRATCH: $(du -hs --inodes $SCRATCH | cut -f1) inodes; \
'
```
This includes the report on usage of personal WORK and SCRATCH partitions.



## Directories

- `$six_ALL_CCFRSCRATCH` - for checkpoints - make sure to copy important ones to WORK or tarball to STORE
- `$six_ALL_CCFRWORK` - for everything else
- `$six_ALL_CCFRSTORE` - for long term storage in tar files (very few inodes!)
- `/gpfsssd/worksf/projects/rech/six/commun/` - for conda and python git clones that take tens of thousands of inodes - it's a small partition with a huge number of inodes. 1TB and 3M inodes.
XXX: update this and above once env var was created.


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


## Efficient tar-balling to STORE

When short on space you don't want to create large tarballs in the WORK dir, instead tar directly to the destination, e.g.

e.g. w/o gzip since we already have arrow binary files

```
mkdir -p $six_ALL_CCFRSTORE/datasets
cd $six_ALL_CCFRWORK/datasets
tar -cvf $six_ALL_CCFRSTORE/datasets/openwebtext.tar openwebtext
```


e.g. w/ gzip for non-binary data
```
tar -czvf $six_ALL_CCFRSTORE/datasets/openwebtext.tar openwebtext
```

If the file is large and takes some resources to build, `tar` will get killed, in such case you can't do it from the login instance and have to use one of the beefier instances. e.g.:
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:0 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
tar ...
```
and if that's not enough do a slurm job
