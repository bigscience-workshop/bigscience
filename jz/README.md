# jay-z
Jean Zay aka JZ pronounced "Jay-Z"

# Doc
- HF Internal: https://github.com/huggingface/conf/wiki/JZ
- Official: http://www.idris.fr/eng/jean-zay/
- Collaborative doc: https://jean-zay-doc.readthedocs.io/en/latest/
- Hackathon action plan: [Gdoc](https://docs.google.com/document/d/1HqUhc2CSSsj_cna1jk3apxUIvORNIUWQce9N1dFDpgk/edit#heading=h.sxozo8oh4fsb)


# accounts

- `ajs` - original dynamic access allocations - use it as we still have resources there
- `six` - the BigScience allocation

# First time setup

Add this to your `~/.bashrc` and run `bash` for the changes to take effect.
```

# start production environment:
# this loads modules, conda and sets all the relevant env vars
alias start-prod="source $six_ALL_CCFRWORK/start-prod"

# our production conda env is here:
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

# SLURM / Account specific settings

# share dirs/files with the group
umask 0007

# eha@gpu is the hackathon account

export SBATCH_ACCOUNT=six@gpu
export SLURM_ACCOUNT=six@gpu
export SALLOC_ACCOUNT=six@gpu

# specific caches
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export DATASETS_CUSTOM=$six_ALL_CCFRWORK/datasets-custom

# shortcut
export BASE=$six_ALL_CCFRWORK

# handy shortcuts
alias myjobs="squeue -u `whoami`"

```

Since most of our work is at `$six_ALL_CCFRWORK` you might want to add something like:
```
ln -s $six_ALL_CCFRWORK ~/base
```
and then you can quickly `cd` there w/o needing to type too much, and with the shortcut `$BASE` env var you now you can do one of 2 ways:
```
cd ~/base
cd $BASE
```


# Production environment

In order to use the production environment, run:

```
start-prod
```
which will:
- activate our custom conda environment which has everything in it
- setup env vars
- configure nice git-prompt with lots of useful info built in
- load the right `module`s

so basically use it when running production scripts.

The alias should have been set in `~/.bashrc` as instructed above.

Note: the fancy [bash-git-prompt](https://github.com/magicmonty/bash-git-prompt) tells you which conda env you are in, and then which branch your are in and a ton of useful git enfo, and it was extended to tell you whether you're in the login instance (prefix `0-0-1`) or whether you're on a GPU instance where it then shows something like `2-0,1,23-10` - the 3 numbers stand for `${SLURM_NNODES}-${SLURM_STEP_GPUS}-${SLURM_CPUS_PER_TASK}` - so you know what `srun` configuration you're logged into (or the login shell where you get 1 cpu and no gpus hence `0-0-1` ).

The production conda env `hf-prod` is too set up already, so you don't need to do anything, but here are some details on how it was done should you want to know.

Our production shared conda env is at `$six_ALL_CCFRWORK/conda`, you can make it visible by either doing this one:
```
conda config --append envs_dirs $six_ALL_CCFRWORK/conda
```
which will add this path to `~/.condarc` or use:
```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda
```
in your `~/.bashrc`.

You can use it for anything but please don't install anything into it (unless coordinating with others), as we want this to be a reliable environment for all to share.

# Monitoring

## nvtop

A nice alternative to `watch -n1 nvidia-smi`

```
module load nvtop
nvtop
```

# Troubleshooting

## pip install

If it's trying to install into your local `~/.local` folder it's because `pip` is in that `$PATH` before
`$six_ALL_CCFRWORK/conda/hf-prod/bin/` - push the last one to be first - or best don't install any python things locally - use conda for that. Check with `which pip` - it should be under `$six_ALL_CCFRWORK/conda/hf-prod/bin/pip`.

# Local resources

For your own personal explorations you can either create your own `conda` envr or use your local python, which has a few of issues, but it allows you to continue using JZ's pytorch `module`.

`pip install` installs into `$HOME/.local/lib/python3.7/site-packages`, however system-wide packages may take precedence. For example to do `develop` install of transformers use this workaround:
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install --user --no-use-pep517 -e .
```

May still have to override `PYTHONPATH=$WORK/hf/transformers-master/src` (edit to wherever your clone is) if you want to emulate `develop` build. Test:
```
export PYTHONPATH=$WORK/hf/transformers-master/src
python -c "import transformers; print(transformers.__version__)"
# 4.6.0.dev0
```

See [`envs`](./envs) for instructions on how to build conda and packages


# quotas

```
idrquota -m # $HOME @ user
idrquota -s # $STORE @ user
idrquota -w # $WORK @ user
idrquota -s -p six # $STORE @ shared
idrquota -w -p six # $WORK @ shared
```
if you prefer it the easy way here is an alias to add to `~/.bashrc`:
```
alias dfi="echo Personal:; idrquota -m; idrquota -s; idrquota -w; echo; echo \"Shared (six):\"; idrquota -w -p six; idrquota -s -p six"
```

# Directories

- `$six_ALL_CCFRSCRATCH` - for checkpoints
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

Personal:

- `$HOME` - 3GB for small files
- `$WORK` - 5TB / 500k inodes → sources, input/output files
- `$SCRATCH` - fastest (full SSD), no quota, files removed after 30 days without access
- `$STORE` - for long term storage in tar files (very few inodes!)


TODO:

- put the specially crafted super do-it-all `$six_ALL_CCFRWORK/envs` under git



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


# Projects

Check which projects one belongs to:

`idrproj`


# Aliases

```
# autogenerate the hostfile for deepspeed
# 1. deals with: SLURM_JOB_NODELIST in either of 2 formats:
# r10i1n8,r10i2n0
# r10i1n[7-8]
# 2. and relies on SLURM_STEP_GPUS=0,1,2... to get how many gpu slots per node
#
# usage:
# makehostfile > hostfile
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0; # workaround 4 gpu machines
while ($ENV{"SLURM_JOB_NODELIST"} =~ m/(\w+)(?:\[([\d-,]+)\])?,?/msg) {
$b=$1; $s=$2||q[""]; $s=~s/-/../g;
print map { "$b$_ slots=$slots\n" } eval $s }'
}
```

```
# auto-extract the master node's address from: SLURM_JOB_NODELIST1 which may contain r10i1n3,r10i1n[5-8],r10i1n7
# so here we want r10i1n3
function get_master_address() {
perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'
}
```
# LM Harness Evaluation

The evaluation harness from EleutherAI is integrated a submodule. We use a fork on [HF's Github](https://github.com/huggingface/lm-evaluation-harness).
To initialize the submodule, run:
```bash
git submodule init
git submodule update
```

Make sure you have the requirements in `lm-evaluation-harness`:
```bash
cd lm-evaluation-harness
pip install -r requirements.txt
```

To launch an evaluation, run:
```bash
python lm-evaluation-harness/main.py \
    --model gpt2 \
    --model_args pretrained=gpt2-xl \
    --tasks cola,mrpc,rte,qnli,qqp,sst,boolq,cb,copa,multirc,record,wic,wsc,coqa,drop,lambada,lambada_cloze,piqa,pubmedqa,sciq \
    --provide_description \ # Whether to provide the task description
    --num_fewshot 3 \ # Number of priming pairs
    --batch_size 2 \
    --output_path eval-gpt2-xl
```

Please note:
- As of now, only single GPU is supported in `lm-evaluation-harness`.
- The coding style is quite funky and can be hard to navigate...
