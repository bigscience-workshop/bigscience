# Work Environment Info


## Users and Accounts

**Accounts:**

- `six` - the BigScience allocation - our main allocation
- `ajs` - original dynamic access allocations - use it if you can as we still have resources there - but it will give low priority on scheduling - hence use primarily for jobs that can be bumped down in the queue for a few days.

To switch to `six` as the main project:
```
idrproj -d six
```
and logout/login.

Check which projects one belongs to: `idrproj`

**Users:**

Use `idracct six` to see which username belongs to which real person.


## First time setup

Add this to your `~/.bashrc` and run `bash` for the changes to take effect.

```
# ~/.bashrc: executed by bash(1) for non-login shells.
[[ $- != *i* ]] && return

# Log in with correct group - relevant to all users as we have multiple groups we belong to
if [[ $(id -gn) != "six" ]]
then
    newgrp six
    exit
fi

# start production environment:
# this loads modules, conda and sets all the relevant env vars
alias start-prod="source $six_ALL_CCFRWORK/start-prod"

# our production conda env is here:
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

# SLURM / Account specific settings

# share dirs/files with the group
umask 0007

# specific caches
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export DATASETS_CUSTOM=$six_ALL_CCFRWORK/datasets-custom

# shortcut
export PROD=$six_ALL_CCFRWORK

# handy shortcuts
alias myjobs="squeue -u `whoami`"
```

note: wrt `newgrp six` - if you want to use it elsewhere and not `~/.bashrc` you may use this `newgrp - six` syntax instead, but don't use it in `~/.bashrc` or it will break many things.

Also since most of our work is at `$six_ALL_CCFRWORK` you may want to add symlinks:
```
ln -s $six_ALL_CCFRWORK ~/prod
ln -s $six_ALL_CCFRSCRATCH ~/prod-scratch
ln -s $six_ALL_CCFRSTORE ~/prod-store
ln -s /gpfsssd/worksf/projects/rech/six/commun ~/prod-worksf
```
and then you can quickly `cd` there w/o needing to type too much, and with the shortcut `$PROD` env var you now you can do one of 2 ways:
```
cd ~/prod
cd $PROD
```

Some users prefer to use the env vars, so let's try to not expect the symlinks to be there for everybody.

If you intend to use `gsutil`, add the following lines:

```
if [ -f '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/path.bash.inc' ]; then . '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/path.bash.inc'; fi
if [ -f '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/completion.bash.inc' ]; then . '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/completion.bash.inc'; fi
```

Without them, `gsutil` on Jean Zay fails with a hard-to-debug `TypeError: argument should be integer or bytes-like object, not 'str'` error.

## Production environment

In order to use the production environment, run:

```
start-prod
```
which will:
- setup env vars
- configure nice git-prompt with lots of useful info built in
- load the right `module`s
- activate our custom production conda environment which has everything in it

so basically use it when running production scripts.

The alias should have been set in `~/.bashrc` as instructed above.

Note: the fancy [bash-git-prompt](https://github.com/magicmonty/bash-git-prompt) tells you which conda env you are in, and then which branch your are in and a ton of useful git enfo, and it was extended to tell you whether you're in the login instance (prefix `0-1`) or whether you're on a GPU instance where it then shows something like `4-40` - the 2 numbers stand for `${SLURM_NNODES}-${SLURM_CPUS_PER_TASK}` - so you know what `srun` configuration you're logged into (or the login shell where you get no nodes, with 0 gpus and 1 cpu hence `0-1`).

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



## Creating production conda env

this should be done on a login instance, since we need the network

```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n hf-prod python=3.8
conda activate hf-prod
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia
# pip doesn't seem to produce the right package - missing libnvrtc
# pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install deepspeed

cd $six_ALL_CCFRWORK/code/transformers
pip install -e .[dev]

cd $six_ALL_CCFRWORK/code/megatron-lm
pip install -r requirements.txt

cd $six_ALL_CCFRWORK/code/apex
./build.sh

cd $six_ALL_CCFRWORK/code/deepspeed
./build.sh

```

while we are going to override some of these with our custom installs, we first install these normally to get all the dependencies right.



## Personal environment

You can use these dirs, which are your private spaces:

- `$WORK`
- `$SCRATCH`
- `$STORE`

So you probably want to mimic the production env,

We also agreed to use

```
ln -s $WORK ~/user
ln -s $SCRATCH ~/user-scratch
ln -s $STORE ~/user-store
```
and then you can quickly `cd` there w/o needing to type too much:
```
cd ~/user
```

Since we are going to use `~/user/...` in scripts, it now should be possible to re-use our scripts w/o modifying them. To change the script to use the production setup, it'll be just `s/user/prod/`.



## Custom private env

If wanting to work with variations of packages, create your own conda env, e.g. env `stas`:

```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n stas python=3.8
conda activate stas
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install deepspeed

cd ~/user/code/transformers
pip install -e .[dev]

cd ~/user/code/Megatron-Deepspeed
pip install -r requirements.txt

cd ~/user/code/apex
./build.sh

cd ~/user/code/deepspeed-big-science
./build.sh
```



## Login node

If the login node is heavily used by someone, one can switch to another node

`host jean-zay.idris.fr` will tell you which login nodes are currently in the alias
￼￼
if the DNS round robin doesn't send you to another login node, you can target a specific login node (`jean-zayN.idris.fr` , with N from 1 to 5, though some might not be available so using the alias is always better)


## Dealing with running out of disc space

Find out where disc space is used up:
```
du -ahd1 $six_ALL_CCFRWORK  | sort -rh
du -ahd1 $six_ALL_CCFRSTORE | sort -rh
```

Find out where inodes are used up:
```
du -ahd1 --inodes $six_ALL_CCFRWORK | sort -rh
du -ahd1 --inodes $six_ALL_CCFRSTORE | sort -rh
```

Some busy git clones can be pruned of unused files with: `git gc`, e.g. to prune a dir with multiple-clones as sub-dirs:

```
cd $six_ALL_CCFRWORK/code
du -hs .
du -hs --inodes .
find . -mindepth 1 -maxdepth 1 -type d -exec bash -c "cd '{}' && git gc" \;
du -hs .
du -hs --inodes .
```

## Finding things

Our WORK is indexed by mlocate, after adding this alias:
```
alias locate="/usr/bin/locate -d $ALL_CCFRWORK/lib/mlocate/work.db:$ALL_CCFRWORK/lib/mlocate/worksf.db"
```
You can now do:
```
locate -i megatron
```
(remove `-i` if you want case-sensitive search)

the index is being updated by `$six_ALL_CCFRWORK/bin/mlocate-update` in a crontab job in `$six_ALL_CCFRWORK/cron/cron.daily/mlocate-update.slurm`.

For more details on the emulated crontab job see: [crontab](../crontab/README.md).


## Syncing the perms

We use `umask 0007` in `~/.bashrc` to get the shared dirs have `g+rwx` perms, so that we can all operate on those, but it doesn't always help. When a tarball is extracted it will often retain the original perms on the files, so if those didn't have `w` for the group it'll remain as such. Therefore occasionally and especially after installing a new dataset please run:

We also need `g+s` on dirs, so that new dirs and files created in the sub-dir get created with the same group as the parent dir (e.g. important when `scp`-ing from outside, but also in many other cases).

Then note that `chgrp` removes the sgid bit,  as it has to be restored immediately, so do not run it alone!

For some reason group perms go wrong at times. We need all files to be `g+wrxs` (dirs), `g+rw` (files), `six` (group name), so here is how to fix things back to normal:

```
find $six_ALL_CCFRWORK    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chgrp six {} \; , -execdir chmod g+rwxs {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRWORK    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type f -execdir chgrp six {} \; , -execdir chmod g+rw {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find /gpfsssd/worksf/projects/rech/six/commun    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chgrp six {} \; , -execdir chmod g+rwxs {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find /gpfsssd/worksf/projects/rech/six/commun    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type f -execdir chgrp six {} \; , -execdir chmod g+rw {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSCRATCH -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chgrp six {} \; , -execdir chmod g+rwxs {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSCRATCH -user `whoami` -type d ! \( -readable -executable \) -prune -o -type f -execdir chgrp six {} \; , -execdir chmod g+rw {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSTORE   -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chgrp six {} \; , -execdir chmod g+rwxs {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSTORE   -user `whoami` -type d ! \( -readable -executable \) -prune -o -type f -execdir chgrp six {} \; , -execdir chmod g+rw {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
```

If somehow we lost the sgid bit on some dirs, to restore just those:
```
find $six_ALL_CCFRWORK    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chmod g+s {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find /gpfsssd/worksf/projects/rech/six/commun    -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chmod g+s {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSCRATCH -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chmod g+s {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
find $six_ALL_CCFRSTORE   -user `whoami` -type d ! \( -readable -executable \) -prune -o -type d -execdir chmod g+s {} \; 2>&1 | egrep -v "(Operation not permitted|cannot operate on dangling symlink)"
```
albeit, the set of commands above should have already done the right thing, as they include `g+rwxs`.



## Activate production script

This can be safely added at the beginning of slurm scripts:

```
source $six_ALL_CCFRWORK/start-prod
```

And if you made the symlink from your `$HOME`, interactively it's easier to remember to type:

```
source $six_ALL_CCFRWORK/start-prod
```



## Building things from source


The building should happen on a beefy instance - or things just get killed

Normally use the free `-p compil` partition:

```
srun --pty -p compil --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

but if it has to be really fast, use a dedicated instance with pre-allocated cpu cores:
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```



`/tmp` is tiny on gpu instances, at least apex needs a big `/tmp` folder:


Quick instructions (detailed listing follow):

```
export TMPDIR=$six_ALL_CCFRWORK/tmp
mkdir -p $TMPDIR

cd $six_ALL_CCFRWORK/code/deepspeed-big-science
./build.sh

cd $six_ALL_CCFRWORK/code/apex
./build.sh
```


### deepspeed


We are using a special branch maintained for us:
```
cd $six_ALL_CCFRWORK/code/
git clone https://github.com/microsoft/deepspeed deepspeed-big-science
cd deepspeed-big-science
git checkout big-science
```

To pre-build deepspeed (as compared to have it built via JIT at runtime):

```
export TMPDIR=$six_ALL_CCFRWORK/tmp
mkdir -p $TMPDIR
cd $six_ALL_CCFRWORK/code/deepspeed-big-science
./build.sh
```

what's in the build:
```
$ cat build.sh
#!/bin/bash

rm -rf build

time TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
```

### apex

To build apex (needed by megatron-lm):

build:
```
cd $six_ALL_CCFRWORK/code/apex
./build.sh
```

what's in the build:
```
$ cat build.sh
#!/bin/bash

pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```

Note that since we are using pt/cuda-11.1 and JZ has cuda-11.2, apex won't build unless we skip the version check (which is totally not necessary - things work just fine), so should you reset the clone and removed the local patch, you can restore it with this diff: https://github.com/NVIDIA/apex/issues/988#issuecomment-726343453



## Aliases

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

## Troubleshooting

### pip install

If it's trying to install into your local `~/.local` folder it's because `pip` is in that `$PATH` before
`$six_ALL_CCFRWORK/conda/hf-prod/bin/` - push the last one to be first - or best don't install any python things locally - use conda for that. Check with `which pip` - it should be under `$six_ALL_CCFRWORK/conda/hf-prod/bin/pip`.



## Older info

Probably of no use any longer, but still here in case it is needed (might move to another file).

## Local resources

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
