# Production environment Info

Users:

- ura81os - Stas
- uhk85as - Teven
- uul91hs - Victor
- ugo23cf - François

To switch to `six` as the main project:
```
idrproj -d six
```
and logout/login.

## Login node

If the login node is heavily used by someone, one can switch to another node

`host jean-zay.idris.fr` will tell you which login nodes are currently in the alias
￼￼
if the DNS round robin doesn't send you to another login node, you can target a specific login node (`jean-zayN.idris.fr` , with N from 1 to 5, though some might not be available so using the alias is always better)

## Syncing the perms

We use `umask 0007` in `~/.bashrc` to get the shared dirs have `g+rwx` perms, so that we can all operate on those, but it doesn't always help. When a tarball is extracted it will often retain the original perms on the files, so if those didn't have `w` for the group it'll remain as such. Therefore occasionally and especially after installing a new dataset please run:

We also need `g+s` to automatically create files with the parent's perm (e.g. when `scp`-ing from outside).

For some reason group perms go wrong at times. We need all files to be `g+wrxs` (dirs), `g+rw` (files), `six` (group name), so here is how to fix things back to normal:

```
find $six_ALL_CCFRWORK -user `whoami` -type d -execdir chmod g+rwxs {} \;
find $six_ALL_CCFRWORK -user `whoami` -type f -execdir chmod g+rw {} \;
chgrp -R six $six_ALL_CCFRWORK
find $six_ALL_CCFRSCRATCH -user `whoami` -type d -execdir chmod g+rwxs {} \;
find $six_ALL_CCFRSCRATCH -user `whoami` -type f -execdir chmod g+rw {} \;
chgrp -R six $six_ALL_CCFRSCRATCH
```

## Activate production script

This can be safely added at the beginning of slurm scripts:

```
source $six_ALL_CCFRWORK/start-prod
```

And if you made the symlink from your `$HOME`, interactively it's easier to remember to type:

```
source ~/prod/start-prod
```


## Creating production conda env

this should be done on a login instance, since we need the network

```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n hf-prod python=3.8
conda activate hf-prod
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install deepspeed

cd ~/prod/code/transformers
pip install -e .[dev]

cd ~/prod/code/megatron-lm
pip install -r requirements.txt

cd ~/prod/code/apex
./build.sh

cd ~/prod/code/deepspeed
./build.sh

```

while we are going to override some of these with our custom installs, we first install these normally to get all the dependencies right.


## Custom private env

If wanting to work with variations of packages, create your own conda env, e.g. env `stas`:

```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n stas python=3.8
conda activate stas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install deepspeed

cd ~/stas/code/transformers
pip install -e .[dev]

cd ~/stas/code/megatron-lm
pip install -r requirements.txt

pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio -f https://download.pytorch.org/whl/torch_stable.html

cd ~/stas/code/apex
./build.sh

cd ~/stas/code/deepspeed-shaden
./build.sh
```


## Building things from source

The building should happen on a beefy instance - or things just get killed

e.g.
```
srun --pty --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

`/tmp` is tiny on gpu instances, at least apex needs a big `/tmp` folder:


Quick instructions (detailed listing follow):

```
mkdir -p ~/prod/tmp
export TMPDIR=~/prod/tmp

cd ~/prod/code/deepspeed
./build.sh

cd ~/prod/code/apex
./build.sh

```

### deepspeed

To pre-build deepspeed (as compared to have it built via JIT at runtime):

```
export TMPDIR=~/prod/tmp
cd ~/prod/code/deepspeed
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
cd ~/prod/code/apex
./build.sh
```

what's in the build:
```
$ cat build.sh
#!/bin/bash

pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```
