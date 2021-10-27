## Instrumenting your run

## How to synch your logs with the hub

## How to synch your checkpoints with the hub
Latest version of what was used in [https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base](training 1).

Go to your `checkpoints` folder, which should contain a bunch of `global_stepXXXXXX` folders. Open a long running interactive shell:
```
srun -p compil --cpus-per-task=40 -A six@cpu --time=6:00:00 --pty bash
```
then convert:

```
time find * -maxdepth 0 -type d -name "global_step*" -exec $six_ALL_CCFRWORK/code/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder {} --output_folder ../hf-fixed/{} \;
```
to prepare the target dir:

```
#git -c http.extraHeader="Authorization: Basic " clone https://huggingface.co/bigscience/<YOUR_REPO>/
cd YOUR_REPO
huggingface-cli lfs-enable-largefiles .
git config --unset user.email
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
```
We are going to put each checkpoint into its own branch with the same name.

```
mv ../hf_fixed/global_step* .
time find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout -b {} \; -exec git add {} \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \;
git checkout main
```
## Fast branch switching in case you messed up and want to fix all your checkpoints
What you want is `export GIT_LFS_SKIP_SMUDGE=1`. Here's an example that changes the activation function in the `config.json` files for each branch:
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3e-1B3-c4-checkpoints
cd tr3e-1B3-c4-checkpoints
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
set +H
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s/gelu(?!_)/gelu_fast/" $1/config.json; git commit -m "gelu_fast is the correct activation_function" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
```
