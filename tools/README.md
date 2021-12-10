## Instrumenting your run
We assume you're following the structure of the [arch-and-scaling template](https://github.com/bigscience-workshop/bigscience/blob/master/train/arch-and-scaling-template.slurm)
Go to https://huggingface.co/ and create two models (currently, under your icon on the top right/new model)
- <YOUR_MODEL_NAME>-checkpoints
- <YOUR_MODEL_NAME>-logs
in your output path (DATA_OUTPUT_PATH in the arch-and-scaling template), `git clone` the logs repo and rename the folder to `logs` (mv `<YOUR_MODEL_NAME>-logs` `logs`)

## How to synch your logs with the hub
`python tools/hub-sync.py --repo-path <DATA_OUTPUT_PATH>/logs/tensorboard/ --patterns "*tfevent*"`

## How to synch your checkpoints with the hub
Latest version of what was used in [training 1](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base).

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
- If you have added tokenizer files:

```
mv ../hf_fixed/global_step* .
time find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout -b {} \; -exec mv {}/config.json . \; -exec mv {}/pytorch_model.bin . \; -exec git add config.json pytorch_model.bin <TOKENIZER_FILES> \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \; --exec mv config.json {}/ --exec mv pytorch_model.bin {}/;
git checkout main
```
- If you just want to add the checkpoints, without tokenizer files:

```
mv ../hf_fixed/global_step* .
time find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout -b {} \; -exec mv {}/config.json . \; -exec mv {}/pytorch_model.bin . \; -exec git add config.json pytorch_model.bin \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \; --exec mv config.json {}/ --exec mv pytorch_model.bin {}/
git checkout main
```
- If you want to add tokenizer files later:

```
time find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout {} \; -exec git add <TOKENIZER_FILES> \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \;
git checkout main
```
## Fast branch switching in case you messed up and want to fix all your checkpoints
What you want is `export GIT_LFS_SKIP_SMUDGE=1`. 
Here's an example that changes the activation function in the `config.json` files for each branch:
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3e-1B3-c4-checkpoints
cd tr3e-1B3-c4-checkpoints
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
set +H
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s/gelu(?!_)/gelu_fast/" $1/config.json; git commit -m "gelu_fast is the correct activation_function" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
```
And an example that fixes checkpoints in the old format (contained within a `global_step` subfolder, no tokenizer files) to be compatible with `from_pretrained`:
```
export GIT_LFS_SKIP_SMUDGE=1
my_callback () {
  INDEX=${1}
  BRANCH=${2}
  if [[ $BRANCH == origin/global_step* ]];
  then
    git checkout "${BRANCH:7}"
    git mv "${BRANCH:7}"/* .
    cp ../gpt2_tokenizer/tokenizer.json .
    git add tokenizer.json
    git commit -m "fixed checkpoints to be from_pretrained-compatible"
    git push
  fi
}
get_branches () {
  git branch --all --format='%(refname:short)'
}
# mapfile -t -C my_callback -c 1 BRANCHES < <( get_branches ) # if you want the branches that were sent to mapfile in a new array as well
# echo "${BRANCHES[@]}"
mapfile -t -C my_callback -c 1 < <( get_branches )
```
