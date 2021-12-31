# Fixing things

## Fix multiple checkpoints per branch on hub

Update all `config.json` files:

```
cd /gpfsssd/scratch/rech/six/commun/experiments/fix-config/
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3e-1B3-c4-checkpoints
cd tr3e-1B3-c4-checkpoints
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
set +H
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s/gelu(?!_)/gelu_fast/" $1/config.json; git commit -m "gelu_fast is the correct activation_function" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
```

```
cd /gpfsssd/scratch/rech/six/commun/experiments/fix-config/
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3d-1B3-oscar-checkpoints
cd tr3d-1B3-oscar-checkpoints
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
set +H
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s/gelu(?!_)/gelu_fast/" $1/config.json; git commit -m "gelu_fast is the correct activation_function" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
```


```
cd /gpfsssd/scratch/rech/six/commun/experiments/fix-config/
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3m-1B3-pile-checkpoints
cd tr3m-1B3-pile-checkpoints
set +H
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s/gelu(?!_)/gelu_fast/" $1/config.json; git commit -m "gelu_fast is the correct activation_function" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
```

## Fix corrupted git


Quite a few times now we had an odd git corruption for the logging repos:


```
OSError: error: invalid object 100644 e69f03783ce2b0af675405f22b49ebeb56d907e5 for '.gitattributes'
error: invalid object 100644 e69f03783ce2b0af675405f22b49ebeb56d907e5 for '.gitattributes'
error: Error building trees
```

Of course, the error can be different.

Perhaps slurm somehow occasionally kills the syncing process while git is doing something internally and thus corrupts it. It's hard to tell.

You can fix these easily but making a new clone and swapping in just the `.git` dir. That fixes it up.

Here is the full process using `tr8b-104B-logs` as an example:

```
cd checkpoints/tr8b-104B/
git clone https://huggingface.co/bigscience/tr8b-104B-logs/ tr8b-104B-logs-new
mkdir trash
mv tr8b-104B-logs/.git trash
cp -r tr8b-104B-logs-new/.git tr8b-104B-logs/.git
# check that it is no longer broken
cd tr8b-104B-logs
git gc
```
