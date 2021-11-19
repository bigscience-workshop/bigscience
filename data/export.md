# Export Data outside of JZ



## Upload to the Hub

First go to https://huggingface.co/bigscience/ and via your username (right upper corner) create "new Model"
while choosing the `bigscience` as org.

Say you created https://huggingface.co/bigscience/misc-test-data/

Now on JZ side

```
module load git-lfs
git lfs install
git clone https://huggingface.co/bigscience/misc-test-data/
cd misc-test-data/
```

Now you can add files which are less than 10M, commit and push.

Make sure that if the file is larger than 10M its extension is tracked by git LFS, e.g. if you're adding `foo.tar.gz` make sure `*gz` is in `.gitattributes` like so:
```
*.gz filter=lfs diff=lfs merge=lfs -text
```
if it isn't add it:
```
git lfs track "*.gz"
git commit -m "compressed files" .gitattributes
git push
```
only now add your large file `foo.tar.gz`
```
cp /some/place/foo.tar.gz .
git add foo.tar.gz
git commit -m "foo.tar.gz" foo.tar.gz
git push
```

Now you can tell the contributor on the other side where they can download the files you have just uploaded by sending them to the corresponding hub repo.


## Automated upload to the hub

Once a repo has been cloned and is used as a destination for checkpoints and log files, the following process will automatically push any new files into it.

1. Auth.

Typically you can skip directly to the stage 2 as stage 1 should already work.

We use a shared auth file located at `$six_ALL_CCFRWORK/auth/.hub_info.json` for all processes syncing to the hub. We use a special account of the `bigscience-bot` user so that the process doesn't depend on personal accounts.

If for some reason you need to override this shared file with a different auth data for a specific project, simply run:

```
tools/hub-auth.py
```

And enter login and password, and email, at prompt. This will create `tools/.hub_info.json` with the username, email and then auth token locally.

2. Now for each tracking repo, run the script with the desired pattern, e.g.:


```
module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon
BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/bigscience

$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $TENSORBOARD_PATH --patterns '*tfevents*'
$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $CODECARBON_PATH  --patterns '*csv'
$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $CHECKPOINT_PATH  --patterns '*pt'
```

Of course this needs to be automated, so we will create slurm jobs to perform all these. These must be run on the `prepost` partition, since it has a limited Internet access.

```
$ cat tr1-13B-hub-sync-tensorboard.slurm
#!/bin/bash
#SBATCH --job-name=tr1-13B-hub-sync-tensorboard  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=prepost

echo "START TIME: $(date)"

module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
BIG_SCIENCE_REPO_PATH=$six_ALL_CCFRWORK/code/bigscience

$BIG_SCIENCE_REPO_PATH/tools/hub-sync.py --repo-path $TENSORBOARD_PATH --patterns '*tfevents*' -d

echo "END TIME: $(date)"
```


XXX: create a slurm script for codecarbon when it starts operating

XXX: create a slurm script for checkpoints once we figure out how to share those

XXX: concern: if this is run from `cron.hourly` what if the first `git push` is still uploading when the next round is pushed?

## Large Text files

Normally `*txt` files aren't LFS tracked, so if your log file gets synced to he hub an it has grown over 10M you will get the next push fail with:

```
* Pushing 1 files
remote: -------------------------------------------------------------------------
remote: Your push was rejected because it contains files larger than 10M.
remote: Please use https://git-lfs.github.com/ to store larger files.
remote: -------------------------------------------------------------------------
remote: Offending files:
remote:  - logs/main_log.txt (ref: refs/heads/main)
To https://huggingface.co/bigscience/tr3n-1B3-pile-fancy-logs
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://bigscience-bot:api_gyGezHBUDEGfyBxlAYTHCxQIbkjMUUEpaK@huggingface.co/bigscience/tr3n-1B3-pile-fancy-logs'
```

So you need to do the following from the cloned repo dir in question:

1. Unstage the commits that weren't pushed:

```
git reset --soft origin/HEAD
```

2. Add `*txt` to LFS-tracking

```
git lfs track "**.txt"
```

this will automatically switch to LFS on the next commit

3. commit/push normally

```
git commit -m "update file" logs/main_log.txt
git push
```

In order to avoid this issue in the first place, it's best to set it up to:

```
git lfs track "**.txt"
```
when you first setup the repo clone.
