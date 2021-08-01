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

One a repo has been cloned and is used as a destination for checkpoints and log files, the following process will automatically push any new files into it.

1. Once you need to auth the application - use the `bigscience-bot` user, so that it doesn't interfere with your work.

Once you have the login and password, and email, run:

```
tools/hub-auth.py
```

which creates `tools/.hub_info.json` with the username, email and then auth token locally. Anybody can do it. It will be the same token, specific to the `bigscience-bot` user.

2. Now for each tracking repo, run the script with the desired pattern, e.g.:


```
module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon

tools/hub-sync.py --repo-path $TENSORBOARD_PATH --patterns '*tfevents*'
tools/hub-sync.py --repo-path $CODECARBON_PATH  --patterns '*csv'
tools/hub-sync.py --repo-path $CHECKPOINT_PATH  --patterns '*pt'
```

Of course this needs to be automated, so we will create slurm jobs to perform all these. These must be run on the `prepost` partition, since it has open Internet.

```
$ cat tr1-13B-hub-sync-tensorboard.slurm
#!/bin/bash
#SBATCH --job-name=tr1-13B-hub-sync-tensorboard  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=40           # number of cores per task
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=prepost

module load git-lfs

DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard

tools/hub-sync.py --repo-path $TENSORBOARD_PATH --patterns '*tfevents*' -d

```

XXX: create a slurm script for codecarbon when it starts operating

XXX: create a slurm script for checkpoints once we figure out how to share those

XXX: concern: if this is run from `cron.hourly` what if the first `git push` is still uploading when the next round is pushed?
