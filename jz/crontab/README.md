# Crontab Jobs

JZ has no crontab so we have to emulate it

Put your slurm scripts into either:
```
$six_ALL_CCFRWORK/cron/cron.hourly
$six_ALL_CCFRWORK/cron/cron.daily
```

Any scripts found in these dirs will be run as `sbatch scriptname`.

## The scheduler

The scheduler isn't run automatically, we have to launch it and make sure it gets restarted manually if SLURM
is restarted (not sure if jobs get preserved or not):

* [cron-hourly.slurm](./cron-hourly.slurm)
* [cron-daily.slurm](./cron-daily.slurm)

If these 2 aren't running when you run:

```
squeue --user=$(getent group six | cut -d: -f4) | grep cron
```
re-launch the missing one(s) with:
```
cd $six_ALL_CCFRWORK/cron/scheduler
sbatch cron-hourly.slurm
sbatch cron-daily.slurm
```

If these scripts aren't there copy them from the folder in the repo where this README.md is located.

## Example daily entry

Here is an example of a job that gets to run daily:

```
$ cat $six_ALL_CCFRWORK/cron/cron.daily/mlocate-update.slurm
#!/bin/bash
#SBATCH --job-name=mlocal-update     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=archive

set -e

# "--require-visibility 0" is required when launching this command as a regular user
updatedb -o $ALL_CCFRWORK/lib/mlocate/mlocate.db -U $ALL_CCFRWORK --require-visibility 0
```

The slurm script `mlocate-update.slurm` has been placed inside `$six_ALL_CCFRWORK/cron/cron.daily`. To stop running it, just move it elsewhere.

Another approach to adding/removing is to keep the slurm scripts elsewhere and symlink to them from either
`$six_ALL_CCFRWORK/cron/cron.daily` or `$six_ALL_CCFRWORK/cron/cron.hourly` according to the need.


## Permissions

The scheduler runs with Unix permissions of the person who launched the SLRUM cron scheduler job and so all other SLURM scripts launched by that cron job.
