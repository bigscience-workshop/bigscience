# Crontab Jobs

JZ has no crontab so we have to emulate it.

Put your slurm scripts into either:
```
$six_ALL_CCFRWORK/cron/cron.hourly
$six_ALL_CCFRWORK/cron/cron.daily
```
depending on whether you want to run those approximately once an hour or once a day.

Any scripts found in these dirs that have `.slurm` extension, will be run as `sbatch scriptname`.

## The scheduler

The scheduler isn't run automatically, we have to launch it and make sure it gets restarted manually if SLURM
is restarted (not sure if jobs get preserved or not):

* [cron-hourly.slurm](./cron-hourly.slurm)
* [cron-daily.slurm](./cron-daily.slurm)

If these 2 aren't running when you run:

```
squeue --user=$(getent group six | cut -d: -f4) | grep cron
```
the re-launch the missing one(s) with:
```
cd $six_ALL_CCFRWORK/cron/scheduler
sbatch cron-hourly.slurm
sbatch cron-daily.slurm
```

If these scripts aren't there, copy them from the folder in the repo where this README.md is located.

XXX: need some kind of a watchdog to ensure the 2 cron scheduler jobs don't disappear.

quick alias to test:
```
alias cron-check="squeue --user=$(getent group six | cut -d: -f4) | grep cron"
```

## Example daily entry

Here is an example of a job that gets to run daily.
```
$ cat $six_ALL_CCFRWORK/cron/cron.daily/mlocate-update.slurm
#!/bin/bash
#SBATCH --job-name=mlocate-update    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=1:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil
#SBATCH --account=six@cpu

set -e
date
echo "updating mlocate db"
# "--require-visibility 0" is required when launching this command as a regular user
/usr/bin/updatedb -o $ALL_CCFRWORK/lib/mlocate/work.db   -U $ALL_CCFRWORK --require-visibility 0
/usr/bin/updatedb -o $ALL_CCFRWORK/lib/mlocate/worksf.db -U /gpfsssd/worksf/projects/rech/six/commun --require-visibility 0
```

This builds an index of the files under WORK which you can then quickly query with:
```
/usr/bin/locate -d /gpfswork/rech/six/commun/lib/mlocate/mlocate.db pattern
```

The slurm script `mlocate-update.slurm` has been placed inside `$six_ALL_CCFRWORK/cron/cron.daily`. To stop running it, just move it elsewhere.

Another approach to adding/removing is to keep the slurm scripts elsewhere and symlink to them from either
`$six_ALL_CCFRWORK/cron/cron.daily` or `$six_ALL_CCFRWORK/cron/cron.hourly` according to the need.


## Permissions

The scheduler runs with Unix permissions of the person who launched the SLRUM cron scheduler job and so all other SLURM scripts launched by that cron job.


## TODO

XXX: need to have a facility to report failures. Which is tricky because the job has to run on a SLURM partition that has Internet and that's just `--partition=prepost` and `--partition=compil`
