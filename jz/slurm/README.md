# SLURM How To


## Partitions

- `-p gpu_p1`: 4x v100-32gb
- `-p gpu_p2`: 8x v100-32gb
- `-p gpu_p3`: 4x v100-16gb
- `-p gpu_p4`: 8x A100-40gb / 48cpu cores (only 3 nodes)

Non-gpu nodes time on which isn't deducted from allocation:

- `-p prepost`: up to 20h - for pre/post-processing
- `-p visu`:    up to 4h  - for visualization
- `-p archive`: up to 20h - for archiving
- `-p compil`:  up to 20h - for compilation

Combos:

- `-p gpu_p13` - all 4x nodes combined - i.e. when either 16gb or 32gb will do

**Constraints**:

- `-C v100-16g` # to select nodes having v100 GPUs with 16 GB of memory (same as `-p gpu_p3`)
- `-C v100-32g` # to select nodes having v100 GPUs with 32 GB of memory (same as `-p gpu_p1`)

If your job can run on both types of GPUs, we recommend not to specify any constraints as it will reduce the waiting time of your jobs before resources are available for the execution.

Special reservation constraint - if a special reservation is made, e.g., `huggingface1`, activate it with: `--reservation=huggingface1`.

**Long running jobs**:

Normal jobs can do max `--time=20:00:00`, for longer jobs up to 100h use `--qos=qos_gpu-t4`. Limit 16 GPUs.

Note: the given node could be already heavily used by any other random users.

Full details http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html


## Priorities

- `--qos=qos_gpu-t3` 20h / 512gpus (default priority)
- `--qos=qos_gpu-t4` 100h / 16gpus - long runnning slow jobs - e.g. preprocessing
- `--qos=qos_gpu-dev`  2h / 32gpus - this is for getting allocation much faster - for dev work!


Full info: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html

## Wait time for resource granting

```
squeue -u `whoami` --start
```
will show when any pending jobs are scheduled to start.

They may start sooner if others cancel their reservations before the end of the reservation.


## Preallocated node without time 60min limit

This is very useful for running repetitive interactive experiments - so one doesn't need to wait for an allocation to progress. so the strategy is to allocate the resources once for an extended period of time and then running interactive `srun` jobs using this allocation.

set `--time` to the desired window (e.g. 6h):
```
salloc --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
now use this reserved node to run a job multiple times, by passing the job id of `salloc`:
```
srun --jobid $SLURM_JOBID --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```
if run from inside `bash` started via `salloc`. But it can be started from another shell, but then explicitly set `--jobid`.

if this `srun` job timed out or manually exited, you can re-start it again in this same reserved node.

`srun` can, of course, call the real training command directly and not just `bash`.

When finished, to release the resources, either exit the shell started in `salloc` or `scancel jobid`.

This reserved node will be counted towards hours usage the whole time it's allocated, so release as soon as done with it.

## Make allocations at a scheduled time

To postpone making the allocation for a given time, use:
```
salloc --begin HH:MM MM/DD/YY
```

It will simply put the job into the queue at the requested time, as if you were to execute this command at this time. If resources are available at that time, the allocation will be given right away. Otherwise it'll be queued up.

## Re-use allocation

e.g. when wanting to run various jobs on identical node allocation.

In one shell:
```
salloc --constraint=v100-32g --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=3:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
echo $SLURM_JOBID
```

In another shell:
```
export SLURM_JOBID=<JOB ID FROM ABOVE>
srun --jobid=$SLURM_JOBID ...
```

## Signal the running jobs to finish

Since each SLURM run has a limited time span, it can be configured to send a signal of choice to the program a desired amount of time before the end of the allocated time.
```
--signal=[[R][B]:]<sig_num>[@<sig_time>]
```
TODO: need to experiment with this to help training finish gracefully and not start a new cycle after saving the last checkpoint.


## Detailed job info

While most useful information is preset in various `SLURM_*` env vars, sometimes the info is missing. In such cases use:
```
scontrol show -d job $SLURM_JOB_ID
```
and then parse out what's needed.


## aliases

Handy aliases

```
alias myjobs="squeue -u `whoami`"
alias groupjobs="squeue --user=$(getent group six | cut -d: -f4)"
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p gpu_p13 -o '%A'"
```

more informative all-in-one myjobs that includes the projected start time for pending jobs

```
alias myjobs='squeue -u `whoami` -o "%.10i %.9P %.20j %.8T %.10M %.6D %.20S %R"'
alias groupjobs='squeue -u $(getent group six | cut -d: -f4) -o "%.10i %.9P %.20j %.8T %.10M %.6D %.20S %R"'
```

## show my jobs

```
squeue -u `whoami`
```


by job id:
```
squeue -j JOBID
```

group's jobs (probably won't include the non-account partitions), including all users is probably better

```
squeue --account=six@gpu,six@cpu
```

group's jobs including all `six`'s users:

```
squeue --user=$(getent group six | cut -d: -f4)

```



## zombies

If there are any zombies left behind across nodes, send one command to kill them all.

```
srun pkill python
```


## queue


### cancel job

To cancel a job:
```
scancel [jobid]
```

To cancel all of your jobs:
```
scancel -u <userid>
```

To cancel all of your jobs on a specific partition:
```
scancel -u <userid> -p <partition>
```

### tips

- if you see that `salloc`'ed interactive job is scheduled to run much later than you need, try to cancel the job and ask for shorter period - often there might be a closer window for a shorter time allocation.


## show the state of nodes
```
sinfo -p PARTITION
```

Very useful command is:
```
sinfo -s
```

and look for the main stat, e.g.:

```
NODES(A/I/O/T) "allocated/idle/other/total".
597/0/15/612
```
So here 597 out of 612 nodes are allocated. 0 idle and 15 are not available for whatever other reasons.

```
sinfo -p gpu_p1 -o "%A"
```

gives:
```
NODES(A/I)
236/24
```

so you can see if any nodes are available on the 4x v100-32g partition (`gpu_p1`)

To check each specific partition:

```
sinfo -p gpu_p1 -o "%A"
sinfo -p gpu_p2 -o "%A"
sinfo -p gpu_p3 -o "%A"
sinfo -p gpu_p13 -o "%A"
```

See the table at the top of this document for which partition is which.


## Job arrays


To run a sequence of jobs, so that the next slurm job is scheduled as soon as the currently running one is over in 20h we use a job array.

Let's start with just 10 such jobs:

```
sbatch --array=1-10%1 array-test.slurm
```

`%1` limits the number of simultaneously running tasks from this job array to 1. Without it it will try to run all the jobs at once, which we may want sometimes (in which case remove %1), but when training we need one job at a time.


Here is toy slurm script, which can be used to see how it works:

```
#!/bin/bash
#SBATCH --job-name=array-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --time 00:02:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@cpu
#SBATCH -p prepost

echo $SLURM_JOB_ID
echo "I am job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
date
sleep 10
date
```

Note `$SLURM_ARRAY_JOB_ID` is the same as `$SLURM_JOB_ID`, and `$SLURM_ARRAY_TASK_ID` is the index of the job.

To see the jobs running:
```
$ squeue -u `whoami` -o "%.10i %.9P %.26j %.8T %.10M %.6D %.20S %R"
     JOBID PARTITION                       NAME    STATE       TIME  NODES           START_TIME NODELIST(REASON)
591970_[2-   prepost                 array-test  PENDING       0:00      1  2021-07-28T20:01:06 (JobArrayTaskLimit)
```
now job 2 is running.

To cancel the whole array, cancel the job id as normal (the number before `_`)
```
scancel 591970
```

If it's important to have the log-file contain the array id, add `%A_%a`

```
#SBATCH --output=%x-%j.%A_%a.log
```

More details https://slurm.schedmd.com/job_array.html


## TODO

absorb more goodies from here: https://ubccr.freshdesk.com/support/solutions/articles/5000686861-how-do-i-check-the-status-of-my-job-s-
