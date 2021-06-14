# SLURM HOWTO

**Partitions**:

- `-p gpu_p1`: 4x v100-32g
- `-p gpu_p2`: 8x v100-32g
- `-p gpu_p3`: 4x v100-16g

Combos:
- `-p gpu_p13` - all 4x nodes combined - i.e. when either 16gb or 32gb will do

**Constraints**:

- `-C v100-16g` # to select nodes having GPUs with 16 GB of memory (same as `-p gpu_p3`)
- `-C v100-32g` # to select nodes having GPUs with 32 GB of memory (same as `-p gpu_p1`)

If your job can run on both types of GPUs, we recommend not to specify any constraints as it will reduce the waiting time of your jobs before resources are available for the execution.

Special reservation constraint - if a special reservation is made, e.g., `huggingface1`, activate it with: `--reservation=huggingface1`.

**Long running jobs**:

Normal jobs can do max `--time=20:00:00`, for longer jobs up to 100h use `--qos=qos_gpu-t4`.

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

## Make allocations for a desired time

Not guaranteed, but can try to start an allocation say from the night before for the next morning:
```
salloc --begin HH:MM MM/DD/YY
```

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
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p gpu_p13 -o '%A'"
```

more informative all-in-one myjobs that includes the projected start time for pending jobs

```
alias myjobs='squeue -u `whoami` -o "%.10i %.9P %.20j %.8T %.10M %.6D %.20S %R"'
```

## show my jobs
```
squeue -u `whoami`
```


by job id:
```
squeue -j JOBID
```

# zombies

If there are any zombies left behind across nodes, send one command to kill them all.

```
srun pkill python
```


# queue

## cancel job

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


# show the state of nodes
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
sinfo -p gpu_p13 -o "%A"
```

gives:
```
NODES(A/I)
597/0
``

so you can see if any nodes are available on the 4-gpu partition (`gpu_p13`)

To check each specific partition:

```
sinfo -p gpu_p1 -o "%A"
sinfo -p gpu_p2 -o "%A"
sinfo -p gpu_p3 -o "%A"
sinfo -p gpu_p13 -o "%A"
```


# TODO:

absorb more goodies from here: https://ubccr.freshdesk.com/support/solutions/articles/5000686861-how-do-i-check-the-status-of-my-job-s-
