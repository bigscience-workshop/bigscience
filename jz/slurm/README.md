# SLURM How To


## Partitions

All types of nodes have 40 CPU cores per node, unless specified differently.

GPU-nodes: `--account=six@gpu`

- `-p gpu_p1`: 4x v100-32GB
- `-p gpu_p2`: 8x v100-32GB
- `-p gpu_p3`: 4x v100-16GB
- `-p gpu_p4`: 8x A100-40GB / 48 CPU cores (only 3 nodes)
- `-p prepost`: 1x V100-16GB + network

Combos:

- `-p gpu_p13` - all 4x nodes combined - i.e. when either 16GB or 32GB will do

CPU-only nodes: `--account=six@cpu`

- `-p cpu_p1`:  up to 100h: this is the default partition for `--account=six@cpu`
only 20h by default, add `--qos=qos_cpu-t4` to use 100h (only available if no more than 4 nodes are used).

**Important: having `#SBATCH --gres=gpu:0` in a slurm file forces gpu allocations as well, ignoring the account specification. So remove those**

The following CPU-only partitions time on which isn't deducted from allocation:

- `-p prepost`: up to 20h - for pre/post-processing + has internet!
- `-p visu`:    up to 4h  - for visualization
- `-p archive`: up to 20h - for archiving
- `-p compil`:  up to 20h - for compilation + has internet!


**Constraints**:

- `-C v100-16g` # to select nodes having v100 GPUs with 16 GB of memory (same as `-p gpu_p3`)
- `-C v100-32g` # to select nodes having v100 GPUs with 32 GB of memory (same as `-p gpu_p1`)

If your job can run on both types of GPUs, we recommend not to specify any constraints as it will reduce the waiting time of your jobs before resources are available for the execution.

Special reservation constraint - if a special reservation is made, e.g., `huggingface1`, activate it with: `--reservation=huggingface1`.

**Long running jobs**:

Normal GPU jobs can do max `--time=20:00:00`, for longer jobs up to 100h use `--qos=qos_gpu-t4`. Limit 16 GPUs.

Note: the given node could be already heavily used by any other random users.

Normal CPU jobs can do max `--time=100:00:00` (only `-p cpu_p1`, other partitions 20h)

Full details per parition type

- CPU: http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-exec_partition_slurm-eng.html and
http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-exec_alloc-mem-eng.html
- GPU: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html


To see all available partitions and their total/idle status:

```
sinfo
```

## Priorities

- `--qos=qos_gpu-t3` 20h / 512gpus (default priority)
- `--qos=qos_gpu-t4` 100h / 16gpus - long runnning slow jobs - e.g. preprocessing
- `--qos=qos_gpu-dev`  2h / 32gpus - this is for getting allocation much faster - for dev work!


Full info: http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html


**Important**: when running non-primary training jobs please use: `--nice=10000` in the slurm instructions to allow the main job to get highest priority. But only if you're using  `-C v100-32g` (`-p gpu_p1`). For other type of nodes there is no need to.

Detailed explanation: using `--nice=10000` for the test jobs should work fine as long as you use the same QoS as the production jobs (`qos_gpu-t3`, if you use the `qos_gpu-dev` partition then the test jobs will always have higher priority). The nice value is chosen so that it always cancels the age factor, since the fairshare is common to all your jobs it should be enough to ensure that jobs with `--nice=10000` always have a lower priority than your other jobs with the same QoS. Since the age factor is only 3% of the priority, it should hurt the priority too much compared to other users. (edited)


**How the job priority is computed**

Currently on Jean Zay:

1. 69.4% of the priority depends directly on the chosen QoS
2. 27.8% is the "fairshare" (see `idr_compuse` for the value)
3. and only 2.8% is the job age in queue



## Consumption report


Run:
```
idr_compuse
```

This provides a report on how heavily we use our allocations. When they are over-consumed we get a lower priority in the scheduler.


## Wait time for resource granting

```
squeue -u `whoami` --start
```
will show when any pending jobs are scheduled to start.

They may start sooner if others cancel their reservations before the end of the reservation.



## Request allocation via dependency

To schedule a new job when one more of the currently scheduled job ends (regardless of whether it still running or not started yet), use the dependency mechanism, by telling `sbatch` to start the new job once the currently running job succeeds, using:

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

Using `--dependency` may lead to shorter wait times that using `--begin`, since if the time passed to `--begin` allows even for a few minutes of delay since the stopping of the last job, the scheduler may already start some other jobs even if their priority is lower than our job. That's because the scheduler ignores any jobs with `--begin` until the specified time arrives.


## Make allocations at a scheduled time

To postpone making the allocation for a given time, use:
```
salloc --begin HH:MM MM/DD/YY
```

Same for `sbatch`.

It will simply put the job into the queue at the requested time, as if you were to execute this command at this time. If resources are available at that time, the allocation will be given right away. Otherwise it'll be queued up.

Sometimes the relative begin time is useful. And other formats can be used. Examples:

```
--begin now+2hours
--begin=16:00
--begin=now+1hour
--begin=now+60  # seconds by default
--begin=2010-01-20T12:34:00
```

the time-units can be `seconds` (default), `minutes`, `hours`, `days`, or `weeks`:

## Preallocated node without time 60min limit

This is very useful for running repetitive interactive experiments - so one doesn't need to wait for an allocation to progress. so the strategy is to allocate the resources once for an extended period of time and then running interactive `srun` jobs using this allocation.

set `--time` to the desired window (e.g. 6h):
```
salloc --account=six@gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
now use this reserved node to run a job multiple times, by passing the job id of `salloc`:
```
srun --jobid $SLURM_JOBID --pty bash --rcfile $six_ALL_CCFRWORK/start-prod
```
if run from inside `bash` started via `salloc`. But it can be started from another shell, but then explicitly set `--jobid`.

if this `srun` job timed out or manually exited, you can re-start it again in this same reserved node.

`srun` can, of course, call the real training command directly and not just `bash`.

Important: when allocating a single node, the allocated shell is not on the node (it never is). You have to find out the hostname of the node (reports when giving the allocation or via `squeue` and `ssh` to it.

When finished, to release the resources, either exit the shell started in `salloc` or `scancel JOBID`.

This reserved node will be counted towards hours usage the whole time it's allocated, so release as soon as done with it.

To get just the CPUs instances :

```
salloc --account=six@cpu --nodes=1 --ntasks=1 --cpus-per-task=10 --hint=nomultithread --time=6:00:00 bash
```
edit `--cpus-per-task` if more cpu cores are needed.

Actually, if this is just one node, then it's even easier to not use `salloc` but to use `srun` in the first place, which will both allocate and give you the shell to use:
```
srun  --account=six@gpu  --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=60 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

And to use a cpu-only node:
```
srun --account=six@cpu --pty --nodes=1 --ntasks=1 --cpus-per-task=40 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```
The `--rcfile` part is optional if you want to pre-run something.


## Re-use allocation

e.g. when wanting to run various jobs on identical node allocation.

In one shell:
```
salloc --account=six@gpu --constraint=v100-32g --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=3:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
echo $SLURM_JOBID
```

In another shell:
```
export SLURM_JOBID=<JOB ID FROM ABOVE>
srun --jobid=$SLURM_JOBID ...
```

You may need to set `--gres=gpu:0` to run some diagnostics job on the nodes. For example, let's check shared memory of all the hosts:
```
srun --jobid 631078 --gres=gpu:0 bash -c 'echo $(hostname) $(df -h | grep shm)'
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

## Aliases

Handy aliases

```
alias myjobs="squeue -u `whoami`"
alias groupjobs="squeue --user=$(getent group six | cut -d: -f4)"
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p gpu_p13 -o '%A'"
```

more informative all-in-one myjobs that includes the projected start time for pending jobs and requested time limit:

```
alias myjobs='squeue -u `whoami` -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"'
alias groupjobs='squeue -u $(getent group six | cut -d: -f4) -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"'
```



## Zombies

If there are any zombies left behind across nodes, send one command to kill them all.

```
srun pkill python
```

## Detailed Access to SLURM Accounting

`sacct` displays accounting data for all jobs and job steps in the Slurm job accounting log or Slurm database.

So this is a great tool for analysing past events.

For example, to see which nodes were used to run recent gpu jobs:

```
sacct -u `whoami` -A six@gpu -ojobid,start,end,state,exitcode --format nodelist%300
```

`%300` here tells it to use a 300 char width for the output, so that it's not truncated.

See `man sacct` for more fields and info fields.



## Queue


### Cancel job

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

### Tips

- if you see that `salloc`'ed interactive job is scheduled to run much later than you need, try to cancel the job and ask for shorter period - often there might be a closer window for a shorter time allocation.


## Logging

If we need to separate logs to different log files per node add `%N` (for short hostname) so that we have:

```
#SBATCH --output=%x-%j-%N.out
```

That way we can tell if a specific node misbehaves - e.g. has a corrupt GPU. This is because currently pytorch doesn't log which node / gpu rank triggered an exception.

Hoping it'll be a built-in feature of pytorch https://github.com/pytorch/pytorch/issues/63174 and then one won't need to make things complicated on the logging side.


## Show the state of nodes
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

Alternatively, as always this param can be part of the script:
```
#SBATCH --array=1-10%1
```

Here is toy slurm script, which can be used to see how it works:

```
#!/bin/bash
#SBATCH --job-name=array-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
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

To cancel the whole array, cancel the job id as normal (the number before `_`):
```
scancel 591970
```

To cancel a specific job:
```
scancel 591970_2
```

If it's important to have the log-file contain the array id, add `%A_%a`:

```
#SBATCH --output=%x-%j.%A_%a.log
```

More details https://slurm.schedmd.com/job_array.html


## Job Array Trains and their Suspend and Release

In this recipe we accomplish 2 things:

1. Allow modification to the next job's slurm script
2. Allow suspending and resuming job arrays w/o losing the place in the queue when not being ready to continue running a job

SLURM is a very unforgiving environment where a small mistake can cost days of waiting time. But there are strategies to mitigate some of this harshness.

SLURM jobs have a concept of "age" in the queue which besides project priority governs when a job gets scheduled to run. If your have just scheduled a new job it has no "age" and will normally be put to run last compared to jobs that have entered the queue earlier. Unless of course this new job comes from a high priority project in which case it'll progress faster.

So here is how one can keep the "age" and not lose it when needing to fix something in the running script or for example to switch over to another script.

The idea is this:

1. `sbatch` a long job array, e.g., `-array=1-50%1`
2. inside the slurm script don't have any code other than `source another-script.slurm` - so now you can modify the target script or symlink to another script before the next job starts
3. if you need to stop the job array train - don't cancel it, but suspend it without losing your place in a queue
4. when ready to continue - unsuspend the job array - only the time while it was suspended is not counted towards its age, but all the previous age is retained.

The only limitation of this recipe is that you can't change the number of nodes, time and hardware and partition constraints once the job array was launched.

Here is an example:

Create a job script:

```
$ cat train-64n.slurm
#!/bin/bash
#SBATCH --job-name=tr8-104B
#SBATCH --constraint=v100-32g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

source tr8-104B-64.slurm
```
Start it as:
```
sbatch --array=1-50%1 train-64.slurm
```

Now you can easily edit `tr8-104B-64.slurm` before the next job run and either let the current job finish if it's desired or if you need to abort it, just kill the currently running job, e.g. `1557903_5` (not job array `1557903`) and have the train pick up where it left, but with the edited script.

The nice thing is that this requires no changes to the original script (`tr8-104B-64.slurm` in this example), and the latter can still be started on its own.

Now, what if something is wrong and you need 10min or 10h to fix something. In this case we suspend the train using:

```
scontrol hold <jobid>
```

with <jobid> being either a "normal" job, the id of a job array or the id for a job array step

and then when ready to continue release the job:

```
scontrol release <jobid>
```


## Troubleshooting


### Kill Switch

Since SLURM doesn't allow one user to kill another user's SLURM job or cancel a job array, we need a way to be able to have the program abort itself quickly in situations where one user started a job and has gone away and the group needs to restart it. For example, this is needed when a model gets started by someone in North America, and while they are asleep, someone in Europe may need to handle a problem with the training and can't wait for the submitter of the job to wake up.

So we had a kill-switch feature implemented in Megatron-Deepspeed. When a file gets created at a pre-determined location, the software will stop its run. Instead of trying to implement a complex thread that will run only one of the dozens of nodes, we simply added a check in 2 strategic locations:

1. startup - to deal with job arrays
2. before each iteration of the train loop - to deal with the current run

Since multiple jobs use the same Megatron-Deepspeed repo clone this kill switch can't be hardcoded, and thus each job needs to "arm" the kill switch and must use a unique path so that unintentionally other instances won't get killed.

To arm:

```
python pretrain_gpt.py ... --kill-switch-path /tmp/kill-switch-tr11-200B-exp1
```

To trigger:
```
touch /tmp/kill-switch-tr11-200B-exp1
```

To deactivate and let new instances of a job run normally:

```
rm  /tmp/kill-switch-tr11-200B-exp1
```

### Mismatching nodes number

If the pytorch launcher fails it often means that the number of SLURM nodes and the launcher nodes are mismatching, e.g.:

```
grep -ir nodes= tr123-test.slurm
#SBATCH --nodes=40
NNODES=64
```

This won't work. They have to match.

You can add a sanity check to your script:

```
#!/bin/bash
#SBATCH --job-name=test-mismatch
#SBATCH --constraint=v100-16g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

[...]

NNODES=2

# sanity check for having NNODES and `#SBATCH --nodes` match, assuming you use NNODES variable
if [ "$NNODES" != "$SLURM_NNODES" ]; then
    echo "Misconfigured script: NNODES=$NNODES != SLURM_NNODES=$SLURM_NNODES"
    exit 1
fi

[...]
```

or you could just do:

```bash
#SBATCH --nodes=2
[...]
NNODES=$SLURM_NNODES
```

and then it will always be correct



### Find faulty nodes and exclude them

Sometimes a node is broken, which prevents one from training, especially since restarting the job often hits the same set of nodes. So one needs to be able to isolate the bad node(s) and exclude it from `sbatch`.

To find a faulty node, write a small script that reports back the status of the desired check.

For example to test if cuda is available on all nodes:
```
python -c 'import torch, socket; print(f"{socket.gethostname()}: {torch.cuda.is_available()}")'
```

and to only report the nodes that fail:
```
python -c 'import torch, socket; torch.cuda.is_available() or print(f"Broken node: {socket.gethostname()}") '
```

Of course, the issue could be different - e.g. gpu can't allocate memory, so change the test script to do a small allocation on cuda. Here is one way:

```
python -c "import torch; torch.ones(1000,1000).cuda()"
```

But since we need to run the test script on all nodes and not just the first node, the slurm script needs to run it via `srun`. So our first diagnostics script can be written as:

```
srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
```

I slightly changed it, due to an issue with quotes.

You can always convert the one liner into a real script and then there is no issue with quotes.

```
$ cat << EOT >> test-nodes.py
#!/usr/bin/env python
import torch, socket
print(socket.gethostname(), torch.cuda.is_available())
EOT
$ chmod a+x ./test-nodes.py
```

Now let's create a driver slurm script. Use a few minutes time for this test so that SLURM yields it faster:
```
#!/bin/bash
#SBATCH --job-name=test-nodes
#SBATCH --partition=gpu_p13
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

source $six_ALL_CCFRWORK/start-prod
srun --jobid $SLURM_JOBID ./test-nodes.py
```
Once it runs check the logs to see if any reported `False`, those are the nodes you want to exclude.

Now once the faulty node(s) is found, feed it to `sbatch`:
```
sbatch --exclude=hostname1,hostname2 ...
```
and `sbatch` will exclude the bad nodes from the allocation.

Additionally please report the faulty nodes to `assist@idris.fr` so that they reboot the machine.

Here are a few more situations and how to find the bad nodes in those cases:

### Broken NCCL

If you're testing something that requires distributed setup, it's a bit more complex. Here is a slurm script that tests that NCCL works. It sets up NCCL and checks that barrier works:

```
#!/bin/bash
#SBATCH --job-name=test-nodes-nccl
#SBATCH --partition=gpu_p13
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

source $six_ALL_CCFRWORK/start-prod

NNODES=2

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export SCRIPT=test-nodes-nccl.py

cat << EOT > $SCRIPT
#!/usr/bin/env python
import torch.distributed as dist
import torch
import socket
import os
import fcntl

def printflock(*msgs):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
header = f"{socket.gethostname()}-{local_rank}"
try:
    dist.barrier()
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is OK")
except:
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is broken")
    raise
EOT

echo $LAUNCHER --node_rank $SLURM_PROCID $SCRIPT

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $SCRIPT'
```
The script uses `printflock` to solve the interleaved print outputs issue.


### GPU Memory Check


This tests if each GPU on the allocated nodes can successfully allocate 77Gb (e.g. to test 80GB A100s) (have to subtract a few GBs for cuda kernels).


```python
import torch, os
import time
import socket
hostname = socket.gethostname()

local_rank = int(os.environ["LOCAL_RANK"]);

gbs = 77
try:
    torch.ones((gbs*2**28)).cuda(local_rank).contiguous() # alloc on cpu, then move to gpu
    print(f"{local_rank} {hostname} is OK")
except:
    print(f"{local_rank} {hostname} failed to allocate {gbs}GB DRAM")
    pass

time.sleep(5)


```


### Broken Network

Yet another issue with a node is when its network is broken and other nodes fail to connect to it.

You're likely to experience it with an error similar to:
```
work = default_pg.barrier(opts=opts)
RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch_1616554793803/work/torch/lib/c10d/ProcessGroupNCCL.cpp:825, unhandled system error, NCCL version 2.7.8
ncclSystemError: System call (socket, malloc, munmap, etc) failed.
```
Here is how to debug this issue:

1. Add:
```
export NCCL_DEBUG=INFO
```
before the `srun` command and re-run your slurm script.

2. Now study the logs. If you find:
```
r11i6n2:486514:486651 [1] include/socket.h:403 NCCL WARN Connect to 10.148.3.247<56821> failed : Connection refused
```
Let's see which node refuses to accept connections. We get the IP address from the error above and reverse resolve it to its name:
```
nslookup 10.148.3.247
247.3.148.10.in-addr.arpa       name = r10i6n5.ib0.xa.idris.fr.
```

Add `--exclude=r10i6n5` to your `sbatch` command and report it to JZ admins.



## TODO

absorb more goodies from here: https://ubccr.freshdesk.com/support/solutions/articles/5000686861-how-do-i-check-the-status-of-my-job-s-
