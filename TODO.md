# Things to do

## Carbon Footprint Tracking

Instrument multi-node carbon footprint tracking. https://github.com/mlco2/codecarbon
Seems like we only need to add about 2 lines in our code.

The decision is to run it on one node (gpu?), and then the results will be multiplied by the number of nodes. It generates a csv results. Need to figure out where to broadcast it to from JZ.

Blocking event: no codebase to yet to add it to - need to fork https://github.com/microsoft/Megatron-DeepSpeed once it's ready.


## Weights-Only checkpoints

Contributors that have no access to JZ will want to have intermediary checkpoints to work with. It'll be very slow to scp full checkpoints. Would it be possible to either post-process the Deepspeed PP checkpoints and extract just the model weights before copying those from JZ?

The current DS PP format saves each layer's state dict in its own file, and they're named differently than the optimizer states. Could be as simple as pattern matching the scp. The pipeline engine selectively loads the files based on pipeline rank, so no need to merge them.

But users outside of JZ will very likely have a different HW setup, so these will need to be re-shaped to match a new PP-degree.

## Overcoming lack of crontab for exporting data outside of JZ

Try out this suggestion:

Use the "compil" partition as a cron-like partition. There are usually some resources available on this partition.

The idea would be submit a job that resubmit itself while the job performing the actual computation runs:

```
  # Resubmit the cron-like job if the compute job is still running
  squeue -j $COMPUTE_JOB_ID >& /dev/null && sbatch --begin=now+3hours cron_job.slurm
```


# TODO

- add fixes to Megatron-Deepspeed to our branch
- exit duration in minutes to match the 20h limit - set it to a slightly shorter duration
- deepspeed: activate contiguous activation checkpoints - less fragmentation!
