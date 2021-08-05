# tr1-13B Chronicles

Notes on the training progress with a particular focus on any encountered problems and their diagnosis and solutions/prevention.

To follow the training progress charts, see: [tensorboard](https://huggingface.co/bigscience/tr1-13B-tensorboard/tensorboard).

To follow the raw training logs see: [logs](https://huggingface.co/bigscience/tr1-13B-logs/).


## Round1 SAVE_INTERVAL=10

NNODES=16

saved checkpoint each 10 steps

`$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/tr1-13B-round1/checkpoints`

10 checkpoints (Every 10 steps 1-100) - 4TB

## Round2 SAVE_INTERVAL=18

NNODES=16

moved the round1's checkpoints away

rerun from scratch with the same seed

saved checkpoint each 18 steps

`$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/tr1-13B-round2/checkpoints`

51 checkpoints (Every 18 steps 101-1000) - 20TB


## Round3 SAVE_INTERVAL=1500 NNODES=16

NNODES=16

moved the round2's checkpoints away

rerun from scratch with the same seed

saved checkpoint each 1500 steps

I did the full re-run because otherwise I couldn't separate the tensorboard logs - it is not possible to restart from a checkpoing using `TRAIN_ITER` or `EXIT_INTERVAL` which is not fixed.

now we started uploading tensorboard logs


## Round3 SAVE_INTERVAL=1500 NNODES=32

Tried to switch to 64 nodes, but the training failed because GBS gets incremented by 16, which limits us to DP_SIZE=16 (with MBS=1) so we can do 32 nodes (128gpus at most).

```
DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE)
16     = 32*4/(4*2)
```

will switch to 64 nodes once GBS reaches 1024.


The training then crashed with shared memory error after some 10h+ of training:
```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Traceback (most recent call last):
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 986, in _try_get_data
Traceback (most recent call last):
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 986, in _try_get_data
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/queue.py", line 179, in get
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/queue.py", line 179, in get
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/threading.py", line 306, in wait
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/threading.py", line 306, in wait
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
  File "/gpfswork/rech/six/commun/conda/hf-prod/lib/python3.8/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
RuntimeError: DataLoader worker (pid 30882) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
RuntimeError
The above exception was the direct cause of the following exception:
: Traceback (most recent call last):
DataLoader worker (pid 30801) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.  File "/gpfswork/rech/six/commun/code/Megatron-DeepSpeed/pretrain_gpt.py", line 215, in <module>
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/gpfswork/rech/six/commun/code/Megatron-DeepSpeed/pretrain_gpt.py", line 215, in <module>
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 144, in pretrain
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 144, in pretrain
        iteration = train(forward_step_func,iteration = train(forward_step_func,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 675, in train
  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 675, in train
    train_step(forward_step_func,
    train_step(forward_step_func,  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 381, in train_step
  File "/gpfsssd/worksf/projects/rech/six/commun/code/Megatron-DeepSpeed/megatron/training.py", line 381, in train_step
    loss = model[0].train_batch(data_iter=data_iterator)
loss = model[0].train_batch(data_iter=data_iterator)
```

Each node has 94GB of /dev/shm, so it's very strange that this happened.

```
df -h | grep shm
tmpfs            94G  336K   94G   1% /dev/shm
```
This is after 2h of training on one node. I wonder if the problem was on some specific node.

Though Remi checked that all nodes used by the training that crashed had this exact setup. And all reported %1 usage.



To continually diagnose the running nodes's shm memory usage:
```
for ((;;)) { (srun --jobid 637799 --gres=gpu:0 $six_ALL_CCFRWORK/bin/report_shm_usage | grep -v "1%"); sleep 10; }
```
after adjusting the jobid number.

where:
```
cat $six_ALL_CCFRWORK/bin/report_shm_usage
#!/usr/bin/bash

# print shared memory usage with the host

echo $(hostname) $(df -h | grep /dev/shm)
```

The shared memory is used by `DataLoader` workers. We just use the default `args.num_workers==2` and 94GB of shm available on each node is a huge amount of shared memory.

And given that we use TP+PP, a single node doesn't have DDP on it, so no multiproc on the local host. Currently one full model replica uses 2 full nodes (`TP*PP = 2*4 = 8`) So it's really a single Dataloader call per each 2 nodes. i.e. tiny tiny needs.

If this happens again, setting `args.num_workers==0` will stop using shared memory, but it'll impact the data loading speed.

Jared hasn't seen this problem in his experience.

So at the moment we don't know what happened.

2 more 20h trainings have been run since then w/o any problems.

## Checking the progress

Someone asked when the current training will complete:

Let's do math:

1. we are currently going at 784 samples in 32 seconds, or 24.5 samples / sec
2. roughly we have 145M samples to go, so at the current speed 32nodes if we manage to have 20h allocation every 24 hours we get about 82 days. (145_000_000/(20*60*60*24.5))
3. we should reach GBS=1024 hopefully today and then we can crank up to 64nodes, which should roughly double the speed, so it'll take 41 days to complete if all goes well and we don't sit in the queue for more than 4 hours.
4. we can dare to try 128 nodes, which would quadruple the speed and we should be done in about 20 days. It's hard to tell how quickly the SLURM scheduler will provide such a large allocation - if more than half-day of wait time, we are probably better off with 64 nodes.


## Round3 SAVE_INTERVAL=1500 NNODES=64

XXX: to be continued
