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

Finally GBS is at 1024, so we can do 64 nodes. Clocking about 23-26 secs / iteration - the performance jumps around quite a lot from run to run. But we know that already about JZ - it's very unsteady and depends on network usage by others.

Created a dedicated branch `tr1-13B`, which allows further development w/o the risk of breaking the current training.

## A huge lm loss spike

The training loss just jumped from ~3 to ~9
```
 iteration    29020/  311541 | consumed samples:     10698064 | elapsed time per iteration (ms): 22306.6 | learning rate: 9.850E-05 | global batch size:  1024 | lm loss: 2.775923E+00 | loss scale: 32768.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29030/  311541 | consumed samples:     10708304 | elapsed time per iteration (ms): 22336.4 | learning rate: 9.849E-05 | global batch size:  1024 | lm loss: 2.772822E+00 | loss scale: 32768.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29040/  311541 | consumed samples:     10718544 | elapsed time per iteration (ms): 22332.6 | learning rate: 9.849E-05 | global batch size:  1024 | lm loss: 2.768131E+00 | loss scale: 65536.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29050/  311541 | consumed samples:     10728784 | elapsed time per iteration (ms): 22148.5 | learning rate: 9.849E-05 | global batch size:  1024 | lm loss: 7.343709E+00 | loss scale: 8192.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29060/  311541 | consumed samples:     10739024 | elapsed time per iteration (ms): 22181.7 | learning rate: 9.849E-05 | global batch size:  1024 | lm loss: 8.715872E+00 | loss scale: 4096.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29070/  311541 | consumed samples:     10749264 | elapsed time per iteration (ms): 22107.1 | learning rate: 9.848E-05 | global batch size:  1024 | lm loss: 7.654131E+00 | loss scale: 4096.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29080/  311541 | consumed samples:     10759504 | elapsed time per iteration (ms): 22131.2 | learning rate: 9.848E-05 | global batch size:  1024 | lm loss: 7.192470E+00 | loss scale: 4096.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
 iteration    29090/  311541 | consumed samples:     10769744 | elapsed time per iteration (ms): 22119.2 | learning rate: 9.848E-05 | global batch size:  1024 | lm loss: 6.849044E+00 | loss scale: 4096.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

You can see the spike at https://huggingface.co/bigscience/tr1-13B-tensorboard/tensorboard

It took some 500 iteration to recover.

There was a second spike a bit later, half the first one this time and recovered very quickly.

We discussed why it may have happened, but we don't have any definitive answer.


## Checkpoint bloat issue

We have an issue with per-layer checkpoints that are way bigger than they should be. They are 10x bigger than what they should be. After some research we discovered that `torch.save()` doesn't save the current view, but the whole tensor with its original tensor storage. So that's why were were getting 10x bigger files than the actual data in the per-layer checkpoints.

We need to `.clone()` the tensors before saving them. and then the checkpoint for layers is just modelsize*2 bytes. The reason they were bloated is because ZeRO-1 pre-allocated large tensor buffers for run-time optimization. So this needs to be fixed in Deepspeed's pipe checkpoing saving.

Also will write a script to fix the already-saved checkpoints to `clone` and re-save.


## old NCCL

Discovered the NCCL was statically linked into the distributed pytorch and it's really old 2.7.9. Supposedly newer NCCL should help with OPA interlink performance. But that means we either need to switch to a more recent pytorch or build our own. This is not resolved yet.


## Watchdog

We created a watchdog, that reports if we are running/scheduled and alerts if neither is happening. E.g. the recent log in the main log file was:

```
 iteration    33240/  311541 | consumed samples:     15019344 | elapsed time per iteration (ms): 23491.4 | learning rate: 9.702E-05 | global batch size:  1024 | lm loss: 2.722675E+00 | loss scale: 32768.0 | grad norm: 0.000 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
saving checkpoint at iteration   33241 to /gpfsscratch/rech/six/commun/checkpoints/tr1-13B/checkpoints
[2021-08-08 01:00:44,221] [INFO] [logging.py:68:log_dist] [Rank 0] Saving model checkpoint: /gpfsscratch/rech/six/commun/checkpoints/tr1-13B/checkpoints/global_step33241/mp_rank_00_model_states.pt
  successfully saved checkpoint at iteration   33241 to /gpfsscratch/rech/six/commun/checkpoints/tr1-13B/checkpoints
time (ms) | save-checkpoint: 57514.53
[exiting program after 1190.0357275923093 minutes] datetime: 2021-08-08 01:00:51
[2021-08-08 01:49:40] ***ALERT: tr1-13B-round3.slurm is not RUNNING or SCHEDULED! Alert someone at Eng WG***
[2021-08-08 02:49:44] ***ALERT: tr1-13B-round3.slurm is not RUNNING or SCHEDULED! Alert someone at Eng WG***
[2021-08-08 03:56:54] tr1-13B-round3 is scheduled to start in 3 days, 7:24:19 (at 2021-08-11T11:21:14) (682842_[1-5%1] on 'gpu_p13' partition)
```

## NNODES=96

We thoughts that trying more nodes would be a good idea, but 96 nodes proved to be unacceptable, since

GBS=1024 is not divisible by 384 (96*4), so there is no way to spread data evenly across all replicas.

We can only have either 256, 512 or 1024 gpus (64, 128, 256 nodes)

## Corrupt GPU crashes the training multiple times

One of the array job trainings crashes after many hours of training:

```
iteration    43680/  311541 | consumed samples:     25709904 | elapsed time per iteration (ms): 25593.4 | learning rate: 9.135E-05 | global batch size:  1024 | lm loss: 2.635663E+00 | loss scale: 131072.0 | grad norm: 17224.723 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
Traceback (most recent call last):
  File "/gpfswork/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/pretrain_gpt.py", line 222, in <module>
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/training.py", line 144, in pretrain
    iteration = train(forward_step_func,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/training.py", line 677, in train
    train_step(forward_step_func,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/training.py", line 381, in train_step
    loss = model[0].train_batch(data_iter=data_iterator)
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/DeepSpeed-big-science/deepspeed/runtime/pipe/engine.py", line 291, in train_batch
    self._exec_schedule(sched)
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/DeepSpeed-big-science/deepspeed/runtime/pipe/engine.py", line 1237, in _exec_schedule
    self._exec_instr(**cmd.kwargs)
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/DeepSpeed-big-science/deepspeed/runtime/pipe/engine.py", line 679, in _exec_backward_pass
    torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))
  File "/gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/autograd/__init__.py", line 145, in backward
    Variable._execution_engine.run_backward(
RuntimeError: transform: failed to synchronize: cudaErrorECCUncorrectable: uncorrectable ECC error encountered
terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: uncorrectable ECC error encountered
Exception raised from create_event_internal at /opt/conda/conda-bld/pytorch_1616554793803/work/c10/cuda/CUDACachingAllocator.cpp:733 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x42 (0x1500fb4d42f2 in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x5b (0x1500fb4d167b in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::CUDACachingAllocator::raw_delete(void*) + 0x809 (0x1500fb72d219 in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10::TensorImpl::release_resources() + 0x54 (0x1500fb4bc3a4 in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #4: <unknown function> + 0x6e0e5a (0x150152432e5a in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #5: <unknown function> + 0x6e0ef1 (0x150152432ef1 in /gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0x1a6b5a (0x56434fce9b5a in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #7: <unknown function> + 0x110b7c (0x56434fc53b7c in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #8: <unknown function> + 0x1105b9 (0x56434fc535b9 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #9: <unknown function> + 0x1105a3 (0x56434fc535a3 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #10: <unknown function> + 0x1105a3 (0x56434fc535a3 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #11: <unknown function> + 0x177917 (0x56434fcba917 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #12: PyDict_SetItemString + 0x4c (0x56434fcbd86c in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #13: PyImport_Cleanup + 0xac (0x56434fd2f0ec in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #14: Py_FinalizeEx + 0x79 (0x56434fd95589 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #15: Py_RunMain + 0x1bc (0x56434fd988fc in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #16: Py_BytesMain + 0x39 (0x56434fd98ce9 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
frame #17: __libc_start_main + 0xf3 (0x150183467873 in /lib64/libc.so.6)
frame #18: <unknown function> + 0x1f7847 (0x56434fd3a847 in /gpfswork/rech/six/commun/conda/tr1-13B/bin/python)
```

Nobody was around to notice and slurm scheduler started the next training job in the array, and it crashed too this time right away on:

```
> initializing tensor model parallel with size 2
> initializing pipeline model parallel with size 4
> setting random seeds to 42 ...
[2021-08-12 08:19:28,225] [INFO] [checkpointing.py:226:model_parallel_cuda_manual_seed] > initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 2760 and data parallel seed: 42
> compiling dataset index builder ...
make: Entering directory '/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/data'
make: Nothing to be done for 'default'.
make: Leaving directory '/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/data'
>>> done with dataset index builder. Compilation time: 0.338 seconds
> compiling and loading fused kernels ...
Traceback (most recent call last):
  File "/gpfswork/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/pretrain_gpt.py", line 222, in <module>
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/training.py", line 95, in pretrain
    initialize_megatron(extra_args_provider=extra_args_provider,
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/initialize.py", line 89, in initialize_megatron
    _compile_dependencies()
  File "/gpfsssd/worksf/projects/rech/six/commun/code/tr1-13B/Megatron-DeepSpeed-tr1-13B/megatron/initialize.py", line 140, in _compile_dependencies
    torch.distributed.barrier()
  File "/gpfswork/rech/six/commun/conda/tr1-13B/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2420, in barrier
    work = default_pg.barrier(opts=opts)
RuntimeError: CUDA error: out of memory
```

We figured one of the gpus had a hardware problem. So it crashed the first time. And then the scheduler allocated the same node and of course, we crashed again.

We contacted JZ admins and indeed one of the nodes was faulty. The next training didn't hit this node and the training continued.

Unfortunately we currently don't have a way to correlate the exceptions to the hostname of the node that it happened on. It's really to have this feature available, since if we don't, we can keep on hitting the faulty node and it'll continue crashing the training. If we know the node's hostname we can exclude it from the `sbatch --exclude=node1,node2,... `.

update: At the moment we have to add `%N` to `#SBATCH --output=%x-%j-%N.out` and then each node will have is own log file and then we can tell which node has a corrupt GPU.

## Really long wait time to get allocation

When a job gets queued we often see 3 days expected wait time before yielding, but most of the time the job comes through in several hours. Sometimes we have to wait for a really long time, like 30h, with scheduler bumping our job down multiple times. This is a big problem as it pushes the finish line away continuously. We aren't anywhere close to being able to train 24/7 despite having many hours allocated to us for this project.

Another problem is that within a project we don't have a way to give the main training job a higher priority than other jobs that we run in parallel on various experiments and small trainings. There really should be a way for a user to say, this is a high priority job amongst all other jobs of the same group. But we didn't find a way to do that.

## Test suite added

A `Megatron-Deepspeed` test suite was finally added. It was odd Megatron-LM didn't have one in the first place, so we had to create our own.

Now need to find some hardware with 2 gpus to create a CI.

## Reduced evaluation iterations

Noticed that somehow it was configured to run eval for 100 iterations, after discussion reduced it to 5, thus saving some resources. While validation iterations are much faster than training, this wasn't really needed.

## NNODES=128

Taking advantage of August's holiday in France was able to switch to 128 nodes.

Observed a further drop in TFLOPs, since now we had even less microbatches to go around. This is because Global BS remained the same (GBS=1024) and we currently use 2 nodes for a single replica (TP=2 * TP=4). So with 128 nodes, we have 64 replicas, which leaves only GAS=16 per replica, and that's too little for an efficient pipeline. The idle bubble is too big.

The benchmarking/tune up was done with GAS=128 (GBS=1024/8) and that's where we were getting high TFLops.

Nevertheless, the training is going much faster now and we will catch up lost time quickly.

## NCCL experiments

It was suggested that newer NCCL will lead to faster inter-node communication.


hypothesis that newer nccl should be faster on JZ, but the short experiments I run didn't support it. I get the same throughput with:

1. pt=1.8.1, cuda=11.1, nccl=2708
2. pt=1.9.0, cuda=11.1, nccl=2708
3. pt=1.10.0.dev20210821, cuda=11.3, nccl=(2, 10, 3)

The experiment was run on the same 4-node allocation with GBS=64, but otherwise everything else was the same as the current training script. The speed was 17-17.5 secs per iteration. Did about 100 iterations.
So we will stick to pt=1.8.1 for now until a need arises to change that.

## SLURM Job Arrays and Dependency

Switched to using SLURM Job Arrays and Dependency to schedule jobs. Since our account has a huge allocation we were able to start new 20h jobs with no delay.

If this approach is not used even a tiny delay between finishing one job and scheduling the next one often lead to 1-30 hours of wait time in the queue. This is because the scheduler was quick to allocate other jobs in the first few seconds of finishing the currently running job.

The problem remained if something goes wrong - e.g. a mistake in a script or some hardware issue, would lead to a delay in staring new jobs and a long long wait time.

This training was getting its software updated a lot as missing features were added, so it wasn't a super-stable polished production environment.

So as long as we had a stable setup using SLURM Job Arrays and Dependency chaining things went well. When we couldn't use those SLURM was delaying our training sometimes by a lot.

Also since we run secondary trainings we learned to use `--nice=10000` for those trainings. Without this method all slurm jobs of the same account had the same priority.

## Added an alert email notification

Previously implemented watchdog now got hooked up to email notifications, so if it detected that no job was running or scheduled it'd let the group know.

## Checkpoint bloat fixed

The Deepspeed team fixed the bloat in the checkpoints, so new checkpoints were taking 10x less space for layer weights.

I then processed all the old checkpoints to remove the bloat using:

```
srun -p prepost  -A six@cpu --time=20:00:00 --pty bash
wget https://raw.githubusercontent.com/stas00/toolbox/master/pytorch/pt-checkpoint-shrink.py
chmod a+x pt-checkpoint-shrink.py
cd checkpoints
find -type d -name "global_step*" -exec pt-checkpoint-shrink.py --checkpoint_dir {} --patterns "layer*pt" \;
```

## CI was added

A CI was implemented using EC2 instance on demand. With the help of https://github.com/machulav/ec2-github-runner


## Training completed

On Sep 6th we reached the 300B tokens and on Sep 7th we stopped the training - It took some ~5 weeks to complete.


## Checkpoint conversion

We still need to figure out how to make the checkpoint available in the HF `transformers` format. This is a work in progress.


XXX: to be continued

stopped at Date: 2021-09-09
