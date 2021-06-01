# Deepspeed notes

A lot of these collected from chats with Samyam, Shaden and Olatunji

## Should I use the `deepspeed` launcher under slurm.

No, it won't work.

Instead use:
```
python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $SLURM_PROCID \
    ....
```

## on 8 gpus I get now: `data_parallel_size: 8, parameter_parallel_size: 8`

In this case seeing that the DP and parameter parallel size match means ZeRO will partition across all gpus

## Memory estimates

As each node has about 160GB of memory, the model size you can run with Z2-Offload is about 8-10B parameters per node. Each of those parameters will require 4 bytes for fp32 momentum, variance, and parameters, gradients so a total of 16 bytes per parameter, for a total of about 160 GB.


# Pipeline + ZeRO

If you're using PP, you'll want to use ZeRO stage 0 or 1.  Pipeline parallelism does weird things with gradients that does not play nicely with Z2+. We assert that when using DS' pipeline parallelism, but I think it's more wild west with Megatron's PP implementation.

```
train_batch_size=$(($WORLD_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))
```

You want to scale by DP size instead of WORLD_SIZE. Let me write down a bit about batch sizes:


# Megatron + Deepspeed


The `batch_size` in our Megatron scripts is the same thing as micro-batch size. That's the size of each batch of data that comes off the data loader and goes through the various kernels. That's usually what you think of when you talk about batch size (then multiplied by the size of data parallelism)

Megatron updated their terminology to match DeepSpeed once they added PP support, which adds the concept of gradient accumulation. Before that, there was no grad accumulation and so the global batch size was assumed to be `DP * batch_size`.

So thinking in terms the three axes of parallelism:

* Each pipeline processes a `gradient_accumulation_steps` (gas) number of micro-batches per training step. There are as many pipelines as the data parallel dimension, so the global batch size of each training step is `microbatch * gas * DP`
* Megatron's model parallelism (renamed to tensor model parallelism) is not in the above formula. You can think of it as splitting batches across the MP group.

A bit on the various batch size parameters and performance:

Increasing micro-batch size increases the arithmetic intensity of individual kernels, increasing throughput and also the memory pressure from activations.

Increasing the gradient accumulation steps decreases the bubble overheads of pipeline parallelism. For DeepSpeed's PP algorithm, if you set `gas=8*PP` you should get 90% pipeline efficiency. Theoretical pipeline efficiency is:

```
efficiency = gas / (gas + PP - 1)
```

Increasing gas relative to PP will asymptotically approach 100% efficiency as you shrink the pipeline bubble overhead.

PyTorch's PP implementation is based on the GPipe algorithm, which still has a clear divide between forward/backward passes:

![gpipe](images/gpipe.png)

Their docs use both chunks/microbatch terminology. I'll use 'mb' for short. The key thing to note is that all the forward passes are done first, then all the backward passes. That means that the pipeline memory overheads (eg., activations from each mb) are kept around and scale linearly with the number of chunks. Since you increase the number of chunks to decrease PP overheads, you pay a linearly increasing memory cost to improve throughput.

DeepSpeed's pipeline parallelism takes another approach, in which the forward/backward passes for different mbs are done in parallel.

![deepspeed pipe](images/deepspeed-pipe.png)

After each backward pass completes, the gradient is accumulated into a single gradient buffer and the corresponding activations are freed. The number of mbs in flight at any time is bounded by the dimension of pipeline parallelism, not the number of gradient accumulation steps (same thing as chunks). That means that you can still increase the gas to improve efficiency, but memory overheads stay constant and only scale with the number of pipeline stages.

Say you split a model across 20 pipeline stages and want 90% PP efficiency... the GPipe approach will need about 8x more memory for activations because each microbatch has to be kept around until all of the backward passes begin.

Activation checkpointing of course reduces activation memory for both, but this applies even with checkpointing each layer. There are also pipeline overheads in which you store the input/output for each mb to pass to the adjacent stages

Though let me add, when I'm tuning perf for PP+DP I usually increase the gas first to get rid of the pipeline bubble overhead. Then you can increase the microbatch size to improve efficiency of individual kernels



## Tuning experiments


Shaden's approach:

- Fix MICRO_BATCH_SIZE=1 until you're set with the model configuration.
- Use TP_SIZE=GPUS_PER_NODE
- If using PP, use PP_SIZE=NNODES and PP_CHUNKS at about 8*PP_SIZE. Larger that that won't hurt if you can spare a larger batch size, but there are diminishing returns. PP_CHUNKS=16*PP_SIZE increases efficiency to 94% for example (vs 90%).
- Increase layer/hidden until you can't ￼
. Load balance is important here, you want the number of layers to be divisible by PP_SIZE. Otherwise the entire pipeline slows down
- You can go back at the end and try to increase MICRO_BATCH_SIZE if you have leftover memory for larger activations. Sometimes I can increase to 2 and get higher throughput


Samyam's approach:

- try to tune up the max micro-bs on 1 node model scaled down to a few layers (Same hidden size)
- experiment in the range of 16 to 64 to get the highest tflops
- how efficient it's running w/o communications
- fit on a single node
- could turn off optimizer step - no communications between gpus
- one more hyper param to experiment with:
  tiled - turn it on - overlapping communication improvement
