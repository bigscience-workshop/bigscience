# Megatron-LM Notes and Nuances


Notes from Jared - to sort:

- batch size

`--global-batch-size` leads to automatic gradient accumulation, so for example on 4-gpu node with:

with only 4-way data parallel using a micro batch size of 16 and global batch size of 2048 it's going to do gradient accumulation on 32 batches for each iteration.

so probably best not to use this argument, unless it's thought through.

--micro-batch-size is always the smallest "batch size", it's what gets sent through the model.

--global-batch-size will default to micro batch size * data parallelism unless specified. With the default value there will be no gradient accumulation. If specified, gradient accumulation will happen to reach the global batch size. The "chunks" you talk about above for PP we see as just gradient accumulation. Without gradient accumulation PP is very inefficient with no overlap of executing the different stages. So the more micro-batches that get accumulated, or the large the global batch size, the more efficient PP will be.
We discussed a lot about how best to expose that in arguments and decided most of the time we care about the micro batch size and the global batch size and don't want to do the math to figure out the number of microbatches done to get to the global batch size. Especially since we will sometimes have a dynamic global batch size

So bottom line under PP number of micro-batches == gradient accumulation
