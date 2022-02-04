# Lessons learned

The following are super-brief summary notes. If you want the details with graphs and full notes, see:

13B:
* [chronicles](./tr1-13B-base/chronicles.md)

104B:
* [chronicles a](./tr8-104B-wide/chronicles.md)
* [chronicles b](./tr8b-104B/chronicles.md)

## How training divergences were overcome

The following are techniques that have to be done before the training starts.

### Using a formulaic std init

Setting `--init-method-std` to `sqrt(2/(NHIDDEN*5))` has made a huge difference to the training stability.

e.g. for `NHIDDEN*5=11600` we used  `--init-method-std 0.006`

We derived this from:

`0.00587220219514703 = sqrt(2/(11600*5))` (from the "Transformers without Tears" paper https://arxiv.org/abs/1910.05895)

If you are wondering why the depth of the model is not included in this month, it's then used by the framework internally via a [second std init function](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/40e8b2a086f98920de692ebc4081bf4229bfa81a/megatron/model/utils.py#L33-L40) which rescales the std of the second layer in the MLP and the output layer of the attention with:
```
std = sigma / math.sqrt(2.0 * num_layers)
```
where `sigma` is the `--init-method-std` argument.



### Adding embed layernorm

Embedding LayerNorm has shown to help a lot with spikes that the training can't recover from. This insight came from experimenting with https://github.com/facebookresearch/bitsandbytes which contains a `StableEmbedding` which is a normal Embedding with layernorm and it uses a uniform xavier initialization.

To activate add `--embed-layernorm`

Note: since this has its weights you can only add it at the beginning of the training

Note: since this is not part of the normal HF GPT2, this will require a new arch or a config that adds a layer-norm to the GPT2 model.


### Using a Lower LR

- halving lr from 6e-5 to 3e-5 also proved fruitful, but it went through a huge spike at iteration 11.5k and took ~2k iterations to recover (exp 11) at which point it was put on hold and other approaches were experimented with.


### Patience

In some cases in the case of a huge spike it was taking 2k iterations for a training to return to the same lm loss it spiked from. And then it'd continue training as if nothing happened.

But more often than not the training won't recover from a spike.

Yet in other situations the training diverged slowly without any spikes.


## How to deal with ongoing instabilities

How to recover from an instability without a full restart.

### Data skipping

1. Roll back to the last checkpoint before the instability
2. skip data samples from the instability window `--skip-train-iteration-range 8401-8800 `

### LR Changing

Normally LR-related params can't be changed once training has started (Megatron asserts) but with `--override-lr-scheduler` we can completely rewrite them and it just works. that is megatron recalculates everything based on cmd line args and sets the LR to the right setting which can be very different from what it'd have normally been.

So for example now we can rollback a bit and change LR if we need to to try to overcome some rough patch of data or some other instability.


## What was tried and it didn't work

- changing seed - the problem usually would just shift elsewhere - but it might work in some situation where data skipping worked

- a more numerically stable self-attention version by multiplying the two matrices passed to `torch.baddbmm` by `1.0/math.sqrt(self.norm_factor)` and then using `alpha=1.0`

- lowering `beta2` to 0.95 (from 0.999)

- changing width/depth ratio

- longer lr warmup

- tried Curriculum Learning
