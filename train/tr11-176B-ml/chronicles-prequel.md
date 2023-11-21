# Prequel

Trials and tribulations prior to the start of training.

For the trials and tribulation during the training see: [chronicles](chronicles.md).

# A100 experiments

200B

torch.optim.Adam:

16 nodes:
- 1st node:  61GB
- all nodes: 47GB
- performance: XXX

apex.optimizers.FusedAdam

16 nodes:
- 1st node:  51GB
- all nodes: 44GB
- performance: XXX



## Size


Here are some existing models around the same size with NLAYERS / NHIDDEN and their ratio:


| origin | size | layers | hidden | ratio |
| ------ | ---  | -----: | -----: | ----: |
| bs     | 104B |     64 |  11600 |   180 |
| meg-lm | 145B |     80 |  12288 |   154 |
| openai | 175B |     96 |  12288 |   128 |
| meg-lm | 310B |     96 |  16384 |   170 |
| msft   | 530B |    105 |  20480 |   195 |
|        |      |        |        |       |




Possible ideas:

- 205B: 112 / 12288 (ratio: 109) narrow
- 206B: 96 / 13312 (ratio: 139) closer to typical 150-200 ratio

Formula to get model size, used 150k dict roughly - need to update:
```
NHIDDEN=12288; NLAYERS=112; SEQ_LEN=2048; VOCAB_SIZE=150257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
```

### 104B topology / memory usage

Looking at the current 104B topology to try to estimate the 200B model, though many things are different.

```
NLAYERS=64
NHIDDEN=11600
NHEADS=80
SEQ_LEN=2048
VOCAB_SIZE=50257
```

32 GB gpus.

TP=4, PP=32

breakdown:

104B:

- embedding size: `v*h`: `50257*11600` = 582_981_200 / 4 (TP=4) => 145_745_300 params per gpu for embedding
- one layer size: `12*h**2 + 13*h`:  1_614_870_800 / 4 (TP=4) => 403_717_700 params per gpu per layer

64 layers over PP=32 => 2 layers per gpu

Total params per gpu:
- gpu w/  emb: `2*403_717_700 + 145_745_300` = 953_180_700 params * 18 bytes = 17_157_252_600 bytes (17GB)
- gpu w/o emb: `2*403_717_700`               = 807_435_400 params * 18 bytes = 14_533_837_200 (15GB)

plus activations memory

Checking the actual GPU allocations (nvidia-smi) - also need to take into account the cuda kernels (1271MiB)

- 22GB (w/  embed) (4GB activations memory)
- 18GB (w/o embed) (2GB activations memory)

## Hardware

384 A100s 80GB / 8 gpus per node

We can plan to use 384 gpus out of 416 as 4 nodes of 8 gpus need to remain reserved for when some nodes happen to be down.

Initially we will have only 144 gpus and then around mid-Feb we should have the rest.

## Possible config:

So a possible config is

- a single replica needs to fit 96 gpus and then we can do DP=4 to a full 384 gpus

- extrapolating from the current 104B setup we can have: TP=4/PP=24 @ 80GB + 150K vocab size (which is different from the 50k vocab in 104B - 3x bigger embed matrix plus bigger hidden size.

- most likely the embedding layer now will need to be partitioned together with the transformer blocks to do a good balancing of resources. e.g. in the current 1.3B ml setup, the 1st and last gpus use all of DRAM, but the rest of gpus use only 1/2 DRAM - and TLOPs are ~21 which is very underutilized.


### Possible topologies for 200B

206B:

```
NLAYERS=96
NHIDDEN=13312
NHEADS=XXX
SEQ_LEN=2048
VOCAB_SIZE=150_000
```

Overall we know that DP is the fastest, then PP, then TP - but for PP to be efficient we need a big bs.

The following math is trying various topologies to fit into 80GB gpus


* TP=4, PP=24

- embedding size: `v*h: 150257*13312` = `2_000_221_184 / 4` (TP=4) =>  500_055_296 params per gpu for embedding
- one layer size: `12*h**2 + 13*h`:     `2_126_685_184 / 4` (TP=4)  => 531_671_296 params per gpu per layer

In other words 2B params per layer w/o TP, or 38GB (`2.12*18`) per layer.

So here we definitely need to balance embedding layer with transformer layers as they are of the same size, so overall 2+layers blocks to balance - and the constraint won't be Layers % PP = 0 but Layers+2 % PP = 0

So probably should do 94 layers?

94+2 layers over PP=24 => 4 layers per gpu

Total params per gpu (considering emb layer on par with transformers block):
- `4*531_671_296` = `2_126_685_184 params * 18` = 38_280_333_312 bytes
plus activations memory

40GB A100 takes 1573MiB for cuda kernels (probably about the same for 80GB? may be a bit larger)
`python -c "import torch; import time; torch.ones(1).cuda(); time.sleep(30)"` + check `nvidia-smi` output.



* TP=1, PP=96

~2B params per layer w/o TP, or 38GB (`2.12*18`) per layer.

but DS breaks if there isn't at least one transformer block per gpu :(
otherwise could do a very efficient:

```
1   | 2      | 3      ... | 95     | 96
emb | transf | transf ....| transf | emb
```

So in this scenario no TP is needed, which should make the assembly much faster. But will require DS fixing their side. or perhaps we could somehow hack on a dummy layer which will be like transformers? e.g. if it's the first or last layer it'd be an identity forward.

Also the pipeline will be super long here, which to make efficient will require a huge global batch size.



* with TP=2, PP=48

1_063_342_592 params per layer, 19_140_166_656 bytes (19GB) per layer

perhaps could squeeze 3 layers per gpu - but of course each gpu will be less efficient since it will have to do 3 pipe stages.

* Other considerations

Of course, we could make the model wider and shallower so for example with TP=1 perhaps we could fit a bit more width and use less layers. e.g. 530B model was NLAYERS=105, NHIDDEN=20480 - so it's much wider.



## Reconsiderations

After discussing the above plans with the NVIDIA and DeepSpeed experts it appears that:

1. on A100 and especially with much larger models TP>1 is much more beneficial and typically NVIDIA almost always uses TP=gpus_per_node for large models.

2. A very deep PP (96) would be very difficult to keep efficient unless the batch size per replica is huge.

3. Too many layers isn't great either:

Jared Casper writes:

> Regarding hidden size vs transformer layer (width vs depth), some feedback I got is that there isn't really a magic formula/process. We increase depth with the width but not as drastically as a typical vision model scaling. So you shouldn't go too crazy with depth. The width is somewhat constrained by sizes good for the GPU, so it seems a strategy is to push out the width but keep it nice numbers, then fill out with depth. You'll notice even at 530B params we only went to 105 layers.


## Existing models

Let's first analyse a few existing models and see how they fit 80GB A100 8-gpu nodes.


* 145B meg-lm

```
NHIDDEN=12288; NLAYERS=80; SEQ_LEN=2048; VOCAB_SIZE=50257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
Model size: 146B, ratio=153
```

```
NHIDDEN=12288; VOCAB_SIZE=50257; TP=8; python -c "h=$NHIDDEN; v=$VOCAB_SIZE; tp=$TP; emb=v*h/10**6; blk=(12*h**2+13*h)/10**6; print(f'emb size: {emb:.2f}M/{emb*18:.2f}GB, per gpu {emb/tp:.2f}M/{emb*18/tp:.2f}GB'); print(f'blk size: {blk:.2f}M/{blk*18:.2f}GB, per gpu {blk/tp:.2f}M/{blk*18/tp:.2f}GB')"
emb size: 617.56M/11116.04GB, per gpu 77.19M/1389.51GB
blk size: 1812.10M/32617.78GB, per gpu 226.51M/4077.22GB
```

MP=64: TP=8, PP=8: one replica 64 gpus

so 80/8=10 PP stages per gpu: `10*4` =40GB of weights/optim states/grads per gpu


* 310B meg-lm

```
NHIDDEN=16384; NLAYERS=96; SEQ_LEN=2048; VOCAB_SIZE=50257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
Model size: 310B, ratio=170
```

MP=128: TP=8, PP=16: one replica 128 gpus

```
NHIDDEN=16384; VOCAB_SIZE=50257; TP=8; python -c "h=$NHIDDEN; v=$VOCAB_SIZE; tp=$TP; emb=v*h/10**6; blk=(12*h**2+13*h)/10**6; print(f'emb size: {emb:.2f}M/{emb*18:.2f}GB, per gpu {emb/tp:.2f}M/{emb*18/tp:.2f}GB'); print(f'blk size: {blk:.2f}M/{blk*18:.2f}GB, per gpu {blk/tp:.2f}M/{blk*18/tp:.2f}GB')"
emb size: 823.41M/14821.39GB, per gpu 102.93M/1852.67GB
blk size: 3221.44M/57985.89GB, per gpu 402.68M/7248.24GB
```

so `96/16=6` PP stages per gpu: `6*7.3` ~44GB of weights/optim states/grads per gpu

* 530B msft


```
NHIDDEN=20480; NLAYERS=105; SEQ_LEN=2048; VOCAB_SIZE=50257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
Model size: 310B, ratio=170
```


MP=280: TP=8, PP=35: one replica 280 gpus

(actually don't know the vocab size here, but it doesn't matter much)

```
NHIDDEN=20480; VOCAB_SIZE=50257; TP=8; python -c "h=$NHIDDEN; v=$VOCAB_SIZE; tp=$TP; emb=v*h/10**6; blk=(12*h**2+13*h)/10**6; print(f'emb size: {emb:.2f}M/{emb*18:.2f}GB, per gpu {emb/tp:.2f}M/{emb*18/tp:.2f}GB'); print(f'blk size: {blk:.2f}M/{blk*18:.2f}GB, per gpu {blk/tp:.2f}M/{blk*18/tp:.2f}GB')"
emb size: 1029.26M/18526.74GB, per gpu 128.66M/2315.84GB
blk size: 5033.43M/90601.76GB, per gpu 629.18M/11325.22GB
```

so 105/35=3 PP stages per gpu: `6*7.3` = ~33.9GB of weights/optim states/grads per gpu


To summarize we can see the setup is so that about half the gpu is loaded with weights / optim states / grad `*18`)

## Possible 200B models


So first let's try to come up with wider and shallower model to fit 200B, or wide if shallow doesn't work out too well topology/efficiency-wise


### 199B: 80 x 14336 (layers x hidden)

```
NHIDDEN=14336; NLAYERS=80; SEQ_LEN=2048; VOCAB_SIZE=150257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
Model size: 199B, ratio=179
```

which gives us:

```
NHIDDEN=14336; VOCAB_SIZE=150257;  TP=8; python -c "h=$NHIDDEN; v=$VOCAB_SIZE; tp=$TP; emb=v*h/10**6; blk=(12*h**2+13*h)/10**6; print(f'emb size: {emb:.2f}M/{emb*18:.2f}GB, per gpu {emb/tp:.2f}M/{emb*18/tp:.2f}GB'); print(f'blk size: {blk:.2f}M/{blk*18:.2f}GB, per gpu {blk/tp:.2f}M/{blk*18/tp:.2f}GB')"
emb size: 2154.08M/38773.52GB, per gpu 269.26M/4846.69GB
blk size: 2466.44M/44395.87GB, per gpu 308.30M/5549.48GB
```

TP=8, PP=10 - 80 gpus for one replica, can fit DP=4 (320/384)

so with PP=10, we get 80/10 = 8 stages per gpu = 44GB for normal layer gpus and 50GB for the 1st/last gpus due to 5G embedding, the remaining 28GB for activations (2GB is cuda kernels) - could be enough, but not sure.

If we are tight, consider giving the embedding its own layer so the total layers will be NLAYERS+2. In which case we need to change NLAYERS to be -2 than the wanted number to be able to spread out the layers evenly across gpus.

Also consider that the more tightly we pack each gpu the more PP stages it'll have - the slower it'll run.

And less GPUs means less processing power - so overall it's likely to be slower.

### 206B: 96 x 13312 (layers x hidden)

```
NHIDDEN=13312; NLAYERS=96; SEQ_LEN=2048; VOCAB_SIZE=150257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
Model size: 206B, ratio=138
```

```
NHIDDEN=13312; VOCAB_SIZE=150257; TP=8; python -c "h=$NHIDDEN; v=$VOCAB_SIZE; tp=$TP; emb=v*h/10**6; blk=(12*h**2+13*h)/10**6; print(f'emb size: {emb:.2f}M/{emb*18:.2f}GB, per gpu {emb/tp:.2f}M/{emb*18/tp:.2f}GB'); print(f'blk size: {blk:.2f}M/{blk*18:.2f}GB, per gpu {blk/tp:.2f}M/{blk*18/tp:.2f}GB')"
emb size: 2000.22M/36003.98GB, per gpu 250.03M/4500.50GB
blk size: 2126.69M/38280.33GB, per gpu 265.84M/4785.04GB
```

TP=8, PP=12 => 96 gpus for one replica, can fit DP=4 (384/384)

96/12 = 8 stages per gpu = ~40GB per gpu, same number of PP stages per gpu and more spare memory

This might be a better fit memory-wise if the one above is too close to being full, especially on gpu 0 and -1.

It also uses the full 384 gpu allocation in a snag way.



## Train time estimation

So A100 spec is 312 TFLOPS for BF16, so probably the best would be 50% of that so 150 TFLOPs (which we probably won't reach, but let's see), so yes I agree 150 is a bit too optimistic, but let's use it as the best case scenario.


Also we still don't know how many gpus we will end up using, but let's say we use them all - 350. Once we decide on the topology we will be able to replace 350 with the actual number of gpus we plan to use.

```
$ python -c 'print(f"{8*300*200_000_000/(350*150)/(3600*24):0.2f}", "days")'
105.82 days
```

so 3.5 months in the best case scenario. But more likely 150-200 days since it'll be less of everything plus potential issues. We will know more once we get access to 1 replica as then we should get a much better TFLOPs estimation, which will then be less for DP>1.

And this estimate is w/o encountering any problems, which is unlikely, so add more overhead for rollbacks and restarts.

Additionally this number is too optimistic since we won't have the full number of GPUs till about some time in end of February.

See [Estimate total training time](../../math#estimate-total-training-time) for details of the math.

XXX: actually are we training for 300B or 400B tokens because of Multi-Lingual? in which case it'll be 1/3 longer!


## Allocated hours sufficiency check

We currently have about 3M gpu hours left in our allocation.

Let's see how many total gpus hours the good estimation is:


```
python -c 'print(f"{8*300*200_000_000/150/3600:0.2f}", "compute hours")'
888888.89 compute hours
```
So if it takes 2x longer than the best case scenario, then we say need about 2M hours, so we are fine there.

Important nuance:

We will have an exclusive access only till May, and in May we will have to share with others.

So at the moment we will have only about 3 months of having access to all gpus.



## Best TFLOPs

To measure best TFLOPs possible use a single, so that it uses all the intra-node connections (NVLink) and doesn't touch the network:

### fp16

- 1 node, 1 replica

20B model: TP=8, PP=1, NLAYERS=8, NHIDDEN=14400, NHEADS=32, SEQ_LEN=2048, VOCAB_LENGTH=250k, GBS=2048

```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 769.99 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.384045E+01 | loss scale: 4096.0 | grad norm: 15906.210 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 2.660 | TFLOPs: 108.47 |
```

- 10 nodes, 1 replica

200B model: TP=8, PP=10, NLAYERS=80, NHIDDEN=14400, NHEADS=96, SEQ_LEN=2048, VOCAB_LENGTH=250k, GBS=2048

```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 844.81 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.373861E+01 | loss scale: 4096.0 | grad norm: 34132.119 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 2.424 | TFLOPs: 98.87 |
```

- 20 nodes, 2 replicas

```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 430.21 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.373876E+01 | loss scale: 4096.0 | grad norm: 34027.311 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.761 | TFLOPs: 97.07 |
```

It was puzzling why much less memory was used for identical set up with DP=2 over DP=1 - but it's because of ZeRO-1 that saves a lot of memory across all GPUs!


| GPUs | Size | DP | TP | PP | MBS | Mem  | TFLOPs | Notes |
| ---: | ---: | -: | -: | -: | --: | ---: | -----: | ----: |
|    8 | 20B  |  1 |  8 |  1 |   1 | 67GB | 108.47 | 02-17 |
|   80 | 200B |  1 |  8 | 10 |   1 | 73GB |  98.87 | 02-17 |
|  160 | 200B |  2 |  8 | 10 |   1 | 51GB |  97.07 | 02-17 |
|      |      |    |    |    |     |      |        |       |

*Mem = max memory used by the first (last) nodes with the word embedding matrix - max is 77GB


### bf16

- 1 node, 1 replica

20B model: TP=8, PP=1, NLAYERS=8, NHIDDEN=14400, NHEADS=32, SEQ_LEN=2048, VOCAB_LENGTH=250k, GBS=2048

```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 777.09 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.381926E+01 | loss scale: 1.0 | grad norm: 2.763 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 2.635 | TFLOPs: 107.48 |
```


- 10 nodes, 1 replica

200B model: TP=8, PP=10, NLAYERS=80, NHIDDEN=14400, NHEADS=96, SEQ_LEN=2048, VOCAB_LENGTH=250k, GBS=2048

```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 853.81 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.369443E+01 | loss scale: 1.0 | grad norm: 4.461 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 2.399 | TFLOPs: 97.82 |
```


- 20 nodes, 2 replicas


```
 iteration        2/   95367 | consumed samples:         4096 | consumed tokens:      8388608 | elapsed time per iteration (s): 434.14 | learning rate: 3.787E-06 | global batch size:  2048 | lm loss: 6.369444E+01 | loss scale: 1.0 | grad norm: 6.314 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.717 | TFLOPs: 96.19 |
```


| GPUs | Size | DP | TP | PP | MBS | Mem  | TFLOPs | Notes |
| ---: | ---: | -: | -: | -: | --: | ---: | -----: | ----: |
|    8 | 20B  |  1 |  8 |  1 |   1 | 68GB | 107.48 | 02-17 |
|   80 | 200B |  1 |  8 | 10 |   1 | 75GB |  97.82 | 02-17 |
|  160 | 200B |  2 |  8 | 10 |   1 | 53GB |  96.19 | 02-17 |
|      |      |    |    |    |     |      |        |       |

*Mem = max memory used by the first (last) nodes with the word embedding matrix - max is 77GB

So we can load more stages as we get higher DP as ZeRO spreads out over more gpus - smaller shards.



## dealing with JZ hanging on the large model

This overcomes the hanging which in general should lead to a slower throughput since all CUDA operations become synchronous and would block until they are done.

```
export CUDA_LAUNCH_BLOCKING=1
```

200B, measuring 2nd iter:

| GPUs | async |  GBS | TFLOPs | Notes        |
| ---: | ----: | ---: | -----: | -----------: |
|   80 | no    |  512 |  91.04 |              |
|   80 | yes   |  512 |  97.20 |              |
|  160 | no    |  512 |  84.59 |              |
|  160 | yes   |  512 |  84.44 |              |
|  160 | no    | 2048 |  90.29 |              |
|  160 | yes   | 2048 |  90.25 | may hang     |
|  320 | no    | 2048 |  87.78 |              |
|  320 | yes   | 2048 |   xxxx | always hangs |
|      |       |      |        |              |

async/yes == `CUDA_LAUNCH_BLOCKING=0`

Interesting. Sometimes `CUDA_LAUNCH_BLOCKING=1` impacts the speed, at other times it doesn't. Perhaps with larger set ups it's barely impacting since there is a lot more comms than the small setup.


## Choosing the fastest 3D Topology

Benchmarking the fastest 3D topology. Constraint: can use at most 48 nodes of 8 gpu a100 80gb nodes.

Note that we want not the highest TFLOPs but the highest speed per iteration, since one can get high TFLOPs on less GPUs and overall slower speed, since we only care about how fast we can finish the training.

Also note that the model size isn't always the same as the number of layers had to be tweaked to fit PP and NHIDDEN was fixed - so speed/tflops can't be exactly compared - but can be brought back to the same size by tweaking NHIDDEN. also since for efficiency of finishing this process I take the snapshot of a single iteration (always 2nd) the data isn't exact and can fluctuate a bit. But the point of this exercise is to get a feel of which topology is superior.


| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 200B | 12 |  8 |  4 |   1 | 2040 | 47GB | 189.06 |  91.67 | 02-20 |
|    45 | 200B |  9 |  8 |  5 |   1 | 2043 | 44GB | 208.40 |  88.84 | 02-20 |
|    48 | 194B |  8 |  8 |  6 |   1 | 2048 | 39GB | 183.64 |  92.38 | 02-20 |
|    42 | 191B |  6 |  8 |  7 |   1 | 2046 | 39GB | 202.99 |  94.20 | 02-20 |
|    48 | 200B |  6 |  8 |  8 |   1 | 2046 | 36GB | 185.75 |  93.59 | 02-20 |
|    45 | 205B |  5 |  8 |  9 |   1 | 2045 | 37GB | 199.14 |  94.23 | 02-20 |
|    40 | 200B |  4 |  8 | 10 |   1 | 2048 | 35GB | 221.21 |  94.39 | 02-20 |
|    44 | 195B |  4 |  8 | 11 |   1 | 2048 | 32GB | 197.15 |  92.67 | 02-20 |
|    48 | 183B |  4 |  8 | 12 |   1 | 2048 | 30GB | 172.40 |  90.84 | 02-20 |
|       |      |    |    |    |     |      |      |        |        |       |

* Sec/it throughput at iteration 2

As you can see the 80GB is totally unnecessary for MBS=1 as we are bound by compute of each gpu and we barely use half the gpu memory and trying to pack more on each gpu slows the ensemble down. This is of course thanks to ZeRO which shards all fp32 optim+grad+params over all gpus - so the more gpus you use the less memory is needed to accommodate the same model size, regardless of DP/TP/PP topology. (with MBS=1 that is so that the activations don't take too much memory)

This table doesn't take into account batch size rampup which needs to be divisible by DP as it progressed from 32, 64, ... so really we have an additional constraint of `DP % 4 = 0` and `GBS % 32 = 0`.

which means from the above list only a few configs are suitable, and these are:

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 194B |  8 |  8 |  6 |   1 | 2048 | 39GB | 183.64 |  92.38 | 02-20 |
|    40 | 200B |  4 |  8 | 10 |   1 | 2048 | 35GB | 221.21 |  94.39 | 02-20 |
|    44 | 195B |  4 |  8 | 11 |   1 | 2048 | 32GB | 197.15 |  92.67 | 02-20 |
|    48 | 183B |  4 |  8 | 12 |   1 | 2048 | 30GB | 172.40 |  90.84 | 02-20 |
|       |      |    |    |    |     |      |      |        |        |       |

Increasing MBS will speed up things a bit and we have a ton of spare memory to accommodate a larger MBS, but have to ensure we get the batch size ramp up sorted out. So if the rampup steps are in increments of 32 with DP=4 highest MBS is 8. and `log2(MBS) % 2 = 0`.


| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 194B |  8 |  8 |  6 |   1 | 2048 | 39GB | 183.64 |  92.38 | 02-20 |
|    48 | 194B |  8 |  8 |  6 |   2 | 2048 | 45GB | 172.36 |  98.43 | 02-20 |
|    48 | 194B |  8 |  8 |  6 |   4 | 2048 | 56GB | 173.92 |  97.55 | 02-20 |
|    48 | 194B |  8 |  8 |  6 |   8 | 2048 | 75GB | 192.42 |  88.17 | 02-20 |
|       |      |    |    |    |     |      |      |        |        |       |


| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs |                  Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ---------------------: |
|    40 | 200B |  4 |  8 | 10 |   1 | 2048 | 35GB | 221.21 |  94.39 |                  02-20 |
|    40 | 200B |  4 |  8 | 10 |   2 | 2048 | 43GB | 207.92 | 100.43 |                  02-20 |
|    40 | 200B |  4 |  8 | 10 |   4 | 2048 | 55GB | 208.18 | 100.30 |                  02-20 |
|    40 | 200B |  4 |  8 | 10 |   8 | 2048 | 76GB | 229.69 |  90.91 | 02-20 too close to OOM |
|       |      |    |    |    |     |      |      |        |        |                        |


| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    44 | 195B |  4 |  8 | 11 |   1 | 2048 | 32GB | 197.15 |  92.67 | 02-20 |
|    44 | 195B |  4 |  8 | 11 |   2 | 2048 | 41GB | 186.65 |  97.89 | 02-20 |
|    44 | 195B |  4 |  8 | 11 |   4 | 2048 | 53GB | 185.79 |  98.34 | 02-20 |
|    44 | 195B |  4 |  8 | 11 |   8 | 2048 | 75GB | 206.62 |  88.42 | 02-20 |
|       |      |    |    |    |     |      |      |        |        |       |


| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 183B |  4 |  8 | 12 |   1 | 2048 | 30GB | 172.40 |  90.84 | 02-20 |
|    48 | 183B |  4 |  8 | 12 |   2 | 2048 | 39GB | 161.96 |  96.69 | 02-20 |
|    48 | 183B |  4 |  8 | 12 |   4 | 2048 | 50GB | 163.32 |  95.89 | 02-20 |
|       |      |    |    |    |     |      |      |        |        |       |

The models are slightly different in size so can't compare absolute numbers.

But clearly MBS=2 is about the best, MBS=4 is close by.

If we utilize all 48 nodes then we have PP6 and PP12 as contenders.


## tile and wave quantization


A100 80GB has 108 SMs

https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#tile-quant

```
nhidden % 128 = 0
```

https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant

```
nhidden % 108 = 0
```

TP=8:

```
nhidden % 8 = 0
```

Combining all 3:

```
nhidden = 108*8*c = 864*c
```

which gives 864*16 = 13824 (187B) => so let's try to compare with 14400 (200B)

XXX: This is a total guestimate - need proper math

| Nodes | Size | NHIDDEN | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: |    ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    40 | 200B |   14400 |  4 |  8 | 10 |   1 | 2048 | 35GB | 221.21 |  94.39 | 02-20 |
|    40 | 187B |   13824 |  4 |  8 | 10 |   1 | 2048 | 33GB | 160.29 | 120.05 | 02-20 |
|    40 | 187B |   13824 |  4 |  8 | 10 |   2 | 2048 | 39GB | 151.07 | 127.38 | 02-20 |
|    40 | 187B |   13824 |  4 |  8 | 10 |   4 | 2048 | 53GB | 147.43 | 130.53 | 02-20 |
|    40 | 187B |   13824 |  4 |  8 | 10 |   8 | 2048 | 73GB | 152.51 | 126.18 | 02-20 |
|       |      |         |    |    |    |     |      |      |        |        |       |


## TFLOPs calculation improved

Until now we used an estimated TFLOPs calculator which was under-reporting the real TFLOPs. And we couldn't compare those to the TFLOPs reported by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM#readme).

Deepak Narayanan fixed this here: https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251

So from here on all the TLOPs reports will be about 3% higher - so can't exactly compare to the earlier numbers in this document.


## 48 node contenders

So we have 2 set ups that fit well into 48 nodes - and that's PP=6/DP=8 or PP=12/DP=4

NHIDDEN=14336 / NLAYERS=72

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 181B |  4 |  8 | 12 |   1 | 2048 | 29GB | 143.31 | 112.49 | 02-21 |
|    48 | 181B |  4 |  8 | 12 |   2 | 2048 | 37GB | 134.02 | 120.29 | 02-21 |
|    48 | 181B |  4 |  8 | 12 |   4 | 2048 | 49GB | 123.69 | 130.34 | 02-21 |
|    48 | 181B |  4 |  8 | 12 |   8 | 2048 | 69GB | 129.26 | 124.72 | 02-21 |
|       |      |    |    |    |     |      |      |        |        |       |

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 181B |  8 |  8 |  6 |   1 | 2048 | 38GB | 139.82 | 115.31 | 02-21 |
|    48 | 181B |  8 |  8 |  6 |   2 | 2048 | 44GB | 131.02 | 123.05 | 02-21 |
|    48 | 181B |  8 |  8 |  6 |   4 | 2048 | 56GB | 121.48 | 132.71 | 02-21 |
|       |      |    |    |    |     |      |      |        |        |       |


So it's either:

* DP=4, PP=12, MBS=4: 123 secs/it | 130 TFLOPS
* DP=8, PP=06, MBS=4: 121 secs/it | 133 TFLOPS

Let's compare again with another setup:

NHIDDEN=13824 / NLAYERS=84

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 196B |  4 |  8 | 12 |   2 | 2048 | 39GB | 143.89 | 121.45 | 02-21 |
|    48 | 196B |  4 |  8 | 12 |   4 | 2048 | 52GB | 133.12 | 131.27 | 02-21 |
|    48 | 196B |  8 |  8 |  6 |   2 | 2048 | 65GB | 141.41 | 123.58 | 02-21 |
|    48 | 196B |  8 |  8 |  6 |   4 | 2048 | 56GB | 130.31 | 134.11 | 02-21 |
|       |      |    |    |    |     |      |      |        |        |       |

This one has 15% more layers than the previous tables, so here the less-PP-stages setup wins, that is:

* DP=8, PP=06, MBS=4: 130.31 secs/it | 134.11 TFLOPS

The following has so far given the highest TFLOPs, as we are packing more into less GPUs so 64 gpus are left out, and of course the total speed for iteration is much slower. So the key is the iteration speed and not TFLOPs.

NHIDDEN=13824 / NLAYERS=80

| Nodes | Size | DP | TP | PP | MBS | GBS  | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | --:  | ---: | -----: | -----: | ----: |
|    40 | 187B | 8  | 8  | 10 | 4   | 2048 | GB   | 147.04 | 135.92 | 02-21 |
|       |      |    |    |    |     |      |      |        |        |       |


Max possible TFLOPs check for `NHIDDEN=14336`:

NHIDDEN=14336 / NLAYERS=6 / GBS=512

| Nodes | Size | Layers | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -----: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|     1 | 18B  |      6 |  8 |  1 |   2 | 2048 | 54GB | 130.43 | 143.48 | 02-21 |
|     1 | 18B  |      6 |  8 |  1 |   2 | 2048 | 54GB | 119.19 | 157.02 | 02-21 |
|     1 | 18B  |     10 |  8 |  1 |   1 | 2048 | 80GB | 205.52 | 142.59 | 02-21 |
|       |      |        |    |    |     |      |      |        |        |       |

Trying with ZeRO_STAGE=0/1

NHIDDEN=14336 / NLAYERS=72

| Nodes | Size | ZS | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 181B |  1 |  4 |  8 | 12 |   2 | 2048 | 37GB | 120.29 | 134.02 | 02-21 |
|    48 | 181B |  0 |  4 |  8 | 12 |   2 | 2048 | 72GB | 137.34 | 113.02 | 02-21 |
|       |      |    |    |    |    |     |      |      |        |        |       |

* ZS = ZERO_STAGE

XXX: currently can't test `ZeRO_STAGE=0` on master, or `ZeRO_STAGE=1` on the special branch - so need to retest the above on the same branch.


## Final round comparison

all NHEADS=64 (above too)

NHIDDEN=12288 / NLAYERS=96

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 177B |  8 |  8 |  6 |   2 | 2048 | GB   | 136.73 | 115.73 | 02-23 |
|    48 | 177B |  8 |  8 |  6 |   4 | 2048 | GB   | 122.96 | 128.69 | 02-23 |
|       |      |    |    |    |     |      |      |        |        |       |
|       |      |    |    |    |     |      |      |        |        |       |

NHIDDEN=13312 / NLAYERS=84

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  4 |  8 | 12 |   4 | 2048 | GB   | 125.52 | 129.29 | 02-23 |
|    48 | 182B |  8 |  8 |  6 |   2 | 2048 | GB   | 135.55 | 119.72 | 02-23 |
|    48 | 182B |  8 |  8 |  6 |   4 | 2048 | GB   | 122.93 | 132.00 | 02-23 |
|       |      |    |    |    |     |      |      |        |        |       |

NHIDDEN=13824 / NLAYERS=78

| Nodes | Size | DP | TP | PP | MBS | GBS  | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | --:  | ---: | -----: | -----: | ----: |
| 48    | 182B | 8  | 8  | 6  | 4   | 2048 | GB   | 121.28 | 133.93 | 02-23 |
|       |      |    |    |    |     |      |      |        |        |       |

NHIDDEN=14336 / NLAYERS=72

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: |  --: | ---: | -----: | -----: | ----: |
|    48 | 181B |  4 |  8 | 12 |   4 | 2048 | GB   | 123.79 | 130.24 | 02-23 |
|    48 | 181B |  8 |  8 |  6 |   4 | 2048 | GB   | 120.85 | 133.40 | 02-23 |
|       |      |    |    |    |     |      |      |        |        |       |


## NHEADs comparison

NHIDDEN=14336 / NLAYERS=72

not many variations around 100 as `14336 = 2**11*7` and the constraint is `(HEADS/TP)*MBS % 4 = 0` or for `MBS=4, TP=8` `HEADS % 16 = 0`

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 181B |  8 |  8 |  6 |   4 |     16 | 2048 | 54GB | 121.03 | 133.20 | 02-24 |
|    48 | 181B |  8 |  8 |  6 |   4 |     32 | 2048 | 55GB | 124.01 | 130.00 | 02-23 |
|    48 | 181B |  8 |  8 |  6 |   4 |     64 | 2048 | 55GB | 120.18 | 134.15 | 02-23 |
|    48 | 181B |  8 |  8 |  6 |   4 |    112 | 2048 | 53GB | 138.72 | 116.21 | 02-23 |
|    48 | 181B |  8 |  8 |  6 |   4 |    128 | 2048 | 55GB | 124.89 | 129.08 | 02-23 |
|    48 | 181B |  8 |  8 |  6 |   4 |    256 | 2048 | 54GB | 132.85 | 121.35 | 02-24 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=13824 / NLAYERS=78

here `13824 = 2**9*3**3`

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  8 |  6 |   4 |     64 | 2048 | GB   | 121.28 | 133.93 | 02-23 |
|    48 | 182B |  8 |  8 |  6 |   4 |     96 | 2048 | 59GB | 124.75 | 130.21 | 02-23 |
|    48 | 182B |  8 |  8 |  6 |   4 |    128 | 2048 | 54GB | 162.72 |  99.82 | 02-23 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHEADS=108 breaks constraints for invoking optimized fused softmax kernel


NHIDDEN=13312 / NLAYERS=84

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  8 |  6 |   4 |     64 | 2048 | GB   | 122.93 | 132.00 | 02-23 |
|    48 | 182B |  8 |  8 |  6 |   4 |    128 | 2048 | GB   | 129.17 | 125.63 | 02-23 |
|       |      |    |    |    |     |        |      |      |        |        |       |


NHIDDEN=12288 / NLAYERS=96

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 177B |  8 |  8 |  6 |   4 |     64 | 2048 | GB   | 122.96 | 128.69 | 02-24 |
|    48 | 177B |  8 |  8 |  6 |   4 |     96 | 2048 | GB   | 145.40 | 108.83 | 02-24 |
|    48 | 177B |  8 |  8 |  6 |   4 |    128 | 2048 | GB   | 129.42 | 122.27 | 02-24 |
|       |      |    |    |    |     |        |      |      |        |        |       |


## GBS Variations

Note: A100s PCI-Express/NUMA was improved today so all TFLOPs have changed for the better (1-5%) - thus do not compare today's numbers to yesterday's.

NLAYERS=72
NHIDDEN=14336
NHEADS=64

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | ---: | ---: | -----: | -----: | ----: |
|    48 | 181B |  8 |  8 |  6 |   4 | 1568 | 56GB | 113.01 | 109.22 | 02-25 |
|    48 | 181B |  8 |  8 |  6 |   4 | 2048 | 55GB | 114.11 | 141.27 | 02-25 |
|    48 | 181B |  8 |  8 |  6 |   6 | 2016 | 66GB | 123.57 | 128.43 | 02-25 |
|    48 | 181B |  4 |  8 | 12 |   4 | 1568 | GB   |  92.75 | 133.08 | 02-25 |
|    48 | 181B |  4 |  8 | 12 |   4 | 2048 | 49GB | 117.07 | 137.70 | 02-25 |
|    48 | 181B |  4 |  8 | 12 |   2 | 1568 | GB   |  99.93 | 123.51 | 02-25 |
|    48 | 181B |  4 |  8 | 12 |   2 | 2048 | GB   | 128.82 | 125.15 | 02-25 |
|       |      |    |    |    |     |      |      |        |        |       |

some more configs with lower PP:

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | ---: | ---: | -----: | -----: | ----: |
|    48 | 181B |  6 |  8 |  8 |   4 | 2016 | 52GB | 113.16 | 140.24 | 02-25 |
|    48 | 181B | 12 |  8 |  4 |   2 | 2016 | 53GB | 125.52 | 126.43 | 02-25 |
|    48 | 181B | 12 |  8 |  4 |   4 | 2016 | 59GB | 114.81 | 138.22 | 02-25 |
|    48 | 181B | 24 |  8 |  2 |   1 | 2016 | 65GB | 145.45 | 109.11 | 02-25 |
|    48 | 181B | 24 |  8 |  2 |   2 | 2016 | 76GB | 136.13 | 116.58 | 02-25 |
|    48 | 181B | 48 |  8 |  1 |   1 | 2016 | OOM  |        |        | 02-25 |
|       |      |    |    |    |     |      |      |        |        |       |

Tweaking TP for the first time from the TP=8 is best assumption. But if the model fits into smaller TP it should be faster!

| Nodes | Size | DP | TP | PP | MBS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | ---: | ---: | -----: | -----: | ----: |
|    48 | 181B |  8 |  4 | 12 |   4 | 2048 | 60GB | 111.89 | 144.08 | 02-25 |
|    48 | 181B |  8 |  4 | 12 |   2 | 2048 | 44GB | 110.48 | 145.92 | 02-25 |
|    48 | 181B |  8 |  4 | 12 |   2 | 2048 | 38GB | 113.54 | 141.99 | 02-25 |
|    48 | 181B | 16 |  4 |  6 |   4 | 2048 | 75GB | 117.11 | 137.66 | 02-25 |
|    48 | 181B | 16 |  4 |  6 |   2 | 2048 | 57GB | 111.71 | 144.31 | 02-25 |
|    48 | 181B | 16 |  2 | 12 |   2 | 2048 | 63GB | 112.50 | 143.30 | 02-25 |
|    48 | 181B | 32 |  2 |  6 |   2 | 2048 | OOM  |        |        | 02-25 |
|    48 | 181B | 32 |  2 |  6 |   1 | 2048 | OOM  |        |        | 02-25 |
|    48 | 181B |  8 |  2 | 24 |   1 | 2048 | 44GB | 119.53 | 134.88 | 02-25 |
|    48 | 181B |  8 |  2 | 24 |   2 | 2048 | 53GB | 122.75 | 131.33 | 02-25 |
|    48 | 181B |  4 |  4 | 24 |   1 | 2048 | GB   | 130.60 | 123.44 | 02-25 |
|       |      |    |    |    |     |      |      |        |        |       |


NHIDDEN=12288 / NLAYERS=96

| Nodes | Size | DP | TP | PP | MBS | GBS  | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | ---: | ---: | -----: | -----: | ----: |
| 48    | 177B | 8  | 1  | 48 | 1   | 2048 | 58GB | 142.17 | 111.30 | 02-25 |
|       |      |    |    |    |     |      |      |        |        |       |


## Another round of NHEADS

to retest with TP<8 variations

NHIDDEN=13824 / NLAYERS=78

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  4 | 12 |   1 |     64 | 2048 |      | 148.24 | 109.57 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |     64 | 2048 | 48GB | 103.51 | 156.92 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |     96 | 2048 | 48GB | 107.12 | 151.64 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |    128 | 2048 |      | 147.41 | 110.19 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   4 |     64 | 2048 |      | 106.72 | 152.21 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   4 |     96 | 2048 |      | 110.31 | 147.25 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   4 |    128 | 2048 |      | 153.90 | 105.54 | 02-26 |
|    48 | 182B |  8 |  8 |  6 |   4 |     96 | 2048 |      | 118.12 | 137.51 | 02-26 |
|    48 | 182B |  8 |  8 |  6 |   4 |    128 | 2048 |      | 156.84 | 103.56 | 02-26 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=14336 / NLAYERS=72

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: | ---: | ---: | -----: | -----: | ----: |
|    48 | 181B |  8 |  4 | 12 |   2 |     64 | 2048 |      | 110.42 | 146.00 | 02-26 |
|    48 | 181B |  8 |  4 | 12 |   2 |    128 | 2048 |      | 114.02 | 141.39 | 02-26 |
|    48 | 181B |  8 |  4 | 12 |   4 |    128 | 2048 |      | 137.53 | 117.23 | 02-26 |
|    48 | 181B |  8 |  8 |  6 |   4 |     64 | 2048 |      | 113.95 | 141.47 | 02-26 |
|    48 | 181B |  8 |  8 |  6 |   4 |    128 | 2048 |      | 116.06 | 138.90 | 02-26 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=13312 / NLAYERS=84

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: | ---: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  4 | 12 |   2 |     64 | 2048 |      | 103.82 | 156.46 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   4 |     64 | 2048 |      | 113.21 | 143.34 | 02-26 |
|    48 | 182B |  8 |  8 |  6 |   2 |     64 | 2048 |      | 129.61 | 125.21 | 02-26 |
|       |      |    |    |    |     |        |      |      |        |        |       |

## Batchsize Warmup

NHIDDEN=13824 / NLAYERS=78

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  4 | 12 |   2 |     96 |  512 |      |  35.77 | 113.52 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |     96 | 1024 |      |  59.65 | 136.15 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |     96 | 1536 |      |  83.11 | 146.59 | 02-26 |
|    48 | 182B |  8 |  4 | 12 |   2 |     96 | 2048 |      | 107.12 | 151.64 | 02-26 |
|       |      |    |    |    |     |        |      |      |        |        |       |

## Re-do

78/12=6.5 - so the last stage has 1 block, while the rest have 7 - which is uneven. So that config is not optimal as it wastes gpus.

NHIDDEN=13824 / NLAYERS=78

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 182B |  8 |  8 |  6 |   2 |     96 | 2048 | GB   | 133.57 | 121.61 | 02-27 |
|    48 | 182B |  8 |  8 |  6 |   4 |     96 | 2048 | 59GB | 118.24 | 137.38 | 02-27 |
|    48 | 182B | 16 |  4 |  6 |   2 |     96 | 2048 | GB   |        |        | 02-27 |
|    48 | 182B | 16 |  4 |  6 |   4 |     96 | 2048 | 75GB | 115.55 | 140.57 | 02-27 |
|       |      |    |    |    |     |        |      |      |        |        |       |

HIDDEN=12288; NLAYERS=106; regex partition_method='type:transformer|embed')

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 195B |  8 |  4 | 12 |   2 |     96 | 2048 | 44GB | 112.69 | 154.86 | 02-27 |
|    48 | 195B |  8 |  4 | 12 |   2 |     64 | 2048 | GB   | 110.96 | 157.27 | 02-27 |
|       |      |    |    |    |     |        |      |      |        |        |       |

## Rebalancing layers

Do not compare these numbers to the previous ones. For 2 reasons:

- First, from now on the testing is happening with BF16 optimizer that was just written to accumulate gradients in fp32, so it is more memory heavy and is a bit slower - this is compared to fp16 which grad accumulates in fp16. The additional memory usage is 4bytes x params and it's not sharded across gpus.
- I implemented and enabled `--pp-partition-method 'type:transformer|embedding'` so we use 2 layers less, to match `2+nlayers*PP` math to get a perfect balance giving each embedding layer its own slot on par with transformer layers. This is because 250k embedding matrix takes as much space as a single transformer layer.

HIDDEN=12288; NLAYERS=106; Model size: 195B, ratio=115

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 195B |  8 |  4 | 12 |   2 |     64 | 2048 | 67GB | 116.54 | 149.75 | 02-28 |
|    48 | 195B |  8 |  4 | 12 |   2 |     96 | 2048 | 65GB | 118.79 | 146.90 | 02-28 |
|    48 | 195B |  8 |  4 | 12 |   2 |    128 | 2048 | 67GB | 121.42 | 143.73 | 02-28 |
|    48 | 195B |  8 |  4 | 12 |   4 |     96 | 2048 | 79GB | 120.34 | 145.01 | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |


HIDDEN=12288; NLAYERS=100; Model size: 184B, ratio=122

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 184B | 16 |  4 |  6 |   2 |     64 | 2048 | OOM  | x      | x      | 02-28 |
|    48 | 184B | 16 |  4 |  6 |   1 |     64 | 2048 | OOM  | x      | x      | 02-28 |
|    48 | 184B |  8 |  8 |  6 |   2 |     64 | 2048 | 61GB | 139.72 | 117.91 | 02-28 |
|    48 | 184B |  8 |  8 |  6 |   4 |     64 | 2048 | 72GB | 120.96 | 136.20 | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |


NHIDDEN=13312; NLAYERS=82; Model size: 178B, ratio=162

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 178B |  4 |  8 | 12 |   4 |     64 | 2048 | 52GB | 111.79 | 141.76 | 02-28 |
|    48 | 178B |  8 |  4 | 12 |   2 |     64 | 2048 | 63GB | 104.45 | 151.71 | 02-28 |
|    48 | 178B |  8 |  4 | 12 |   2 |    104 | 2048 | 62GB | 123.71 | 128.10 | 02-28 |
|    48 | 178B |  8 |  4 | 12 |   2 |    128 | 2048 | 60GB | 108.78 | 145.68 | 02-28 |
|    48 | 178B |  8 |  4 | 12 |   4 |     64 | 2048 | 74GB | 104.82 | 151.18 | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=13312; NLAYERS=94 Model size: 203B, ratio=141

| Nodes | Size | DP | TP | PP | MBS | NHEADS | GBS  | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: | --:  | ---: | -----: | -----: | ----: |
| 48    | 203B | 8  | 4  | 12 | 2   | 128    | 2048 | 67GB | 124.10 | 146.12 | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=14336; NLAYERS=70; Model size: 176B, ratio=204

| Nodes | Size | DP | TP | PP | MBS | NHEADS |  GBS | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: |  --: | ---: | -----: | -----: | ----: |
|    48 | 176B |  4 |  8 | 12 |   2 |     64 | 2048 | 40GB | 121.63 | 128.92 | 02-28 |
|    48 | 176B |  8 |  4 | 12 |   2 |     64 | 2048 | 59GB | 102.03 | 153.68 | 02-28 |
|    48 | 176B |  8 |  4 | 12 |   2 |    112 | 2048 | 59GB | 104.50 | 150.05 | 02-28 |
|    48 | 176B |  8 |  4 | 12 |   2 |    128 | 2048 | 60GB | 105.89 | 148.08 | 02-28 |
|    48 | 176B |  8 |  4 | 12 |   4 |     64 | 2048 | 73GB | 102.27 | 153.33 | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |

NHIDDEN=14336; NLAYERS=82; Model size: 206B, ratio=174

| Nodes | Size | DP | TP | PP | MBS | NHEADS | GBS  | Mem  | Sec/it | TFLOPs | Notes |
| ----: | ---: | -: | -: | -: | --: | -----: | --:  | ---: | -----: | -----: | ----: |
| 48    | 206B | 8  | 4  | 12 | 2   | 128    | 2048 | OOM  |        |        | 02-28 |
|       |      |    |    |    |     |        |      |      |        |        |       |



(was quickly getting the memory snapshot with: `pdsh -w jean-zay-iam01 "source ~/.pdshrc; nvidia-smi"`)


## Hanging Issue

Here we are dealing with 320-384 A100 GPUs working in ensemble.

It appears that the system can't handle heavy NCCL traffic or something of sorts. It can handle less than 100B model over 40nodes (TP=8/PP=10/DP=4). It can handle 200B over 10 nodes. At 100B over 20-40 nodes random GPUs start not to respond and the whole system hangs until it times out. I was able to test with the same NHIDDEN and growing the model on the layer dimension:

- 10 layers - 25B works
- 20 layers - 50B works
- 40 layers - 100B hangs after succeeding iteration 1

I was just starting to diagnose on the hidden dimension and now 13/52 nodes are down and so I can't continue with this line of work, since 40 nodes gave me a reliable failure and 20 nodes is intermittent failure, so it's not good for diagnosing.

This is for a single replica of 10 nodes with 200B model + 250k vocab.

I think the failed nodes that crashed and didn't recover are high suspects for having internal problems. Even though when I tested in groups of 10 nodes everything was dandy - note - the same 200B model.
One more data point - Deepspeed ZeRO shards data over all gpus - so the more GPUs are involved the more communication happens. This is totally orthogonal to DP.

The next day:

Most of the nodes have come back this morning so continuing the dimensional growing experiments.
To remind, growing on the layer dimension and keeping hidden at `1024*14` worked until 40 layers were reached where it was hanging. So it couldn't handle 100B model in this dimension.
Now I'm keeping the layers dimension frozen to 80 and growing the nhidden dimension, starting from `1024*4` - proving that it works and then incrementing the size until it hangs:

- `1024*10` works (100B model)
- `1024*12` hangs (145B model)

So these 2 experiments both show that when the inter-node traffic exceeds certain level - the system is fails.

So it's not the size of each `all_reduce`/`broadcast` packet since at full NHIDDEN but only 1/4 of layers everything is just fine.

And BTW to get a quick success/failure indication I'm working with `GLOBAL_BATCH_SIZE=64` so PP is very inefficient, but it doesn't matter for the purpose of this experiment.

Using `py-spy` on the processes to dump python call stacks I have derived the same story on each node:

On each node with TP=8 - i.e. each node is only TP - the same situation: (checked nodes 0 and 1 only)

6 processes are in:

```
Thread 835990 (active): "MainThread"
    train (megatron/training.py:915)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
2 processes are in:
```
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```

so 6 processes finished `train_step` and now are trying to:
```
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
```
but for some reason 2 processes never finished the `train_step` and are stuck broadcasting I presume to the other 6 processes, which have long gone.

So this hanging happens partially in Deepspeed and partially in Megatron-LM, somehow processes get out of sync even though everything works just fine on a smaller scale. But the issue could be brought on by apex's `FusedAdam` as we have dealt with a serious issue in it as well a week earlier, but it could also be pytorch, NCCL or some internal system issue. It's very hard to find the cause.

As I shared earlier the problem doesn't exist or goes away if either of 2 things happens:

- the model is under 100B (short stack of layer or narrow hidden) and 20 or more nodes are used in a single job
- `CUDA_LAUNCH_BLOCKING=1`

Topology is TP=8, PP=10, DP=4

It has been very difficult to work on diagnosing this issue since every time I run the hanging setup I would lose a few nodes and since I'm 10h behind JeanZay, nobody is around there to reboot the nodes.

So first of all it appears that `CUDA_LAUNCH_BLOCKING=1` removes the hanging issue and I did several performance checks and it surprisingly has no impact on this framework at this scale. Normally, it should make things much slower as it makes CUDA ops synchronous.

### py-spying all processes

After discussing this issue with Samyam I first run `py-spy` on all processes, but alas several processes weren't responding, so we had no idea how to tell where they were hanging.

For posterity here is the process:


In one console, first allocate the gpus:
```
salloc --partition=gpu_p5 --constraint=a100 --reservation=hug --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 --hint=nomultithread --gres=gpu:8 --time 20:00:00 --account=six@a100
```
We are doing that so that if SLURM kills the processes we could still access those.

Now run the training job, which calls the main `srun` with all the gpus:
```
bash 200B-n40-bf16-mono.slurm
```

Wait till the program hangs.

Now in another console get the `SLURM_JOBID` (or get it from `salloc` log):
```
squeue -u `whoami` -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"
```

Adjust jobid with `SLURM_JOBID` from above:
```
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

Must use `--gres=gpu:0` for the monitor `srun` or otherwise it will block until the first `srun` exits

I also attempted using `pdsh` via `ds_ssh`, but somehow I wasn't able to run `py-spy` remotely - the main issue was that remote `ssh` command wasn't giving the same env as when I was logged in interactively via `ssh`. But if you have `sudo` access on the compute nodes than you could do:

First prepare `hostfile`:
```
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile
```

Now run the `py-spy` extraction command over all nodes:
```
ds_ssh -f hostfile "source ~/.pdshrc; ps aux | grep python | grep -v grep | grep `whoami` | awk '{print \$2}' | xargs -I {} sudo py-spy dump --pid {} "
```

### python trace

So next came the idea of tracing all calls like one does with `strace(1)`, I researched python calls tracing facilities and have discovered that python has a `trace` sub-system.

This code will trace all python calls and log them to the console and into a dedicated per process log file, via a custom `Tee` module I added.

This then can help to understand where some processes stopped responding, since we will have the log of the last call before it went unresponsive.

```
$ cat pretrain_gpt.py
[...]

def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

import re
class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __name__ == "__main__":

    import sys
    import trace
    import socket
    import os

    # enable to trace
    if 0:
        cwd = os.path.realpath('.')
        pid = os.getpid()
        hostname = socket.gethostname()
        local_rank = int(os.environ["LOCAL_RANK"])
        trace_output_file = f"{cwd}/trace-{hostname}-{local_rank}-{pid}.txt"

        # create a Trace object, telling it what to ignore, and whether to
        # do tracing or line-counting or both.
        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix],
            trace=1,
            count=1,
        )
        #    outfile=trace_output_file)

        # run the new command using the given tracer
        sys.stdout = Tee(trace_output_file)
        tracer.run('main()')
    else:
        main()

```

This code doesn't require any special handing other than enabling the trace by changing `if 0` to `if 1`.

Of course, this will now dump all python calls. I was worried that the slowdown will mask the issue causing the hanging, but surprisingly it didn't.

I got 14GB (!) of data logged of just python calls from 320 processes.

In retrospect I probably should have started the tracing at a later place, probably just before `train_step` - otherwise we have gotten a lot of useless traces of the dataloader and other preliminary code.

I wish I could tell `trace` which packages to follow, but alas it only supports dirs to ignore, which is much more difficult to set, and thus you end up with a lot more data than one needs. But still this is a super useful tool for debugging hanging processes.


### To be continued

We needed to do some more tweaks to get to the root of it.

Unfortunately I had to pause here, since I had to switch to testing the final version of the code and I couldn't risk losing nodes.

With having `CUDA_LAUNCH_BLOCKING=1` workaround providing a robust solution we will use that for a time being.

# a few preliminary runs


## main-1

While the final data is being cleaned up we are doing a few preliminary runs with data that still has some issues.

GBS ramp up of `--rampup-batch-size 16 16 9_765_625` - the first few stages starting with GBS=16 are really slow (8 TFLOPs). The pipeline doesn't have enough data to even fill all the stages once, so it's super inefficient and it'll take days until we start hitting 100 TFLOPs.

But there were no spikes during this brief experiment.



## main-2

Trying `--rampup-batch-size 384 16 9_765_625` since 384 is the first GBS where the pipe is filled up fully for the first time. `12*2*4=384` (`PP*MBS*DP`). The throughput start at 100 TFLOPs right away (and it should be 150 TFLOPS once we reach GBS=2048).

Found a bug: tied weights weren't getting reduced - was getting a spike on restart, fixed at
https://github.com/microsoft/DeepSpeed/pull/1801/commits/37011a92bad42b07c2cb742751873ef7073d84b8

So only the front embed matrix grad updates were making, the end one were ignored.

Will do a totally new run to compare that it's similar or better.




## main-3

Trying the rebased to master version 61d51fd62141ddb51b629b785af256fac407e048 and it has serious issues - the learning is much much slower

## main-4

So rolling back `olruwase/bf16-updates` branch to the fix:

37011a92bad42b07c2cb742751873ef7073d84b8 Reduce tied weight gradients

This time the learning is just a tad slower than main-2, so either deepspeed@master introduced some regression or the merge didn't go well.

additionally going to try the latest checkpoint from `main-3` as it's more progressive to check `main-4` for spikes.

same spike.

After analysis the module weights, it's clear we have 2 distinct issues:
1. `module.tied_modules.embed.word_embeddings.norm` is not taken care of at all in pp rank -1 and they get reset to defaults on load
2. `module.tied_modules.embed.word_embeddings.weight` is mismatching on pp rank -1 between after iteration of last step before save and before iteration of first step after load

This was derived with this debug instrumentation:

```
diff --git a/megatron/training.py b/megatron/training.py
index fd65ae9..fd76d28 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -58,6 +58,23 @@ from megatron.data.dataset_utils import analyze_data_prefix

 import deepspeed

+def dump_weights(preamble, iteration, model):
+
+    import os
+    import socket
+    hostname = socket.gethostname()
+    pid = os.getpid()
+
+    tp_rank = mpu.get_tensor_model_parallel_rank()
+    pp_rank = mpu.get_pipeline_model_parallel_rank()
+    dp_rank = mpu.get_data_parallel_rank()
+    global_rank = torch.distributed.get_rank()
+
+    fn = f"debug-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-global{global_rank}-{preamble}-{pid}.txt"
+    #print(fn)
+    with open(fn, "w") as fh:
+        for n, p in model[0].named_parameters():
+            fh.write(f"{n}={p}\n")

 def print_datetime(string):
     """Note that this call will sync across all ranks."""
@@ -426,6 +443,8 @@ def setup_model_and_optimizer(model_provider_func):
         if args.fp16:
             optimizer.reload_model_params()

+    #optimizer.update_lp_params()
+
     return model, optimizer, lr_scheduler


@@ -848,12 +867,18 @@ def train(forward_step_func, model, optimizer, lr_scheduler,
             args.pipeline_model_parallel_size >= 1:
             args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                     args.iteration + 1)
+
+        dump_weights("before-iteration", iteration+1, model)
+
         loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
             train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        lr_scheduler)
+
+        dump_weights("after-iteration", iteration+1, model)
+
         iteration += 1
         args.iteration = iteration
         new_samples = mpu.get_data_parallel_world_size() * \
```

and then
1. run 5 iterations and saved checkpoint, then run:
```
mkdir a; mv debug-* a
```
2. restarted and run a few iterations, then run:

```
mkdir b; mv debug-* b
```

I basically dumped weights for all ranks before and after train_step

Now let's compared them all. Comparing:
1. the after iteration of the last step before save (iteration 805 in this example)
2. the before iteration step after the load (on restart) (iteration 806 in this example)

with the help of:
```
perl -le 'print qx[diff -u a/debug-805-*global$_-after-iteration-*.txt b/debug-806-*-global$_-before-iteration-*.txt] for 0..383'
```

Result: all `a/debug-805-pp11-*-after-iteration-*.txt` and corresponding `b/debug-806-pp11-*-before-iteration-*.txt` mismatch.

so here is a sample diff:
```
--- a/debug-805-pp11-tp1-dp4-global369-after-iteration-377074.txt       2022-03-06 05:44:06.074835000 +0100
+++ b/debug-806-pp11-tp1-dp4-global369-before-iteration-378990.txt      2022-03-06 05:48:24.842635000 +0100
@@ -1,21 +1,15 @@
 module.tied_modules.embed.word_embeddings.weight=Parameter containing:
-tensor([[-3.1090e-04,  4.6082e-03, -2.3499e-03,  ..., -1.1292e-02,
-          2.1667e-03, -2.7313e-03],
-        [-1.1353e-02,  9.9487e-03, -1.9684e-03,  ..., -5.4550e-04,
-         -2.3460e-04,  4.2114e-03],
-        [ 3.2806e-03, -3.4332e-04, -5.5847e-03,  ...,  7.6294e-03,
-          1.7853e-03,  2.5868e-05],
+tensor([[-0.0006,  0.0046, -0.0024,  ..., -0.0114,  0.0014, -0.0030],
+        [-0.0109,  0.0096, -0.0020,  ..., -0.0005, -0.0001,  0.0041],
+        [ 0.0027, -0.0004, -0.0056,  ...,  0.0070,  0.0017,  0.0003],
         ...,
-        [ 1.6098e-03,  4.1809e-03, -2.4567e-03,  ..., -4.6692e-03,
-         -4.5776e-03,  1.7090e-03],
-        [ 5.7373e-03,  3.5858e-03, -1.7471e-03,  ...,  2.3041e-03,
-         -6.4392e-03,  1.0223e-03],
-        [-1.6937e-03, -1.4038e-02,  2.1057e-03,  ..., -3.6011e-03,
-          1.3275e-03, -5.8594e-03]], device='cuda:1', dtype=torch.bfloat16,
-       requires_grad=True)
+        [ 0.0018,  0.0039, -0.0026,  ..., -0.0051, -0.0043,  0.0016],
+        [ 0.0051,  0.0039, -0.0015,  ...,  0.0027, -0.0063,  0.0008],
+        [-0.0018, -0.0142,  0.0021,  ..., -0.0035,  0.0015, -0.0060]],
+       device='cuda:1', dtype=torch.bfloat16, requires_grad=True)
 module.tied_modules.embed.word_embeddings.norm.weight=Parameter containing:
-tensor([0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961], device='cuda:1',
-       dtype=torch.bfloat16, requires_grad=True)
+tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:1', dtype=torch.bfloat16,
+       requires_grad=True)
 module.tied_modules.embed.word_embeddings.norm.bias=Parameter containing:
 tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:1', dtype=torch.bfloat16,
        requires_grad=True)
```


## main-5

trying a new baseline with rampup starting from 192



## main-6

trying https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/260 - comparing with main-5

tracks exactly main-5 - merged.


## main-7

Running with https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/261

Don't allocate embed LN on pp rank -1, - different checkpoint

still spikes on restart


# main-no-emb-norm

disable `--embed-layernorm` completely, check if spikes on restart

no spikes on restart

## main-8

1. test https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/262

2. At 1438 switched to deepspeed@ab61edb02a137d91b61bd416b4e8d3eb287b0eba of olruwase/bf16-updates - let's see if it tracks still the previous runs - yes it does.

So the restart spike's cause was this: the framework was putting `LayerNorm` that I added for the embedding layr into the wrong param group [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/dd06ea32e014d8db6cdaf5e6839071d6523ca83c/megatron/optimizer/__init__.py#L31-L45).

it should have been in `no_weight_decay_params` but ended up in `weight_decay_params` because in this module `LayerNorm` is an alias for `MixedFusedLayerNorm`, so if `isinstance(module_, LayerNorm)` was `False`.

So if we want to use `torch.nn.LayerNorm` we have to change the code above to additionally check for ` or isinstance(module_, torch.nn.LayerNorm).`

## main-9

re-running with  deepspeed@77b649d160c1cd86f33415e2a7deab50c45fba16 of olruwase/bf16-updates which fixed the tied-embedding desynchronization bug due to clip grads not running on the last pp rank for tied embeddings.
