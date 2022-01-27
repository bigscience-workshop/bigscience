# tr11 200B ML

final size to be defined


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

- embedding size: v*h: 50257*11600 = 582_981_200 / 4 (TP=4) => 145_745_300 params per gpu for embedding
- one layer size: 12*h**2 + 13*h:  1_614_870_800 / 4 (TP=4) => 403_717_700 params per gpu per layer

64 layers over PP=32 => 2 layers per gpu

Total params per gpu:
- gpu w/  emb: 2*403_717_700 + 145_745_300 = 953_180_700 params * 18 bytes = 17_157_252_600 bytes (17GB)
- gpu w/o emb: 2*403_717_700               = 807_435_400 params * 18 bytes = 14_533_837_200 (15GB)

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

- embedding size: v*h: 150257*13312 = 2_000_221_184 / 4 (TP=4) =>  500_055_296 params per gpu for embedding
- one layer size: 12*h**2 + 13*h:     2_126_685_184 / 4 (TP=4)  => 531_671_296 params per gpu per layer

In other words 2B params per layer w/o TP, or 38GB (2.12*18) per layer.

So here we definitely need to balance embedding layer with transformer layers as they are of the same size, so overall 2+layers blocks to balance - and the constraint won't be Layers % PP = 0 but Layers+2 % PP = 0

So probably should do 94 layers?

94+2 layers over PP=24 => 4 layers per gpu

Total params per gpu (considering emb layer on par with transformers block):
- 4*531_671_296 = 2_126_685_184 params * 18 = 38_280_333_312 bytes
plus activations memory

40GB A100 takes 1573MiB for cuda kernels (probably about the same for 80GB? may be a bit larger)
`python -c "import torch; import time; torch.ones(1).cuda(); time.sleep(30)"` + check `nvidia-smi` output.



* TP=1, PP=96

~2B params per layer w/o TP, or 38GB (2.12*18) per layer.

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

so 80/8=10 PP stages per gpu: 10*4 =40GB of weights/optim states/grads per gpu


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

so 96/16=6 PP stages per gpu: 6*7.3 ~44GB of weights/optim states/grads per gpu

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

so 105/35=3 PP stages per gpu: 6*7.3 ~33.9GB of weights/optim states/grads per gpu


To summarize we can see the setup is so that about half the gpu is loaded with weights / optim states / grad *18)

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


-------------
