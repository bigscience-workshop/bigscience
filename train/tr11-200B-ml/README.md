# tr11 200B ML

final size to be defined


## Size


Here are some existing models around the same size with NLAYERS / NHIDDEN and their ratio:

- 104B-ml: 64 / 11600 (ratio: 180)
- gpt3 175B: 96 / 12288 (ratio: 128)
- meg 145B 80 / 12288 (ratio: 154)
- meg 310B 96 / 16384 (ratio: 170)

Possible ideas:

- 205B: 112 / 12288 (ratio: 109) narrow
- 206B: 96 / 13312 (ratio: 139) closer to typical 150-200 ratio

Formula to get model size, used 150k dict roughly - need to update:
```
NHIDDEN=12288;NLAYERS=112;SEQ_LEN=2048;VOCAB_SIZE=150257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B')"
```

## Hardware


We can plan to use 384 gpus out of 416 as 4 nodes of 8 gpus need to remain reserved for when some nodes happen to be down.

Initially we will have only 144 gpus and then around mid-Feb we should have the rest.

## Possible config:

So a possible config is

- a single replica needs to fit 96 gpus and then we can do DP=4 to a full 384 gpus

- extrapolating from the current 104B setup we can have: TP=4/PP=24 @ 80GB + 150K vocab size (which is different from the 50k vocab in 104B - 3x bigger embed matrix plus bigger hidden size.

- most likely the embedding layer now will need to be partitioned together with the transformer blocks to do a good balancing of resources. e.g. in the current 1.3B ml setup, the 1st and last gpus use all of DRAM, but the rest of gpus use only 1/2 DRAM - and TLOPs are ~21 which is very underutilized.

104B:

```
NLAYERS=64
NHIDDEN=11600
NHEADS=80
SEQ_LEN=2048
VOCAB_SIZE=50257
```

206B:

```
NLAYERS=96
NHIDDEN=13312
NHEADS=XXX
SEQ_LEN=2048
VOCAB_SIZE=150_000
```

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


### 200B

now let's estimate the same for 200B:

* TP=4

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

* TP=1

2. ~2B params per layer w/o TP, or 38GB (2.12*18) per layer.

but DS breaks if there isn't at least one transformer block per gpu :(
otherwise could do a very efficient:

```
1   | 2      | 3      ... | 95     | 96
emb | transf | transf ....| transf | emb
```

So in this scenario no TP is needed, which should make the assembly much faster. But will require DS fixing their side.

Also the pipeline will be super long here, which to make efficient will require a huge global batch size.

* with TP=2, PP=48

1_063_342_592 params per layer, 19_140_166_656 bytes (19GB) per layer

perhaps could squeeze 3 layers per gpu - then
