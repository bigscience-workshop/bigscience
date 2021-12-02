
# Benchmarking setup 1B3


benchmarked with this slurm setup [tr3m-1B3-emb-norm-pile.slurm](../train/tr3-1B3-baseline/tr3m-1B3-emb-norm-pile.slurm)


# 32GB node

Benchmarking on 2 nodes to make sure we catch the inter-node slowdown.

Measuring w/o BS-rampup so with full GBS

```
salloc --account=six@gpu --constraint=v100-32g --nodes=2 --ntasks=2 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=2:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

measuring w/o rampup

| NNODES |  TP |  PP |  DP | MBS | Speed | TFlops | Notes                 |
| -----: | --: | --: | --: | --: | ----: | -----: | --------------------: |
|      2 |   1 |   1 |   8 |   1 |    29 |   47.0 | 16GB                  |
|      2 |   1 |   1 |   8 |   2 |    29 |   47.0 | 17GB                  |
|      2 |   1 |   1 |   8 |   4 |    28 |   48.7 | 20GB                  |
|      2 |   1 |   1 |   8 |   8 |    28 |   48.7 | 25GB                  |
|      2 |   1 |   2 |   4 |   1 |    30 |   45.4 | 10GB                  |
|      2 |   1 |   2 |   4 |   2 |    29 |   47.0 | 11GB                  |
|      2 |   1 |   2 |   4 |   8 |    29 |   47.0 | 15GB                  |
|      2 |   1 |   2 |   4 |  16 |     x |      x | OOM                   |
|      2 |   1 |   4 |   2 |   1 |    32 |   42.6 | 9GB                   |
|      2 |   1 |   4 |   2 |   8 |    32 |   42.6 | 13GB                  |
|      2 |   2 |   1 |   4 |   1 |    53 |   25.7 | 11GB                  |
|        |     |     |     |     |       |        |                       |


```
perl -le '$ng=8; $sp=29; $ms=1.3; $gbs=512; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```

After removing `--checkpoint-activations` (which changes the factor to 3 from 4 for TFLOPs calculation)

| NNODES |  TP |  PP |  DP | MBS | Speed | TFlops | Notes                 |
| -----: | --: | --: | --: | --: | ----: | -----: | --------------------: |
|      2 |   1 |   1 |   8 |   1 |    23 |   44.4 | 27GB                  |
|      2 |   1 |   2 |   4 |   1 |    23 |   44.4 | 21GB                  |
|      2 |   1 |   4 |   2 |   1 |    25 |   40.8 | 19GB                  |
|      2 |   1 |   4 |   2 |   2 |    24 |   42.5 | 30GB                  |
|      2 |   2 |   1 |   4 |   1 |    39 |   26.2 | 21GB                  |
|        |     |     |     |     |       |        |                       |


factor = 3 here (not 4)

```
perl -le '$ng=8; $sp=; $ms=1.3; $gbs=512; $seqlen=2048; print $ms*3*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```

So the best throughput is with

1. removing `--checkpoint-activations`
2. config

```
PP_SIZE=1 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=1
MICRO_BATCH_SIZE=1
```

Which means that one replica is 1 gpu.

If BS rampup is used, e.g. starting from 32, that means that you can use max 32/1 = 32 gpus or 8 nodes.

This of course can be manually adjusted to more nodes once BS is larger. Here is a possible schedule:


| NNODES  |   BS |  MBS |
| ------: | ---: | ---: |
| 8       |   32 |    1 |
| 16      |   64 |    1 |
| 32      |  128 |    1 |


# 16GB node

Same as above but with 16GB gpus

```
salloc --account=six@gpu --constraint=v100-16g --nodes=2 --ntasks=2 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=2:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


| NNODES |  TP |  PP |  DP | MBS | Speed | TFlops | Notes                 |
| -----: | --: | --: | --: | --: | ----: | -----: | --------------------: |
|      2 |   1 |   1 |   8 |   1 |    29 |   47.0 | 16GB borderline OOM   |
|      2 |   1 |   2 |   4 |   1 |    30 |   45.4 | 11GB                  |
|      2 |   1 |   2 |   4 |   2 |    29 |   47.0 | 12GB                  |
|      2 |   1 |   2 |   4 |   4 |    28 |   48.7 | 13GB                  |
|      2 |   1 |   2 |   4 |   8 |     x |      x | OOM                   |
|      2 |   1 |   4 |   2 |   1 |    32 |   42.6 | 9GB                   |
|      2 |   1 |   4 |   2 |   4 |    30 |   45.4 | 11GB                  |
|      2 |   1 |   4 |   2 |   8 |     x |        | OOM                   |
|      2 |   1 |   8 |   1 |   1 |    37 |   36.8 | 9GB                   |
|      2 |   1 |   8 |   1 |   4 |    35 |   38.9 | 11GB                  |
|      2 |   1 |   8 |   1 |   8 |     x |        | OOM                   |
|        |     |     |     |     |       |        |                       |


```
perl -le '$ng=8; $sp=29; $ms=1.3; $gbs=512; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```

So the best throughput is with:

```
PP_SIZE=2 # NLAYERS must be a multiple of PP_SIZE here
TP_SIZE=1
MICRO_BATCH_SIZE=4
```

Which means that one replica is 2 gpus. But there is the BS rampup constraint for first values.

but if BS rampup is used, e.g. starting from 32, that means that you can use max 32/4 = 8 replicas, 16 gpus, 4 nodes only.

To use 8 nodes use MBS=2 and so it's just slightly slower (32/2=16 replicas or 32 gpus or 8 nodes).

To use 16 nodes use MBS=1 and so it's again slightly slower (32/1=32 replicas or 64 gpus or 16 nodes).

It's also possible to start with MBS=1 and then down the road switch to MBS=2 and then finally MBS=4 and later use even more nodes.

So here is a possible schedule that will require manual adjustments of the slurm file as the BS is going through a rampup to get the maximum speeds.


| NNODES  |   BS |  MBS |
| ------: | ---: | ---: |
| 16      |   32 |    1 |
| 16      |   64 |    2 |
| 16      |  128 |    4 |
| 32      |  256 |    4 |
| 64      |  512 |    4 |


## calibration


Cuda kernels:
```
python -c "import torch; x = torch.ones(1).cuda(); import time; time.sleep(100)" &
```

V100 16GB 1113MiB
V100 32GB 1113MiB

(same memory!)
