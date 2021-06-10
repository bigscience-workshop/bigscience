# GPT2 Experiments

Scripts and logs of GPT2 experiments on Jean Zay HPC.

Using 4x VT100 32GB nodes.

(add `-C v100-32g` for 32gb nodes.)

## Apples and Oranges

JZ seems to give us inconsistent performance - so each allocation may give performance that can vary as much as 40%, so the numbers in the summaries of this document are very hard to compare. We thought it had to do with the proximity of the allocated nodes but it proved to vary randomly through the day, most likely highly dependening on the traffic on the JZ network.

Therefore any results you will find in this summary are +/-40% correct. An identical test scored 40% faster or slower on the same allocation at different times of the day.

## Megatron-LM

Constants:

- `TP_SIZE` = tensor parallel
- `PP_SIZE` = pipeline parallel
- `DP_SIZE` = data parallel is derived automatically from `WORLD_SIZE / (TP_SIZE * PP_SIZE)`
- `WORLD_SIZE` = total number of GPUs

According to Megatron-LM paper the highest degree of TP we can use is 4 for 4-gpu nodes - crossing nodes would slow things down a lot. So max `TP_SIZE=4`. So the full 4 gpu node is used only for tensor parallel dimension.

## Metrics

TFlops: `model_size_in_B * 4 * 2 * seq * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

The factor of 4 is when used with activation check-pointing,
otherwise it will be 3, but for 200B model, activation check-pointing will always be on.

The peak of V100 32gb gpu is about 125 TFlops/sec [spec](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf). But we cannot get the peak. The max achievable performance will be 30-60TFlops depending on the model size. So if you see low 20s, the model is not tuned well, if you see, over 100 then there is a bug in the calculation. ￼

For v100 16gb gpus the max spec is 120 TFlops/sec.

## Allocation

```
salloc --constraint=v100-32g --nodes=16 --ntasks=16 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=6:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


### Megatron

The full slurm scripts and log files are at [`gpt2-meg`](./gpt2-meg):
- scripts starting with `meg_gpt2_base_` are for getting the baseline with tiny BS
- scripts starting with `meg_gpt2_perf_` are for smaller model, and tuned for high performance

Not yet optimized with NVIDIA team!

Metrics can be calculated in bash after figuring out the throughput (in seconds):

```
THROUGHPUT=122
NNODES=16
MSIZE=52
MICRO_BATCH_SIZE=4
DP_SIZE=1
PP_CHUNKS=256
echo "($MSIZE*4*2*1024*$MICRO_BATCH_SIZE*$DP_SIZE*$PP_CHUNKS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
55.86675409836065573770
```

**Max model size**

These first results are all about how big of a model can be fit into the given the hardware on the smallest batch size, disregarding throughput.

16GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 | 7.5B |  1 |  4 |         4 |      1 |       4 | 0.661s     |   23.2 |
|   64 |  30B |  1 | 16 |         4 |      1 |       4 | 1.439s     |   10.7 |
|  128 |  50B |  1 | 32 |         4 |      1 |       4 | 2.124s     |    6.0 |
|  256 |  78B |  1 | 64 |         4 |      1 |       4 | 2.953s     |    3.4 |
|  256 |  22B |  4 | 16 |         4 |      1 |       4 | 1.826s     |    1.5 |
|      |      |    |    |           |        |         |            |        |

32GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 |  18B |  1 |  4 |         4 |      1 |       4 | 1.381s     |   26.7 |
|   32 |  30B |  1 |  8 |         4 |      1 |       4 | 1.618s     |   19.0 |
|   64 |  65B |  1 | 16 |         4 |      1 |       4 | 2.738s     |   12.2 |
|  128 | 116B |  1 | 32 |         4 |      1 |       4 | 4.234s     |    7.0 |
|  256 | 206B |  1 | 64 |         4 |      1 |       4 | 6.736s     |    3.9 |
|      |      |    |    |           |        |         |            |        |

The TFLops are very low because there are too few PP chunks/micro-batches (4) (gradient accumulation size / GAS) and so the bubble takes a lot of overhead, increasing PP chunks should dramatically improve performance but also need to lower the max model size to have memory to hold those chunks in.

**Performance**

These experiments are to try a lower model size, but much higher TFlops performance

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: | ----: |
|   16 | 18B  |  1 |  8 |        64 |      4 |     256 | 90.5s      |   26.1 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      4 |     512 | 177s       |   26.7 | 05-26 |
|   16 | 18B  |  1 |  8 |       256 |      4 |    1024 | 356s       |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   16 | 18B  |  1 |  4 |       128 |      4 |     512 | 179s       |   26.4 | 05-26 |
|   16 | 18B  |  1 |  4 |       128 |      6 |     768 | 262s       |   27.0 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      6 |     768 | 259s       |   27.3 | 05-26 |
|   16 | 18B  |  1 |  8 |        32 |      8 |     256 |  89s       |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   32 | 39B  |  1 |  8 |       128 |      4 |     512 |  82s       |   62.3 | 05-26 |
|   32 | 39B  |  1 |  8 |       128 |      6 |     768 | 123s       |   62.3 | 05-26 |
|   32 | 39B  |  1 |  8 |       256 |      6 |    1536 | 241s       |   63.6 | 05-26 |
|   32 | 39B  |  1 |  8 |       512 |      6 |    3072 | 478s       |   64.2 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 52B  |  1 | 16 |       256 |      4 |    1024 | 129s       |   52.8 | 05-25 |
|   64 | 52B  |  1 | 16 |       256 |      4 |    1024 | 217s       |   31.4 | 05-26 |
|   64 | 52B  |  1 | 16 |       256 |      4 |    1024 | 125s       |   54.5 | 05-27 |
|   64 | 52B  |  1 | 16 |       256 |      4 |    1024 | 225s       |   30.3 | 05-28 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 52B  |  1 | 16 |       256 |      6 |    1536 | 328s       |   31.2 | 05-26 |
|   64 | 52B  |  1 | 16 |       256 |      8 |    2048 | 435s       |   31.3 | 05-26 |
|   64 | 52B  |  1 | 16 |       512 |      6 |    3072 | 650s       |   31.5 | 05-26 |
|   64 | 52B  |  1 | 16 |       512 |      8 |    4096 | 870s       |   31.3 | 05-26 |
|   64 | 52B  |  1 | 32 |       256 |      4 |    1024 | 220s       |   31.0 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |


data:
- Size = Model Size
- `TP=4` in all of entries
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * pp_chunks * dp_size`
- PP chunks is the number of PP stages, so each pipeline handles `micro-batch-size * pp_chunks`
- Seq length is 1024

notes:
- 32gpus had a very snag fit for gpu memory for 39B model (others were in ~75%) so it might be a bit too risky to OOM-borderline


```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=127; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```



#### Megatron + Deepspeed 3D (new branch)

This is new branch synced with Megatron

DeepSpeed branch: https://github.com/ShadenSmith/DeepSpeed/tree/megatron2.4-3d
Megatron branch: https://github.com/jeffra/DSE/tree/megatron-2.4-ds-pipe

This script can now launch Meg alone or Meg + Deepspeed 3D (ignore the zero options it doesn't work yet):
https://github.com/jeffra/DSE/blob/megatron-2.4-ds-pipe/run.sh

```
git clone https://github.com/ShadenSmith/DeepSpeed/ deepspeed-shaden
cd deepspeed-shaden
git checkout megatron2.4-3d
```

```
git clone https://github.com/jeffra/DSE megator-jeffra
cd megator-jeffra
git checkout megatron-2.4-ds-pipe
```


| GPUs | Size | GPU Mem | DP | PP | PP chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -----:  | -: | -: | --------: | -----: | ------: | ---------: | -----: | ----: |
| 64   | 52B  |  26GB   | 1  | 16 | 256       | 4      | 1024    |            |        | 06-10 |
|      |      |         |    |    |           |        |         |            |        |       |

```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=146; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```



### Megatron + Deepspeed 3D (old branch)


**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism` is not in sync with M-LM master - so several config args don't match. It's about 8 months old.

See scripts and logs under [gpt2-meg-ds-3d-old](./gpt2-meg-ds-3d-old).

Uses 3D:
- TP: tensor parallelism
- PP: pipeline parallelism
- DP: data parallelism

same features as Megatron's native, but improved by Deepspeed

**Performance**

| GPUs | Size | DP | PP | PP chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | --------: | -----: | ------: | ---------: | -----: | ----: |
| 64   | 52B  | 1  | 16 | 256       | 4      | 1024    | 146s       | 46.7   | 05-27 |
|      |      |    |    |           |        |         |            |        |       |


- GAS = Gradient Accumulation size (same as PP_chunks / number of PP stages)
- Global_bs = pp_chunks*micro_bs*dp_size
- `TP_SIZE=4` (size of the node)

```
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=146; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```






#### Megatron + Deepspeed ZeRO (old branch)


**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3` is not in sync with M-LM master - so several config args don't match. It's about 8 months old.

See scripts and logs under [gpt2-meg-ds-zero](./gpt2-meg-ds-zero).

This one uses only TP from Megatron (no PP)

Not yet optimized with Deepspeed team!

**With Offload off**

**Performance**
| GPUs | Size  | DP | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ----: | -: |   ---: |  -----: | ---------: | -----: | ----: |
|   64 | 52B   | 16 |     48 |     768 | 122s       |   41.9 | 05-25 |
|   64 | 52B   | 16 |     48 |     768 | 127s       |   40.3 | 05-27 |
|      |       |    |        |         |            |        |       |


```
perl -le '$ng=64; $ms=52; $gbs=768; $sp=122; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```
- Seq length is 1024
- `TP=4` in all of entries
- `DP` is number of nodes here
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * dp-size`

- tried w/ and w/o Tiling once but saw no difference - perhaps would be more important on larger collections

| GPUs | Size | TP | DP | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | -----: | ------: | ---------: | -----: | ----: |
|   64 | 52B  |  4 | 16 |     48 |     768 | 127s       |   40.3 |       |
|   64 | 52B  |  2 | 32 |     32 |    1024 | 167s       |   40.8 |       |
|   64 | 52B  |  1 | 64 |     16 |    1024 | 184s       |   37.0 |       |
|   64 | 24B  |  1 | 64 |     16 |    1024 | 89.0s      |   35.3 |       |
|   64 | 24B  |  2 | 32 |     32 |    1024 | 85.7s      |   36.7 |       |


**With full cpu offload**

| GPUs | Size | TP | DP | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | -----: | ------: | ---------: | -----: |
| 64   | 52B  | 4  | 16 | 64     | 1024    | 171s       | 39.9   |
|      |      |    |    |        |         |            |        |


Olatunji requested the following experiments:

- enabled/set: `--split-transformers --checkpoint-num-layers=2`
- removed: `--synchronize-each-layer   --contigious-checkpointing`

| GPUs | Size | ScatEmb | TP | DP | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | ------: | -: | -: | -----: | ------: | ---------: | -----: |
|   64 | 52B  | N       |  4 | 16 |     48 |     768 | 119s       |   43.0 |
|   64 | 52B  | Y       |  4 | 16 |     48 |     768 | 115s       |   44.5 |
|   64 | 52B  | Y       |  4 | 16 |     52 |     832 | 124s       |   44.7 |
|   64 | 52B  | N       |  2 | 32 |     32 |    1024 | 159s       |   42.9 |
|   64 | 52B  | Y       |  2 | 32 |     32 |    1024 | 158s       |   43.1 |
|   64 | 52B  | Y       |  2 | 32 |     36 |    1152 | 176s       |   43.6 |
|   64 | 52B  | Y       |  4 | 16 |     56 |     896 | 161s       |   37.0 |
|   64 | 52B  | Y       |  2 | 32 |     38 |    1216 | 178s       |   45.5 |
|   64 | 52B  | Y       |  1 | 64 |     18 |    1152 | 197s       |   38.9 |
|   64 | 52B  | Y       |  1 | 64 |     20 |    1280 | 219s       |   38.9 |
|   64 | 52B  | Y       |  1 | 64 |     22 |    1408 | OOM        |        |



Automatically process slurm/ megatron log files, average the througput (prints 'fail' on when the training failed):
```
find . -type f -exec perl -lne 'm|elapsed time per iteration .ms.: ([\d\.]+)| &&  do {$x+=$1; $c++}; END { print "$ARGV " . ($c ? int($x/$c/1000) : "fail")}' {} \; | sort | grep -v fail
```

### HF + Deepspeed Zero 3 + Full Offload

See scripts and logs under [gpt2-hf-ds](./gpt2-hf-ds).

Not yet optimized with Deepspeed team!

**Max model size**

| GPUs | Size  | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ----: | -----: | ------: | ---------: | -----: |
|   16 | 25B   |      4 |      64 | 58.72s     |   14.0 |
|   32 | 52B   |      4 |     128 | 114.56s    |   14.9 |
|   64 | 97B   |      4 |     256 | 222.72s    |   14.3 |
|      |       |        |         |            |        |


**Performance**

| GPUs | Size  | Zero | Opt Offl | Par Offl | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ----: |  --: | -------: | -------: | -----: | ------: | ---------: | -----: | ----: |
|   64 | 52B   |    3 | N        | N        |      8 |     512 | 139s       |   24.5 | 05-25 |
|   64 | 52B   |    3 | N        | N        |      4 |     256 | 185s       |    9.2 | 05-27 |
|   64 | 52B   |    3 | N        | N        |      8 |     512 | 118s       |   28.9 | 05-27 |
|      |       |      |          |          |        |         |            |        |       |
|   64 | 52B   |    3 | N        | N        |      8 |     512 | 117s       |   29.1 | 05-28 |
|   64 | 52B   |    3 | N        | N        |      6 |     384 | 111s       |   23.0 | 05-28 |
|   64 | 52B   |    3 | N        | N        |     10 |     640 | 150s       |   28.4 | 05-28 |
|   64 | 52B   |    3 | Y        | N        |     12 |     768 | 183s       |   27.9 | 05-28 |
|   64 | 52B   |    3 | Y        | N        |     12 |     768 | 175s       |   29.2 | 05-28 |
|   64 | 52B   |    3 | Y        | Y        |     12 |     768 | 177s       |   28.9 | 05-28 |
|      |       |      |          |          |        |         |            |        |       |
|   64 | 52B   |    2 | Y        | N        |        |         | OOM        |        | 05-28 |
|      |       |      |          |          |        |         |            |        |       |


- DP=GPUs
- global bs = micro bs * DP
- Throughput reported by HF Trainer metrics is `samples_per_second` - So total throughput in the table is `glob_bs/samples_per_second`

notes:
- gradient checkpointing activated


```
perl -le '$ng=64; $ms=52; $gbs=512; $sp=139.52; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
22
```

ZeRO-2 with model of this size I can't fit into this setup at all - even BS=4 - it keeps getting on getting killed by cgroups - i.e. it's asking for more than 40GB general RAM per gpu. Same story w/ or w/o offload.


## Maintenance

A secret formula to re-generate tflops column:
```
perl -ne 's#^(\| +(\d+) +\| +(\d+)B.*? +(\d+) +\| +([\d\.]+)s) +\| +[\d\.]+ +(.*?)$#"$1 | ".sprintf("%.01f", $3*4*2*1024*$4 / ($5*$2*1e3))." $6"#e && print ' gpt2.md
```

I originally had a mistake in model size calculation script - which has been fixed in tables and the scripts, but many logs still have the old formula - I used G `(2**30)` instead of B `(10**9)` so the model size was getting reported small than it is.

Now it's the correct version:
```
NHIDDEN=4096
NLAYERS=36
SEQ_LEN=512
VOCAB_SIZE=50257
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B')"
```
