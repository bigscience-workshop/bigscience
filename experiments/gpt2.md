# GPT2 Experiments

Scripts and logs of GPT2 experiments on Jean Zay HPC.

Using 4x VT100 32GB nodes.

(add `-C v100-32g` for 32gb nodes.)

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


### Megatron

The full slurm scripts and log files are at [`gpt2-meg`](./gpt2-meg):
- scripts starting with `meg_gpt2_base_` are for getting the baseline with tiny BS
- scripts starting with `meg_gpt2_perf_` are for smaller model, and tuned for high performance

Not yet optimized with NVIDIA team!

**Max model size**

These first results are all about how big of a model can be fit into the given the hardware on the smallest batch size, disregarding throughput.

16GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 | 7.5B |  1 |  4 |         4 |      1 |       4 | 0.661s     |        |
|   64 | 30B  |  1 | 16 |         4 |      1 |       4 | 1.439s     |        |
|  128 | 50B  |  1 | 32 |         4 |      1 |       4 | 2.124s     |        |
|  256 | 78B  |  1 | 64 |         4 |      1 |       4 | 2.953s     |        |
|  256 | 22B  |  4 | 16 |         4 |      1 |       4 | 1.826s     |        |
|      |      |    |    |           |        |         |            |        |

32GB nodes:

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: |
|   16 | 18B  |  1 |  4 |         4 |      1 |       4 | 1.381s     | 26.693 |
|   32 | 28B  |  1 |  8 |         4 |      1 |       4 | 1.618s     | 17.720 |
|   64 | 61B  |  1 | 16 |         4 |      1 |       4 | 2.738s     | 11.406 |
|  128 | 109B |  1 | 32 |         4 |      1 |       4 | 4.234s     |  6.590 |
|  256 | 193B |  1 | 64 |         4 |      1 |       4 | 6.736s     |  3.667 |
|      |      |    |    |           |        |         |            |        |

The TFLops are very low because there are too few PP chunks/micro-batches (4) (gradient accumulation size / GAS) and so the bubble takes a lot of overhead, increasing PP chunks should dramatically improve performance but also need to lower the max model size to have memory to hold those chunks in.

**Performance**

These experiments are to try a lower model size, but much higher TFlops performance

| GPUs | Size | DP | PP | PP Chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | --------: | -----: |  -----: | ---------: | -----: | ----: |
|   16 | 18B  |  1 |  8 |        64 |      4 |     256 | 90.5s      |   26.1 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      4 |     512 | 177s       |   26.5 | 05-26 |
|   16 | 18B  |  1 |  8 |       256 |      4 |    1024 | 356s       |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   16 | 18B  |  1 |  4 |       128 |      4 |     512 | 179s       |   26.3 | 05-26 |
|   16 | 18B  |  1 |  4 |       128 |      6 |     768 | 262s       |   27.0 | 05-26 |
|   16 | 18B  |  1 |  8 |       128 |      6 |     768 | 259s       |   27.3 | 05-26 |
|   16 | 18B  |  1 |  8 |        32 |      8 |     256 | 89s        |   26.5 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   32 | 36B  |  1 |  8 |       128 |      4 |     512 | 82s        |   57.5 | 05-26 |
|   32 | 36B  |  1 |  8 |       128 |      6 |     768 | 123s       |   57.5 | 05-26 |
|   32 | 36B  |  1 |  8 |       256 |      6 |    1536 | 241s       |   58.7 | 05-26 |
|   32 | 36B  |  1 |  8 |       512 |      6 |    3072 | 478s       |   59.2 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 48B  |  1 | 16 |       256 |      4 |    1024 | 129s       |   48.7 | 05-25 |
|   64 | 48B  |  1 | 16 |       256 |      4 |    1024 | 217s       |   29.0 | 05-26 |
|   64 | 48B  |  1 | 16 |       256 |      4 |    1024 | 125s       |   50.3 | 05-27 |
|      |      |    |    |           |        |         |            |        |       |
|   64 | 48B  |  1 | 16 |       256 |      6 |    1536 | 328s       |   28.7 | 05-26 |
|   64 | 48B  |  1 | 16 |       256 |      8 |    2048 | 435s       |   28.9 | 05-26 |
|   64 | 48B  |  1 | 16 |       512 |      6 |    3072 | 650s       |   29.0 | 05-26 |
|   64 | 48B  |  1 | 16 |       512 |      8 |    4096 | 870s       |   28.9 | 05-26 |
|   64 | 48B  |  1 | 32 |       256 |      4 |    1024 | 220s       |   28.6 | 05-26 |
|      |      |    |    |           |        |         |            |        |       |


data:
- Size = Model Size
- `TP=4` in all of entries
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * pp_chunks * dp_size`
- PP chunks is the number of PP stages, so each pipeline handles `micro-batch-size * pp_chunks`
- Seq length is 1024

notes:
- 32gpus had a very snag fit for gpu memory for 36B model (others were in ~75%) so it might be a bit too risky to OOM-borderline


```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=127; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```


### Megatron + Deepspeed 3D


**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism` is not in sync with M-LM master - so several config args don't match. It's about 8 months old.

See scripts and logs under [gpt2-meg-ds-3d](./gpt2-meg-ds-3d).

Uses 3D:
- TP: tensor parallelism
- PP: pipeline parallelism
- DP: data parallelism

same features as Megatron's native, but improved by Deepspeed

**Performance**

| GPUs | Size | DP | PP | PP chunks | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | --------: | -----: | ------: | ---------: | -----: | ----: |
| 64   | 48B  | 1  | 16 | 256       | 4      | 1024    | 146s       | 43     | 05-27 |
|      |      |    |    |           |        |         |            |        |       |


- GAS = Gradient Accumulation size (same as PP_chunks / number of PP stages)
- Global_bs = pp_chunks*micro_bs*dp_size
- `TP_SIZE=4` (size of the node)

```
perl -le '$ng=64; $ms=48; $gbs=1024; $sp=146; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```




#### Megatron + Deepspeed ZeRO


**Important**: `DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3` is not in sync with M-LM master - so several config args don't match. It's about 8 months old.

See scripts and logs under [gpt2-meg-ds-zero](./gpt2-meg-ds-zero).

This one uses only TP from Megatron (no PP)

Not yet optimized with Deepspeed team!

**With Offload off**

**Performance**
| GPUs | Size  | DP | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ----: | -: |   ---: |  -----: | ---------: | -----: | ----: |
|   64 | 48B   | 16 |     48 |     768 | 122s       |  38.67 | 05-25 |
|   64 | 48B   | 16 |     48 |     768 | 127s       |  37.15 | 05-27 |
|      |       |    |        |         |            |        |       |
|      |       |    |        |         |            |        |       |


```
perl -le '$ng=64; $ms=48; $gbs=768; $sp=122; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```
- Seq length is 1024
- `TP=4` in all of entries
- `DP` is number of nodes here
- Throughput is time per iteration - to complete global batch size
- Global batch size is `micro-batch-size * dp-size`

- tried w/ and w/o Tiling once but saw no difference - perhaps would be more important on larger collections



| GPUs | Size | TP | DP | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ---: | -: | -: | -----: | ------: | ---------: | -----: | ----: |
|      |      |    |    |        |         |            |        |       |
|   64 | 48B  |  4 | 16 |     48 |     768 |        127 |  37.15 |       |
|   64 | 48B  |  2 | 32 |     32 |    1024 |        167 |  37.67 |       |
|   64 | 48B  |  1 | 64 |     16 |    1024 |        184 |  34.99 |       |
|   64 | 24B  |  1 | 64 |     16 |    1024 |       89.0 |  35.34 |       |
|   64 | 24B  |  2 | 32 |     32 |    1024 |       85.7 |  36.70 |       |

**With full cpu offload**

| GPUs | Size | TP | DP | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ---: | -: | -: | -----: | ------: | ---------: | -----: |
| 64   | 48B  | 4  | 16 | 64     | 1024    | 171s       | 39.71  |
|      |      |    |    |        |         |            |        |


**With full optim cpu offload**




### HF + Deepspeed Zero 3 + Full Offload

See scripts and logs under [gpt2-hf-ds](./gpt2-hf-ds).

Not yet optimized with Deepspeed team!

**Max model size**

| GPUs | Size  | Mic-BS | Glob-BS | Throughput | TFlops |
| ---: | ----: | -----: | ------: | ---------: | -----: |
|   16 | 23B   |      4 |      64 |  58.72s    |  25.66 |
|   32 | 48B   |      4 |     128 | 114.56s    |  13.72 |
|   64 | 91B   |      4 |     256 | 222.72s    |  13.38 |
|      |       |        |         |            |        |


**Performance**

| GPUs | Size  | Mic-BS | Glob-BS | Throughput | TFlops | Notes |
| ---: | ----: | -----: | ------: | ---------: | -----: | ----: |
|   64 | 48B   |      8 |     512 | 139.52s    |  22.54 | 05-25 |
|   64 | 48B   |      4 |     256 | 185s       |   8.50 | 05-27 |
|   64 | 48B   |      8 |     512 | 118.38     |  26.57 | 05-27 |
|      |       |        |         |            |        |       |

Don't seem to be able to enlarge the global bs here - OOMing

- gradient checkpointing activated

- DP=GPUs

- global bs = micro bs * DP

- Throughput reported by HF Trainer is samples_per_second per node - So total throughput in the table is `glob_bs/samples_per_second`

- TFlops: `model_size_in_B * 4 * 2 * seq * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`
```
perl -le '$ng=64; $ms=48; $gbs=512; $sp=139.52; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
22
```
