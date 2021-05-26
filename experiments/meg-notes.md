# Megatron-LM notes

## config

- Data Parallel: `data-parallel-size = world_size / (pipeline_model_parallel_size * tensor_model_parallel_size)`
   By default, `pipeline_model_parallel_size=`` and `tensor_model_parallel_size=1`


## some general performance notes:

NVIDIA paper: https://arxiv.org/abs/2104.04473v2

- they used 80GB A100s with 312TFlops/gpu (and achieved about 50% of that in the largest model/batch size (163TFlops)

- we are using 32GB V100s with 125TFlops/gpu (so if we reach about 60TFlops that would be fantastic)

Their main scaling table:

- model parallel size = tensor model parallel * pipeline model parallel

where tensor parallel is 8 at the most

So for example for 76B it says MP=32, which means 8 * 4 - so `PP_size=4` and `TP_size=8`

Basically use tensor model parallelism within a node, then use pipeline model parallelism for larger models
- So if MP size <= 8, tensor MP = MP size, pipeline MP = 1
- Otherwise, tensor MP = 8, pipeline MP = (MP size // 8 )

DataParallel isn't not in the table, it's:

DP = (total number of GPUs // MP size)

Here is the main table from the paper with added breakdown of TP/PP/DP:

| Model | Atten | Hidden | Lay | TP | PP | DP |  MP | GPUs | Micro | Global | TFlops | TFlops | TFlops |
| size  | heads |   size | ers |    |    |    |     |      |    BS |     BS |   /GPU |      % | Aggreg |
| ---:  | ----: | -----: | --: | -: | -: | -: | --: | ---: |  ---: | -----: |  ----: |  ----: | -----: |
| 1.7B  |    24 |   2304 |  24 |  1 |  1 | 32 |   1 |   32 |    16 |    512 |    137 |    44% |    4.4 |
| 3.6B  |    32 |   3072 |  30 |  2 |  1 | 32 |   2 |   64 |    16 |    512 |    138 |    44% |    8.8 |
| 7.5B  |    32 |   4096 |  36 |  4 |  1 | 32 |   4 |  128 |    16 |    512 |    142 |    46% |   18.2 |
| 18B   |    48 |   6144 |  40 |  8 |  1 | 32 |   8 |  256 |     8 |   1024 |    135 |    43% |   34.6 |
| 39B   |    64 |   8192 |  48 |  8 |  2 | 32 |  16 |  512 |     4 |   1536 |    138 |    44% |   70.8 |
| 76B   |    80 |  10240 |  60 |  8 |  4 | 32 |  32 | 1024 |     2 |   1792 |    140 |    45% |  143.8 |
| 145B  |    96 |  12288 |  80 |  8 |  8 | 24 |  64 | 1536 |     2 |   2304 |    148 |    47% |  227.1 |
| 310B  |   128 |  16384 |  96 |  8 | 16 | 15 | 128 | 1920 |     1 |   2160 |    155 |    50% |  297.4 |
| 530B  |   128 |  20480 | 105 |  8 | 35 |  9 | 280 | 2520 |     1 |   2520 |    163 |    52% |  410.2 |
| 1T    |   160 |  25600 | 128 |  8 | 64 |  6 | 512 | 3072 |     1 |   3072 |    163 |    52% |  502.0 |
|       |       |        |     |    |    |    |     |      |       |        |        |        |        |
