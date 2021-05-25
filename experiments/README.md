# Experiments and Results

(please edit this doc as we gather info)

Model

* Basically GPT, but with slightly different attention/loss mask (see agenda above)
* Targeting 175-200B parameters, seq length 512 or 1024, adam optimizer
* Note: our nodes have 128GB total gpu memory each -> need sharding (model weights alone are 350GB in fp16!)
* Note 2: may need to adjust expectations based on estimated training time (pending)
* Subsequent runs: consider encoder-decoder arch? longer sequences? adafactor?

Making it train:

* Candidates for run1:  deepspeed-gpt and megatron-gpt. Save fairscale/t5/custom for later. Use reference training code instead of integrating.
* Only consider training nodes with 4x V100-32GB because there are too few 8x nodes in JZ
* Current strategy: fully sharded data-parallel between nodes, arbitrary black magic within one node.

For each framework, we need to measure (hackathon participants + Sidd + Laurel)
* maximum number of parameters that we can train on 1 node
* minimum number of nodes to fit 175B-parameter model  (num_layers/dmodel/dhead: from gpt3 table 2.1)
* training throughput (i.e. training tokens per second in both configurations)
* Once we get through the initial debugging, ask microsoft/nvidia teams to sanity-check our configs for DS/Megatron respectively


Investigate:
* @Stas measured allreduce bandwidth between two nodes as ~135Gbps, now he's is investigating if we can get closer to the theoretical max = 400Gbps
* Estimated training time for 175B parameters - based on training throughput in megatron/deepspeed tech reports; estimated time to "reasonable performance". (TODO @Yozh and @Max Ryabinin). Sanity-check whether the 8-node servers are indeed too few to train in reasonable time.
* Figure out the how to introduce semi-autoregressive attention mask (and loss mask) to the code of deepspeed and megatron-LM training configs (looking for volunteers!)

Next steps:
* After Stas figures out the bandwidth, we need to measure how performance scales with #nodes using data-parallel training.
* Ask/measure how many server nodes can we reliably provision with JZ. Find the optimal trade-off between #nodes vs training time.
* a quick preliminary run with the same configuration, but fewer parameters ... and we can unleash the beast!


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
