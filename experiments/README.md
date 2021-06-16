# Experiments and Results


- [GPT2 Experiments](./gpt2.md) - main working doc with all the experiment results




The following is mostly outdated and should be recycled/updated:


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
