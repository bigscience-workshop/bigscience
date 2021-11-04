## Rule of thumb for optimizing TFLOPS

Given n gpus, we're interested in finding the configuration that allows us to run the model the fastest:

When to use DP:
 - Whenever you can. Use as much DP as you can.
 - It does have a negative impact if the number `$GBS / $MBS` is close to DP as you end up losing pipeline efficiency

When to use TP:
 - When the largest layer does not fit into a single gpu (along with all the activation, optimizer states and gradient memory).
 - TP is communication heavy, so you should never go beyond the number of gpus available in a single node

When to use PP:
 - When the entire model doesn't fit in a single gpu.

The recipe goes as follow:
 1) Determine TP*PP (we'll refer to this value as MP later on):
    1) Try and compare with some existing similar architecture (13B GPT needed 8 GPUS for one replica, ie TP*DP = 8)
       1) The factor in model size should be roughly the same as the factor in gpus
    2) Empiric rule: model_size*18 < 75% of gpu (to take in account additional activation memory)
       1) If that is `True` then you don't need any model parallelism
    3) Test different configurations with a single replica starting from TP=1/PP=1 with a single replica (DP=1) until you don't have OOM errors
 2) You usually want PP=$MP unless a single layer doesn't fit in a single gpu in which case TP is necessary:
    1) You can use the rule the empiric rule in 1.ii for single layers to get an idea if you need TP or not.

Bear in mind that this is just guidelines that will help you to quickly narrow down the configuration options. But you should still try a few different configurations and see which one gives the best throughput.

Also watch the gpu memory usage by logging into one of the nodes. You don't want it to be too close to the max. And also you don't want to have a lot of free gpu memory available. If there is then you can tune things up more to squeeze a higher gpu utilization. e.g. you can benchmark raising MBS to use the free memory. But test the impact, since it doesn't always make things faster.

Additionally, be aware of [the different constraints](https://github.com/bigscience-workshop/bigscience/blob/master/train/sanity-checks.md).

Here is [how to calculate TFLOPs](https://github.com/bigscience-workshop/bigscience/tree/master/math#calculate-tflops).

