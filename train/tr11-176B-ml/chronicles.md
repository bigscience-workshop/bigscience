# The final training

Trials and tribulations during the 176B 250k multi-lingual training.

For the trials and tribulation during the preparation stage before the launch  see: [chronicles-prequel](chronicles-prequel.md).

## main



### 2022-03-11

Launch

The training launched on March 11, 2022 11:42am PST



### 2022-03-14

Switched from MBS=1 to MBS=2 at GBS=784, as now there is enough microbatches in each of the 8 replicas to fill each pipeline 4 times. With MBS=2 each replica has `784/(2*8) = 49` microbatches over 12 PP stages which is already pretty efficient. (`GBS/(MBS*DP)`)




### 2022-03-18

7 days after the launch the training finished the batch size ramp up stage and we are now at full throttle of GBS=2048 and at 149-150 TFLOPs.

A single train iteration is about 105 secs.

A single eval iteration is about 12 min (1 iteration eval on each of 29 datasets) - we perform it once every 1k iteration and it "costs" about 7 iterations or 0.7% of training.

A checkpoint is saved in 40 secs, which is about 1/2 iteration duration.

We consumed 20B/450B tokens.

At the current speed and no downtime we need 125 days more `(115311-12695)*105/(3600*24)=124.7` to finish the plan.

the math is based on the recent log:

```
 [default7]: iteration    12695/  115311 | consumed samples:      9841584 | consumed tokens:  20155564032 | elapsed time per iteration (s): 105.22 | learning rate: 5.969E-05 | global batch size:  2048 | lm loss: 2.463556E+00 | grad norm: 0.174 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.463 | TFLOPs: 149.02 |
```


### What makes 176B-ml training so stable?

To compare: at [104B-en experiments](https://huggingface.co/bigscience/tr8b-104B-logs/tensorboard) we failed to cross the 24B-tokens barrier.

![104B-en-24B-tokens-fail](images/104B-en-24B-tokens-fail.png)

At 176B-ml we have crossed [24B-tokens barrier](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard#scalars&tagFilter=loss%20vs%20tokens&_smoothingWeight=0) w/o a single instability issue:

![176B-ml-24B-tokens-succeed](images/176B-ml-24B-tokens-succeed.png)

It's hard to tell if there is one specific improvement that made the biggest impact w/o doing ablation studies, which would be both expensive and time consuming.

It's probably a combination of a few or all of the following improvements:

1. Very clean data
2. BF16 mixed precision regime
3. FP32 grad accumulation for the pipeline
4. A very low standard initialization `sqrt(0.3333/NHIDDEN)` - in 104B-en used `sqrt(0.4/NHIDDEN)`
5. Word embedding layer norm - but it was also used in 104B-en

If you believe in metaphysics perhaps one more important factor is the "intentional" contribution of many thousands of ML enthusiasts around the world who want this largest open source multilingual language model training to succeed.
