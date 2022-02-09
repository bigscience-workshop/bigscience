# Chronicles

Same as tr8-104B but using new additions such as:

1. Curriculum Learning (CL) https://arxiv.org/abs/2108.06084
2. BitsNBytes (BNB) https://github.com/facebookresearch/bitsandbytes

https://huggingface.co/bigscience/tr8b-104B-logs/tensorboard

## CL Experiment 1

Trying to figure out good baseline settings for CL

[baseline script](
https://github.com/bigscience-workshop/bigscience/blob/82fe642fb1eedd0361bac6899b79769e2c842c9f/train/tr8b-104B/tr8b-104B-cl.slurm)

Stopped training at iter 500:

![tr8b-104B-cl-exp-01.png](images/tr8b-104B-cl-exp-01.png)


## CL Experiment 2

finetuned exp 1 for more optimal performance

> Conglong Li

Here are my recommendation for next time: GPT-3 uses 375M token for LR warmup. Assuming the average seqlen is about 100 during LR warmup for CL (a very rough estimation), then we should set LR_WARMUP_SAMPLES= 375e6/100 = 3_750_000, this leads to 375e6/100/2048 = 1.8K warmup steps which sounds good to me

For peak LR, yeah 1e-4 might be a good next candidate, together with LR_WARMUP_SAMPLES=3_750_000

GPT-3 175B uses 6e-5 for batch size 1.6K, so 1e-4 for batch size 2K seems to be an appropriate/moderate increase.

Also change eval from every 1k to 150, since we can't tell from lm loss what's going on - we need the eval loss as it is reported for the full SEQLEN (whereas train lm loss just for the current CL SEQLEN instead).
150 since that's the current period between switching seqlen.

```
perl -pi -e 's|--lr 6e-5|--lr 1e-4|' *slurm
perl -pi -e 's|LR_WARMUP_SAMPLES=216_320|LR_WARMUP_SAMPLES=3_750_000|' *slurm
perl -pi -e 's|--eval-interval 1000|--eval-interval 150|' *slurm
```

[script](https://github.com/bigscience-workshop/bigscience/blob/d5fc4b22d7e88e87b4b9ec610b6c522b9a8c7a8d/train/tr8b-104B/tr8b-104B-cl.slurm)


## CL Experiment 3

Same as exp-2, but

```
    --lr 6e-5 \
    --embed-layernorm \
```

that is activating Embed LayerNorm that we found to be superior to all other experiments so far, and lowering `lr` to the same as the emb-norm experiments so that it's easier to compare the performance and quality.

```
perl -pi -e 's|--lr 1e-4|--lr 6e-5|' *cl*slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --embed-layernorm \\|' *cl*slurm
```

[script](https://github.com/bigscience-workshop/bigscience/blob/5bc0d43cb782291b48c98cfba2d55ce0188f9961/train/tr8b-104B/tr8b-104B-cl.slurm)



## BNB Experiment 1

[script](https://github.com/bigscience-workshop/bigscience/blob/7a1481355a1abe097a9fb2c9021c292cb9971da3/train/tr8b-104B/tr8b-104B-bnb.slurm)

![tr8b-104B-bnb-exp-01.png](images/tr8b-104B-bnb-exp-01.png)

Tim:

what I have seen before with linear quantization, is that a smaller Adam eps is needed for stability. I never see this to be required for 8-bit Adam with dynamic quantization, but in the beginning of training the optimizer is a bit more unstable

For linear I found that stability started from `adam-eps=1e-6`

I think 1e-5 degraded performance quite a bit, so I would try 1e-6 and 1e-7

I am not sure how the initialization is done. It could be that the initial initialization for the embedding layer is overwritten and that may cause instabilities
￼￼
I also see that you are using weight decay. I have not run many experiments with that and so unsure how the behavior is. For weight decay the AdamW formulation is used

When I tried 8-bit adam with fully sharded parallelism by just replacing the optimizer it did not work for me and I actually had a similar behavior as you see. Short decrease in loss and then stagnation. I think this could be related to the quantization statistics which are not properly synchronized across shards. But this is just a hunch. I think this could be tested by running a small model (maybe something like 500M params) and see if 8-bit Adam works there. If it does not work, it might be related to the quantization statistics
￼￼
So with xavier the max value for the embedding layer is 0.0106 and the 99% percentile value for N(0, 0.006) is 0.18 which is much larger. So it could just be the initialization
￼￼
I think 0.006 is still very high for the embedding layer. So that might be the issue, but could also the other things mentioned. I would leave the init value for the other layers if that worked for you

Stas:

I will add an experiment to leave the default init for the embed layer, and keep our 0.006 for the rest.

## BNB Experiment 2

So trying lower Adam eps:

```
--adam-eps 1e-6 \
```

```
perl -pi -e 's|--adam-eps 1e-8|--adam-eps 1e-6|' *bnb*.slurm
```

this made no difference, got an identical loss as exp 1


## BNB Experiment 3

Rollback to Exp 01, restore `--adam-eps`

```
perl -pi -e 's|--adam-eps 1e-6|--adam-eps 1e-8|' *bnb*.slurm
```

Try to turn optimizer sharding off - turn off ZeRO-1 - perhaps it doesn't work well with the 8-bit optimizer.

```
perl -pi -e 's|ZERO_STAGE=1|ZERO_STAGE=0|' *bnb*.slurm
```

Not sure if the setup won't OOM now. Got 31.7GB memory - it's borderline OOM.

no change, same trajectory

ZeRO-1's optim state sharding should be totally transparent, since it unshards the states before the optimizer gets to see them. But it was good to validate that in an experiment.

## BNB Experiment 4

Rollback to Exp 01,

Let's do a quick test with `--init-method-std 0.02` - we know it's not good for most of the model, but let's see if it impacts for the better the really early issue with BNB. If it does make things better then we can do the different init for different layers, so changing:

```
perl -pi -e 's|--init-method-std 0.006|--init-method-std 0.02|' *bnb*.slurm
```

Something is wrong there, as it very quickly stopped improving and got stuck at loss 8

![tr8b-104B-bnb-exp-04.png](images/tr8b-104B-bnb-exp-04.png)



## BNB Experiment 5

Discovered `StableEmbedding` wasn't integrated correctly in the original BNB PR, as it wasn't doing the right thing for split word embedding under TP>1, so fixing it in [PR182](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/182).

Rollback to Exp 01, no config change this time around.


We did emb-norm and bnb experiments (BNB Exp 5) in parallel and both tracked the same lm loss trajectory, here is the combination:

![tr8b-104B-emb-norm-exp-01-bnb-05.png](images/tr8b-104B-emb-norm-exp-01-bnb-05.png)

BNB started diverging just before. So we can tell BNB is more susceptible to instabilities.

Here is a zoomed in version:

![tr8b-104B-emb-norm-exp-01-bnb-05-zoom-in.png](images/tr8b-104B-emb-norm-exp-01-bnb-05-zoom-in.png)


## BNB Experiment 6

Tim suggested that dropping to a lower LR faster and having the min-lr lower helped a lot in his experiments, so let's try that:



```
 iteration     6502/  159576 | consumed samples:       216960 | consumed tokens:    444334080 | elapsed time per iteration (ms): 31074.1 | learning rate: 5.997E-05 | global batch size:    80 | lm loss: 3.876781E+00 | loss scale: 4096.0 | grad norm: 3246.595 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6503/  159576 | consumed samples:       217040 | consumed tokens:    444497920 | elapsed time per iteration (ms): 31065.8 | learning rate: 6.000E-05 | global batch size:    80 | lm loss: 4.023108E+00 | loss scale: 4096.0 | grad norm: 3670.127 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6504/  159576 | consumed samples:       217120 | consumed tokens:    444661760 | elapsed time per iteration (ms): 31073.4 | learning rate: 6.000E-05 | global batch size:    80 | lm loss: 4.030526E+00 | loss scale: 4096.0 | grad norm: 2954.856 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |

 ...


  iteration     8600/  159576 | consumed samples:       464560 | consumed tokens:    951418880 | elapsed time per iteration (ms): 66451.1 | learning rate: 6.000E-05 | global batch size:   160 | lm loss: 3.407058E+00 | loss scale: 8192.0 | grad norm: 3035.816 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
  # the log seems to still report 6.000E-05 for a while but TB starts decaying already here - probably a rounding to 3 points issue 6564 is the last one of 6e-5 on TB
```

but I don't have those older checkpoints, the only one I have is `global_step6000` so will use it.

will try: 1% from lr: `--min_lr 6e-7` so it should decay 10x faster.

And reducing `--lr-decay-samples` from `126_953_125` to `12_695_312`

But one can't change the lr config once the training started, so getting:

```
AssertionError: AnnealingLR: class input value 1e-06 and checkpointvalue 1e-05 for minimum learning rate do not match
```

But discovered a new option which seems to allow an override:

```
    --override-lr-scheduler \
```

So going to change the plan and switch to a recent `global_step12000` checkpoint instead, some time before the divergence:

```
 iteration    12000/  159576 | consumed samples:      1519744 | consumed tokens:   3112435712 | elapsed time per iteration (ms): 207967.5 | learning rate: 5.999E-05 | global batch size:   528 | lm loss: 2.985657E+00 | loss scale: 262144.0 | grad norm: 79683.827 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

so the next step with:

```
    --min-lr 6e-7 \
    --lr-decay-samples 12_695_312 \
    --override-lr-scheduler \
```

dropped the learning rate to `5.842E-05` from `5.999E-05`

```
 iteration    12001/  159576 | consumed samples:      1520272 | consumed tokens:   3113517056 | elapsed time per iteration (ms): 279206.6 | learning rate: 5.842E-05 | global batch size:   528 | lm loss: 3.029124E+00 | loss scale: 262144.0 | grad norm: 60983.653 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

so let's see if that makes a difference.


Clearly LR didn't get low fast enough and it diverged too, but even sooner than bnb-exp-5!

![tr8b-104B-bnb-exp-06.png](images/tr8b-104B-bnb-exp-06.png)



## Embed-Norm Experiment 1


Since we discovered BNB did so well, we decided to try just adding Embedding LayerNorm to the normal training. So did an experiment that is the same as Exp12 but with `--embed-layernorm` enabled.

[tr8b-104B-emb-norm-64n.slurm](/.tr8b-104B-emb-norm-64n.slurm)


It worked really well till 14k and then diverged

![tr8b-104B-emb-norm-exp-01.png](images/tr8b-104B-emb-norm-exp-01.png)



## Embed-Norm Experiment 2

Let's try to first restart with some data skipping to see if data was the issue:

1. Rollback to 13250 and skip data till 14k:

```
 iteration    13251/  159576 | consumed samples:      2333840 | consumed tokens:   4779704320 | elapsed time per iteration (ms): 220631.9 | learning rate:
 5.996E-05 | global batch size:   800 | lm loss: 2.924229E+00 | loss scale: 524288.0 | grad norm: 122160.116 | num zeros: 0.0 | number of skipped iteratio
ns:   0 | number of nan iterations:   0 |
...
 iteration    14000/  159576 | consumed samples:      3014320 | consumed tokens:   6173327360 | elapsed time per iteration (ms): 255453.3 | learning rate: 5.994E-05 | global batch size:  1024 | lm loss: 2.898812E+00 | loss scale: 4096.0 | grad norm: 2553.971 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

so will repeat Exp 1 with `--skip-train-iteration-range 13251-14000`

Worked well for a while and then started flatting out around 17k and then went on a roller coaster around 18k.

![tr8b-104B-emb-norm-exp-02.png](images/tr8b-104B-emb-norm-exp-02.png)


## Embed-Norm Experiment 3

Repeat the same as last time

1. Rollback to 16651 and skip data till 18500:

so will repeat Exp 2 with `--skip-train-iteration-range 13251-14000 16652-18500`

Actually got the same problem as exp 2 but arriving even sooner:


![tr8b-104B-emb-norm-exp-03.png](images/tr8b-104B-emb-norm-exp-03.png)



## Embed-Norm Experiment 4

Repeat the same as last time but let's try another data range. Seeing how the rollercoaster started around 18k, let's go for 19500.

1. Rollback to 16651 and skip data till 19500:

so will repeat Exp 2 with `--skip-train-iteration-range 13251-14000 16651-19500`

It didn't help. It exhibited the same behavior as Exp 2 and 3.

![tr8b-104B-emb-norm-exp-04.png](images/tr8b-104B-emb-norm-exp-04.png)

Next, will try to reset the optimizer.


## Embed-Norm Experiment 5

(this experiment was doomed to fail since later we discovered that `--no-load-optim` was being silently ignored with `--deepspeed` - i.e. wasn't ported, so skip to exp 6, which repeats this one identically except after fixing Meg-DS to respect`--no-load-optim` with  `--deepspeed`)

After 100 iterations on lr=0 after reset (which wasn't reset as discovered later), resume started with a spike, started recovering from it but started diverging again. It's still interesting to observe so saving the outcome:

![tr8b-104B-emb-norm-exp-05.png](images/tr8b-104B-emb-norm-exp-05.png)

So Exp-6 is what exp-5 was meant to be after making `--no-load-optim` work.



## Embed-Norm Experiment 6

So 2 data skipping attempts didn't help. Let's try resetting the optimizer states next.

**Half-way optimizer reset method**:

- reset optimizer - don't load the previous states from the checkpoint with the help of `--no-load-optim`﻿
- since we can't do lr warm up half-way through the training we will cheat and simply run the optimizer w/o updates to the weights by setting `lr=0` - now let it train for this number of iterations to emulate warm up (1/(1-0.95)) * 5 = 100 (beta2 = 0.95)
- then resume normal training, after restoring the setup to normal

Note:
- Make sure the `step` counter for bias correction are reset when starting the optimizer from scratch - checked that apex's `FusedAdam` does that already.


Steps:

1. Rollback tb/checkpoint to 16800 (last stable low loss point)
2. Calculate how to get the framework to run for 100 extra iterations and stop

```
 iteration    16800/  159576 | consumed samples:      7594208 | consumed tokens:  15552937984 | elapsed time per iteration (ms): 384505.8 | learning rate: 5.955E-05 | global batch size:  2048 | lm loss: 2.682074E+00 | loss scale: 524288.0 | grad norm: 180376.315 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 35.24 |
 iteration    16801/  159576 | consumed samples:      7596256 | consumed tokens:  15557132288 | elapsed time per iteration (ms): 400291.6 | learning rate: 5.955E-05 | global batch size:  2048 | lm loss: 2.657616E+00 | loss scale: 524288.0 | grad norm: 226760.401 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 33.85 |
```

Each iteration is 2048 samples at this point and thus we want to run for an additional 204800 samples, and thus we know we want to stop at 7799008 (7594208+204800). 7594208 was consumed samples counter at iteration 16800. i.e. the new setting is `--train-samples 7799008`

For the optimizer reset run we need to add:
```
    --no-load-optim \
    --override-lr-scheduler \
```
and change:
```
    --lr 0 \
    --min-lr 0 \
    --train-samples 7799008 \
```

Automating the change:
```
git checkout tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --no-load-optim \\|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--lr 6e-5|--lr 0|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--min-lr 6e-6|--min-lr 0|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--train-samples 300_000_000|--train-samples 7_799_008|' tr8b-104B-emb-norm-64n.slurm
```


1. Run the optimizer reset job once as set up above

2. The next job should run mostly on the normal slurm script to the original as the optimizer should have been warmed up

once (1) started running, back it up and restore the original:
```
cp tr8b-104B-emb-norm-64n.slurm tr8b-104B-emb-norm-64n.slurm.reset-optim
git checkout tr8b-104B-emb-norm-64n.slurm
```
but the checkpoint from step (1) will now have wrong lr info, so we again need to tell megatron to ignore it and use the normal lr setup from the command line:

```
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr8b-104B-emb-norm-64n.slurm
```

3. Once (2) has started running and all looks good we can then fully reset it to normal to remove `--override-lr-scheduler`

```
git checkout tr8b-104B-emb-norm-64n.slurm
```

So on step 2 when resuming from a warmed up reset optimizer there was a huge spike and then mostly recovery followed by a divergence.

Exp 5 and 6 have a strangely almost identical trajectory, even though exp 5 didn't actually reset the optimizer, whereas exp 6 did.

![tr8b-104B-emb-norm-exp-06.png](images/tr8b-104B-emb-norm-exp-06.png)


## Embed-Norm Experiment 7

We continue with the optimizer reset experiment.

There was a bug in deepspeed that wasn't saving/restoring optimizer's `group['step']` so on every resume bias_correction was significant, but it should only happen in the first 1k iterations or so (or after optimizer was reset).

So let's try again with the fixed deepspeed and will do a longer warmup.

Let's 300 steps on lr=0 to make the reset optimizer warm up longer.

Steps:

1. Rollback tb/checkpoint to 16800 (like exp 6)
2. Calculate how to get the framework to run for 300 iterations and stop

Repeating from previous step we know we want to stop at 8208608 (7594208+204800*3).

For the optimizer reset run we need to add:
```
    --no-load-optim \
    --override-lr-scheduler \
```
and change:
```
    --lr 0 \
    --min-lr 0 \
    --train-samples 8_208_608 \
```

Automating the change:
```
git checkout tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --no-load-optim \\|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--lr 6e-5|--lr 0|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--min-lr 6e-6|--min-lr 0|' tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|--train-samples 300_000_000|--train-samples 8_208_608|' tr8b-104B-emb-norm-64n.slurm
```

Once this 1st job is running, the next 2nd job is just to override the lr-scheduler:

```
cp tr8b-104B-emb-norm-64n.slurm tr8b-104B-emb-norm-64n.slurm.reset-optim-try-3
git checkout tr8b-104B-emb-norm-64n.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr8b-104B-emb-norm-64n.slurm
```

And once the 2nd job starts, and it looks good we can then resume normally from the checkpoint for the subsequent job:

```
git checkout tr8b-104B-emb-norm-64n.slurm
```

Actually due to SLURM and not being able to be awake 24/7, I managed to get the first step to run for iterations 16801-17059 - so 259 warmup iterations instead of planned 300, so this is close enough for the sake of the experiment. And thus the resume with the warmed up optimizer happened at iteration 17060.

After warm up, just like with the 100-iteration long optimizer reset/warmup there was a spike but much smaller this time - only 2.7 to 4.4 and then quickly recovering to 2.7:
```
 iteration    17060/  159576 | consumed samples:      8126688 | consumed tokens:  16643457024 | elapsed time per iteration (ms): 573125.8 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 2.665295E+00 | loss scale: 1048576.0 | grad norm: 453082.517 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.004 | TFLOPs: 23.64 |
 iteration    17061/  159576 | consumed samples:      8128736 | consumed tokens:  16647651328 | elapsed time per iteration (ms): 423042.3 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 2.667121E+00 | loss scale: 1048576.0 | grad norm: 464441.887 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.03 |
[2022-01-19 13:09:13,305] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576.0, reducing to 1048576.0
 iteration    17062/  159576 | consumed samples:      8130784 | consumed tokens:  16651845632 | elapsed time per iteration (ms): 422832.0 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.374269E+00 | loss scale: 1048576.0 | grad norm: 464441.887 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.04 |
[2022-01-19 13:16:17,683] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576.0, reducing to 524288.0
 iteration    17063/  159576 | consumed samples:      8132832 | consumed tokens:  16656039936 | elapsed time per iteration (ms): 424377.8 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.365822E+00 | loss scale: 524288.0 | grad norm: 464441.887 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 31.93 |
[2022-01-19 13:23:19,412] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 524288.0, reducing to 262144.0
 iteration    17064/  159576 | consumed samples:      8134880 | consumed tokens:  16660234240 | elapsed time per iteration (ms): 421729.4 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.391485E+00 | loss scale: 262144.0 | grad norm: 464441.887 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.13 |
[2022-01-19 13:30:23,180] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144.0, reducing to 131072.0
 iteration    17065/  159576 | consumed samples:      8136928 | consumed tokens:  16664428544 | elapsed time per iteration (ms): 423768.1 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.404639E+00 | loss scale: 131072.0 | grad norm: 464441.887 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 31.97 |
 iteration    17066/  159576 | consumed samples:      8138976 | consumed tokens:  16668622848 | elapsed time per iteration (ms): 421634.1 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.398458E+00 | loss scale: 131072.0 | grad norm: 879946.622 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.13 |
[2022-01-19 13:44:28,157] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072.0, reducing to 65536.0
 iteration    17067/  159576 | consumed samples:      8141024 | consumed tokens:  16672817152 | elapsed time per iteration (ms): 423342.2 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.096038E+00 | loss scale: 65536.0 | grad norm: 879946.622 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.00 |
[2022-01-19 13:51:30,908] [INFO] [stage_1_and_2.py:1644:step] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536.0, reducing to 32768.0
 iteration    17068/  159576 | consumed samples:      8143072 | consumed tokens:  16677011456 | elapsed time per iteration (ms): 422752.4 | learning rate: 5.948E-05 | global batch size:  2048 | lm loss: 4.137348E+00 | loss scale: 32768.0 | grad norm: 879946.622 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 0.005 | TFLOPs: 32.05 |
```

This attempt failed too - the model had a huge spike and diverged into NaN loss.

![tr8b-104B-emb-norm-exp-07.png](images/tr8b-104B-emb-norm-exp-07.png)

So at this moment we are stopping this experiment as we must focus on the 200B setup and we are going to use bf16 there and hope that things will be much better with that dtype.





## On the importance of saving and restoring the optimizer step counter

Since SLURM limits us to 20h-runs we have to save the optim states and everything else at the end of each run and then resume from the checkpoint on the next run. What happens if some things fail to be restored on resume?

While investigating the optimizer reset I've found a bug in Deepspeed (on its own and as part of Meg-DS). As it doesn't saves/restores `group['step']` and so` bias_correction` is not done according to the algorithm.

In pytorch's `Adam`, step is part of a param's `state_dict` and so it gets saved and restored with the `state_dict`:
https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adamw.py#L81-L85

In apex's `FusedAdam`, which is what we use now, `step` is not part of the each param's state, but wisely is maintained only in one global copy per param group:
https://github.com/NVIDIA/apex/blob/b88c507edb0d067d5570f7a8efe03a90664a3d16/apex/optimizers/fused_adam.py#L111-L114

The problem is that because it isn't part of the param's state it doesn't get saved and restored.

So on every resume one starts with `step=1` and not `step=iteration`. I verified that empirically by dumping `optimizer.optimizer.param_groups[0]['step']`

It's pretty clear that Deepspeed itself isn't aware of this state variable:
https://github.com/microsoft/DeepSpeed/blob/3293cf72a0abd5cf77a831996bd054bc908476a6/deepspeed/runtime/zero/stage_1_and_2.py#L1958-L1980

Which leads us to:
```
bias_correction1 = 1 - beta1 ** state['step']
bias_correction2 = 1 - beta2 ** state['step']
```
so on each resume we end up with a smaller bias_correction than it should normally be. since `beta2**1 >> beta2**150000` or `0.95 >> 0`
So typically `bias_correction` becomes `1` in about 1k iterations, but here it's not and on every resume there is a `bias_correction` of `[0.1,0.05]` in the first iteration after resume.

This bug has been fixed in Deepspeed here: https://github.com/microsoft/DeepSpeed/pull/1525 and the
fix should be available in DeepSpeed v0.5.10+.

Follow up notes from Tim Dettmers:

> I wonder if this a confounding factor for the "decrease learning rate from checkpoint" experiments. The bias corrections are there to have the adam states move more quickly away from zero at the start from training. However, if you reload and optimizer and the gradient has similar magnitude as the adam state (this would be expected in most cases) then learning rate is increased by 2.2x - 1.15x for the first 5 iterations and it stabilizes around less extreme values after 10 iterations. Not sure if that initial high learning rate could move the weights into a space of instability


## 1.3B optim-reset experiment

Let's train a very small model and see how optimizer reset goes there in various stages.

1. Train from scratch for 5M samples w/ beta2=0.95 to match our 104B experiment

This will run iterations till iteration 19686

train/tr3-1B3-baseline/tr3m-1B3-emb-norm-pile-optim-reset.slurm

lm loss: 2.134260E+00


2. optimizer reset and 100 iterations warmup (beta2=0.95)

100 iterations @ bs=512 51_200 samples: 5_000_000+51_200=5_051_200

```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --no-load-optim \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|--lr 2e-4|--lr 0|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|--min-lr 1e-5|--min-lr 0|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|TRAIN_ITER=5_000_000|TRAIN_ITER=5_051_200|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
```

This will run iterations 19687 - 19786

3. resume after reset - let's do 1000 iterations +512_000 samples: 5_563_200


```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|TRAIN_ITER=5_000_000|TRAIN_ITER=5_563_200|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
```

This will run iterations 19786 - 20786

4. restore to normal once the above completes


```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
```


5. Let's train some longer - to 20M samples and repeat the rest at a different point in the training curve

```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|TRAIN_ITER=5_000_000|TRAIN_ITER=20_000_000|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
```

6. repeat reset and warm up

and 100 iterations warmup (beta2=0.95)

100 iterations @ bs=512 51_200 samples: 20_000_000+51_200=20_051_200


```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --no-load-optim \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|--lr 2e-4|--lr 0|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|--min-lr 1e-5|--min-lr 0|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|TRAIN_ITER=5_000_000|TRAIN_ITER=20_051_200|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
```

This will run iterations 48984 - 49084

7. resume after reset - let's do 1000 iterations +512_000 samples: 20_563_200


```
git checkout tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|(--checkpoint-activations \\)|$1\n    --override-lr-scheduler \\|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
perl -pi -e 's|TRAIN_ITER=5_000_000|TRAIN_ITER=20_563_200|' tr3m-1B3-emb-norm-pile-optim-reset.slurm
```

This seems to have worked just fine in both cases, the model just continued training normally, w/o any spikes or lost progress.
