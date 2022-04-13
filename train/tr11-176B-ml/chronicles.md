# The final training

Trials and tribulations during the 176B 250k multi-lingual training.

For the trials and tribulation during the preparation stage before the launch  see: [chronicles-prequel](chronicles-prequel.md).

The full spec is [here](README.md) and the script is [here](tr11-176B-ml.slurm).

## main



### 2022-03-11

Launch

The training launched on March 11, 2022 11:42am PST



### 2022-03-14

Switched from MBS=1 to MBS=2 at GBS=784, as now there is enough microbatches in each of the 8 replicas to fill each pipeline 4 times. With MBS=2 each replica has `784/(2*8) = 49` microbatches over 12 PP stages which is already pretty efficient. (`GBS/(MBS*DP)`)




### 2022-03-18

7 days after the launch the training finished the batch size ramp up stage and we are now at full throttle of GBS=2048 and at 149-150 TFLOPs per GPU.

A single train iteration is about 105 secs.

A single eval iteration is about 12 min (1 iteration eval on each of 29 datasets) - we perform it once every 1k iteration and it "costs" about 7 iterations or 0.7% of training.

A checkpoint is saved in 40 secs, which is about 1/2 iteration duration.

We consumed 20B/450B tokens.

At the current speed and no downtime we need 125 days more `(115311-12695)*105/(3600*24)=124.7` to finish the plan.

the math is based on the recent log:

```
 [default7]: iteration    12695/  115311 | consumed samples:      9841584 | consumed tokens:  20155564032 | elapsed time per iteration (s): 105.22 | learning rate: 5.969E-05 | global batch size:  2048 | lm loss: 2.463556E+00 | grad norm: 0.174 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.463 | TFLOPs: 149.02 |
```


### What makes the 176B-ml training so stable?

To compare: at [104B-en experiments]
(https://huggingface.co/bigscience/tr8b-104B-logs/tensorboard?tab=scalars&runSelectionState=eyJ0ZW5zb3Jib2FyZC9iYXNlLWV4cC0xMSI6ZmFsc2UsInRlbnNvcmJvYXJkL2NsLWV4cC0wMSI6ZmFsc2UsInRlbnNvcmJvYXJkL2NsLWV4cC0wMiI6ZmFsc2UsInRlbnNvcmJvYXJkL2JuYi1leHAtMDEiOmZhbHNlLCJ0ZW5zb3Jib2FyZC9ibmItZXhwLTAyIjpmYWxzZSwidGVuc29yYm9hcmQvYm5iLWV4cC0wMyI6ZmFsc2UsInRlbnNvcmJvYXJkL2JuYi1leHAtMDQiOmZhbHNlLCJ0ZW5zb3Jib2FyZC9iYXNlLWV4cC0xMiI6ZmFsc2UsInRlbnNvcmJvYXJkL2JuYiI6ZmFsc2UsInRlbnNvcmJvYXJkL2JuYi1leHAtMDUiOmZhbHNlLCJ0ZW5zb3Jib2FyZC9lbWItbm9ybSI6dHJ1ZSwidGVuc29yYm9hcmQvZW1iLW5vcm0tMDEiOnRydWUsInRlbnNvcmJvYXJkL2VtYi1ub3JtLTAyIjp0cnVlLCJ0ZW5zb3Jib2FyZC9lbWItbm9ybS0wMyI6dHJ1ZSwidGVuc29yYm9hcmQvZW1iLW5vcm0tMDQiOnRydWUsInRlbnNvcmJvYXJkL2VtYi1ub3JtLTA1Ijp0cnVlLCJ0ZW5zb3Jib2FyZC9lbWItbm9ybS0wNiI6dHJ1ZSwidGVuc29yYm9hcmQvY2wiOmZhbHNlLCJ0ZW5zb3Jib2FyZC9jbC1hMTAwIjpmYWxzZX0%253D&tagFilter=loss%2520vs%2520tokens#scalars&runSelectionState=eyJ0ZW5zb3Jib2FyZC9iYXNlLWV4cC0xMSI6dHJ1ZSwidGVuc29yYm9hcmQvY2wtZXhwLTAxIjpmYWxzZSwidGVuc29yYm9hcmQvY2wtZXhwLTAyIjpmYWxzZSwidGVuc29yYm9hcmQvYm5iLWV4cC0wMSI6ZmFsc2UsInRlbnNvcmJvYXJkL2JuYi1leHAtMDIiOmZhbHNlLCJ0ZW5zb3Jib2FyZC9ibmItZXhwLTAzIjpmYWxzZSwidGVuc29yYm9hcmQvYm5iLWV4cC0wNCI6ZmFsc2UsInRlbnNvcmJvYXJkL2Jhc2UtZXhwLTEyIjp0cnVlLCJ0ZW5zb3Jib2FyZC9ibmIiOnRydWUsInRlbnNvcmJvYXJkL2JuYi1leHAtMDUiOnRydWUsInRlbnNvcmJvYXJkL2VtYi1ub3JtIjp0cnVlLCJ0ZW5zb3Jib2FyZC9lbWItbm9ybS0wMSI6dHJ1ZSwidGVuc29yYm9hcmQvZW1iLW5vcm0tMDIiOnRydWUsInRlbnNvcmJvYXJkL2VtYi1ub3JtLTAzIjp0cnVlLCJ0ZW5zb3Jib2FyZC9lbWItbm9ybS0wNCI6dHJ1ZSwidGVuc29yYm9hcmQvZW1iLW5vcm0tMDUiOnRydWUsInRlbnNvcmJvYXJkL2VtYi1ub3JtLTA2Ijp0cnVlLCJ0ZW5zb3Jib2FyZC9jbCI6ZmFsc2UsInRlbnNvcmJvYXJkL2NsLWExMDAiOmZhbHNlfQ%3D%3D&tagFilter=loss%20vs%20tokens) we failed to cross the 24B-tokens barrier. You can find the indepth details in [the 104B chronicles](../tr8b-104B/chronicles.md).

![104B-en-24B-tokens-fail](images/104B-en-24B-tokens-fail.png)

At 176B-ml we have crossed [24B-tokens barrier](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard?tab=scalars&tagFilter=loss%2520vs%2520tokens) w/o a single instability issue:

![176B-ml-24B-tokens-succeed](images/176B-ml-24B-tokens-succeed.png)

Here is an overlapping image:

![104B-176B-overlap-24B-tokens](images/104B-176B-overlap-24B-tokens.png)

It's hard to tell if there is one specific improvement that made the biggest impact w/o doing ablation studies, which would be both expensive and time consuming.

It's probably a combination of a few or all of the following improvements:

1. Very clean data
2. BF16 mixed precision regime
3. FP32 grad accumulation for the pipeline
4. A very low standard initialization `sqrt(0.3333/NHIDDEN)` - in 104B-en used `sqrt(0.4/NHIDDEN)`
5. Word embedding layer norm - but it was also used in 104B-en

If you believe in metaphysics perhaps one more important factor is the "intentional" contribution of many thousands of ML enthusiasts around the world who want this largest open source multilingual language model training to succeed.



### 2022-03-21

One of the GPUs crashed at 2022-03-21_13:34:34 JZ time:

```
[default3]:terminate called after throwing an instance of 'c10::CUDAError'
[default3]:  what():  CUDA error: unknown error
[default3]:Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:95 (most recent call first):
[default3]:frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x42 (0x1518a06137d2 in /gpfswork/rech/six/commun/conda/tr11-176B-ml/lib/python3.8/site-packages/torch/lib/libc10.so)
[default3]:frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x11a (0x15190c6c29ea in /gpfswork/rech/six/commun/conda/tr11-176B-ml/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
[default3]:frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x50 (0x15190c6c4fd0 in /gpfswork/rech/six/commun/conda/tr11-176B-ml/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
[default3]:frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x145 (0x15190c6c6265 in /gpfswork/rech/six/commun/conda/tr11-176B-ml/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
[default3]:frame #4: <unknown function> + 0xc2ba3 (0x15191b339ba3 in /lib64/libstdc++.so.6)
[default3]:frame #5: <unknown function> + 0x814a (0x15191c77d14a in /lib64/libpthread.so.0)
[default3]:frame #6: clone + 0x43 (0x15191c4acdc3 in /lib64/libc.so.6)
```

we lost 7.3h of work, it failed 2 iterations before checkpoint saving time :(

Will switch to more frequent checkpoint saving of 200 iterations and will lower it further if the hardware failures continue. With 2.3TB checkpoint size and 40 secs to save a checkpoint we don't want to do it too often.



### 2022-03-22

Switched to `SAVE_INTERVAL=100` as JeanZay had some emergency maintenance downtime and we lost some hours again, so now with 100 iterations per checkpoint we would lose at most 3h.



### 2022-03-24 grad clip TP sync bug fixing

We discovered a bug in BF16Optimizer that didn't clip gradients in TP ranks > 1, leading to layer norm weights and biases getting out of sync. Mathematically they all should have the same layer norm weights since the TP splitting is virtual. This doesn't impact the model while it's training, as each TP rank just learns its own weights. But it will be a problem once the model training has been finished and TP ranks need to be merged from 4 to 1. Since if we use one of them or average the model's correctness will be impacted.

Thomas Wang wrote a test that reproduces the problem with a small test:
`test_layer_norm_consistent` in https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/271

And the bug in BF16Optimizer was then quickly identified and fixed [here](https://github.com/microsoft/DeepSpeed/pull/1801/commits/e24814a10de04ce280efe2adb027b023e3336493).

Now the challenge is how do we sync the live weights.

As a temporary band-aid Thomas Wang added a code to sync bf16 layer norm weights before each forward:
https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/272
and we used this temporary work-around for now.

last checkpoint before restart that applied this workaround: global_step17235


### 2022-03-28 Fixing out of sync layer norms

So we developed a new Deepspeed code that allows one to have accessors to the previously inaccessible fp32 and optimizer state discreet entries hidden inside the flattened 1 per param group tensor.

Once this was written it was now possible to sync b16, fp32 layer norm weights on the checkpoint loading, including syncing the optimer states corresponding to layer norm params.
https://github.com/microsoft/DeepSpeed/pull/1801/commits/4e8f7fff9a1575f16191e16d2d161af4e6b52b51

Then the code was written to sync b16, fp32 and optim states for 4 types of layer norm:
https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/274

last good checkpoint before restart with the sync: global_step20448

Switched to this branch:
https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/274

Restarted from it, run for 4 iterations, saved the checkpoint

First checkpoint saved after restart: global_step20452

Result 60% of `post_attention_layernorm.weight` are in sync across TP dimension, some layers are not in sync again - some partially - some all of TP ranks.

So one bug in BF16 optimizer was solved and the live model has been fixed, but clearly there is at least one more bug somewhere. And it predates BF16 as I have discovered this same problem in all 104B checkpoints. I didn't find any problems in 13B model.

So for now restarted again from the branch with per iteration bf16 layer norm weight syncing:
https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/272
so that we don't continue getting out of sync.



### 2022-03-30

At iteration 21680 switched from https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/272
to a different version of the layer-norm auto-sync band-aid, which performs `all_reduce` with `ReduceOp.AVG` - no manual division by `TP_size=4`, which corrects the gradients back to an identity and not 1/4th. `all_reduce` is not an `AutoGrad` function, but division is.

This version is slightly faster as well as it doesn't clone the tensors.

```
diff --git a/megatron/model/fused_layer_norm.py b/megatron/model/fused_layer_norm.py
index 0ae519a..6305bcd 100644
--- a/megatron/model/fused_layer_norm.py
+++ b/megatron/model/fused_layer_norm.py
@@ -85,6 +85,16 @@ class MixedFusedLayerNorm(torch.nn.Module):


   def forward(self, input):

    torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
    torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

    return FusedLayerNormAffineFunction.apply(
      input, self.weight, self.bias, self.normalized_shape, self.eps)
```




### 2022-04-04

Switched to deepspeed@olruwase/bf16-updates 5ea1c60f8 (HEAD) - which includes a few minor fixes which don't impact the main training, but are needed if we want to resync weights or dump fp32 weights and optim states.

Saved checkpoint iteration before the switch: 25956



### 2022-04-06

First restarted with https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/274 to sync layer norm weights and RNG states.

Saved checkpoint iteration before the switch: 26343

Then restarted after adding the RNG sync code: https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/276. But

Saved checkpoint iteration before the switch: 26345

### 2022-04-12 First lm loss spike

While there have been a few grad norm spikes so far, this is the first time we had an lm loss spike. Luckily it recovered in ~30 iterations:
```
[default7]: iteration    31214/  115311 | consumed samples:     47768496 | consumed tokens:  97829879808 | elapsed time per iteration (s): 106.46 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 2.198845E+00 | grad norm: 0.242 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.238 | TFLOPs: 147.29 |
[default7]: iteration    31215/  115311 | consumed samples:     47770544 | consumed tokens:  97834074112 | elapsed time per iteration (s): 105.90 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 2.220424E+00 | grad norm: 0.947 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.338 | TFLOPs: 148.06 |
[default7]: iteration    31216/  115311 | consumed samples:     47772592 | consumed tokens:  97838268416 | elapsed time per iteration (s): 106.40 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 2.595213E+00 | grad norm: 2.390 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.248 | TFLOPs: 147.37 |
[default7]: iteration    31217/  115311 | consumed samples:     47774640 | consumed tokens:  97842462720 | elapsed time per iteration (s): 106.33 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 2.241558E+00 | grad norm: 15.886 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.260 | TFLOPs: 147.46 |
[default7]: iteration    31218/  115311 | consumed samples:     47776688 | consumed tokens:  97846657024 | elapsed time per iteration (s): 107.53 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 2.345279E+00 | grad norm: 145.119 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.046 | TFLOPs: 145.82 |
[default7]: iteration    31219/  115311 | consumed samples:     47778736 | consumed tokens:  97850851328 | elapsed time per iteration (s): 106.36 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 5.098124E+00 | grad norm: 960.351 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.256 | TFLOPs: 147.43 |
[default7]: iteration    31220/  115311 | consumed samples:     47780784 | consumed tokens:  97855045632 | elapsed time per iteration (s): 106.30 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 3.492029E+00 | grad norm: 353.201 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.266 | TFLOPs: 147.51 |
[default7]: iteration    31221/  115311 | consumed samples:     47782832 | consumed tokens:  97859239936 | elapsed time per iteration (s): 106.13 | learning rate: 5.279E-05 | global batch size:  2048 | lm loss: 3.213059E+00 | grad norm: 379.330 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.297 | TFLOPs: 147.74 |
[default7]: iteration    31222/  115311 | consumed samples:     47784880 | consumed tokens:  97863434240 | elapsed time per iteration (s): 106.44 | learning rate: 5.278E-05 | global batch size:  2048 | lm loss: 2.804948E+00 | grad norm: 650.246 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.240 | TFLOPs: 147.31 |
[...]
[default7]: iteration    31250/  115311 | consumed samples:     47842224 | consumed tokens:  97980874752 | elapsed time per iteration (s): 106.09 | learning rate: 5.277E-05 | global batch size:  2048 | lm loss: 2.215699E+00 | grad norm: 0.267 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.304 | TFLOPs: 147.80 |
[default7]: iteration    31251/  115311 | consumed samples:     47844272 | consumed tokens:  97985069056 | elapsed time per iteration (s): 106.13 | learning rate: 5.277E-05 | global batch size:  2048 | lm loss: 2.235667E+00 | grad norm: 0.205 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 19.297 | TFLOPs: 147.75 |
```
