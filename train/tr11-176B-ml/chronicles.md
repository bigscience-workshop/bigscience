# The final training

Trials and Tribulations during the 176B 250k multi-lingual training.

## main-1

While the final data is being cleaned up we are doing a few preliminary runs with data that still has some issues.

GBS ramp up of `--rampup-batch-size 16 16 9_765_625` - the first few stages starting with GBS=16 are really slow (8 TFLOPs). The pipeline doesn't have enough data to even fill all the stages once, so it's super inefficient and it'll take days until we start hitting 100 TFLOPs.

But there were no spikes during this brief experiment.



## main-2

Trying `--rampup-batch-size 384 16 9_765_625` since 384 is the first GBS where the pipe is filled up fully for the first time. `12*2*4=384` (`PP*MBS*DP`). The throughput start at 100 TFLOPs right away (and it should be 150 TFLOPS once we reach GBS=2048).

Found a bug: tied weights weren't getting reduced - was getting a spike on restart, fixed at
https://github.com/microsoft/DeepSpeed/pull/1801/commits/37011a92bad42b07c2cb742751873ef7073d84b8

So only the front embed matrix grad updates were making, the end one were ignored.

Will do a totally new run to compare that it's similar or better.




## main-3

Trying the rebased to master version 61d51fd62141ddb51b629b785af256fac407e048 and it has serious issues - the learning is much much slower

## main-4

So rolling back `olruwase/bf16-updates` branch to the fix:

37011a92bad42b07c2cb742751873ef7073d84b8 Reduce tied weight gradients

This time the learning is just a tad slower than main-2, so either deepspeed@master introduced some regression or the merge didn't go well.

additionally going to try the latest checkpoint from `main-3` as it's more progressive to check `main-4` for spikes.

same spike.

After analysis the module weights, it's clear we have 2 distinct issues:
1. `module.tied_modules.embed.word_embeddings.norm` is not taken care of at all in pp rank -1 and they get reset to defaults on load
2. `module.tied_modules.embed.word_embeddings.weight` is mismatching on pp rank -1 between after iteration of last step before save and before iteration of first step after load

This was derived with this debug instrumentation:

```
diff --git a/megatron/training.py b/megatron/training.py
index fd65ae9..fd76d28 100644
--- a/megatron/training.py
+++ b/megatron/training.py
@@ -58,6 +58,23 @@ from megatron.data.dataset_utils import analyze_data_prefix

 import deepspeed

+def dump_weights(preamble, iteration, model):
+
+    import os
+    import socket
+    hostname = socket.gethostname()
+    pid = os.getpid()
+
+    tp_rank = mpu.get_tensor_model_parallel_rank()
+    pp_rank = mpu.get_pipeline_model_parallel_rank()
+    dp_rank = mpu.get_data_parallel_rank()
+    global_rank = torch.distributed.get_rank()
+
+    fn = f"debug-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-global{global_rank}-{preamble}-{pid}.txt"
+    #print(fn)
+    with open(fn, "w") as fh:
+        for n, p in model[0].named_parameters():
+            fh.write(f"{n}={p}\n")

 def print_datetime(string):
     """Note that this call will sync across all ranks."""
@@ -426,6 +443,8 @@ def setup_model_and_optimizer(model_provider_func):
         if args.fp16:
             optimizer.reload_model_params()

+    #optimizer.update_lp_params()
+
     return model, optimizer, lr_scheduler


@@ -848,12 +867,18 @@ def train(forward_step_func, model, optimizer, lr_scheduler,
             args.pipeline_model_parallel_size >= 1:
             args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                     args.iteration + 1)
+
+        dump_weights("before-iteration", iteration+1, model)
+
         loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
             train_step(forward_step_func,
                        train_data_iterator,
                        model,
                        optimizer,
                        lr_scheduler)
+
+        dump_weights("after-iteration", iteration+1, model)
+
         iteration += 1
         args.iteration = iteration
         new_samples = mpu.get_data_parallel_world_size() * \
```

and then
1. run 5 iterations and saved checkpoint, then run:
```
mkdir a; mv debug-* a
```
2. restarted and run a few iterations, then run:

```
mkdir b; mv debug-* b
```

I basically dumped weights for all ranks before and after train_step

Now let's compared them all. Comparing:
1. the after iteration of the last step before save
2. the before iteration step after the load (on restart)

with the help of:
```
perl -le 'print qx[diff -u a/debug-805-*global$_-after-iteration-*.txt b/debug-806-*-global$_-before-iteration-*.txt] for 0..383'
```

so here is a sample diff:
```
--- a/debug-805-pp11-tp1-dp4-global369-after-iteration-377074.txt       2022-03-06 05:44:06.074835000 +0100
+++ b/debug-806-pp11-tp1-dp4-global369-before-iteration-378990.txt      2022-03-06 05:48:24.842635000 +0100
@@ -1,21 +1,15 @@
 module.tied_modules.embed.word_embeddings.weight=Parameter containing:
-tensor([[-3.1090e-04,  4.6082e-03, -2.3499e-03,  ..., -1.1292e-02,
-          2.1667e-03, -2.7313e-03],
-        [-1.1353e-02,  9.9487e-03, -1.9684e-03,  ..., -5.4550e-04,
-         -2.3460e-04,  4.2114e-03],
-        [ 3.2806e-03, -3.4332e-04, -5.5847e-03,  ...,  7.6294e-03,
-          1.7853e-03,  2.5868e-05],
+tensor([[-0.0006,  0.0046, -0.0024,  ..., -0.0114,  0.0014, -0.0030],
+        [-0.0109,  0.0096, -0.0020,  ..., -0.0005, -0.0001,  0.0041],
+        [ 0.0027, -0.0004, -0.0056,  ...,  0.0070,  0.0017,  0.0003],
         ...,
-        [ 1.6098e-03,  4.1809e-03, -2.4567e-03,  ..., -4.6692e-03,
-         -4.5776e-03,  1.7090e-03],
-        [ 5.7373e-03,  3.5858e-03, -1.7471e-03,  ...,  2.3041e-03,
-         -6.4392e-03,  1.0223e-03],
-        [-1.6937e-03, -1.4038e-02,  2.1057e-03,  ..., -3.6011e-03,
-          1.3275e-03, -5.8594e-03]], device='cuda:1', dtype=torch.bfloat16,
-       requires_grad=True)
+        [ 0.0018,  0.0039, -0.0026,  ..., -0.0051, -0.0043,  0.0016],
+        [ 0.0051,  0.0039, -0.0015,  ...,  0.0027, -0.0063,  0.0008],
+        [-0.0018, -0.0142,  0.0021,  ..., -0.0035,  0.0015, -0.0060]],
+       device='cuda:1', dtype=torch.bfloat16, requires_grad=True)
 module.tied_modules.embed.word_embeddings.norm.weight=Parameter containing:
-tensor([0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961], device='cuda:1',
-       dtype=torch.bfloat16, requires_grad=True)
+tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:1', dtype=torch.bfloat16,
+       requires_grad=True)
 module.tied_modules.embed.word_embeddings.norm.bias=Parameter containing:
 tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:1', dtype=torch.bfloat16,
        requires_grad=True)
```


## main-5

trying a new baseline with rampup starting from 192
