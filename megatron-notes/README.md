# Various Megatron-specific notes

HOWTOs, Troubleshooting, etc.


## Propagating local bug fixes upstream

If you find bugs in Megatron-LM and commit fixes, please add your fix to https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/10 so that we could then send all those upstream.


## Highlights of insights from Megatron-LM developers

Mohammad Shoeybi:
- With respect to reproducibility, we have done a lot of work to make sure Megatron is reproducible, meaning that if you resume from an earlier checkpoint and run on the same number of GPUs, you should see EXACTLY the same behaviour. This implies that dataloaders are also reproducible.
- The spikes sometimes happen during the training and if the loss quickly recovers, it is generally ok. Sometimes it might be due to a set of bad samples but most of the time it is due to optimizers being in a bad state and having values that might underflow in the gradients. What we found that was helpful is to use a lower beta2 in the adam optimizer. Basically the closer beta2 is to beta1, the less chances of these spikes happening. Definitely we don’t want to use a very low value for beta2 (for example beta2=beta1=0.9) as it will slow down the convergence.
- Large learning rate can cause instabilities in the fp16 training (fp16 training is more sensitive to learning rate). I don’t have a solid explanation for this but we found this empirically.
- We also found that the larger the model, the lower the initialization std should be. A rule of thumb is to scale it down bu sqrt of hidden size. This also helps with the stability.

## Troubleshooting

If the trainer hangs in `compiling and loading fused kernels` it means it dropped a lock file, delete it and restart:

```
rm ./megatron/fused_kernels/build/lock
```
