# Various Megatron-specific notes

HOWTOs, Troubleshooting, etc.


## Propagating local bug fixes upstream

If you find bugs in Megatron-LM and commit fixes, please add your fix to https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/10 so that we could then send all those upstream.


## Troubleshooting

If the trainer hangs in `compiling and loading fused kernels` it means it dropped a lock file, delete it and restart:

```
rm ./megatron/fused_kernels/build/lock
```
