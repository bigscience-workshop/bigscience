# Lessons learned


## How training divergences were overcome

The following are techniques that have to be done before the training starts.

### using a formulaic std init

Setting `--init-method-std` to `sqrt(2/(NHIDDEN*5))` has made a huge difference to the training stability.

e.g. for `NHIDDEN*5=11600` we used  `--init-method-std 0.006`

We derived this from:

`0.00587220219514703 = sqrt(2/(11600*5))` (from the ScaleNorm paper https://arxiv.org/abs/1910.05895)


### adding embed layernorm

Emedding LayerNorm has shown to help a lot with spikes that the training can't back from:

To activate add `--embed-layernorm`

Note: since this has its weights you can only add it at the beginning of the training




## How to deal with ongoing instabilities

How to recover from an instability without a full restart.

### Data skipping

1. Roll back to the last checkpoint before the instability
2. skip data samples from the instability window `--skip-train-iteration-range 8401-8800 `
