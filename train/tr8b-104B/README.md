# Variations to experiments with 104B


## Curriculum Learning

- [slurm script](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8b-104B/tr8b-104B-cl.slurm)
- [TB](https://huggingface.co/bigscience/tr8b-104B-logs/tensorboard)
- [chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8b-104B/chronicles.md)


## BitsNBytes (BNB)

- [slurm script](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8b-104B/tr8b-104B-bnb.slurm)
- [TB](https://huggingface.co/bigscience/tr8b-104B-logs/tensorboard)
- [chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8b-104B/chronicles.md)


https://github.com/facebookresearch/bitsandbytes

Needs:

```
pip install bitsandbytes-cuda111
```
and adding:
```
--use-bnb-optimizer \
```

since the optimizer uses almost 3/4 less memory, was able to reconfigure the setup to give faster training (about 10% speed-up +4 TFLOPs leading to 44 TFLOPs). For tune-up experiments see [this doc](https://github.com/bigscience-workshop/bigscience/blob/master/experiments/tr8-104B.md#bnb)

Notes from Tim:

> It seems that 8-bit Adam is better than 32-bit Adam for larger learning rates, but that difference decreases once the learning rate is decayed to zero. I have not seen any instabilities, and no instabilities have been reported to me, but that might be an indicator that 8-bit Adam might have some issues when learning rates are small at the end of training.
> Another view is that this might also just indicate the updates are so small that they are dominated by the noise from quantization and no instability is present. It is so difficult to say because so few models were run at a very large scale, and even though MoEs are large, the likely behave very differently than large dense transformers

> I think it could be worth a try to try a higher lr. This could counteract potential degradation at the end of training when the lr is too small
