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

since the optimizer uses almost 3/4 less memory, was able to reconfigure the setup to give faster training (about 10% speed-up). For tune-up experiments see [this doc](https://github.com/bigscience-workshop/bigscience/blob/master/experiments/tr8-104B.md#bnb)
