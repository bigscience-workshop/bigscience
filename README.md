# bigscience

[Research workshop on large language models - The Summer of Language Models 21](https://bigscience.huggingface.co/)

At the moment we have 2 code repos:

1. https://github.com/bigscience-workshop/Megatron-DeepSpeed - this is our flagship code base
2. https://github.com/bigscience-workshop/bigscience - (this repo) for everything else - docs, experiments, etc.

Currently, the most active segments of this repo are:

- [JZ](./jz/) - Lots of information about our work environment which helps evaluate, plan and get things done
- [Experiments](./experiments) - many experiments are being done. Documentation, result tables, scripts and logs are all there
- [Datasets info](./data/)

## Trainings

### Train 1 - 13B - unmodified Megatron gpt2 - baseline

* [the full spec and discussions](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base)
* [the training script](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm)
