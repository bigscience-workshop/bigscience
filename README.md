# bigscience

[Research workshop on large language models - The Summer of Language Models 21](https://bigscience.huggingface.co/)

At the moment we have 2 code repos:

1. https://github.com/bigscience-workshop/Megatron-DeepSpeed - this is our flagship code base
2. https://github.com/bigscience-workshop/bigscience - (this repo) for everything else - docs, experiments, etc.

Currently, the most active segments of this repo are:

- [JZ](./jz/) - Lots of information about our work environment which helps evaluate, plan and get things done
- [Experiments](./experiments) - many experiments are being done. Documentation, result tables, scripts and logs are all there
- [Datasets info](./data/)


## Contribute

This is a community project and we would love to have your help. If you are inspired to contribute please see the following entries:

Megatron-DeeepSpeed:

- [Megatron-DeepSpeed Issues](https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues)
- [Good First Issues](https://github.com/bigscience-workshop/Megatron-DeepSpeed/contribute)

General BigScience:

- [bigscience Issues](https://github.com/bigscience-workshop/bigscience/issues)
- [Good First Issues](https://github.com/bigscience-workshop/bigscience/contribute)


## Trainings

### Train 1 - 13B - unmodified Megatron gpt2 - baseline

* [the full spec and discussions](./train/tr1-13B-base)
* [the training script](./train/tr1-13B-base/tr1-13B-round1.slurm)
* checkpoints and logs:
   - [tensorboard](https://huggingface.co/bigscience/tr1-13B-tensorboard/tensorboard)
   - [logs](https://huggingface.co/bigscience/tr1-13B-logs/)
* [chronicles](./train/tr1-13B-base/chronicles.md)

You can watch the training logs live by running this `tail -f` like script over remote log file that gets synced to the hub once an hour:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/x-linked-size: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr1-13B-logs/resolve/main/main_log.txt
```

### Train 2

* [todo](./train/tr2/TODO.md)
