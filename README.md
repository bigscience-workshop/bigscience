# bigscience

[Research workshop on large language models - The Summer of Language Models 21](https://bigscience.huggingface.co/)

At the moment we have 2 code repos:

1. https://github.com/bigscience-workshop/Megatron-DeepSpeed - this is our flagship code base
2. https://github.com/bigscience-workshop/bigscience - (this repo) for everything else - docs, experiments, etc.

Currently, the most active segments of this repo are:

- [JZ](./jz/) - Lots of information about our work environment which helps evaluate, plan and get things done
- [Experiments](./experiments) - many experiments are being done. Documentation, result tables, scripts and logs are all there
- [Datasets info](./data/)
- [Train](./train) - all the information about the current trainings (see below for the most important ones)

We have READMEs for specific aspects, such as:
- [hub integration](./tools/README.md)


## Trainings

While we keep detailed chronicles of experiments and findings for some of the main trainings, here is a doc that contains a summary of the most important findings: [Lessons learned](train/lessons-learned.md)


### Train 1 - 13B - unmodified Megatron gpt2 - baseline

* [the full spec and discussions](./train/tr1-13B-base)
* [the training script](./train/tr1-13B-base/tr1-13B-round1.slurm)
* checkpoints and logs:
   - [tensorboard](https://huggingface.co/bigscience/tr1-13B-tensorboard/tensorboard)
   - [logs](https://huggingface.co/bigscience/tr1-13B-logs/)
* [chronicles](./train/tr1-13B-base/chronicles.md)

You can watch the training logs live by running this `tail -f` like script over remote log file that gets synced to the hub once an hour:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/content-length: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr1-13B-logs/resolve/main/main_log.txt

```

### Train 3

Architecture and scaling baseline runs: no fancy tricks, just GPT2. Here are links to the respective tensorboards:

| Size                	| 1B3 	| 760M 	| 350M 	| 125M 	|
|---------------------	|-----	|------	|------	|------	|
| C4 + low warmup     	| [a](https://huggingface.co/bigscience/tr3-1B3-modeling-baseline-tensorboard)   	| [b](https://huggingface.co/bigscience/tr3b-760M-modeling-baseline-tensorboard)    	| [c](https://huggingface.co/bigscience/tr3c-350M-modeling-baseline-tensorboard)    	|      	|
| OSCAR + low warmup  	| [f](https://huggingface.co/bigscience/tr3f-1B3-diagnostic2-low-warmup-oscar-tensorboard)   	|      	|      	|      	|
| C4 + high warmup    	| [e](https://huggingface.co/bigscience/tr3e-1B3-diagnostic1-warmup-c4-tensorboard)   	|      	|      	|      	|
| OSCAR + high warmup 	| **[d (current baseline)](https://huggingface.co/bigscience/tr3d-1B3-more-warmup-tensorboard)**   	| [g](https://huggingface.co/bigscience/tr3g-760M-v2-tensorboard)    	| [h](https://huggingface.co/bigscience/tr3h-350M-v2-tensorboard)    	| [i](https://huggingface.co/bigscience/tr3i-125M-v2-tensorboard)    	|
| Pile + high warmup  	| [m](https://huggingface.co/bigscience/tr3m-1B3-pile-tensorboard)   	| [j](https://huggingface.co/bigscience/tr3j-760M-pile-tensorboard)    	| [k](https://huggingface.co/bigscience/tr3k-350M-pile-tensorboard)    	| [l](https://huggingface.co/bigscience/tr3l-125M-pile-tensorboard)    	|


### Train 8

104B - unmodified Megatron gpt2 - with extra-wide hidden size to learn how to deal with training instabilities

* [the full spec and discussions](./train/tr8-104B-wide)
* [the training script](./train/tr8-104B-wide/tr8-104B.slurm)
* checkpoints and logs:
   - [tensorboard](https://huggingface.co/bigscience/tr8-104B-logs/tensorboard)
   - [logs](https://huggingface.co/bigscience/tr8-104B-logs/tree/main/logs)
* [chronicles](./train/tr8-104B-wide/chronicles.md)

You can watch the training logs live by running this `tail -f` like script over remote log file that gets synced to the hub once an hour:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/content-length: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://cdn-lfs.huggingface.co/bigscience/tr8-104B-logs/b2cc478d5ae7c9ec937ea2db1d2fe09de593fa2ec38c171d6cc5dca094cd79f9
```

### Train 11

**This is the current main training**

tr11-176B-ml

* [the full spec and discussions](./train/tr11-176B-ml/)
* [the training script](./train/tr11-176B-ml/tr11-176B-ml.slurm)
* checkpoints and logs:
   - [tensorboard](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard)
   - [logs](https://huggingface.co/bigscience/tr11-176B-ml-logs/tree/main/logs/main)
* [chronicles-prequel](./train/tr11-176B-ml/chronicles-prequel.md)
* [chronicles](./train/tr11-176B-ml/chronicles.md)

You can watch the training logs live by running this `tail -f` like script over remote log file that gets synced to the hub once an hour:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -LsI $u]=~/2 200.*?content-length: (\d+)/s; \
print qx[curl -Lsr $b-$e $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr11-176B-ml-logs/resolve/main/logs/main/main_log.txt
```
