# Train 1 - 13B - unmodified Megatron gpt2 - baseline


## Task

Auto-regressive objective using regular Megatron-LM GPT2 language model

## Environment

To launch the environment use [start-tr1-13B](./start-tr1-13B)

```
source $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B
```

We are using the following branches specific to this training:

- `$six_ALL_CCFRWORK/code/tr1-13B/Megatron-DeepSpeed-tr1-13B` a frozen `tr1-13B` branch - can cherry pick from `main` if need be.
- `$six_ALL_CCFRWORK/code/tr1-13B/DeepSpeed-big-science` - a mostly frozen `big-science` branch - under Deepspeed's team control - so it may also require a specific SHA if something gets broken upstream.


How the environment was built:
```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n tr1-13B python=3.8
conda activate tr1-13B
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install deepspeed
pip install tensorboard

mkdir $six_ALL_CCFRWORK/code/tr1-13B

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/bigscience-workshop/bigscience

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed Megatron-DeepSpeed-tr1-13B
cd Megatron-DeepSpeed-tr1-13B
git checkout tr1-13B
pip install -r requirements.txt
pip install -e .
mkdir data
cd data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

`apex` and `deepspeed` build require an instance w/ beefy cpu and internet (unless cloned beforehand), so continue on the `prepost` partition:

```
ssh jean-zay-pp
conda activate tr1-13B
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/microsoft/DeepSpeed DeepSpeed-big-science
cd DeepSpeed-big-science
git checkout big-science
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log

#cp $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B ...

```


## Architecture

40 layers | 40 heads (128d each) | hid size 5120 | ffn size 20480


config:
```
NLAYERS=40
NHIDDEN=5120
NHEADS=32
#FFN_HIDDEN_SIZE=20480

#    --ffn_hidden_size $FFN_HIDDEN_SIZE \
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    [...]
    "
```

XXX: for some reason `--ffn_hidden_size` doesn't work, but its default `args.ffn_hidden_size = 4 * args.hidden_size` leads to the same number, ok for the first traiing. But should still fix it.



Sanity check:
```
$ VOCAB_SIZE=50257 NLAYERS=40 NHIDDEN=5120 NHEADS=32 SEQ_LEN=1024; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"
Model size: 13B
```



## Sequence Length

Default Megatron-LM LM with 1024 tokens

All dataset samples are at least 1024 tokens long (filtered via `transformers`'s `GPT2TokenizerFast`).

```
SEQ_LEN=2048

    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \

```


## Global batch size

GBS = Global Batch Size

Use a schedule:

- start from 32k tokens (gbs=16)
- increase linearly to 2048k (gbs=1024) over 5M samples (for a total of ~10B tokens / 10k steps)
- then continue at 2048k  (gbs=1024) for 145M samples (290B tokens / 145K steps)

Total: 300B tokens (150K steps)

syntax:
```
--rampup-batch-size <start batch size>  <batch size increment> <ramp-up samples>
```

At seqlen 2048 (1k tokens is bs=1), we get:

```
    --rampup-batch-size 16 16 5_000_000 \
    --global-batch-size 1024 \
```

This means it will start with global batch size 16 and over 63 (`(1024-16)/16`) intervals will increase the
batch size by 16 linearly to 1024.

79365 (`5_000_000/63`) is the number of samples before the next GBS increment. That is we run at GBS=16 for 79365 samples, or 4960 steps (`79365/16`). Then we run at GBS=32 for 79365 samples, or 2480 steps. Then 1653 steps at GBS=48, 1240 at GBS=64, etc....

Notes:
* `--rampup-batch-size` requires the use of `--train-samples` and can't be used with `--train-iters`.
* global batch size has to be divisible by micro-batch-size * DP_SIZE

Important:  the software will fail if GBS is not divisible by `MBS * DP_SIZE`.
Though Jared's recommendation is to use MBS=1 and then it's much easier to match GBS/DP_SIZE even at GBS=16.

`DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE)`

Since the increments are in GBS=16, we can do only DP_SIZE=16, which means that at most we can use 32 nodes (`32*4/(4*2)=16`).

Once GBS reaches 1024, we can use up to 8192 GPUs (1024*2*4), so we will be able to switch to 64 nodes or may be even 128 nodes (4 gpus each).




## Checkpoints

We need the checkpoints:

1. in order to be able to resume the training when the training is prematurely stopped for whatever reason.
2. In addition a special saving schedule has been requested by the interpretabity group.

Because there are 3 different schedules, and Megatron-LM has only fixed checkpoint saving schedule, we will need 3 different run scripts, to be launched in a sequence, each starting once the previous has finished.

1. steps 1-100 - 10 checkpoints, interval 10 steps
2. steps 101-1000 - 50 checkpoints, interval 18 steps
3. steps 1001-300K - 100+ checkpoints, interval 1500 steps
4. if still needed, can continue with schedule 3

note: the interoperability study doesn't care for checkpoints in the range of 1k-20k, so we only save those to be able to restart the training.

It'd have been
```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then TRAIN_ITER=100    SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then TRAIN_ITER=1000   SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then TRAIN_ITER=300000 SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi
    --train-iter $TRAIN_ITER  \
    --save-interval $SAVE_INTERVAL  \
```

Unfortunately, `--rampup-batch-size` can't work with `--train-iter` and we have to use `--train-samples` instead. It has to be fixed through all of trainings and can't be changed, otherwise resume from checkpoint will break.

So the only thing left is to use `--exit-interval` which is in steps.

Which gives us the three rounds:

```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then EXIT_INTERVAL=100 SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then EXIT_INTERVAL=900 SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then                   SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

    --train-samples 300_000_000 \
    --exit-interval $EXIT_INTERVAL \
    --save-interval $SAVE_INTERVAL  \
```

`--exit-interval` counts steps only for the current run, regardless of previous steps. So to stop at effective step 1000, the second round we tell it to exit at 900 (the first round did the first 100).

And unfortunately, this proved to be not supported by Megatron-LM either at the moment. There are a few possible ways to approach this:

1. One approach is to simply use 3 independent trainings, while using the same `--seed ` and just have `--exit_interval` as above. Though after each training moving the checkpoints away.

2.
XXX: Also megatron code could be extended to implement `--exit-samples` - so sample-based exit strategy

3. Yet another approach is to do it manually. Kill the training after 100, and then restart and kill after 900 iterations, while changing the save interval, and manually fixing up the `checkpoints/latest` to point to the correct checkpoint - since the manual killing might have a few extra checkpoints. So the recipe to follow:

```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

    --train-samples 300_000_000 \
    --save-interval $SAVE_INTERVAL  \
```

(could also do it with 3 parallel jobs by using the same seed!)

```
--seed 42
```

Therefore do this manually:

0.
* delete the old checkpoints `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints`

1.

* set to `ROUND=1`
* `sbatch tr1-13B-round1.slurm`
* run for 100+ steps
* scancel the job
* clean up `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints` to remove any checkpoints beyond 100
* make sure `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/latest` contains 100


2.

* set to `ROUND=2`
* `sbatch tr1-13B-round1.slurm`
* run for the additional 900+ steps (it's incremental, so the script already knows it started at 100)
* scancel the job
* clean up `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints` to remove any checkpoints beyond 1000
* make sure `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/latest` contains 1000


3.

* set to `ROUND=3`
* `sbatch tr1-13B-round1.slurm`
* run normally



Because it'd be potentially too demanding to export TBs of data and the intended users might not be even able to download all that data, most likely we will need to run the interpretabity post-analysis experiments on JZ and send the reports to those who need the reports.

Megatron-LM resumes from the most recent checkpoint by default. Does it need the exact path or does it auto-discover the latest checkpoint by default.

```
--load path_to_check_point \
```


Remi suggests 100TB on SCRATCH shouldn't be a problem.


## Optimizer

- AdamW,  β1=0.9, β2=0.999 eps=1e−8
- learning rate:
   * peak=1e-4
   * warmup over 2000 steps
   * cosine decay for learning rate down to 10% of its value, over 260B tokens (after 260 billion tokens, training continues at 10% of the original learning rate)
- clipping by global norm of 1 (as in GPT-3)
- weight decay of 0.1

We need lr-decay in samples, so tokens2samples = 260B / 2048 = 126_953_125

We need lr-warmup in samples, so doing the math again as in checkpoints

2000=160*12+80

so we will get to 2000 in 216_320 samples `16*160*12*(12+1)/2+16*13*80`



```
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
```


## Logging


For now enable all tensorboard features, later we might decide to not log it all.

We are logging:

- lr (enabled by default)
- bs (enabled)
- loss (always)
- loss-scale (log_loss) (enabled by default)
- grad-norm (always)
- num-zeros (always)
- param-norm (always)
- timers (enabled)
- validation loss (always)
- validation ppl (perplexity) (enabled)

almost all of these are also logged as a comparison to consumed_train_samples

XXX: nice to have:
- throughput - Tflops/gpu or tokens


**Tensorboard config**:

```
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard

    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
```

**CodeCarbon config**:

```
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon

    --codecarbon-dir $CODECARBON_PATH \
```

**Training logs**

All training logs are piped into `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs/main_log.txt`.


## Exporting

Before starting training insert cloned git repos to where output data will go.

The last 3 should all be git repo clones
```
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon
LOGS_PATH=$DATA_OUTPUT_PATH/logs
```

I created 4 repos on https://huggingface.co/bigscience/ and now we can clone those as the folders data will be output into:

```
cd $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
git clone https://huggingface.co/bigscience/tr1-13B-checkpoints checkpoints
git clone https://huggingface.co/bigscience/tr1-13B-tensorboard tensorboard
git clone https://huggingface.co/bigscience/tr1-13B-codecarbon codecarbon
git clone https://huggingface.co/bigscience/tr1-13B-logs logs
```

If this is your first time running git-lfs on this system, you need to init it once:
```
module load git-lfs
git lfs install
```

Most of the data types we are going to sync will be large or huge, and most are already lfs-tracked by default, so no setup is required. Except our log file which too can grow large, so we need to set it up:

```
cd logs
git-lfs track *.txt
git commit -m "large text files" .gitattributes
git push
```

### Cronjobs to auto-sync the hub

Now we just need a cronjob to automatically do for each type of data to export:

```
cd checkpoints
git add */*.pt
git commit -am "new data"
git push
```

**Weights checkpoints**

Currently, we aren't exporting checkpoints.

**Tensorboard**

Here is the slurm script to sync the tensorboard data: [tr1-13B-hub-sync-tensorboard.slurm](./tr1-13B-hub-sync-tensorboard.slurm)

**CodeCarbon**

Currently is not running, so nothing to log.

**Log of logs**

Let's also create a log of logs. We will pipe all the logs in there and also the various statuses - e.g. while SLURM is queued the training and it's not running.

Here is the slurm script to sync the raw logs data: [tr1-13B-hub-sync-logs.slurm](./tr1-13B-hub-sync-logs.slurm)

The main source of logs is the training scripts. The logs are gathered via
```
$CMD ... 2>&1 | tee -a $LOGS_PATH/main_log.txt
```
in the training slurm script.

XXX: add a pulse script that will report to the outside world when the training job is on the backburner (or perhaps not even scheduled).

XXX: we could also add various other diagnostics appended to the main log file. e.g. shared memory, etc.




## Deepspeed config

Using Deepspeed's activation checkpointing to use a lot less GPU memory

```
    --deepspeed-activation-checkpointing \
```

Possible extras:

- Enabling `"contiguous_memory_optimization": true,` can help to reduce memory fragmentation, but it requires￼setting `number_checkpoints`. This should be set to be equal to number of transformer blocks per pipeline stage times the number of pipeline parallel stage. Samyam says: Full disclaimer: I have only used this with ZeRO but not with pipeline parallelism. But by setting the number_checkpoints as described, it should work for PP too. The benefit of using it is usually only apparent when running very close to the memory limit.



## Dataset


- Full 304.2M version (529GB) : `$six_ALL_CCFRWORK/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k`

We are using english-only OSCAR with full documents (*not* individual sentences).

We have about 300M records in 1.2TB of jsonl data (about 3/4 of which are smaller than 1K tokens), which amounts to about 280B tokens (estimated at about 4.5chars/word).

For more information on the pre-processing process and various estimations see: [OSCAR](../../data/oscar/README.md)



## Dealing with 20h SLURM limit

First, let's ensure we save a checkpoint just before SLURM kills the job

Let's try 19:50 1190=60*24-10

```
    --exit-duration-in-mins 1190 \
```

For the bigger models 10min might not be long enoug to finish an iteration (assume the limit hits right as one starts) and write out a checkpoint.

Then we need to figure out how to schedule the next slurm job as soon as the currently running one is over in 20h.

We will use job arrays, to solve this. Let's start with just 10 such jobs:

```
sbatch --array=1-10%1 tr1-13B-round1.slurm
```

`%1` limits the number of simultaneously running tasks from this job array to 1, since we want them to run in a sequence.

Alternatively, as always this param can be part of the script:
```
#SBATCH --array=1-10%1
```

## Crontab

JZ doesn't have a user-accessible crontab facility, so we have to emulate it with a self-restarting slurm job.

I'm thinking of having a slurm script that will poll some dir and if it finds any scripts in it will run those. perhaps we can emulate `/etc/cron.hourly` and `/etc/cron.daily`
XXX:



## Estimated run time

Best case scenario:
```
$ python -c 'Btokens=300; Bmodel=13; n_gpus=256; Tflops=45; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
31.35 days
```

You will find the detailed explanation of the estimation formula [here](../../math/README.md#estimate-model-training-time).

Of course, the training will be much slower in the first 10k steps because of the batch size rampup, where the pipeline will be very inefficient.



## Exports

- GCS https://console.cloud.google.com/storage/browser/bigscience
- The Hub https://huggingface.co/bigscience


## Training scripts

[tr1-13B-round1.slurm](./tr1-13B-round1.slurm)



## Extras
