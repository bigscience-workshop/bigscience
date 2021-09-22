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
conda install pytorch==1.8.1 torchvision cudatoolkit=10.2 -c pytorch -y
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

Config:

```
NLAYERS=40
NHIDDEN=5120
NHEADS=32
FFN_HIDDEN_SIZE=20480

#    --ffn_hidden_size $FFN_HIDDEN_SIZE \
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    [...]
    "
```

Sanity check:
```
$ VOCAB_SIZE=50257 NLAYERS=40 NHIDDEN=5120 NHEADS=32 SEQ_LEN=2048; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"
Model size: 13B
```



## Sequence Length

Default Megatron-LM language model with 2048 tokens sequence length

```
SEQ_LEN=2048

    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \

```


## Global batch size

GBS = Global Batch Size

Use a schedule:

- start from 32k tokens (gbs=16)
- increase linearly to 2048k (gbs=1024) over 5M samples (for a total of ~10B tokens / 5k steps)
- then continue at 2048k  (gbs=1024) for 145M samples (290B tokens / 145K steps)

Total: 300B tokens (150K steps)

Note: the training script wasn't updated when we flipped seqlen/gbs from 1024/2048 to 2048/1024, so we are currently planning to train for 300K steps (samples) and 600B tokens. But since longer doesn't impact anything, we will just stop at half the time. I updated the document to use the right 150K number so we don't repeat this mistake in the next training.

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

Once GBS reaches 1024, we can use up to 8192 GPUs (1024*2*4), so we will be able to switch to 64 nodes or may be even 128 nodes (4 gpus each). We can't use any number of nodes between 64 and 128 though, because the number has to be 2**X. So 96 nodes won't work, because it has a multiplier of 3 there.




## Checkpoints

We need the checkpoints:

1. in order to be able to resume the training when the training is prematurely stopped for whatever reason.
2. In addition a special saving schedule has been requested by the interpretabity group.

Because there are 3 different schedules, and Megatron-LM has only fixed checkpoint saving schedule, we will need 3 different run scripts, to be launched in a sequence, each starting once the previous has finished.

1. steps 1-100 - 10 checkpoints, interval 10 steps
2. steps 101-1000 - 50 checkpoints, interval 18 steps
3. steps 1001-150K - 100+ checkpoints, interval 1500 steps
4. if still needed, can continue with schedule 3

note: the interoperability study doesn't care for checkpoints in the range of 1k-20k, so we only save those to be able to restart the training.

It'd have been
```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then TRAIN_ITER=100    SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then TRAIN_ITER=1000   SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then TRAIN_ITER=150000 SAVE_INTERVAL=1500
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

    --train-samples 150_000_000 \
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

    --train-samples 150_000_000 \
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

Before starting training create cloned git repos to where output data will go.

The last 4 should all be git repo clones
```
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon
LOGS_PATH=$DATA_OUTPUT_PATH/logs
```

I created 4 repos at https://huggingface.co/bigscience/ and now we can clone those as the dirs data will be output into:

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

This job is performed automatically by `hub-sync.py`. For full details see: [Automated upload to the hub](../../data/export.md#automated-upload-to-the-hub).

**Weights checkpoints**

Currently, we aren't exporting checkpoints.

**Tensorboard**

Here is the slurm script to sync the tensorboard data: [tr1-13B-hub-sync-tensorboard.slurm](./tr1-13B-hub-sync-tensorboard.slurm)

**CodeCarbon**

Currently the feature is not enabled, so there is nothing to log.

**Log of logs**

Let's also create a log of logs. We will pipe all the logs in there and also the various status reports - e.g. while SLURM is queued the training and it's not running.

Here is the slurm script to sync the raw logs data: [tr1-13B-hub-sync-logs.slurm](./tr1-13B-hub-sync-logs.slurm)

The main source of logs is the training scripts. The logs are gathered via
```
$CMD ... 2>&1 | tee -a $LOGS_PATH/main_log.txt
```
in the training slurm script.

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

We are using English-only subset of [the OSCAR dataset](https://huggingface.co/datasets/oscar) with full documents (*not* individual sentences).

We have about 300M records in 1.2TB of jsonl data (about 3/4 of which are smaller than 1K tokens), which amounts to about 280B tokens (estimated at about 4.5chars/word).

Megatron's preprocessing tool indexes everything and then at training time the Dataloader serves chunks of the desired fixed sequence length (2048 tokens in our case).

For more information on the pre-processing process and various estimations see: [OSCAR](../../data/oscar/README.md).



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

JZ doesn't have a user-accessible crontab facility, so we have to emulate it with a self-restarting slurm job that polls some dir for new jobs to run. For full details on how this works please see [Crontab Jobs](../../jz/crontab/).

But to use it simply put your slurm scripts into either:
```
$six_ALL_CCFRWORK/cron/cron.hourly
$six_ALL_CCFRWORK/cron/cron.daily
```

and the jobs will be run on hourly or daily basis. This is similar to Linux's `/etc/cron.*` setup. Except the jobs aren't guaranteed to start on the hour, but should be around that time.

Currently we have:

```
ls -1 $six_ALL_CCFRWORK/cron/cron.hourly/*slurm
tr1-13B-hub-sync-logs.slurm
tr1-13B-hub-sync-tensorboard.slurm
tr1-13B-slurm-status.slurm
```

The first 2 sync log files to the hub and the last one monitors the health of the training and alerts of any problems.


## Estimated run time

Best case scenario when training 24/7 on 64 nodes with 4 gpus each:
```
$ python -c 'Btokens=300; Bmodel=13; n_gpus=256; Tflops=45; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
31.35 days
```

You will find the detailed explanation of the estimation formula [here](../../math/README.md#estimate-model-training-time).

The training was much slower in the first 10k steps because of the batch size rampup, where the pipeline was very inefficient.

And then we were only able to use 20h slurm jobs, with unpredictable gaps of wait time in between (1-30 hours!), so it's impossible to predict when the finish line will be finished.


## Memory usage

During training currently we use 256GB (8x 32GB gpus) per each full replica (TP=2 + PP=4), the rest are ZeRO-DP. So if we throw x times more GPUs we just speed things up by having more 2-node replicas.
The required memory breakdown:

1. 4B for fp32 weights
2. 2B for fp16 weights
3. 8B for optimizer states.
4. 4B for gradients (we don't save these in the checkpoint)
5. plus memory for activations and temps, which total majorly depends on the seqlen and mini batch size - and since we use activation checkpointing this memory need is quite small.

Total: 234GB (18*13) plus activations and temps memory. So we are close to 256GB here.

Activation memory would have been much much bigger if it weren't for activation checkpointing.


## Checkpoint Back Up

To copy multiple checkpoints excluding optimizer states. First move the desired checkpoints to back up to some dedicated dir, e.g. `tr1-13B-round2/checkpoints`, then copy just the needed files:

```
srun -p prepost  -A six@cpu --time=20:00:00 --pty bash
mkdir to-upload
rsync -acvhu --no-compress --info=progress2 --exclude "zero*pt" tr1-13B-round2/checkpoints/ to-upload
```

then to back those up:

```
cp -arun $six_ALL_CCFRSCRATCH/checkpoints/to-upload/* $six_ALL_CCFRSTORE/checkpoints/tr1-13B
```


**Final checkpoint with optimizer states:**

```
mkdir $six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim
cp -arun $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/global_step168000 $six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim/
```

This is the final checkpoint, that can be resumed from at will:

```
$six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim/global_step168000
```

Here is the corresponding log:
```
 iteration   168000/  311541 | consumed samples:    153013584 | elapsed time per iteration (ms): 13248.2 | learning rate: 1.000E-05 | global batch size:  1024 | lm loss: 2.376641E+00 | loss scale: 131072.0 | grad norm: 19767.052 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
--------------------------------------------------------------------------------------------------
 validation loss at iteration 168000 | lm loss value: 2.342049E+00 | lm loss PPL: 1.040253E+01 |
--------------------------------------------------------------------------------------------------
```

## Checkpoint Conversion and Upload


Open a long running interactive shell:
```
srun -p compil --cpus-per-task=40 -A six@cpu --time=6:00:00 --pty bash
```
then convert:

```
cd $six_ALL_CCFRSCRATCH/checkpoints/to-upload
time find * -maxdepth 0 -type d -name "global_step*" -exec $six_ALL_CCFRWORK/code/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder {} --output_folder hf/{} \;
```

It takes about 100sec per 26GB checkpoint.

The results will be all under `hf/`.

Now to uploading to the hub.

Prepare the target dir:

```
git clone https://huggingface.co/bigscience/tr1-13B-checkpoints/
cd tr1-13B-checkpoints
transformers-cli lfs-enable-largefiles .
```
We are going to put each checkpoint into its own branch with the same name.

```
mv ../hf/global_step* .
find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout -b {} \; -exec git add {} \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \;

```


## Other backups

Logs:

```
mkdir $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/
tar -zcvf $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/tensorboard.tgz $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/tensorboard
tar -zcvf $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/logs.tgz $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs
```

note: codecarbon wasn't ready during this training, so nothing to back up there.


## Exports

- GCS https://console.cloud.google.com/storage/browser/bigscience
- The Hub https://huggingface.co/bigscience


## Training scripts

The training script is:

- [tr1-13B-round1.slurm](./tr1-13B-round1.slurm)

We also have:

- [tr1-13B-short.slurm](./tr1-13B-short.slurm)

which is a very small model to do quick testing and debug, but otherwise the same as the main script.

The scripts are located at:

```
cd $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base
```

When no jobs are scheduled, currently we launch the main training script using:

```
sbatch --array=1-5%1 tr1-13B-round1.slurm
```
This will schedule 5 20h-trainings which will run one at a time, once the scheduler yields to the request, with unknown wait time in between each job.

If there is a job running already, **do not use the above command** as we can't have 2 trainings overlap. If there is a training already running you can:

1. either tell `sbatch` to start the new job once the currently running job succeeds, using:

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID --array=1-5%1 tr1-13B-round1.slurm
```

Where `CURRENTLY_RUNNING_JOB_ID` is the job being reported running. For example if the report of the last job is:
```
[2021-08-16 22:08:01] tr1-13B-round3 is running for 18:15:59 since 2021-08-16T03:52:02 (711114_4 on 'gpu_p13' partition (r7i4n[1-7],r7i7n[1-8],r8i0n0,r8i5n[3-8],r8i6n[0-8],r9i0n8,r9i1n[0-8],r9i2n[7-8],r9i3n[0-8],r9i4n[0-8],r9i5n[0-2])
```
then the currently running job ID is `711114_4`. You can also gather the same info about the current scheduler status using `squeue`:

```
squeue --user=$(getent group six | cut -d: -f4) | grep tr1-13B
```

2. you could also see how much time is left before the current job finished (based on training log files) and then pass that many hours to `sbatch`. For example, if the job has **less** than 2 hours to run, but more than 1 hour, you want to launch it `now+2hours` from now:

```
sbatch --begin now+2hours --array=1-5%1 tr1-13B-round1.slurm
```

Using `--dependency` may lead to shorter wait times, since if the time passed to `--begin` allows even for a few minutes of delay since the stopping of the last job, the scheduler may already start some other jobs even if their priority is lower than our job. That's because the scheduler ignores any jobs with `--begin` until the specified time arrives.


## On Call

When a person is on call, they need to watch that the training is either running or scheduled to run. If neither is happening they need to schedule a new training. When this situation occurs the log file will report:

```
***ALERT: tr1-13B-round3.slurm is not RUNNING or SCHEDULED! Alert someone at Eng WG***
```

XXX: will soon add an email alert as well. bigscience-jean-zay@groups.google.com


The next section explains how to watch the logs.


Other than waiting for the watchdog which runs once an hour, one can immediately see if anything is scheduled with:

```
$six_ALL_CCFRWORK/code/tr1-13B/bigscience/tools/slurm-status.py --job-name tr1-13B-round3
```

If for some reason the training is not scheduled or running, to schedule a new training:

```
cd $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base
sbatch --array=1-5%1 tr1-13B-round1.slurm
```

This will schedule a job array of 5 jobs of 20h each, so if all goes well, that's at least 4 days of not needing to do anything other than being on the lookout for potential crashes.

XXX: need a troubleshooting section, but elsewhere in the document that is not this training specific.

1. if one of the nodes gets a corrupted gpu, and the training crashes there is a risk that the next job in the training will get allocated the same node, in which case it'll crash again. We need a method to identify which node is corrupted, report that to assist@idris.fr so they know to fix it and exclude this node from the slurm job by adding a list of nodes to exclude as following:

```
sbatch --exclude=r7i5n2,r7i5n6 ...
```
but we currently have no way to identify which node is faulty. I think if we switch to pt-1.9.0 or higher where torch elastic replaces the usual launcher. Or we have to use dedicated log files per node via: `#SBATCH --output=%x-%j-%N.out`.


## Watching the training logs

On JZ:
```
tail -f $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs/main_log.txt
```

Outside of JZ:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/x-linked-size: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr1-13B-logs/resolve/main/main_log.txt
```
Currently the updates happen hourly, so this is a delayed version of `tail -f`.


## CodeCarbon


CodeCarbon wasn't ready until the training was over so we only did an additional 10h run to measure with and the to extrapolate to the whole training.

https://huggingface.co/bigscience/tr1-13B-codecarbon

This set of records captures the startup time and 2499 iterations in 2 records per gpu, since there was also an intermediary checkpoint saved half-way and we flush the CC records on each checkpoint saving.

The training had 168000 iterations. Therefore multiply the reported data by 67. This would be quite approximate since we were using 16 nodes when doing the ramp up, then 64 and only the last 3 weeks 128 nodes.

Caveat emptor: I'm not sure whether CC-reports overlap since each report is per gpu and I think they may be measuring the same thing, other than the gpu itself. So this requires research.

Each csv file contains a report for a single gpu/process. There are 512 reports.


## Extras
