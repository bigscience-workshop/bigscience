# tr11 176B ML

Large multilingual language model training

## Task

Auto-regressive objective using regular Megatron-LM GPT2 language model w/o multi-lingual dataset

Model size: 176B

The training started on March 11, 2022 11:42am PST

## Main info

Important links:

- [tensorboard](https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard)
- [log file](https://huggingface.co/bigscience/tr11-176B-ml-logs/tree/main/logs/main) or [watch it live](watching-the-training-logs)
- [training slurm script](./tr11-176B-ml.slurm)
- [hub sync script](./tr11-176B-ml-hub-sync-logs.slurm) which lives at `$six_ALL_CCFRWORK/cron/cron.hourly`
- [slurm pulse script](./tr11-176B-ml-slurm-status.slurm) which lives at `$six_ALL_CCFRWORK/cron/cron.hourly`
- each checkpoint with fp32 optim states and bf16+fp32 weights is 2.3TB - just the bf16 weights are 329GB.

XXX: correct this
Languages: ar, ca, code, en, es, eu, fr, id, indic-as, indic-bn, indic-gu, indic-hi, indic-kn, indic-ml, indic-mr, indic-ne, indic-or, indic-pa, indic-ta, indic-te, indic-ur, nigercongo-all, oscar-en, oscar-zh, pt, vi, zhs, zht

Hardware:

- GPUs: 416 A100 80GB GPUs (52 nodes) - using 384 gpus (48 nodes) and keeping 32 gpus (4 nodes) in reserve
- 8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links
- CPU: AMD
- CPU memory: 512GB per node
- GPU memory: 640GB per node
- Inter-node connect: Omni-Path Architecture (OPA)
- NCCL-communications network: a fully dedicated subnet
- Disc IO network: shared network with other types of nodes

Software:

- [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) @ master / BigScience fork
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) @ master (soon)
- [PyTorch](https://github.com/pytorch/pytorch)-1.11 w/ CUDA-11.5
- [apex](https://github.com/NVIDIA/apex) @ master


## Environment

To launch the environment use [start-tr11-176B-ml](./start-tr11-176B-ml)

```
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml
```

See [Environment setup](#environment-setup) for how it was set up.

There is an hourly [pulse checking script](./tr11-176B-ml-slurm-status.slurm) running that checks that the training is either running or scheduled.
XXX: this needs to be updated since we have an exclusive access now and if the training is scheduled but not running this is no longer OK and should fire an email alert.


## Watching the training logs

On JZ:
```
tail -F $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/tr11-176B-ml-logs/logs/main/main_log.txt
```

Outside of JZ:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -LsI $u]=~/2 200.*?content-length: (\d+)/s; \
print qx[curl -Lsr $b-$e $u] if $e>$b; $b=$e; sleep 300}' https://huggingface.co/bigscience/tr11-176B-ml-logs/resolve/main/logs/main/main_log.txt
```
Currently the updates happen hourly, so this is a delayed version of `tail -f`.


## Model Setup

### Packages

- `deepspeed` uses the `olruwase/bf16-updates` branch at the moment - XXX: hopefully it should be merged soon.

- pytorch-1.11-to-be (using a release candidate and will update to final release when it's out) - we must use it for its NCCL version which supports BF16 comms (the NCCL version that comes with pt-1.10 doesn't)

- `tokenizers` requires a special branch `bigscience_fork` which also requires manual building:

```
# to build custom tokenizers make sure that if run on JZ your `~/.cargo/config.toml` contains the following:
[net]
git-fetch-with-cli = true

# if needed first:
# git clone https://github.com/huggingface/tokenizers $six_ALL_CCFRWORK/code/tokenizers
cd $six_ALL_CCFRWORK/code/tokenizers
git checkout bigscience_fork
module load rust
pip install setuptools_rust
pip install -e bindings/python
```
- `transformers` - any version

- `datasets` - any version

- `apex` - any version


### Architecture

```
NHIDDEN=14336
NLAYERS=70
NHEADS=112

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --pp-partition-method 'type:transformer|embedding' \
    [...]
    "
```

`--pp-partition-method 'type:transformer|embedding'` tells the framework to consider the tied embedding layers for partitioning, as the latter are approximately of the same size as the transformer blocks due to the huge 250k dictionary (`250k*HIDDEN_LENTH`). So now the partitioning is:

```
pp rank 0:  [embed | 5 transformer blocks]
pp rank 1:  [6 transformer blocks]
pp rank 2:  [6 transformer blocks]
[...]
pp rank 10: [6 transformer blocks]
pp rank 11: [5 transformer blocks | embed]
```
and each gpu has about the same amount of memory used. Without this rebalancing gpus for pp rank 0 and 11 were using much more gpu memory than the rest - that setup was slower too.


Sanity check:
```
$ NHIDDEN=14336; NLAYERS=70; NHEADS=112; SEQ_LEN=2048; VOCAB_SIZE=250000; python -c "h=$NHIDDEN; l=$NLAYERS; n=$NHEADS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, hidden/layers ratio: {int(h/l)}, hidden/heads ratio: {int(h/n)}')"
Model size: 176B, hidden/layers ratio: 204, hidden/heads ratio: 128
```



### Sequence Length

Default Megatron-LM language model with 2048 tokens sequence length

```
SEQ_LEN=2048

    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \

```



### Replica setup

GPUs = 384 (48 nodes of 8)

```
TP=4
PP=12
DP=8
MBS=2
```

One replica is 48 GPUs (`TP*PP=4*12`)

MBS=2 performs the fastest in this setup w/o using too much additional memory.

Note that due to ZeRO-1 sharding if one decides to run on less GPUs (smaller DP) they may not fit into the smaller collective memory.

We started with MBS=1 as it was faster for smaller GBS (better pipe fill) and switched to MBS=2 at around GBS=784.


### Global batch size

GBS = Global Batch Size

8 replicas -> with MBS=2 (8*2) can do GBS increments of 16 (2 samples per replica).

Use a schedule:

- start from 32k tokens (GBS=16)
- increase linearly to 4.2M tokens/step (GBS=2048) over 9_765_625 samples (~20B tokens)
- then continue at 4.2M tokens/step (GBS=2048) for 210M samples (430B tokens / ~102K steps)

Total: 450B tokens / 220M samples

syntax:
```
--rampup-batch-size <start batch size>  <batch size increment> <ramp-up samples>
```

At seqlen 2048 (1k tokens is bs=1), we get:

```
GLOBAL_BATCH_SIZE=2048
TRAIN_SAMPLES=220_000_000

    --rampup-batch-size 16 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
```


This means it will start with global batch size 16 and over 127 (`(2048-16)/16`) intervals will increase the
batch size by 16 linearly to 2048.

76894 (`9_765_625/127`) is the number of samples before the next GBS increment. That is we run at GBS=16 for 76894 samples, or 4805 steps (`76894/16`). Then we run at GBS=32 for 76894 samples, or 2402 steps (`76894/32`). Then 1600 steps at GBS=48, 1200 at GBS=64, etc....

To calculate how many steps it'll take to reach a specific GBS, use this one-liner. For example to reach GBS=384:
```
perl -le '$x+=76894/(16*$_) for 1..$ARGV[0]/16; print int $x' 384
18146
```

To run to completion the slowest GBS=16, which will take 4805 steps, with 15 sec/step (8 TFLOPs) for GBS=16 (measured on our setup)
```
python -c 'print(f"{4805*15/3660:.2f}h")'
19.69h
```

the data comes from:
```
iteration     3707/  128728 | consumed samples:        59312 | consumed tokens:    121470976 | elapsed time per iteration (s): 15.23 | learning rate: 1.944E-05 | global batch size:    16 | lm loss: 5.396257E+00 | grad norm: 0.765 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 1.051 | TFLOPs: 8.04 |
```

The next step still remains at about the same speed, even though it processes 2x data:
```
python -c 'print(f"{2402*15/3660:.2f}h")'
9.84h
```

That is it'll take about 30h to get to GBS=48.

So we have 18146 to reach gbs=384, and keeping the same speed until the pipeline is filled:

```
python -c 'print(f"{18146*15/3660:.2f}h")'
74.37h
```
so more than 3 days at slow speed.

So it'll take several days of very inefficient run. We know we get 113 TFLOPs at iteration 512, and since PP=12 and MBS=2, only at 384 `12*2*16` it'll be the first time all pipeline stages will be filled and that's when the performance should be much better, probably around 90 TFLOPs.


Notes:
* `--rampup-batch-size` requires the use of `--train-samples` and can't be used with `--train-iters`.
* global batch size has to be divisible by micro-batch-size * DP_SIZE


Update: at the end we decided to start with GBS=192 and MBS=1, as 16 was too too slow, so the current setup starts with 73 TFLOPs as compared to 8 with GBS=16.



### Optimizer

`apex.optimizers.FusedAdam` is used.

- AdamW, β1=0.9, β2=0.95, eps=1e−8
- learning rate:
   * peak=6e-5
   * warmup over 183_105 samples (375M tokens)
   * cosine decay for learning rate down to 10% of its value, over 410B tokens (after 410B tokens, training continues at 10% of the original learning rate, that is fixed `--min-lr`)
- clipping by global norm of 1 (as in GPT-3)
- weight decay of 0.1

We need lr-decay in samples, so tokens2samples = 410B / 2048 = ~200_000_000


```
LR_DECAY_SAMPLES=200_000_000
LR_WARMUP_SAMPLES=183_105  # 375M tokens

    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
```


### std Init

This proved to be a very crucial setting in our 104B experiments and we couldn't break past the first few thousands iterations until we figured out the 0.02 default `--init-method-std` was a way too big.

1. "Transformers without Tears" paper https://arxiv.org/abs/1910.05895 prescribes: `sqrt(2/(NHIDDEN*5))`

2. The 530B training paper https://arxiv.org/abs/2201.11990 they used an even smaller init formula: `sqrt(1/(NHIDDEN*3))`

and we decided to go with the 530B one as it leads to an even smaller init value.

To make it easier to compare the two formulas, they can be rewritten as:
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

Thus: `sqrt(1/(14336*3)) = 0.00482197968631537`

```
    --init-method-std 0.0048 \
```

### Misc features

* **Positional encoding**:

   We use the added by us AliBi implementation:

   ```
       --position-embedding-type alibi \
   ```
Paper: [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

* **Embed LayerNorm**:

   We use the added by us embedding layer norm which makes the training more stable at a small training slowdown cost and a tiny additional amount of memory.

   ```
       --embed-layernorm \
   ```

   This insight came from experimenting with https://github.com/facebookresearch/bitsandbytes which contains a `StableEmbedding` which is a normal Embedding with layernorm and it uses a uniform xavier initialization.


* **Activation Function**:

   `torch.nn.functional.gelu`

   Various activation functions were experimented with and GeLU was the best, when considering outcome quality and training throughput.


### Data and Tokenizer


```
BIGSCIENCE_REPO=$six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience
TRAIN_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/train-splits.txt
VALID_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/valid-splits.txt
CATALOGUE_JSON_PATH=$BIGSCIENCE_REPO/data/catalogue/training_dataset_ratios_merged_nigercongo.json
LOAD_RATIOS_SCRIPT=$BIGSCIENCE_REPO/data/catalogue/load_ratios_meg_ds_format.py
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split train --output-meg-ds-ratio-file $TRAIN_DATA_PATH
python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split valid --output-meg-ds-ratio-file $VALID_DATA_PATH

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
```

Backups of data:

- `$six_ALL_CCFRWORK/bigscience-training/merged-meg-ds_v2` is backed up at `$six_ALL_CCFRSTORE/datasets/merged-meg-ds_v2`. These paths are inside `data/*-splits.txt` files.

XXX: need to back up v3



### Data type

We are using `bfloat16` since it's supposed to deliver a training experience with less instabilities as compared to `float16`, due to the former's better numerical range (i.e. no overflow risks).

```
    --bf16 \
```
and the rest is in the Deepspeed config


### Deepspeed config


The new `BF16_Optimizer` implements its own ZeRO Stage 1, hence until it gets its own stage number, we must use:
```
ZERO_STAGE=0
config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

```

The new `BF16_Optimizer` accumulates grads in fp32. It doesn't shard the static fp32 buffer it reuses, which consumes `4 bytes * params` of additional memory, but since it's not sharding it saves on communications overhead. Down the road, it'll be expanded to support sharding on demand.

Using Deepspeed's activation checkpointing to use a lot less GPU memory:

```
    --deepspeed-activation-checkpointing \
```



### Important environment variables

The usual set to tell where things are and that we are working w/o internet on the compute nodes:

```
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

```


There is some complex hanging problem that occurs under certain conditions with 40+ nodes, which the following settings solves:

```
export CUDA_LAUNCH_BLOCKING=1
```
in theory it should make everything much slower but makes a tiny impact or no impact at all to the throughput.


To hide duplicated errors using this hack - will be properly fixed in pt-1.12

```
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
```
using `/tmp/` on purpose here so that each node will have a different target.




### Launcher

We are using the latest elastic-based launcher with `c10d` backend.

```
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
```

`--tee 3` prefixes all logs with the local rank, which helps to unravel interleaved error messages by grepping for one of the local rank prefixes, e.g.:
```
grep `[default7]` main_log.txt
```




### Cronjobs to auto-sync the hub

TB and the log files are hourly synced to https://huggingface.co/bigscience/tr11-176B-ml-logs via `$six_ALL_CCFRWORK/cron/cron.hourly/tr11-176B-ml-hub-sync-logs.slurm`.

if you want to do an additional manual sync on demand at any moment you can do:

```
cd $six_ALL_CCFRWORK/cron/cron.hourly
sh tr11-176B-ml-hub-sync-logs.slurm
```



### Dealing with 100h SLURM limit

First, let's ensure we save a checkpoint just before SLURM kills the job

Let's try 99:50 5990=60*100-10

```
    --exit-duration-in-mins 5990 \
```

We need about 2 min per cycle plus a few minutes to save the checkpoint.

We will use job arrays, to automatically start a new job. Let's start with just 10 such jobs:

```
sbatch --array=1-10%1 tr11-176B-ml.slurm
```

`%1` limits the number of simultaneously running tasks from this job array to 1, since we want them to run in a sequence.


As we have full control over the slurm we don't need to create the train concept to be able to modify the slurm script w/o losing a place in the queue, so just unschedule all jobs if changing the script and then re-schedule them again.

Also remember that if it's not you who started the job you can't kill it. And you have to use the [kill switch](#kill-switch) workaround instead.



### Kill Switch

This is a feature that allows us to "kill" a SLURM job started by a user who isn't around at the moment, since SLURM doesn't support groups and we don't have `sudo` access. But basically we get the program to poll for a file at startup and before each iteration and it'll quit if it finds this file.

For an explanation on how it works see: [Kill Switch](../../jz/slurm/README.md#kill-switch)

Note that it saves the checkpoint before exiting, so nothing gets lost.

To arm:

```
KILL_SWITCH_PATH=$MEGATRON_DEEPSPEED_REPO/kill-switch-tr11-176B-exp1
[...]
    --kill-switch-path $KILL_SWITCH_PATH \
```

To trigger:

```
touch $MEGATRON_DEEPSPEED_REPO/kill-switch-tr11-176B-exp1
```

To deactivate and let new instances of a job run normally:

```
rm  $MEGATRON_DEEPSPEED_REPO/kill-switch-tr11-176B-exp1
```

So if there is an array of jobs that belong to another user, each job will now try to start and it'll abort once it detects a kill switch - this process can take a few minutes per job. If there is only one job it'll exit when it saved the checkpoint.

Sometimes the job still doesn't exit after it saved the checkpoint and had so to be killed manually, which might not be possible if the job isn't yours. Then it'll eventually time out in some 15min or so and will exit.


### Checkpoints

XXX:
During bs ramp up 250 should be ok, but once we reach GBS=2048, should save more frequently until we know the hardware holds. Just last night the training crashed loosing a few hours of work.  At full GBS=2048 we have about 2min/iteration so 250 iterations is about ~8h. So if the training crashes at 7:50, we lose 8h of work. So probably we need to create the checkpoints more frequently than that, but that also requires that we delete the many checkpoints pretty often as well. Probably every 4h is sane enough of a compromise.

```
SAVE_INTERVAL=250
    --save-interval $SAVE_INTERVAL \
```

If we want to save just the weights w/o optimizer states then saving just these 2 groups:

```
ls -l layer_* mp_*
```

Here is the breakdown:

- bf16 weights - 2 bytes per param: `176*10**9*2/2**30 = 327.82GB`
- the whole checkpoint - 8 bytes for fp32 optim, 4+2 bytes for weights (fp32+bf16) - total 14 bytes per param: `176*10**9*14/2**30=2294.77GB`

Let's validate:

```
$ cd main/global_step1000
$ du -ch mp* layer*
329G    total
$ du -sh .
2.3T
```

So it all checks out.


Preserving checkpoints:

The least is to store a full checkpoint every 10% of the training. More frequent than that is better, but 10% is the minimum



### Watchdogs

1. We have the filesystem watchdog running at `$six_ALL_CCFRWORK/cron/cron.daily/fs-watchdog.slurm`
2. We have the is-training-scheduled-or-running watchdog running at `$six_ALL_CCFRWORK/cron/cron.hourly/tr11-176B-ml-slurm-status.slurm`



### Estimated run time

Best case scenario when training 24/7 on 48 nodes with 8 gpus each:
```
$ python -c 'Btokens=450; Bmodel=167; n_gpus=384; Tflops=150; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
120.80 days
```

Since this doesn't include the batch size rampup when we run on average at half speed - add a few more days to that.



### On Call

When a person is on call, they need to watch that the training is either running or scheduled to run. If neither is happening they need to schedule a new training. When this situation occurs the log file will report:

```
***ALERT: tr11-176B-ml is not RUNNING or SCHEDULED! Alert someone at Eng WG***
```

An email alert is sent as well to `bigscience-jean-zay@groups.google.com`.

The next section explains how to watch the logs.

Other than waiting for the watchdog which runs once an hour, one can immediately see if anything is scheduled with:

```
$six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/tools/slurm-status.py --job-name tr11-176B-ml
```

To see if it's your job that's scheduled:

```
squeue -u `whoami` -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"
```

To see if you or anybody else in the group scheduled this job:
```
squeue -u $(getent group six | cut -d: -f4) -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"'
```

If you have to kill a slurm job launched or scheduled by someone else you need to read about the [kill switch](#kill-switch).

If for some reason the training is not scheduled or running, to schedule a new training:

```
cd $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml
sbatch --array=1-3%1 tr11-176B-ml.slurm
```

This will schedule a job array of 3 jobs of 100h each, so if all goes well, that's at least 12 days of not needing to do anything other than being on the lookout for potential crashes.

To see the availability of the gpus, do:

```
sinfo -p gpu_p5
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
gpu_p5       up 4-04:00:00      1  drain jean-zay-iam27
gpu_p5       up 4-04:00:00     48  alloc jean-zay-iam[01-26,28-49]
gpu_p5       up 4-04:00:00      3   idle jean-zay-iam[50-52]
```
so here we have one broken node (state:`drain`), 48 being used and 3 are idle and can be used. Note that if we have less than 48 nodes we can't continue training. Notify the sysadmins if there are many unavailable gpus.

XXX: need a troubleshooting section, but elsewhere in the document that is not this training specific.

1. if one of the nodes gets a corrupted gpu, and the training crashes there is a risk that the next job in the training will get allocated the same node, in which case it'll crash again. We need a method to identify which node is corrupted, report that to assist@idris.fr so they know to fix it and exclude this node from the slurm job by adding a list of nodes to exclude as following:

```
sbatch --exclude=jean-zay-iam34,jean-zay-iam35 ...
```
or:
```
sbatch --exclude=jean-zay-iam[34-35] ...
```

but we currently have no way to identify which node is faulty. I think if we switch to pt-1.9.0 or higher where torch elastic replaces the usual launcher. Or we have to use dedicated log files per node via: `#SBATCH --output=%x-%j-%N.out`.


When doing `git pull` or `scp`, `gsutil`, etc - anything creating or updating files and dirs, please make sure the permissions are such that they are set to be `rw` by the group and the group is set to `six` - if this is messed up others might not be able to edit files. Here is how to fix the perms [Syncing the perms](../../jz/envs#syncing-the-perms).

If you found this situation where someone's files have wrong perms, often you can work around it by moving the "bad" files away and replacing those with new files with the correct permissions - e.g. via the repo. e.g. this will restore the original slurm script to the git version:

```
mv tr11-176B-ml.slurm tr11-176B-ml.slurm.bad
git checkout tr11-176B-ml.slurm
```

### Testing new changes to the script

Before making changes to the main training script apply those to the 2 node script [setup-test-n2.slurm](setup-test-n2.slurm).

Then:

```
sbatch setup-test-n2.slurm
```
watch:
```
tail -F $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml-test-setup/tr11-176B-ml-logs/logs/main-test-setup/main_log.txt
```
then, of course, kill the 2-node job as soon as testing is complete.
```
squeue -u `whoami` -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R" | grep setup-test-n2
scancel <jobid>
```



### Backups

We need to back up checkpoints, logs and tensorboard files.

Most of the time you need to use a non-login shell to do the job or it will get killed:
```
srun --pty -A six@cpu -p compil --hint=nomultithread --time=06:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

Backing up to STORE: root dir: `$six_ALL_CCFRSTORE/checkpoints/tr11-176B-ml`

```
cp -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step3000 $six_ALL_CCFRSTORE/checkpoints/tr11-176B-ml/checkpoints/
```

Backing up to GCS: root dir: `gs://bigscience-backups/tr11-176B-ml`


```
gsutil -m rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step3000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step3000

gusitil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/tr11-176B-ml-logs gs://bigscience-backups/tr11-176B-ml/tr11-176B-ml-logs
```



### Environment setup

Please do not run any of these instructions unless you know what you're doing. The environment has already been set up and the information below is for fixes/updates/rebuilding env.

To launch the environment use [start-tr11-176B-ml](./start-tr11-176B-ml)

```
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml
```

The git clones that we install or run from are under `$six_ALL_CCFRWORK/code/tr11-176B-ml/`.


```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n tr11-176B-ml python=3.8
conda activate tr11-176B-ml

pip install transformers

# switch to a `compil` interactive node where we don't get killed by cgroups
srun --pty -A six@cpu -p compil --hint=nomultithread --time=60 bash

conda activate tr11-176B-ml

# pt-1.11.0 / cuda 11.5
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/test/cu115/torch_test.html -U

# XXX: will change on Mar-11
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install deepspeed
cd $six_ALL_CCFRWORK/code/tr11-176B-ml/DeepSpeed
./build.sh

cd $six_ALL_CCFRWORK/code/tr11-176B-ml/Megatron-DeepSpeed
pip install -r requirements.txt

cd $six_ALL_CCFRWORK/code/tr11-176B-ml/apex
./build.sh

# to build custom tokenizers make sure that if run on JZ your `~/.cargo/config.toml` contains the following:
[net]
git-fetch-with-cli = true

# if needed first:
# git clone https://github.com/huggingface/tokenizers $six_ALL_CCFRWORK/code/tr11-176B-ml/tokenizers
cd $six_ALL_CCFRWORK/code/tr11-176B-ml/tokenizers
git checkout bigscience_fork
module load rust
pip install setuptools_rust
pip install -e bindings/python

```
