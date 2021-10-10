# Train 8 - 104B - unmodified Megatron gpt2 - baseline - monolingual

Note that since this is a very stop-n-go experimental training with no intention to complete it, you will find extensive notes on how to do almost anything in [tr1-13B-base](../tr1-13B-base), here we only list what's different.

## Intention

While waiting for the main new features to be developed and tested this training is an experiment at a much bigger model size and we are likely to start encountering training instabilities at the 100+B range.

It was suggested that there are 2 ways to get to possible instabilities:
- go deep (many layers)
- go wide (huge hidden size)

For this experiment we chose to 'go wide' and thus made the hidden size extra wide and had to adjust depth to still remain at 104B. Using the hidden/layers ration of 512 instead of the usual 150-200.

## Environment

For this training currently :
- using the same `tr1-13B` conda env
- made a copy of `$six_ALL_CCFRWORK/code/tr1-13B` to `$six_ALL_CCFRWORK/code/tr1-104B`
- copied `tr1-13B` branch of Meg-DS to `tr8-104B` branch (and later made some changes to it - see Corby's PR below)
the setup is the same as fully documented in  [tr1-13B-base](../tr1-13B-base).

## Memory usage


```
# Let h = hidden size, n = num_layers, k = num_heads, s = sequence length, v = vocabulary size
total_params = n * (12h^2 + 13h) + (v * h) + (s * h) + 2*h
```

- 0.8x times layers=32 than 13B (40)
- 3.2x times NHIDDEN=16384 than 13B (5120)

While the 104B model is 8x times bigger than 13B param-wise, the model grows quadratically with NHIDDEN size, so each layer will require ~10x (3.2**2) more gpu memory plus more memory per activations. We double TP from 2 to 4 as 4 is a max we can use on a 4-gpu node. So we have to 5x the PP then, so we need at least PP=20, and to work with NLAYERS=32, it takes us to PP=32.

So:
```
TP_SIZE=4
PP_SIZE=32
```

so 13B took 8 gpus for a single replica, and 104B needs 128 gpus (16x times)


During training currently we use 32 nodes or 4096GB (128x 32GB gpus) per each full replica (TP=4 + PP=16), the rest are ZeRO-DP. So if we throw x times more GPUs we just speed things up by having more 32-node replicas.

The required memory breakdown:

1. 4B for fp32 weights
2. 2B for fp16 weights
3. 8B for optimizer states.
4. 4B for gradients (we don't save these in the checkpoint)
5. plus memory for activations and temps, which total majorly depends on the seqlen and mini batch size - and since we use activation checkpointing this memory need is quite small.

Total: 1872GB (18*104) plus activations and temps memory. The param-needed memory is much less than 4096GB, because we don't need that much memory, but because we have to double memory for each increment we go from PP=16 to PP=32, whereas PP=20 should have been enough.

Activation memory would have been much much bigger if it weren't for activation checkpointing.



## 104B Training

XXX: a lot of the following is no longer correct as we are changing the model a lot due to instabilities, so need to review once we have it sorted out. [chronicles.md](chronicles.md) keeps track of the changes and of course the latest setup is reflected in [tr8-104B.slurm](tr8-104B.slurm).

Comparison with [tr1-13B-base](../tr1-13B-base):
- changed model shape/size to be extra wide NHIDDEN=16384, which makes the hidden/layers ratio of 512 (the normal ratio in Megatron paper is 150-200)
- doubled GBS (Global batch size)
- changed lr and min-lr from --lr 1e-4 --min-lr 1e-5 to --lr 6e-5 --min-lr 6e-6
- doubled batch size rampup to 32 from 16, since PP=32 and we can't stretch bs=16 over 32 gpus.

Later during experiments changed to `--adam-beta2 0.95` as it proved to train faster.

Additionally Corby Rosset suggested we try a more numerically stable self-attention version, which was implemented [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118). Note, that hasn't been merged into the `main` tree, it's currently only in the `tr8-104B` branch of Meg-DS



everything else is the same.

Let's check the model size:

```
VOCAB_SIZE=50257 NLAYERS=32 NHIDDEN=16384 NHEADS=32 SEQ_LEN=2048; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"
104B
```

```
SEQLEN=2048

VOCAB_SIZE=50257
NLAYERS=32
NHIDDEN=16384
NHEADS=32
SEQ_LEN=2048
    --rampup-batch-size 32 32 6_000_000 \
    --global-batch-size 2048 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
```

Saving checkpoints every 300 iterations so that if we have to recover a training we don't have to roll back far.

Switched to logging on every single iteration, in case we need to remap samples to text to find a really bad text input in case of a huge glitch in lm loss.



## Watching the training logs

On JZ:
```
tail -f $six_ALL_CCFRSCRATCH/checkpoints/tr8-104B/tr8-104B-logs/logs/main_log.txt
```

Outside of JZ:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/content-length: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://cdn-lfs.huggingface.co/bigscience/tr8-104B-logs/b2cc478d5ae7c9ec937ea2db1d2fe09de593fa2ec38c171d6cc5dca094cd79f9
```
Currently the updates happen hourly, so this is a delayed version of `tail -f`.


## log files

```
cd $six_ALL_CCFRSCRATCH//checkpoints/tr8-104B
mkdir checkpoints
git clone https://huggingface.co/bigscience/tr8-104B-logs
cd tr8-104B-logs
mkdir tensorboard codecarbon logs
git lfs track "*.csv"
git lfs track "*.txt"
huggingface-cli lfs-enable-largefiles .
```




## Running

With dependency and array (remove unneeded parts)

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID --array=1-10%1 tr8-104B.slurm
```

```
sbatch --array=1-10%1 tr8-104B.slurm
```

For 64 nodes (not under git, local adjusted copy)

```
sbatch --array=1-10%1 tr8-104B-64.slurm
```


## Syncing / monitoring

```
cd $six_ALL_CCFRWORK/cron/cron.hourly
ls -1 tr8*slurm
tr8-104B-hub-sync-logs.slurm
tr8-104B-slurm-status.slurm
```

Here is the slurm script to sync the tensorboard/codecarbon/logs data: [tr1-104B-hub-sync-logs.slurm](./tr1-104B-hub-sync-logs.slurm)

SLURM status and alerts script: [tr8-104B-slurm-status.slurm](tr8-104B-slurm-status.slurm)


## Curriculum learning

For full details see:
- [guide](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/examples/curriculum_learning/README.md)
- [paper](https://arxiv.org/abs/2108.06084)

Transitioning from a BS-rampup based setup like tr1-13B

Changes to cl args:

1. remove --rampup-batch-size
2. if possible to increase your micro batch size if it fits at max seqlen/gbs - but needs to be tested w/o CL enabled, otherwise a false lower memory usage may occur under early steps of CL
3. add --train-tokens n_samples * SEQ_LEN
4. double --train-samples
5. add --lr-decay-tokens: SEQ_LEN * --lr-decay-samples

Changes to DS config file:

1. `total_curriculum_step` - recommendation: `~TRAIN_TOKENS/GLOBAL_BATCH_SIZE/SEQ_LEN/2`

the last `/2` is a rough approximation.

Here total iterations `300B/2K/2K = 71525` steps. So `total_curriculum_step ~= 36_000`

4. `min_difficulty` 64 (recommended for large model)

3. `max_difficulty`: $SEQ_LEN (always)

also an important constraint `min_difficulty % 8 = 0` (to enable Tensor Core acceleration)

4. `difficulty_step` is 8 (always)
