# Train 8 - 104B - unmodified Megatron gpt2 - baseline - monolingual

## Intention

While waiting for the main new features to be developed and tested this training is an experiment at a much bigger model size and we are likely to start encountering training instabilities at the 100+B range.

It was suggested that there are 2 ways to get to possible instabilities:
- go deep (many layers)
- go wide (wide model)

For this experiment we chose to 'go wide' and thus made the hidden size extra wide and had to adjust depth to still remain at 104B.


## 104B Training

Comparison with [tr1-13B-base](../tr1-13B-base):
- changed model shape/size
- doubled GBS
- changed lr and min-lr


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
    --rampup-batch-size 16 16 6_000_000 \
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



## Watching the training logs

On JZ:
```
tail -f $six_ALL_CCFRSCRATCH/checkpoints/tr8-104B/tr8-104B-logs/logs/main_log.txt
```

Outside of JZ:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/content-length: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr8-104B-logs/resolve/main/logs/main_log.txt
```
Currently the updates happen hourly, so this is a delayed version of `tail -f`.


## log files

```
cd $six_ALL_CCFRSCRATCH//checkpoints/tr8-104B
mkdir checkpoints
git clone https://huggingface.co/bigscience/tr8-104B-logs
cd tr8-104B-logs
mkdir tensorboard codecarbon logs

transformers-cli lfs-enable-largefiles .



```




## Running

For 64 nodes (not under git)

```
sbatch --array=1-10%1 tr8-104B.slurm
```

For 64 nodes (not under git)
```
sbatch --array=1-10%1 tr8-104B-64.slurm
```

With dependency:

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID --array=1-10%1 tr8-104B-64.slurm
```
