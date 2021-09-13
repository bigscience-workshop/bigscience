# Handy Math


## Estimate model training time

in days:
```
 (X billion tokens)*(8* M billion parameters)/(N_GPUs * Achieved_TFlops * 1e12*60*60*24)
```

`Achieved_TFlops` is measured by running experiments that tune up the setup for the best throughput performance.

For example, for a 13 billion parameter model, trained for 300 billion tokens, on 256 GPUs at 45 TFlops would take: `(300 billion)*(8*13 billion)/(256*45*1 trillion *60*60*24) = ~31 days`

```
$ python -c 'Btokens=300; Bmodel=13; n_gpus=256; Tflops=45; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
31.35 days
```

Notes:

- the factor of 8 can be broken into `(2 x (1+2+1))` where the factor of 2 is for multiple+add, the two ones are for forward propagation and recomputation in the backward and the 2 is for the backward propagation.

contributed by Samyam Rajbhandari


## Calculate TFlops


TFlops: `model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

The factor of 4 is when used with activation check-pointing,
otherwise it will be 3, but for 200B model, activation check-pointing will always be on.

```
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=127; print $ms*4*2*1024*$gbs / ( $sp * $ng * 1e3)'
```
(ng = total gpus, ms = model size in B, gbs = global batch size, sp = throughput in seconds)

same with bash env vars and broken down GBS into mbs*dp*gas (gas=pp_chunks):
```
echo "($MSIZE*4*2*1024*$MICRO_BATCH_SIZE*$DP_SIZE*$GAS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
```

- Automatically process slurm/ megatron log files, average the throughput (prints 'fail' on when the training failed w/o producing a single iteration stat):
```
find . -type f -name "*out" -exec perl -lne 'm|elapsed time per iteration .ms.: ([\d\.]+)| &&  do {$x+=$1; $c++}; END { print "$ARGV " . ($c ? int($x/$c/1000) : "fail")}' {} \; | sort | grep -v fail
```

## Calculate model size

```
NHIDDEN=4096
NLAYERS=36
SEQ_LEN=512
VOCAB_SIZE=50257
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B')"
```



For full details see [Calculate model size](./experiments/gpt2-utils.md).
