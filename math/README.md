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
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=127; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```
(ng = total gpus, ms = model size in B, gbs = global batch size, sp = throughput in seconds)

same with bash env vars and broken down GBS into mbs*dp*gas (gas=pp_chunks):
```
echo "($MSIZE*4*2*SEQLEN*$MICRO_BATCH_SIZE*$DP_SIZE*$GAS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
```

- Automatically process slurm/ megatron log files, average the throughput (prints 'fail' on when the training failed w/o producing a single iteration stat):
```
find . -type f -name "*out" -exec perl -lne 'm|elapsed time per iteration .ms.: ([\d\.]+)| &&  do {$x+=$1; $c++}; END { print "$ARGV " . ($c ? int($x/$c/1000) : "fail")}' {} \; | sort | grep -v fail
```

## Model sizing

### Params as a function of the network size hyperparams

```
NHIDDEN=4096; NLAYERS=36; SEQ_LEN=512; VOCAB_SIZE=50257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B, ratio={int(h/l)}')"
```

For full details see [Calculate model size](../experiments/gpt2-utils.md).

### Width-depth tradeoff

From [The Depth-to-Width Interplay in Self-Attention](https://arxiv.org/abs/2006.12467):

```
NLAYERS=70; python -c "import math; l=$NLAYERS; a = 5.039; b = 5.55e-2; print(f'Optimal n_params: {12 * l * math.exp(2*a) * math.exp(2*b*l) / 10**9 :.0f}B')"
```
This seems to be less important as the number of parameters scales up, but is useful to ground the discussion.


## Estimate total training time

Training Time Estimates. Given these throughputs, we can also estimate the total amount of time needed for end-to-end training on 𝑇 tokens. Training requires 𝐼 = 𝑇 /(𝐵 · 𝑠) iterations. Using the value of 𝐹 from equation (3) and empirical end-to-end throughputs from Table 1 (denoted by 𝑋), we can estimate total training time. We note that for the configurations in Table 1, we have 6ℎ ≫ 𝑠, 16𝑙ℎ ≫ (𝑉 + 𝑠), and 12𝑙ℎ ≫ 𝑉 . Combining these observations with equations (2) and (3), we arrive at:

End-to-end training time (seconds) ≈ 8𝑇𝑃/𝑛𝑋

Let us consider the GPT-3 model with 𝑃 =175 billion parameters as an example. This model was trained on 𝑇 = 300 billion tokens. On 𝑛 = 1024 A100 GPUs using batch size 1536, we achieve 𝑋 = 140 teraFLOP/s per GPU. As a result, the time required to train this model is 34 days. For the 1 trillion parameter model, we assume that 450 billion tokens are needed for end-to-end training. With 3072 A100 GPUs, we can achieve a per-GPU throughput of 163 teraFLOP/s, and end-to-end training time of 84 days. We believe these training times (using a reasonable number of GPUs) are practical.


This math and discussion is quoted from [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473).

Let's explain the formula: `8𝑇𝑃/𝑛𝑋`

In the formula:

- T: number of tokens used for training in Billions
- P: number of parameters in normal numbers
- n: number of GPUs
- X: throughput per GPU in TFlops
- The result is in seconds, so divide by 3600*24 to get days

Example:

- T = 300B
- P = 200_000_000
- X = 150 TFLOPs (more or less the best one can get on an efficient setup on A100)
- n = 350

gives us:

```
$ python -c 'print(f"{8*300*200_000_000/(350*150)/(3600*24):0.2f}", "days")'
105.82 days
```
