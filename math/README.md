# Handy Math


## Estimate model training time

in days:
```
 (X billion tokens)*(8* M billion parameters)/(N_GPUs * Achieved_TFlops * 1e12*60*60*24)
```

`Achieved_TFlops` is measured by running experiments that tune up the setup for the best throughput performance.

For example, for a 13 billion parameter model, trained for 300 billion tokens, on 256 GPUs at 45 TFlops would take: `(300 billion)*(8*13 billion)/(256*45*1 trillion *60*60*24) = ~31 days`

```
python -c 'print(300*1e9*8*13*1e9/(256*45*1e12*60*60*24))'
```

Notes:

- the factor of 8 can be broken into `(2 x (1+2+1))` where the factor of 2 is for multiple+add, the two ones are for forward propagation and recomputation in the backward and the 2 is for the backward propagation.
