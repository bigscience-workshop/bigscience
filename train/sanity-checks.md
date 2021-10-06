# Sanity Checks


1.

```
NHIDDEN % NHEADS = 0
```

2.

GLOBAL_BATCH_SIZE has to be divisible by MICRO_BATCH_SIZE*DP_size

```

```

3.

NLAYERS must be a multiple of PP_SIZE
