# Sanity Checks

When configuring the slurm script must ensure the following is strictly exact:


1.

players:
- NHIDDEN
- NHEADS

```
NHIDDEN % NHEADS == 0
```

2.

players:
- GLOBAL_BATCH_SIZE
- MICRO_BATCH_SIZE
- DP_SIZE

```
GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE) == 0
```

3.

players:
- NLAYERS
- PP_SIZE

```
NLAYERS % PP_SIZE == 0
```

4.




5. Curriculum Learning Constraints

- min_difficulty % 8 = 0 (to enable Tensor Core acceleration)

- json ds config can't have numbers with '_' in them - invalid json - careful with substitutions.
