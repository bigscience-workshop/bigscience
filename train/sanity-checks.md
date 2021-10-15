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


## Restaring from existing checkpoint constraints

XXX: quite a few of these - need to start collecting them all

- can't change TP-size (But ok to change PP)

- can't change max-lr or will get:

```
AnnealingLR: class input value 1e-05 and checkpointvalue 3e-05 for learning rate do not match
```
