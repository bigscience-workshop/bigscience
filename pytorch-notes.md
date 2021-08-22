# PyTorch Notes

This document lists nuances of pytorch relevant to our work.

## Distributed Launcher

### pt <= 1.8.1

The good old `torch.distributed.launch` works here:

```
export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "
```


### pt >= 1.9

pytorch switched to elastic in 1.9. `torch.distributed.launch` is supposed to be backward compatible, but it's not. Under multi-node it results in `RuntimeError: Address already in use` error.

Therefore for pt 1.9 and higher you must use the following launcher syntax:

```
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    "
```

For full details see: https://pytorch.org/docs/1.9.0/elastic/quickstart.html

Note: If you're using the `deepspeed` launcher (which we can't use in the slurm environment), it should continue working as before with either pytorch version.
