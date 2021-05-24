# Deepspeed notes



## Should I use the `deepspeed` launcher under slurm.

No, it won't work.

Instead use:
```
python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $SLURM_PROCID \
    ....
```

## on 8 gpus I get now: `data_parallel_size: 8, parameter_parallel_size: 8`

In this case seeing that the DP and parameter parallel size match means ZeRO will partition across all gpus



##
