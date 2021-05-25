# Bandwidth tests



## Single node

```
export NCCL_DEBUG=info
python -m torch.distributed.launch --nproc_per_node=4 all_reduce_bench.py
```

Results:
- [16gp](./n1_16gb_all_reduce_bench.txt)
- [32gp](./n1_32gb_all_reduce_bench.txt)

## 16 nodes

```
export NCCL_DEBUG=info
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.launch --nnodes 16 --nproc_per_node=4 --node_rank $SLURM_PROCID --master_addr r7i4n1 --master_port 12345 all_reduce_bench.py' > 16_node_32gb_all_reduce_bench.txt
```

Results:

- [32gp](./n16_32gb_all_reduce_bench.txt)
