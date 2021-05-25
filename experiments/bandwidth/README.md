# Bandwidth tests



## Single node

```
export NCCL_DEBUG=info
python -m torch.distributed.launch --nproc_per_node=4 all_reduce_bench.py
```

Results:
- [16gp](./n1_16gb_all_reduce_bench.txt) - `algo throughput: 1329.4242 Gbps`
- [32gp](./n1_32gb_all_reduce_bench.txt) - `algo throughput: 1323.6244 Gbps`

## 16 nodes

```
export NCCL_DEBUG=info
export MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.launch --nnodes 16 --nproc_per_node=4 --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port 12345 all_reduce_bench.py' > 16_node_32gb_all_reduce_bench.txt
```

Results:

- [32gp](./n16_32gb_all_reduce_bench.txt) - `algo throughput: 55.0766 Gbps`
