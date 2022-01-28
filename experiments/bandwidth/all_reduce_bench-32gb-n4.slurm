#!/bin/bash
#SBATCH --job-name=all_reduce_bench_n4_32gb
#SBATCH --constraint=v100-32g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu


export NNODES=4
export GPUS_PER_NODE=4
export NCCL_DEBUG=info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.launch --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port 12345 all_reduce_bench.py'  2>&1 | tee all_reduce_bench_n4_32gb.txt
