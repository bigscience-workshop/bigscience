# python -m torch.distributed.launch --nproc_per_node=2 all_reduce_bench.py

import torch
import torch.distributed as dist
import time
import argparse
import os
import fcntl

TRIALS = 5

N = 500000
M = 2000

def printflock(*msgs):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def timed_allreduce(mat, global_rank):
    pre = time.perf_counter()
    dist.all_reduce(mat)
    printflock(f"ignore me {int(mat[0][0])}")  # required due to lazy evaluation
    duration = time.perf_counter() - pre
    tput = ((M*N*4*2)/duration)*8
    size = M * N * 4
    n = dist.get_world_size()
    busbw = (size / duration) * (2 * (n - 1) / n) * 8
    printflock(f"{global_rank}:\n",
               f"duration: {duration:.4f} sec\n",
               f"algo throughput: {tput:.4f} bps, {tput/1e9:.4f} Gbps\n",
               f"busbw: {busbw / 1e9:.4f}  Gbps"
    )

def run(local_rank):
    global_rank = dist.get_rank()
    printflock(f"{global_rank} data size: {M*N*4/1e9} GB")
    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    for _ in range(TRIALS):
        timed_allreduce(mat, global_rank)

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    rank = args.local_rank
    printflock("local_rank: %d" % rank)
    init_processes(local_rank=rank, fn=run)
