# Performance

## Network

The state of the network can hugely impact the performance of the training - to the tune of 40% difference in throughput.

When making slurm allocations, use `--contiguous` to request nodes to be close to each other. Unless reserved ahead of time by the admins, such constraint may add a huge delay for when such requests will be granted.
