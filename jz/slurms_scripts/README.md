# Slurm scripts

Mainly here as indicative. Adapt to current traning.

- `cpu.slurm` -> for data preprocessing
- `gpu.slurm` -> arguments are adapted to maximize the gpu mem of the 8 32GB GPU requested

# Shared disk spaces

We are using common disk spaces for datasets, caches and experiment dumps:

- Model cache and datasets -> `$six_ALL_CCFRWORK/models` and `$six_ALL_CCFRWORK/datasets`
- Experiment dumps -> `$six_ALL_CCFRWORK/experiments`

`SCRATCH` disk spaces are wiped regularly (wiping every file that was not accessed in the past 30 days) so we have S3 buckets (https://console.cloud.google.com/storage/browser/bigscience-experiments and https://console.cloud.google.com/storage/browser/bigscience-datasets) as shared storage that is accessible from JZ but from others instances too.
