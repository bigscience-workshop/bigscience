# Slurm scripts

Mainly here as indicative. Adapt to current traning.

- `cpu.slurm` -> for data preprocessing
- `gpu.slurm` -> arguments are adapted to maximize the gpu mem of the 8 32GB GPU requested




We are using common disk spaces for datasets, caches and experiment dumps:


- Experiment dumps -> `$six_ALL_CCFRWORK/experiments`

`SCRATCH` disk spaces are wiped regularly (wiping every file that was not accessed in the past 30 days) so we have S3 buckets (https://console.cloud.google.com/storage/browser/bigscience-experiments and https://console.cloud.google.com/storage/browser/bigscience-datasets) as shared storage that is accessible from JZ but from others instances too.
