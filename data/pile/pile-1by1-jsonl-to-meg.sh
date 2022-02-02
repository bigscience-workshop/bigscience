#!/bin/bash

for i in {0..29}
do
    sbatch $six_ALL_CCFRWORK/code/bigscience/data/pile/pile-1by1-jsonl-to-meg.slurm $i
done
