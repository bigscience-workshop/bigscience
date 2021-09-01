#!/bin/bash
#SBATCH --job-name=oscar-fast-shuffle      # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=40           # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu            # allocation account
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

# must set tmp to SCRATCH to be fast
export TMPDIR=$six_ALL_CCFRSCRATCH/tmp
mkdir -p $TMPDIR

# memory to use in GBs float
export MEMORY=150.0

input=oscar-en.jsonl
output=oscar-en-shuffled.jsonl

cd $six_ALL_CCFRSCRATCH/datasets/oscar-small
cat oscar-[0-4].jsonl > oscar-en.jsonl
/usr/bin/time -v $six_ALL_CCFRWORK/bin/terashuf < $input > $output
