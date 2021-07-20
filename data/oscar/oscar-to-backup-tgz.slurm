#!/bin/bash
#SBATCH --job-name=oscar-to-backup-tgz # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.out           # error file name (same to watch just one file)
#SBATCH --account=six@cpu

set -x -e

cd $six_ALL_CCFRSCRATCH/datasets/oscar-small

# plain text -> gz
gzip oscar-en-shuffled.jsonl
mv oscar-en-shuffled.jsonl.gz $six_ALL_CCFRSTORE/datasets/

# already binary -> tar
tar -cvf $six_ALL_CCFRSTORE/datasets/oscar-en-cache.tar cache
