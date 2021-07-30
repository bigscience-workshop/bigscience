#!/bin/bash
#SBATCH --job-name=oscar-to-backup-tgz # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10           # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=cpu_p1
#SBATCH --account=six@cpu
#SBATCH --qos=qos_cpu-t4

# 20h is not enough to gzip 1.2TB file, so have to use the other allocation
##SBATCH --partition=archive


set -x -e

cd $six_ALL_CCFRSCRATCH/datasets/oscar-small

# plain text -> gz
gzip oscar-en-shuffled.jsonl
mv oscar-en-shuffled.jsonl.gz $six_ALL_CCFRSTORE/datasets/

# already binary -> tar
tar -cvf $six_ALL_CCFRSTORE/datasets/oscar-en-cache.tar cache
