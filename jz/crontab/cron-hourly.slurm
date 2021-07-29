#!/bin/bash
#SBATCH --job-name=cron-hourly       # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=compil

# do not set -e - we must run all of it
# set -x -e

cd $six_ALL_CCFRWORK/cron/scheduler

# ensure to restart self first
sbatch --begin=now+1hour cron-hourly.slurm

# now launch any slurm scripts in cron.hourly
cd $six_ALL_CCFRWORK/cron/cron.hourly
for f in *.slurm; do
  sbatch "$f"
done