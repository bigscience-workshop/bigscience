#!/bin/bash
#SBATCH --job-name=tar_experiments    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3
#SBATCH --output=../%x-%j.out           # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

for filename in *; do
    tar -cvf "$filename.tar" "$filename"
    mv "$filename.tar" $ALL_CCFRSTORE/arch_scaling_experiments_store/"$filename.tar"
done
