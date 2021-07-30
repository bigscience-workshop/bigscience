#!/bin/bash
#SBATCH --job-name=make-huge-tarball # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=archive

set -x -e

cd $six_ALL_CCFRWORK/datasets
tar -cvf $six_ALL_CCFRSTORE/datasets/openwebtext.tar openwebtext

# if it's plain text and not arrow or another binary format, use gzip
# tar -czvf $six_ALL_CCFRSTORE/datasets/openwebtext.tar.gz openwebtext
