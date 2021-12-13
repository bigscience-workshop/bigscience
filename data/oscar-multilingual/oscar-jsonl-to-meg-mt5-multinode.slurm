#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-mt5-multinode # job name
#SBATCH --ntasks-per-node=1                   # number of MP tasks
#SBATCH --nodes=8
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate teven-temp-backcompat
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

input=$six_ALL_CCFRSCRATCH/datasets/oscar-multilingual/oscar-en-shuffled.jsonl
output=$six_ALL_CCFRSCRATCH/datasets/oscar-multilingual/oscar-meg-mt5-multinode

cd $SCRATCH/Megatron-DeepSpeed
python -m torch.distributed.launch --nproc_per_node 75 --nnodes 8 tools/preprocess_data_dist.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path google/mt5-small \
       --append-eod \
       --scratch $six_ALL_CCFRSCRATCH/datasets/oscar-multilingual/merge_dir

#echo "now copy the results to $six_ALL_CCFRWORK/datasets-custom/oscar/ from $six_ALL_CCFRSCRATCH/datasets/oscar-multilingual/oscar-meg-mt5"
