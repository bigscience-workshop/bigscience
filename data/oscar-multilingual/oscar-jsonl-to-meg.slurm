#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-equal # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

input=$six_ALL_CCFRWORK/datasets-custom/oscar-multilingual/oscar_${1}.jsonl
output=$six_ALL_CCFRWORK/datasets-custom/oscar-multilingual-${2}-tok/oscar_${1}_${2}

cd $ALL_CCFRWORK/code/Megatron-DeepSpeed
/usr/bin/time -v python tools/preprocess_data.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path bigscience/oscar_13_languages_${2}_weight \
       --append-eod \
       --workers 25
done
