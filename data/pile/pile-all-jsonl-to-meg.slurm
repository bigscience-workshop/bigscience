#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-mt5 # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t4
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

input=$six_ALL_CCFRWORK/datasets-custom/pile/train/pile_train.jsonl
output==$six_ALL_CCFRWORK/datasets-custom/pile/train/pile_train_alpha

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
/usr/bin/time -v python tools/preprocess_data.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path bigscience/oscar_13_languages_alpha_weight \
       --append-eod \
       --workers 25
done
