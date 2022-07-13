#!/bin/bash
#SBATCH --job-name=evaluate_t0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --time 20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-0

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate thomas_t_zero_evaluation

# CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/6b3global_step169250
CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/6b3global_step163750
WORKDIR=$WORK/code/big_science

pushd $WORKDIR

# OUTPUT_DIR=$WORK/t_zero_evaluations/mtf-6B3
OUTPUT_DIR=$WORK/t_zero_evaluations/pretrain-6B3
mkdir -p $OUTPUT_DIR

CONFIG_NAMES=(
ar
bg
de
el
en
es
fr
hi
ru
sw
th
tr
ur
vi
zh
)

CONFIG_NAME=${CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}

# Run XNLI evaluation

python eval_t0_deepspeed/run_eval_xnli.py \
	--model_name_or_path $CHECKPOINT_PATH \
	--output_dir $OUTPUT_DIR \
	--dataset_name xnli \
	--dataset_config_name $CONFIG_NAME \
	--dataset_split_name test \
	--per_device_eval_batch_size 1 \
	--max_length 512 \
	--tokenizer_name bigscience/tokenizer
