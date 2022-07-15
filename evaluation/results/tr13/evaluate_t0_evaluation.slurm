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
#SBATCH --array=0-24

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate thomas_t_zero_evaluation

CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/6b3global_step169250
# CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6B3-ml-t0-lmtoks168B-t0toks0
WORKDIR=$WORK/code/big_science

pushd $WORKDIR

OUTPUT_DIR=$WORK/t_zero_evaluations/mtf-6B3
# OUTPUT_DIR=$WORK/t_zero_evaluations/pretrain-6B3
mkdir -p $OUTPUT_DIR

# TODO @thomasw21 find all the prompts to run.
DATASETS_AND_CONFIGS=(
super_glue,rte,None
anli,dev_r1,None
anli,dev_r2,None
anli,dev_r3,None
super_glue,cb,None
super_glue,rte,None
super_glue,wsc.fixed,None
winogrande,winogrande_xl,None
super_glue,wic,None
hellaswag,None,None
xnli,ar,en
xnli,bg,en
xnli,de,en
xnli,el,en
xnli,en,en
xnli,es,en
xnli,fr,en
xnli,hi,en
xnli,ru,en
xnli,sw,en
xnli,th,en
xnli,tr,en
xnli,ur,en
xnli,vi,en
xnli,zh,en
)

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

# Run T0 evaluation
IFS=',' read dataset_name dataset_config_name template_config_name <<< "${DATASET_AND_CONFIG}"
python t-zero/evaluation/run_eval.py \
        --dataset_name $dataset_name \
        --dataset_config_name $dataset_config_name \
        --template_config_name $template_config_name \
        --model_name_or_path $CHECKPOINT_PATH \
        --output_dir $OUTPUT_DIR
