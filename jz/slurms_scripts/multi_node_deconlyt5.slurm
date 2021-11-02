#!/bin/bash
#SBATCH --job-name=deconlyt5
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1                  # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=40            # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time=50:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%j.out               # output file name
#SBATCH --error=%j.out                # error file name (same to watch just one file)
#SBATCH --account=six@gpu
#SBATCH --mail-type=ALL

GPUS_PER_NODE=8
NNODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

set -x -e

source $six_ALL_CCFRWORK/start-prod

cd $six_ALL_CCFRWORK/code/transformers
export PYTHONPATH=$six_ALL_CCFRWORK/code/transformers
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=13370

export LAUNCHER=" \
    python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

DATASET=openwebtext
LOGG_FREQUENCY=125
SAVE_FREQUENCY=250
EVAL_FREQUENCY=1000
SERIALIZATION_DIR=${ALL_CCFRSCRATCH}/experiments/dec_only_t5-xl-multinode
LOGGING_DIR=${ALL_CCFRSCRATCH}/tensorboard/dec_only_t5-xl-multinode

export CMD=" \
    ${SCRATCH}/code/bigscience/jz/scripts/run_clm.py \
    --deepspeed ${six_ALL_CCFRWORK/code/bigscience/jz/configs/deepspeed/ds_zero3.json \
    --model_type decoder_only_t5 \
    --tokenizer_name t5-small \
    --config_name ${six_ALL_CCFRWORK/code/bigscience/jz/configs/dec_only_t5/decoder_only_t5-xl.json \
    --dataset_name ${DATASET} --block_size 1024 \
    --preprocessing_num_workers 76 \
    --do_train --do_eval \
    --max_steps 34000 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 6e-4 \
    --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1 \
    --warmup_steps 800 \
    --max_grad_norm 1.0 \
    --output_dir ${SERIALIZATION_DIR} --overwrite_output_dir \
    --report_to tensorboard \
    --logging_strategy steps --logging_first_step --logging_dir ${LOGGING_DIR} --logging_steps ${LOGG_FREQUENCY} \
    --eval_steps ${EVAL_FREQUENCY} --evaluation_strategy steps --max_val_samples 10000 \
    --save_strategy steps --save_steps ${SAVE_FREQUENCY} --save_total_limit 200
    "

# to debug - add echo (it exits and prints what it would have launched)
srun bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
