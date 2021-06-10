#!/bin/bash
#SBATCH --job-name=hf_ds_gpt2_base_n4
#SBATCH --constraint=v100-32g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 00:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e
export PYTHONUNBUFFERED=1

source $six_ALL_CCFRWORK/start-prod

nvidia-smi

cd ~/base/code/transformers-clm-any-model-config/

export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

DATASET="stas/openwebtext-10k"

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

# adjust depending on the number of the nodes

NNODES=4
MICRO_BATCH_SIZE=4

# succeeded:

# to try:
MSIZE=25

if   [[ ${MSIZE} == 7 ]];    then NHIDDEN=4096;  NLAYERS=36
elif [[ ${MSIZE} == 14 ]];   then NHIDDEN=6144;  NLAYERS=32
elif [[ ${MSIZE} == 18 ]];   then NHIDDEN=6144;  NLAYERS=40
elif [[ ${MSIZE} == 25 ]];   then NHIDDEN=7168;  NLAYERS=40
elif [[ ${MSIZE} == 30 ]];   then NHIDDEN=7168;  NLAYERS=48
elif [[ ${MSIZE} == 39 ]];   then NHIDDEN=8192;  NLAYERS=48
elif [[ ${MSIZE} == 52 ]];   then NHIDDEN=8192;  NLAYERS=64
elif [[ ${MSIZE} == 65 ]];   then NHIDDEN=9216;  NLAYERS=64
elif [[ ${MSIZE} == 81 ]];   then NHIDDEN=10240; NLAYERS=64
elif [[ ${MSIZE} == 97 ]];   then NHIDDEN=11264; NLAYERS=64
elif [[ ${MSIZE} == 116 ]];  then NHIDDEN=12288; NLAYERS=64
elif [[ ${MSIZE} == 136 ]];  then NHIDDEN=13312; NLAYERS=64
elif [[ ${MSIZE} == 158 ]];  then NHIDDEN=14336; NLAYERS=64
elif [[ ${MSIZE} == 181 ]];  then NHIDDEN=15360; NLAYERS=64
elif [[ ${MSIZE} == 206 ]];  then NHIDDEN=16384; NLAYERS=64
else echo "invalid MSIZE: $MSIZE"
fi


GPUS_PER_NODE=4
NHEADS=32
SEQ_LEN=1024
VOCAB_SIZE=50257

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "


config_json="./ds_z3_cpu_offload.json"
cat <<EOT > $config_json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 8,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e14,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": false
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT

export PYTHONPATH=src
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0

export CMD=" \
    examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --config_overrides "n_embd=$NHIDDEN,n_head=$NHEADS,n_layer=$NLAYERS,n_positions=$SEQ_LEN,gradient_checkpointing=true,use_cache=False" \
    --dataset_name $DATASET \
    --output_dir output_dir \
    --overwrite_output_dir \
    --do_train \
    --max_train_samples 1000 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --num_train_epochs 1 \
    --warmup_steps 8 \
    --fp16 \
    --report_to none \
    --deepspeed $config_json \
    "

# clear old checkpoint as it'd mismatch while we sort things out
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-1-node

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

# to debug - add echo (it exits and prints what it would have launched)
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
