#!/bin/bash
#SBATCH --job-name=meg_ds_zero_gpt2_perf_n16_offload
#SBATCH --constraint=v100-32g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.out            # error file name (same to watch just one file)
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod

nvidia-smi


cd $six_ALL_CCFRWORK/code/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3

CHECKPOINT_PATH=$six_ALL_CCFRWORK/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRWORK/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

# adjust depending on the number of the nodes

NNODES=16
MICRO_BATCH_SIZE=96 # works at 64, OOMs at

# at mbs96 full offload it get cgroup killed - overcomes 40gb/gpu limit

# succeeded:
#MSIZE=30 # @ mbs 16 # gpu ~17gb, cpu 5gb res per gpu, 20TFlops
MSIZE=52 # @ mbs 48 gpu ~30gb, cpu 5gb res per gpu, 43TFlops

# to try:


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

TP_SIZE=4 # always fixed to the size of a single node

# Here TP takes over each nodes so DP sees only 16 "gpus"
# So total batch size is MICRO_BATCH_SIZE*NNODES

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --batch-size $MICRO_BATCH_SIZE \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup 0.01 \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    "

#ZeRO Configs
gradient_accumulation_steps=1
reduce_bucket_size=$(($NHIDDEN*$NHIDDEN))
stage3_prefetch_bucket_size=$(($NHIDDEN*$NHIDDEN*9/10))
stage3_param_persistence_threshold=$((10*$NHIDDEN))

# Here it is different from the other setup
# not using this anymore
#train_batch_size=$(($WORLD_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))

config_json="./ds_zero_stage_3_config.json"


    # "offload_param": {
    #     "device": "none"
    # },
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $gradient_accumulation_steps,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
    },
    "offload_param": {
        "device": "cpu",
        "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": $stage3_prefetch_bucket_size,
    "stage3_param_persitence_threshold": $stage3_param_persistence_threshold,
    "reduce_bucket_size": $reduce_bucket_size,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 10,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": false
}
EOT

MP_SIZE=$TP_SIZE

stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# TiledLinear splits, "true" to enable
TILED_LINEAR="true"
TILE_DIM=1


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${stage} \
    --zero-reduce-bucket-size ${rbs} \
    --zero-allgather-bucket-size ${agbs} \
    "

if [ "${contigious_gradients}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
DEEPSPEED_ARGS="${DEEPSPEED_ARGS} \
    --zero-reduce-scatter"
fi

CHKP_ARGS=" \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

#    --tensor-model-parallel-size $TP_SIZE \
#    --pipeline-model-parallel-size $PP_SIZE \
export CMD=" \
    `pwd`/pretrain_gpt2.py \
    --model-parallel-size $TP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $SAVE_CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
     $CHKP_ARGS \
    "


# clear old checkpoint as it'd mismatch while we sort things out
rm -rf $six_ALL_CCFRWORK/checkpoints/gpt2-meg-ds

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee meg_ds_zero_gpt2_perf_n16_offload.out


#  iteration        2/    1000 | elapsed time per iteration (ms): 122204.8 | learning rate: 3.750E-05 | lm loss: 1.251770E+01 | loss scale: 1024.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
# time (ms) | forward: 34007.93 | backward: 87798.87 | backward-backward: 87798.82 | backward-allreduce: 0.00 | optimizer: 393.85 | batch generator: 3.51
# Effective Tera Flops per GPU: 41.83 and total parameters 52.005 B
