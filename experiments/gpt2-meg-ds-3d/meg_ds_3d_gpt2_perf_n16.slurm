#!/bin/bash
#SBATCH --job-name=meg_ds_3d_gpt2_perf_n16
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

source $ALL_CCFRSCRATCH/start-prod

nvidia-smi

cd ~/base/code/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism

CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/models-custom/megatron-gpt2/megatron_lm_345m_v0.0/release
VOCAB_FILE=$CHECKPOINT_PATH/gpt2-vocab.json
MERGE_FILE=$CHECKPOINT_PATH/gpt2-merges.txt
DATA_PATH=$six_ALL_CCFRSCRATCH/datasets-custom/openwebtext-10k/meg-gpt2_text_document
SAVE_CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

# adjust depending on the number of the nodes

NNODES=16
PP_SIZE=16 # NLAYERS must be a multiple of PP_SIZE here
MICRO_BATCH_SIZE=4 # works at 4, OOMs at
PP_CHUNKS=256 # GAS

MSIZE=52

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

DP_SIZE=1
TP_SIZE=4 # always fixed to the size of a single node

# --gas added by Deepspeed only for PP (this option is not in MegLM master) - same as PP_CHUNKS

#    --micro-batch-size $MICRO_BATCH_SIZE \
#    --lr-warmup-fraction .01 \
#    --global-batch-size $GLOBAL_BATCH_SIZE
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --batch-size $MICRO_BATCH_SIZE \
    --gas $PP_CHUNKS \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-iters 800 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
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
#train_batch_size=$(($DP_SIZE*$MICRO_BATCH_SIZE*$gradient_accumulation_steps))
#  "train_batch_size": $train_batch_size,

config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $PP_CHUNKS,
  "steps_per_print": 1,
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

stage=0
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

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
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
CHKP_ARGS="${CHKP_ARGS} \
        --partition-activations"
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

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

# new names:
#    --tensor-model-parallel-size $TP_SIZE \
#    --pipeline-model-parallel-size $PP_SIZE \
export CMD=" \
    `pwd`/pretrain_gpt2.py \
    --model-parallel-size $TP_SIZE \
    --pipe-parallel-size $PP_SIZE \
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
rm -rf $six_ALL_CCFRSCRATCH/checkpoints/gpt2-meg-ds

# model size
python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"

echo $CMD

# to debug - add echo (it exits and prints what it would have launched)
clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee meg_ds_3d_gpt2_perf_n16.out
