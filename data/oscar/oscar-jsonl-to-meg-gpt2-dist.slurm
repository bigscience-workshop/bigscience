#!/bin/bash
#SBATCH --job-name=oscar-jsonl-to-meg-gpt2-dist # job name
#SBATCH --ntasks=320                 # number of MP tasks
#SBATCH --nodes=8
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=8:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod

input=$six_ALL_CCFRSCRATCH/datasets/oscar-small/oscar-en-shuffled-p1.jsonl
output=$six_ALL_CCFRSCRATCH/datasets/oscar-small/meg-gpt2-dist

##########################
# Setup to run preprocess_data_dist as an mpi4py job
##########################

# Note: if mpi4py works with the system MPI,
# and if srun is configured to launch system MPI jobs,
# then I think the following should be sufficient:
#
# cd $six_ALL_CCFRWORK/code/megatron-lm
# <set CMD just as below>
# srun -n 320 -N 8 python $CMD --mpi4py

##########################
# Setup to run preprocess_data_dist as a torch.distributed job
##########################

# I use a line like the following.  Perhaps easier to read if this works for you?
#MASTER_ADDR=`scontrol show hostname ${SLURM_NODELIST} | head -n1`
MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000

NNODES=$SLURM_NNODES

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node 40 \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

cd $six_ALL_CCFRWORK/code/megatron-lm
CMD="tools/preprocess_data_dist.py \
       --input $input \
       --output-prefix $output \
       --vocab data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod
       "

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

#echo "now copy the results to $six_ALL_CCFRWORK/datasets-custom/oscar/ from $six_ALL_CCFRSCRATCH/datasets/oscar-small/meg-gpt2"
