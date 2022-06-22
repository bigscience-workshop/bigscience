#!/bin/bash
#SBATCH --job-name=tr11c_move_to_tar # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-239%1
#SBATCH --partition=cpu_p1

# DEBUG
# SLURM_ARRAY_TASK_ID=0 # 0-6549

pushd $six_ALL_CCFRWORK/checkpoints
# readarray CHECKPOINTS < <(find . -regex '\./tr11[a-z].*/global_step[0-9]*')
# DEBUG regex to test out only on tr11e-350
# readarray CHECKPOINTS < <(find . -regex '\./tr11e-350M-ml/.*/global_step[0-9]*')
# batch size 512 -> one out of 4 checkpoints for 1B tokens
readarray CHECKPOINTS < <(find . -regex '\./tr11c-2B5-ml/.*/global_step[0-9]*000')

echo "Total number of checkpoints to tar: ${#CHECKPOINTS[@]}"

CHECKPOINT_TO_TAR=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
echo "Checkpoint to tar: $CHECKPOINT_TO_TAR"

TEMPNAME=$(dirname $CHECKPOINT_TO_TAR)
DIRNAME=${TEMPNAME:2}
BASENAME=$(basename $CHECKPOINT_TO_TAR)

CHECKPOINT_TO_TAR=$DIRNAME/$BASENAME
CHECKPOINT_TAR_TO_FOLDER=$six_ALL_CCFRSTORE/checkpoints/$DIRNAME
CHECKPOINT_TAR_TO=$CHECKPOINT_TAR_TO_FOLDER/$BASENAME.tar

mkdir -p $CHECKPOINT_TAR_TO_FOLDER
echo $CHECKPOINT_TO_TAR
echo $CHECKPOINT_TAR_TO

# cvfj for bz2 compression; won't change much
tar cvf $CHECKPOINT_TAR_TO $CHECKPOINT_TO_TAR

popd

