#!/bin/bash
#SBATCH --job-name=move_first_checkpoints_to_tar # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --partition=cpu_p1

# DEBUG
# SLURM_ARRAY_TASK_ID=0 # 0-149


# you have to also pass --array=0-<desired_number>%1 as an sbatch flag to compress everything, eg sbatch --array=0-149%1 move_first_150_checkpoints_to_store.slurm tr11b-1B3-ml 150

pushd $six_ALL_CCFRWORK/checkpoints
# readarray CHECKPOINTS < <(find . -regex '\./tr11[a-z].*/global_step[0-9]*')
# DEBUG regex to test out only on tr11e-350
# readarray CHECKPOINTS < <(find . -regex '\./tr11e-350M-ml/.*/global_step[0-9]*')
# batch size 512 -> first 150 checkpoints for 39B tokens, batch size 256 -> 300
readarray CHECKPOINTS < <(ls -v ./"${1}"/checkpoints/main/ | head -"${2}")

echo "Total number of checkpoints to tar: ${#CHECKPOINTS[@]}"

CHECKPOINT_TO_TAR="./${1}/checkpoints/main/${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
echo "Checkpoint to tar: $CHECKPOINT_TO_TAR"

TEMPNAME=$(dirname $CHECKPOINT_TO_TAR)
DIRNAME=${TEMPNAME:2}
BASENAME=$(basename $CHECKPOINT_TO_TAR)

CHECKPOINT_TO_TAR=$DIRNAME/$BASENAME
CHECKPOINT_TAR_TO_FOLDER=$six_ALL_CCFRSTORE/checkpoints/$DIRNAME
CHECKPOINT_TAR_TO=$CHECKPOINT_TAR_TO_FOLDER/$BASENAME.tar

mkdir -p $CHECKPOINT_TAR_TO_FOLDER
echo $CHECKPOINT_TO_TAR
echo $CHECKPOINT_TAR_TO_FOLDER

# cvfj for bz2 compression; won't change much
tar cvf $CHECKPOINT_TAR_TO $CHECKPOINT_TO_TAR

popd
