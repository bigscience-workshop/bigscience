#!/bin/bash
#SBATCH --job-name=upload-hub-for-loop
#SBATCH --partition=compil
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --output=output-update-hub-for.out
#SBATCH --account=six@cpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

# source younes-model-conversion
# echo $WORK
source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
#source $HOME/start-train
#conda activate younes-eval-opt

module load git-lfs

#PATH_COPY=/gpfsscratch/rech/six/commun/uan68tv-model-conversion/intermediate-checkpoints
PATH_COPY=/gpfsscratch/rech/six/commun/experiments/muennighoff/intermedckpts/ckpts
#PATH_REPO=/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom-176-intermediate
PATH_REPO=/gpfsscratch/rech/six/commun/experiments/muennighoff/intermedckpts/bloom-2b5-intermediate
#PATH_CKPTS=/gpfsdsstore/projects/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/
PATH_CKPTS=/gpfsdsstore/projects/rech/six/commun/checkpoints/tr11c-2B5-ml/checkpoints/main
PATH_CKPTS_NOTTAR=/gpfsscratch/rech/six/commun/checkpoints/tr11c-2B5-ml/checkpoints/main

PATH_CODE_SYNC=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0/bigscience/tools
cd $PATH_COPY
for VARIABLE in global_step100000
do
    eval "cp -r $PATH_CKPTS/$VARIABLE.tar $PATH_COPY/"
    eval "tar xvf $VARIABLE.tar"
    #eval "python /gpfswork/rech/six/uan68tv/code/integration/transformers/src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py --bloom_checkpoint_path $PATH_COPY/$VARIABLE --pytorch_dump_folder_path $PATH_REPO --pretraining_tp 4 --shard_model"
    eval "python /gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/transformers_clone/src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py --bloom_checkpoint_path $PATH_COPY/tr11c-2B5-ml/checkpoints/main/$VARIABLE --pytorch_dump_folder_path $PATH_REPO --pretraining_tp 4 --bloom_config_file /gpfsscratch/rech/six/commun/experiments/muennighoff/intermedckpts/2b5config.json"
    cd $PATH_CODE_SYNC
    eval "python $PATH_CODE_SYNC/hub-sync.py --repo-path $PATH_REPO --patterns '*json'"
    eval "python $PATH_CODE_SYNC/hub-sync.py --repo-path $PATH_REPO --patterns '*bin'"
    cd $PATH_REPO
    eval "git tag -a $VARIABLE -m 'checkpoint from $VARIABLE'"
    eval "git push origin $VARIABLE"
    cd $PATH_COPY
    # pwd
    # eval "$(pwd)"
    # eval "rm -r $VARIABLE.tar"
    # eval "rm -r $VARIABLE"
done



# first update the repo
# cd $PATH_REPO
#eval $SYNC_CMD_JSON
# eval $SYNC_CMD_BIN
# CMD="git push --set-upstream origin gs47400"
# eval $CMD
# eval "git push"
