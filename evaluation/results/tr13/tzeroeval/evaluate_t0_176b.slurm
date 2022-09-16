#!/bin/bash
#SBATCH --job-name=evaluate_t0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --time 20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-164

# VALIDATION:
# --array=0-168

# L1
# --array=0-159

# L2
# --array=0-79

# XNLITR:
# --array=0-79

# Fixed prompts subset:
# --array=0-63

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate thomas_t_zero_evaluation

CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom

WORKDIR=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0

pushd $WORKDIR

OUTPUT_DIR=$CHECKPOINT_PATH/evaluation
mkdir -p $OUTPUT_DIR


DATASETS_AND_CONFIGS_L1=(
super_glue,copa,None,"best_option",validation
super_glue,copa,None,"C1 or C2? premise, so/because…",validation
super_glue,copa,None,"i_am_hesitating",validation
super_glue,copa,None,"cause_effect",validation
super_glue,copa,None,"plausible_alternatives",validation
super_glue,rte,None,"MNLI crowdsource",validation
super_glue,rte,None,"GPT-3 style",validation
super_glue,rte,None,"does it follow that",validation
super_glue,rte,None,"should assume",validation
super_glue,rte,None,"guaranteed true",validation
anli,dev_r1,None,"guaranteed/possible/impossible",dev_r1
anli,dev_r1,None,"MNLI crowdsource",dev_r1
anli,dev_r1,None,"GPT-3 style",dev_r1
anli,dev_r1,None,"justified in saying",dev_r1
anli,dev_r1,None,"can we infer",dev_r1
anli,dev_r2,None,"guaranteed/possible/impossible",dev_r2
anli,dev_r2,None,"MNLI crowdsource",dev_r2
anli,dev_r2,None,"GPT-3 style",dev_r2
anli,dev_r2,None,"justified in saying",dev_r2
anli,dev_r2,None,"can we infer",dev_r2
anli,dev_r3,None,"guaranteed/possible/impossible",dev_r3
anli,dev_r3,None,"MNLI crowdsource",dev_r3
anli,dev_r3,None,"GPT-3 style",dev_r3
anli,dev_r3,None,"justified in saying",dev_r3
anli,dev_r3,None,"can we infer",dev_r3
super_glue,cb,None,"guaranteed/possible/impossible",validation
super_glue,cb,None,"MNLI crowdsource",validation
super_glue,cb,None,"GPT-3 style",validation
super_glue,cb,None,"justified in saying",validation
super_glue,cb,None,"can we infer",validation
winogrande,winogrande_xl,None,"underscore refer to",validation
winogrande,winogrande_xl,None,"Replace",validation
winogrande,winogrande_xl,None,"stand for",validation
winogrande,winogrande_xl,None,"does underscore refer to",validation
winogrande,winogrande_xl,None,"True or False",validation
story_cloze,2016,None,"Story Continuation and Options",validation
story_cloze,2016,None,"Answer Given options",validation
story_cloze,2016,None,"Novel Correct Ending",validation
story_cloze,2016,None,"Generate Ending",validation
story_cloze,2016,None,"Choose Story Ending",validation
Muennighoff/xstory_cloze,ar,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,ar,en,"Answer Given options",validation
Muennighoff/xstory_cloze,ar,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,ar,en,"Generate Ending",validation
Muennighoff/xstory_cloze,ar,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,es,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,es,en,"Answer Given options",validation
Muennighoff/xstory_cloze,es,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,es,en,"Generate Ending",validation
Muennighoff/xstory_cloze,es,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,eu,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,eu,en,"Answer Given options",validation
Muennighoff/xstory_cloze,eu,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,eu,en,"Generate Ending",validation
Muennighoff/xstory_cloze,eu,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,id,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,id,en,"Answer Given options",validation
Muennighoff/xstory_cloze,id,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,id,en,"Generate Ending",validation
Muennighoff/xstory_cloze,id,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,hi,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,hi,en,"Answer Given options",validation
Muennighoff/xstory_cloze,hi,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,hi,en,"Generate Ending",validation
Muennighoff/xstory_cloze,hi,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,zh,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,zh,en,"Answer Given options",validation
Muennighoff/xstory_cloze,zh,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,zh,en,"Generate Ending",validation
Muennighoff/xstory_cloze,zh,en,"Choose Story Ending",validation
xnli,ar,en,"guaranteed/possible/impossible",validation
xnli,ar,en,"MNLI crowdsource",validation
xnli,ar,en,"GPT-3 style",validation
xnli,ar,en,"justified in saying",validation
xnli,ar,en,"can we infer",validation
xnli,en,en,"guaranteed/possible/impossible",validation
xnli,en,en,"MNLI crowdsource",validation
xnli,en,en,"GPT-3 style",validation
xnli,en,en,"justified in saying",validation
xnli,en,en,"can we infer",validation
xnli,es,en,"guaranteed/possible/impossible",validation
xnli,es,en,"MNLI crowdsource",validation
xnli,es,en,"GPT-3 style",validation
xnli,es,en,"justified in saying",validation
xnli,es,en,"can we infer",validation
xnli,fr,en,"guaranteed/possible/impossible",validation
xnli,fr,en,"MNLI crowdsource",validation
xnli,fr,en,"GPT-3 style",validation
xnli,fr,en,"justified in saying",validation
xnli,fr,en,"can we infer",validation
xnli,hi,en,"guaranteed/possible/impossible",validation
xnli,hi,en,"MNLI crowdsource",validation
xnli,hi,en,"GPT-3 style",validation
xnli,hi,en,"justified in saying",validation
xnli,hi,en,"can we infer",validation
xnli,sw,en,"guaranteed/possible/impossible",validation
xnli,sw,en,"MNLI crowdsource",validation
xnli,sw,en,"GPT-3 style",validation
xnli,sw,en,"justified in saying",validation
xnli,sw,en,"can we infer",validation
xnli,ur,en,"guaranteed/possible/impossible",validation
xnli,ur,en,"MNLI crowdsource",validation
xnli,ur,en,"GPT-3 style",validation
xnli,ur,en,"justified in saying",validation
xnli,ur,en,"can we infer",validation
xnli,vi,en,"guaranteed/possible/impossible",validation
xnli,vi,en,"MNLI crowdsource",validation
xnli,vi,en,"GPT-3 style",validation
xnli,vi,en,"justified in saying",validation
xnli,vi,en,"can we infer",validation
xnli,zh,en,"guaranteed/possible/impossible",validation
xnli,zh,en,"MNLI crowdsource",validation
xnli,zh,en,"GPT-3 style",validation
xnli,zh,en,"justified in saying",validation
xnli,zh,en,"can we infer",validation
xcopa,id,en,"best_option",validation
xcopa,id,en,"C1 or C2? premise, so/because…",validation
xcopa,id,en,"i_am_hesitating",validation
xcopa,id,en,"cause_effect",validation
xcopa,id,en,"plausible_alternatives",validation
xcopa,sw,en,"best_option",validation
xcopa,sw,en,"C1 or C2? premise, so/because…",validation
xcopa,sw,en,"i_am_hesitating",validation
xcopa,sw,en,"cause_effect",validation
xcopa,sw,en,"plausible_alternatives",validation
xcopa,ta,en,"best_option",validation
xcopa,ta,en,"C1 or C2? premise, so/because…",validation
xcopa,ta,en,"i_am_hesitating",validation
xcopa,ta,en,"cause_effect",validation
xcopa,ta,en,"plausible_alternatives",validation
xcopa,vi,en,"best_option",validation
xcopa,vi,en,"C1 or C2? premise, so/because…",validation
xcopa,vi,en,"i_am_hesitating",validation
xcopa,vi,en,"cause_effect",validation
xcopa,vi,en,"plausible_alternatives",validation
xcopa,zh,en,"best_option",validation
xcopa,zh,en,"C1 or C2? premise, so/because…",validation
xcopa,zh,en,"i_am_hesitating",validation
xcopa,zh,en,"cause_effect",validation
xcopa,zh,en,"plausible_alternatives",validation
Muennighoff/xwinograd,en,en,"underscore refer to",test
Muennighoff/xwinograd,en,en,"Replace",test
Muennighoff/xwinograd,en,en,"stand for",test
Muennighoff/xwinograd,en,en,"does underscore refer to",test
Muennighoff/xwinograd,en,en,"True or False",test
Muennighoff/xwinograd,fr,en,"underscore refer to",test
Muennighoff/xwinograd,fr,en,"Replace",test
Muennighoff/xwinograd,fr,en,"stand for",test
Muennighoff/xwinograd,fr,en,"does underscore refer to",test
Muennighoff/xwinograd,fr,en,"True or False",test
Muennighoff/xwinograd,pt,en,"underscore refer to",test
Muennighoff/xwinograd,pt,en,"Replace",test
Muennighoff/xwinograd,pt,en,"stand for",test
Muennighoff/xwinograd,pt,en,"does underscore refer to",test
Muennighoff/xwinograd,pt,en,"True or False",test
Muennighoff/xwinograd,zh,en,"underscore refer to",test
Muennighoff/xwinograd,zh,en,"Replace",test
Muennighoff/xwinograd,zh,en,"stand for",test
Muennighoff/xwinograd,zh,en,"does underscore refer to",test
Muennighoff/xwinograd,zh,en,"True or False",test
)

DATASETS_AND_CONFIGS_L2=(
Muennighoff/xstory_cloze,ru,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,ru,en,"Answer Given options",validation
Muennighoff/xstory_cloze,ru,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,ru,en,"Generate Ending",validation
Muennighoff/xstory_cloze,ru,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,my,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,my,en,"Answer Given options",validation
Muennighoff/xstory_cloze,my,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,my,en,"Generate Ending",validation
Muennighoff/xstory_cloze,my,en,"Choose Story Ending",validation
xnli,bg,en,"guaranteed/possible/impossible",validation
xnli,bg,en,"MNLI crowdsource",validation
xnli,bg,en,"GPT-3 style",validation
xnli,bg,en,"justified in saying",validation
xnli,bg,en,"can we infer",validation
xnli,de,en,"guaranteed/possible/impossible",validation
xnli,de,en,"MNLI crowdsource",validation
xnli,de,en,"GPT-3 style",validation
xnli,de,en,"justified in saying",validation
xnli,de,en,"can we infer",validation
xnli,el,en,"guaranteed/possible/impossible",validation
xnli,el,en,"MNLI crowdsource",validation
xnli,el,en,"GPT-3 style",validation
xnli,el,en,"justified in saying",validation
xnli,el,en,"can we infer",validation
xnli,ru,en,"guaranteed/possible/impossible",validation
xnli,ru,en,"MNLI crowdsource",validation
xnli,ru,en,"GPT-3 style",validation
xnli,ru,en,"justified in saying",validation
xnli,ru,en,"can we infer",validation
xnli,th,en,"guaranteed/possible/impossible",validation
xnli,th,en,"MNLI crowdsource",validation
xnli,th,en,"GPT-3 style",validation
xnli,th,en,"justified in saying",validation
xnli,th,en,"can we infer",validation
xnli,tr,en,"guaranteed/possible/impossible",validation
xnli,tr,en,"MNLI crowdsource",validation
xnli,tr,en,"GPT-3 style",validation
xnli,tr,en,"justified in saying",validation
xnli,tr,en,"can we infer",validation
Muennighoff/xwinograd,ru,en,"underscore refer to",test
Muennighoff/xwinograd,ru,en,"Replace",test
Muennighoff/xwinograd,ru,en,"stand for",test
Muennighoff/xwinograd,ru,en,"does underscore refer to",test
Muennighoff/xwinograd,ru,en,"True or False",test
Muennighoff/xwinograd,jp,en,"underscore refer to",test
Muennighoff/xwinograd,jp,en,"Replace",test
Muennighoff/xwinograd,jp,en,"stand for",test
Muennighoff/xwinograd,jp,en,"does underscore refer to",test
Muennighoff/xwinograd,jp,en,"True or False",test
xcopa,et,en,"best_option",validation
xcopa,et,en,"C1 or C2? premise, so/because…",validation
xcopa,et,en,"i_am_hesitating",validation
xcopa,et,en,"cause_effect",validation
xcopa,et,en,"plausible_alternatives",validation
xcopa,ht,en,"best_option",validation
xcopa,ht,en,"C1 or C2? premise, so/because…",validation
xcopa,ht,en,"i_am_hesitating",validation
xcopa,ht,en,"cause_effect",validation
xcopa,ht,en,"plausible_alternatives",validation
xcopa,it,en,"best_option",validation
xcopa,it,en,"C1 or C2? premise, so/because…",validation
xcopa,it,en,"i_am_hesitating",validation
xcopa,it,en,"cause_effect",validation
xcopa,it,en,"plausible_alternatives",validation
xcopa,qu,en,"best_option",validation
xcopa,qu,en,"C1 or C2? premise, so/because…",validation
xcopa,qu,en,"i_am_hesitating",validation
xcopa,qu,en,"cause_effect",validation
xcopa,qu,en,"plausible_alternatives",validation
xcopa,th,en,"best_option",validation
xcopa,th,en,"C1 or C2? premise, so/because…",validation
xcopa,th,en,"i_am_hesitating",validation
xcopa,th,en,"cause_effect",validation
xcopa,th,en,"plausible_alternatives",validation
xcopa,tr,en,"best_option",validation
xcopa,tr,en,"C1 or C2? premise, so/because…",validation
xcopa,tr,en,"i_am_hesitating",validation
xcopa,tr,en,"cause_effect",validation
xcopa,tr,en,"plausible_alternatives",validation
)

DATASETS_AND_CONFIGS_XNLITR=(
xnli,ar,ar,"guaranteed/possible/impossible_arht",validation
xnli,ar,ar,"MNLI crowdsource_arht",validation
xnli,ar,ar,"GPT-3 style_arht",validation
xnli,ar,ar,"justified in saying_arht",validation
xnli,ar,ar,"can we infer_arht",validation
xnli,ar,ar,"guaranteed/possible/impossible_armt",validation
xnli,ar,ar,"MNLI crowdsource_armt",validation
xnli,ar,ar,"GPT-3 style_armt",validation
xnli,ar,ar,"justified in saying_armt",validation
xnli,ar,ar,"can we infer_armt",validation
xnli,es,es,"guaranteed/possible/impossible_esht",validation
xnli,es,es,"MNLI crowdsource_esht",validation
xnli,es,es,"GPT-3 style_esht",validation
xnli,es,es,"justified in saying_esht",validation
xnli,es,es,"can we infer_esht",validation
xnli,es,es,"guaranteed/possible/impossible_esmt",validation
xnli,es,es,"MNLI crowdsource_esmt",validation
xnli,es,es,"GPT-3 style_esmt",validation
xnli,es,es,"justified in saying_esmt",validation
xnli,es,es,"can we infer_esmt",validation
xnli,fr,fr,"guaranteed/possible/impossible_frht",validation
xnli,fr,fr,"MNLI crowdsource_frht",validation
xnli,fr,fr,"GPT-3 style_frht",validation
xnli,fr,fr,"justified in saying_frht",validation
xnli,fr,fr,"can we infer_frht",validation
xnli,fr,fr,"guaranteed/possible/impossible_frmt",validation
xnli,fr,fr,"MNLI crowdsource_frmt",validation
xnli,fr,fr,"GPT-3 style_frmt",validation
xnli,fr,fr,"justified in saying_frmt",validation
xnli,fr,fr,"can we infer_frmt",validation
xnli,hi,hi,"guaranteed/possible/impossible_hiht",validation
xnli,hi,hi,"MNLI crowdsource_hiht",validation
xnli,hi,hi,"GPT-3 style_hiht",validation
xnli,hi,hi,"justified in saying_hiht",validation
xnli,hi,hi,"can we infer_hiht",validation
xnli,hi,hi,"guaranteed/possible/impossible_himt",validation
xnli,hi,hi,"MNLI crowdsource_himt",validation
xnli,hi,hi,"GPT-3 style_himt",validation
xnli,hi,hi,"justified in saying_himt",validation
xnli,hi,hi,"can we infer_himt",validation
xnli,ur,ur,"guaranteed/possible/impossible_urht",validation
xnli,ur,ur,"MNLI crowdsource_urht",validation
xnli,ur,ur,"GPT-3 style_urht",validation
xnli,ur,ur,"justified in saying_urht",validation
xnli,ur,ur,"can we infer_urht",validation
xnli,ur,ur,"guaranteed/possible/impossible_urmt",validation
xnli,ur,ur,"MNLI crowdsource_urmt",validation
xnli,ur,ur,"GPT-3 style_urmt",validation
xnli,ur,ur,"justified in saying_urmt",validation
xnli,ur,ur,"can we infer_urmt",validation
xnli,sw,sw,"guaranteed/possible/impossible_swht",validation
xnli,sw,sw,"MNLI crowdsource_swht",validation
xnli,sw,sw,"GPT-3 style_swht",validation
xnli,sw,sw,"justified in saying_swht",validation
xnli,sw,sw,"can we infer_swht",validation
xnli,sw,sw,"guaranteed/possible/impossible_swmt",validation
xnli,sw,sw,"MNLI crowdsource_swmt",validation
xnli,sw,sw,"GPT-3 style_swmt",validation
xnli,sw,sw,"justified in saying_swmt",validation
xnli,sw,sw,"can we infer_swmt",validation
xnli,vi,vi,"guaranteed/possible/impossible_viht",validation
xnli,vi,vi,"MNLI crowdsource_viht",validation
xnli,vi,vi,"GPT-3 style_viht",validation
xnli,vi,vi,"justified in saying_viht",validation
xnli,vi,vi,"can we infer_viht",validation
xnli,vi,vi,"guaranteed/possible/impossible_vimt",validation
xnli,vi,vi,"MNLI crowdsource_vimt",validation
xnli,vi,vi,"GPT-3 style_vimt",validation
xnli,vi,vi,"justified in saying_vimt",validation
xnli,vi,vi,"can we infer_vimt",validation
xnli,zh,zh,"guaranteed/possible/impossible_zhht",validation
xnli,zh,zh,"MNLI crowdsource_zhht",validation
xnli,zh,zh,"GPT-3 style_zhht",validation
xnli,zh,zh,"justified in saying_zhht",validation
xnli,zh,zh,"can we infer_zhht",validation
xnli,zh,zh,"guaranteed/possible/impossible_zhmt",validation
xnli,zh,zh,"MNLI crowdsource_zhmt",validation
xnli,zh,zh,"GPT-3 style_zhmt",validation
xnli,zh,zh,"justified in saying_zhmt",validation
xnli,zh,zh,"can we infer_zhmt",validation
)

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS_L1[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

# Run T0 evaluation
# For PrefixLM add --prefixlm
# bfloat16 for 176B ; float16 for smaller models
IFS=',' read dataset_name dataset_config_name template_config_name template_name split <<< "${DATASET_AND_CONFIG}"
python t-zero/evaluation/run_eval.py \
        --dataset_name $dataset_name \
        --dataset_config_name $dataset_config_name \
        --template_config_name $template_config_name \
        --template_name "$template_name" \
        --split $split \
        --model_name_or_path $CHECKPOINT_PATH \
        --output_dir $OUTPUT_DIR \
        --per_device_eval_batch_size 8 \
        --max_length 2048 \
        --dtype bfloat16
