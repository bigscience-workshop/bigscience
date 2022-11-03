DATASETS_AND_CONFIGS=(
super_glue,copa,None
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
story_cloze,2016,None
Muennighoff/xstory_cloze,2016,None
Muennighoff/xstory_cloze,2016,None
Muennighoff/xstory_cloze,2016,None
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
xcopa,id,en
xcopa,sw,en
xcopa,ta,en
xcopa,vi,en
xcopa,zh,en
Muennighoff/xwinograd,en,en
Muennighoff/xwinograd,fr,en
Muennighoff/xwinograd,jp,en
Muennighoff/xwinograd,pt,en
Muennighoff/xwinograd,ru,en
Muennighoff/xwinograd,zh,en
)

# Unique ones: 0 1 2 5 6 7 8 9 10 11
for val in {0..37}; do
    DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$val]}
    IFS=',' read dataset_name dataset_config_name template_config_name <<< "${DATASET_AND_CONFIG}"
    echo $dataset_config_name
    python select_templates.py \
            --dataset_name $dataset_name \
            --dataset_config_name $dataset_config_name \
            --template_config_name $template_config_name
done

