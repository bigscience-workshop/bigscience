DATASETS_AND_CONFIGS=(
piaf,None,None
GEM/wiki_lingua,ar,ar
GEM/wiki_lingua,en,en
GEM/wiki_lingua,es,es
GEM/wiki_lingua,fr,fr
GEM/wiki_lingua,hi,hi
GEM/wiki_lingua,id,id
GEM/wiki_lingua,pt,pt
GEM/wiki_lingua,vi,vi
GEM/wiki_lingua,zh,zh
GEM/web_nlg,en,en
GEM/web_nlg,ru,ru
wmt14,fr-en,fr-en
)

# Unique ones: 0 1 2 5 6 7 8 9 10 11
for val in {0..12}; do
    DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$val]}
    IFS=',' read dataset_name dataset_config_name template_config_name <<< "${DATASET_AND_CONFIG}"
    echo $dataset_config_name
    python evaluation/results/tr13/tzeroeval/get_templates.py \
            --dataset_name $dataset_name \
            --dataset_config_name $dataset_config_name \
            --template_config_name $template_config_name
done

