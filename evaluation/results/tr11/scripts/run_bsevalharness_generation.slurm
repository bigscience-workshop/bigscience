#!/bin/bash
#SBATCH --job-name=genbseval
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --qos=qos_gpu-gc             # up to 100h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                  # number of gpus
#SBATCH --time 100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
conda activate muennighofflmevalgen

echo "START TIME: $(date)"

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Converted transformer checkpoint
MODEL_CKPT=/gpfsscratch/rech/six/commun/uan68tv-model-conversion/bloom

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/bslmevalgeneration/lm-evaluation-harness


DATASETS_AND_CONFIGS = (
piaf,Given_above_context
piaf,after_reading
piaf,context_follow_q
piaf,extract_the_answer
piaf,given_passage_answer
GEM/wiki_lingua_ar,article_summary_ar
GEM/wiki_lingua_ar,rephrase_ar
GEM/wiki_lingua_ar,summarize_above_ar
GEM/wiki_lingua_ar,tldr_ar
GEM/wiki_lingua_ar,write_abstract_ar
GEM/wiki_lingua_en,article_summary_en
GEM/wiki_lingua_en,rephrase_en
GEM/wiki_lingua_en,summarize_above_en
GEM/wiki_lingua_en,tldr_en
GEM/wiki_lingua_en,write_abstract_en
GEM/wiki_lingua_es,article_summary_es
GEM/wiki_lingua_es,rephrase_es
GEM/wiki_lingua_es,summarize_above_es
GEM/wiki_lingua_es,tldr_es
GEM/wiki_lingua_es,write_abstract_es
GEM/wiki_lingua_fr,article_summary_fr
GEM/wiki_lingua_fr,rephrase_fr
GEM/wiki_lingua_fr,summarize_above_fr
GEM/wiki_lingua_fr,tldr_fr
GEM/wiki_lingua_fr,write_abstract_fr
GEM/wiki_lingua_hi,article_summary_hi
GEM/wiki_lingua_hi,rephrase_hi
GEM/wiki_lingua_hi,summarize_above_hi
GEM/wiki_lingua_hi,tldr_hi
GEM/wiki_lingua_hi,write_abstract_hi
GEM/wiki_lingua_id,article_summary_id
GEM/wiki_lingua_id,rephrase_id
GEM/wiki_lingua_id,summarize_above_id
GEM/wiki_lingua_id,tldr_id
GEM/wiki_lingua_id,write_abstract_id
GEM/wiki_lingua_pt,article_summary_pt
GEM/wiki_lingua_pt,rephrase_pt
GEM/wiki_lingua_pt,summarize_above_pt
GEM/wiki_lingua_pt,tldr_pt
GEM/wiki_lingua_pt,write_abstract_pt
GEM/wiki_lingua_vi,article_summary_vi
GEM/wiki_lingua_vi,rephrase_vi
GEM/wiki_lingua_vi,summarize_above_vi
GEM/wiki_lingua_vi,tldr_vi
GEM/wiki_lingua_vi,write_abstract_vi
GEM/wiki_lingua_zh,article_summary_zh
GEM/wiki_lingua_zh,rephrase_zh
GEM/wiki_lingua_zh,summarize_above_zh
GEM/wiki_lingua_zh,tldr_zh
GEM/wiki_lingua_zh,write_abstract_zh
e2e_nlg_cleaned,area_qa
e2e_nlg_cleaned,coherent_text
e2e_nlg_cleaned,create_text_for_me
e2e_nlg_cleaned,eat_type_qa
e2e_nlg_cleaned,family_friendly_yes_no
e2e_nlg_cleaned,food_qa
e2e_nlg_cleaned,generate_gramatically_correct_text
e2e_nlg_cleaned,generate_text_restaurant
e2e_nlg_cleaned,name_qa
e2e_nlg_cleaned,near_qa
e2e_nlg_cleaned,price_range_qa
e2e_nlg_cleaned,rating_qa
GEM/web_nlg_en,PALM_prompt
GEM/web_nlg_en,explicit-graph-description
GEM/web_nlg_en,explicit-graph-description2
GEM/web_nlg_en,implicit-graph-description
GEM/web_nlg_en,non-explicit-description
GEM/web_nlg_en,use-category
GEM/web_nlg_en,very-explicit-description
)



DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

IFS=',' read dataset_name template_name <<< "${DATASET_AND_CONFIG}"

# Use this fork of lm-eval: https://github.com/bigscience-workshop/lm-evaluation-harness/pull/109
python main.py \
    --model_api_name 'hf-causal' \
    --model_args pretrained=$MODEL_CKPT,use_accelerate=True,tokenizer=$MODEL_CKPT,dtype=bfloat16 \
    --device cuda \
    --batch_size 16 \
    --no_tracking \
    --task_name $dataset_name \
    --template_names $template_name \

echo "END TIME: $(date)"
