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
GEM/wiki_lingua_ar,ar,"article_summary_ar"
GEM/wiki_lingua_ar,ar,"write_abstract_ar"
GEM/wiki_lingua_ar,ar,"summarize_above_ar"
GEM/wiki_lingua_ar,ar,"rephrase_ar"
GEM/wiki_lingua_ar,ar,"tldr_ar"
GEM/wiki_lingua_en,en,"article_summary_en"
GEM/wiki_lingua_en,en,"write_abstract_en"
GEM/wiki_lingua_en,en,"summarize_above_en"
GEM/wiki_lingua_en,en,"rephrase_en"
GEM/wiki_lingua_en,en,"tldr_en"
GEM/wiki_lingua_es,es,"article_summary_es"
GEM/wiki_lingua_es,es,"write_abstract_es"
GEM/wiki_lingua_es,es,"summarize_above_es"
GEM/wiki_lingua_es,es,"rephrase_es"
GEM/wiki_lingua_es,es,"tldr_es"
GEM/wiki_lingua_fr,fr,"article_summary_fr"
GEM/wiki_lingua_fr,fr,"write_abstract_fr"
GEM/wiki_lingua_fr,fr,"summarize_above_fr"
GEM/wiki_lingua_fr,fr,"rephrase_fr"
GEM/wiki_lingua_fr,fr,"tldr_fr"
GEM/wiki_lingua_hi,hi,"article_summary_hi"
GEM/wiki_lingua_hi,hi,"write_abstract_hi"
GEM/wiki_lingua_hi,hi,"summarize_above_hi"
GEM/wiki_lingua_hi,hi,"rephrase_hi"
GEM/wiki_lingua_hi,hi,"tldr_hi"
GEM/wiki_lingua_id,id,"article_summary_id"
GEM/wiki_lingua_id,id,"write_abstract_id"
GEM/wiki_lingua_id,id,"summarize_above_id"
GEM/wiki_lingua_id,id,"rephrase_id"
GEM/wiki_lingua_id,id,"tldr_id"
GEM/wiki_lingua_pt,pt,"article_summary_pt"
GEM/wiki_lingua_pt,pt,"write_abstract_pt"
GEM/wiki_lingua_pt,pt,"summarize_above_pt"
GEM/wiki_lingua_pt,pt,"rephrase_pt"
GEM/wiki_lingua_pt,pt,"tldr_pt"
GEM/wiki_lingua_vi,vi,"article_summary_vi"
GEM/wiki_lingua_vi,vi,"write_abstract_vi"
GEM/wiki_lingua_vi,vi,"summarize_above_vi"
GEM/wiki_lingua_vi,vi,"rephrase_vi"
GEM/wiki_lingua_vi,vi,"tldr_vi"
GEM/wiki_lingua_zh,zh,"article_summary_zh"
GEM/wiki_lingua_zh,zh,"write_abstract_zh"
GEM/wiki_lingua_zh,zh,"summarize_above_zh"
GEM/wiki_lingua_zh,zh,"rephrase_zh"
GEM/wiki_lingua_zh,zh,"tldr_zh"
)



DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

IFS=',' read dataset_name lang template_name <<< "${DATASET_AND_CONFIG}"

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
