#!/bin/bash
#SBATCH --job-name=evaluate_t0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --time 20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-9

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
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmixlossseq
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmix
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3/bloom-6b3
MODEL_CKPT=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/bslmevalgeneration/lm-evaluation-harness

# WMT19 ZH-EN does not work
DATASETS_AND_CONFIGS=(
wmt19_zh_en,zh-en,"version-en-zh-target"
wmt19_zh_en,zh-en,"a_good_translation-en-zh-target"
wmt19_zh_en,zh-en,"a_good_translation-en-zh-source+target"
wmt19_zh_en,zh-en,"xglm-en-zh-target"
wmt19_zh_en,zh-en,"gpt3-en-zh"
wmt19_zh_en,zh-en,"version-zh-en-target"
wmt19_zh_en,zh-en,"a_good_translation-zh-en-target"
wmt19_zh_en,zh-en,"a_good_translation-zh-en-source+target"
wmt19_zh_en,zh-en,"xglm-zh-en-target"
wmt19_zh_en,zh-en,"gpt3-zh-en"
)

DATASETS_AND_CONFIGS=(
wmt14_fr_en,fr-en,"version-en-fr-target"
wmt14_fr_en,fr-en,"a_good_translation-en-fr-target"
wmt14_fr_en,fr-en,"a_good_translation-en-fr-source+target"
wmt14_fr_en,fr-en,"xglm-en-fr-target"
wmt14_fr_en,fr-en,"gpt3-en-fr"
wmt14_fr_en,fr-en,"version-fr-en-target"
wmt14_fr_en,fr-en,"a_good_translation-fr-en-target"
wmt14_fr_en,fr-en,"a_good_translation-fr-en-source+target"
wmt14_fr_en,fr-en,"xglm-fr-en-target"
wmt14_fr_en,fr-en,"gpt3-fr-en"
)

# Use --limit 3000
DATASETS_AND_CONFIGS=(
mlsum_es,"es","layman_summ_es"
mlsum_es,"es","palm_prompt"
mlsum_es,"es","summarise_this_in_es_few_sentences"
)

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

IFS=',' read dataset_name lang template_name <<< "${DATASET_AND_CONFIG}"

# Use this fork of lm-eval: https://github.com/bigscience-workshop/lm-evaluation-harness/pull/109
python main.py \
    --model_api_name 'hf-causal' \
    --model_args pretrained=$MODEL_CKPT,use_accelerate=True,tokenizer=$MODEL_CKPT,dtype=float16 \
    --device cuda \
    --batch_size 16 \
    --no_tracking \
    --task_name $dataset_name \
    --template_names $template_name \
    --bootstrap_iters 10 \
    --limit 3000

echo "END TIME: $(date)"
