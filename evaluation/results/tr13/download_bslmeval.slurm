#!/bin/bash
#SBATCH --job-name=download-bslmeval
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

echo "START TIME: $(date)"

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate muennighofflmeval

#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export TOKENIZERS_PARALLELISM=false

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/lm-evaluation-harness

#  GEM/web_nlg_en,GEM/web_nlg_en_challenge_test_numbers,GEM/web_nlg_en_challenge_test_scramble,GEM/web_nlg_en_challenge_validation_sample,GEM/web_nlg_ru,GEM/web_nlg_ru_challenge_test_scramble,GEM/web_nlg_ru_challenge_validation_sample,GEM/wiki_auto_asset_turk_challenge_test_asset_backtranslation,GEM/wiki_auto_asset_turk_challenge_test_asset_bfp02,GEM/wiki_auto_asset_turk_challenge_test_asset_bfp05,GEM/wiki_auto_asset_turk_challenge_test_asset_nopunc,GEM/wiki_auto_asset_turk_challenge_test_turk_backtranslation,GEM/wiki_auto_asset_turk_challenge_test_turk_bfp02,GEM/wiki_auto_asset_turk_challenge_test_turk_bfp05,GEM/wiki_auto_asset_turk_challenge_test_turk_nopunc,GEM/wiki_auto_asset_turk_test_asset,GEM/wiki_auto_asset_turk_test_turk,GEM/wiki_lingua_ar,GEM/wiki_lingua_cs,GEM/wiki_lingua_de,GEM/wiki_lingua_en,GEM/wiki_lingua_es,GEM/wiki_lingua_fr,GEM/wiki_lingua_hi,GEM/wiki_lingua_id,GEM/wiki_lingua_it,GEM/wiki_lingua_ja,GEM/wiki_lingua_ko,GEM/wiki_lingua_nl,GEM/wiki_lingua_pt,GEM/wiki_lingua_ru,GEM/wiki_lingua_th,GEM/wiki_lingua_tr,GEM/wiki_lingua_vi,GEM/wiki_lingua_zh,gem_xsum,gem_xsum_challenge_sample,gem_xsum_challenge_test_backtranslation,gem_xsum_challenge_test_bfp_02,gem_xsum_challenge_test_bfp_05,gem_xsum_challenge_test_covid,gem_xsum_challenge_test_nopunc \
python3 main.py --model hf-causal \
    --model_args pretrained=hf-internal-testing/tiny-random-gpt2,use_accelerate=True,tokenizer=hf-internal-testing/tiny-random-gpt2,dtype=float16 \
    --tasks wmt14_fr_en,wmt19_ru_en,wmt19_zh_en \
    --device cuda \
    --limit 1 \
    --no_cache \
    --num_fewshot 0
